"""Offline close-window liquidation proofs; no broker network or real orders."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from dockdack import (AccountSnapshot, BrokerAPIError, Market, OpenOrder, OrderOutcomeUnknown,
                     OrderExecution, OrderSide, Position, TradingMode)
from dockdack.autotrade import AutoTrader
from dockdack.gui_service import Instrument
from dockdack.http import _ORDER_SEND_GUARD
from dockdack.lstm30_close import CloseLiquidator
from dockdack.market_schedule import session_on
from dockdack.signal_bridge import ExternalPolicy
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService


def held(market=Market.US, symbol="HELD", exchange=None, *, quantity=5, sellable=None):
    exchange = exchange or ("KRX" if market is Market.DOMESTIC else "ND")
    sellable = quantity if sellable is None else sellable
    currency = "KRW" if market is Market.DOMESTIC else "USD"
    raw = ({"stk_cd": "A" + symbol, "rmnd_qty": str(quantity), "trde_able_qty": str(sellable)}
           if market is Market.DOMESTIC else
           {"stk_cd": symbol, "stex_nm": {"ND": "NASDAQ", "NY": "NYSE", "NA": "AMEX"}.get(exchange, exchange),
            "crnc_code": currency, "poss_qty": str(quantity), "sell_alowq": str(sellable)})
    return Position(market, symbol, "Verified holding", exchange, currency, Decimal(quantity),
                    Decimal(sellable), Decimal(100), Decimal(500), Decimal(quantity) * 500,
                    Decimal(0), Decimal(0), raw=raw)


class CloseService(FakeTradingService):
    def __init__(self):
        super().__init__()
        self.positions_by_market = {market: () for market in Market}
        self.before_send = lambda: None
        self.account_transform = lambda account: account
        self.account_calls = []
        self.prices = [Decimal(500)]
        self.ensure_common_equity = Mock(side_effect=AssertionError("held liquidation must not use the BUY common-stock filter"))

    def safety_account(self, instrument):
        self.account_calls.append(instrument.market)
        self.on_account()
        positions = self.positions_by_market[instrument.market]
        key = "acnt_evlt_remn_indv_tot" if instrument.market is Market.DOMESTIC else "result_list"
        account = AccountSnapshot(instrument.market, instrument.currency, positions, available_to_order=Decimal(10000),
                                  raw={"balance": [{key: [dict(position.raw) for position in positions]}]})
        return self.account_transform(account)

    def safety_orders(self, instrument):
        return tuple(order for order in self.open_orders
                     if order.market is instrument.market and order.symbol == instrument.symbol)

    def submit(self, request):
        self.before_send()
        guard = _ORDER_SEND_GUARD.get()
        if guard is not None:
            guard()
        return super().submit(request)


class CloseLiquidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "watch.sqlite3")
        self.service = CloseService()
        self.watch = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "Normal model watch", 31)
        self.store.save_item(self.watch)
        self.session = session_on(Market.US, date(2026, 9, 14))
        self.now = self.session.closed - timedelta(minutes=5)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: self.now)
        self.engine.external_only = True
        self.policy = ExternalPolicy("lstm30-mark0", 1, Decimal(500000), Decimal(1000))
        self.engine.external_policy = self.policy
        self.network_patch = patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden"))
        self.network = self.network_patch.start()
        self.addCleanup(self.network_patch.stop)

    def liquidator(self, *, enabled=True, confirmation="DEMO_CLOSE_ALL_SELLABLE", **overrides):
        options = dict(enabled=enabled, confirmation=confirmation, clock=lambda: self.now)
        options.update(overrides)
        return CloseLiquidator(self.service, self.store, self.engine, **options)

    def arm(self):
        self.engine.enable_orders("DEMO_AUTOTRADE")

    def add_holding(self, market=Market.US, symbol="HELD", exchange=None, quantity=5, sellable=None):
        item = held(market, symbol, exchange, quantity=quantity, sellable=sellable)
        self.service.positions_by_market[market] += (item,)
        return item

    def close_attempts(self):
        return tuple(attempt for attempt in self.store.attempts() if attempt["rule_id"].startswith("close-"))

    def test_exact_five_minute_boundary_is_inclusive_but_exchange_close_is_exclusive(self):
        liquidator = self.liquidator()
        for market in Market:
            session = session_on(market, date(2026, 9, 14))
            for moment, expected in ((session.closed - timedelta(minutes=5, microseconds=1), False),
                                     (session.closed - timedelta(minutes=5), True),
                                     (session.closed - timedelta(microseconds=1), True),
                                     (session.closed, False)):
                with self.subTest(market=market, moment=moment):
                    self.assertEqual(market in liquidator.closing_markets(moment), expected)
                    # SELL ends exactly at close; new BUY remains blocked.
                    self.assertEqual(liquidator.buy_blocked(market, moment),
                                     moment >= session.closed - timedelta(minutes=5))

    def test_holidays_dst_and_early_close_use_actual_exchange_sessions(self):
        liquidator = self.liquidator()
        for day, expected_utc_hour in ((date(2026, 7, 6), 19), (date(2026, 11, 30), 20),
                                       (date(2026, 11, 27), 17)):
            session = session_on(Market.US, day)
            starts = session.closed - timedelta(minutes=5)
            self.assertEqual(starts.astimezone(timezone.utc).hour, expected_utc_hour)
            self.assertIn(Market.US, liquidator.closing_markets(starts))
        self.assertNotIn(Market.US, liquidator.closing_markets(datetime(2026, 9, 7, 19, 56, tzinfo=timezone.utc)))
        self.assertNotIn(Market.DOMESTIC, liquidator.closing_markets(datetime(2026, 9, 25, 6, 26, tzinfo=timezone.utc)))
        self.assertTrue(liquidator.buy_blocked(Market.US, datetime(2026, 9, 7, 19, 56, tzinfo=timezone.utc)))
        self.assertTrue(liquidator.buy_blocked(Market.DOMESTIC, datetime(2026, 9, 25, 6, 26, tzinfo=timezone.utc)))

    def test_disabled_has_no_buy_block_or_account_queries_or_orders(self):
        liquidator = self.liquidator(enabled=False, confirmation=None)
        self.add_holding()
        self.arm()
        self.assertFalse(liquidator.buy_blocked(Market.US))
        liquidator.tick()
        self.assertEqual(self.service.account_calls, [])
        self.assertEqual(self.service.submitted, [])

    def test_enabled_requires_exact_separate_confirmation(self):
        for confirmation in (None, "", "DEMO_AUTOTRADE", "REAL_AUTOTRADE", True):
            with self.subTest(confirmation=confirmation), self.assertRaises(ValueError):
                self.liquidator(confirmation=confirmation)
        self.assertEqual(self.service.account_calls, [])

    def test_real_service_is_rejected_before_any_account_or_order_call(self):
        self.service.mode = TradingMode.REAL
        with self.assertRaises(ValueError):
            self.liquidator()
        self.assertEqual(self.service.account_calls, [])
        self.assertEqual(self.service.submitted, [])

    def test_engine_off_never_liquidates_even_when_feature_is_authorized(self):
        self.add_holding()
        self.liquidator().tick()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts(), ())

    def test_all_sellable_nonwatch_holding_exceeds_normal_caps_without_changing_them(self):
        self.add_holding(quantity=7)
        before = self.store.items()
        self.arm()
        self.liquidator().tick()
        self.assertEqual(len(self.service.submitted), 1)
        order = self.service.submitted[0]
        self.assertEqual((order.side, order.symbol, order.quantity), (OrderSide.SELL, "HELD", 7))
        self.assertGreater(order.estimated_notional, self.policy.max_usd)
        self.assertEqual(self.engine.external_policy, self.policy)
        self.assertEqual(self.engine.external_policy.max_quantity, 1)
        self.assertEqual(self.store.items(), before)
        self.assertEqual(self.close_attempts()[0]["status"], "accepted")
        self.assertFalse(any(rule.status == "ready" for rule in self.store.rules()))
        self.service.ensure_common_equity.assert_not_called()

    def test_domestic_all_holdings_also_use_sellable_quantity_above_one(self):
        self.session = session_on(Market.DOMESTIC, date(2026, 9, 14))
        self.now = self.session.closed - timedelta(minutes=5)
        self.add_holding(Market.DOMESTIC, "006400", quantity=3)
        self.service.prices = [Decimal(250000)]
        self.arm()
        self.liquidator().tick()
        self.assertEqual(len(self.service.submitted), 1)
        order = self.service.submitted[0]
        self.assertEqual((order.market, order.side, order.quantity), (Market.DOMESTIC, OrderSide.SELL, 3))
        self.assertGreater(order.estimated_notional, self.policy.max_krw)
        self.assertEqual(self.store.items(), (self.watch,))

    def test_verified_held_etf_does_not_use_buy_common_equity_filter(self):
        self.add_holding(symbol="SPY", exchange="NA", quantity=4)
        self.arm()
        self.liquidator().tick()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.service.submitted[0].symbol, "SPY")
        self.service.ensure_common_equity.assert_not_called()

    def test_every_supported_account_holding_is_considered_not_only_watchlist_members(self):
        self.add_holding(symbol="HELD", quantity=7)
        self.add_holding(symbol="SPY", exchange="NA", quantity=4)
        self.arm()
        self.liquidator().tick()
        self.assertEqual({order.symbol: order.quantity for order in self.service.submitted}, {"HELD": 7, "SPY": 4})
        self.assertTrue(all(order.side is OrderSide.SELL for order in self.service.submitted))
        self.assertEqual(self.store.items(), (self.watch,))

    def test_partial_unsellable_holding_sells_only_available_and_reports_remainder(self):
        self.add_holding(quantity=5, sellable=3)
        self.arm()
        status = self.liquidator().tick()
        self.assertEqual(self.service.submitted[0].quantity, 3)
        rows = [row for row in status["unsold"] if row["symbol"] == "HELD"]
        self.assertTrue(rows)
        self.assertEqual(Decimal(rows[0]["quantity"]) - Decimal(rows[0]["sellable_quantity"]), Decimal(2))
        self.assertEqual(rows[0]["reason"], "ACCEPTED_PENDING_FILL")

    def test_zero_sellable_position_never_creates_an_order(self):
        self.add_holding(quantity=5, sellable=0)
        self.arm()
        status = self.liquidator().tick()
        self.assertEqual(self.service.submitted, [])
        self.assertTrue(status["unsold"])

    def test_unknown_exchange_is_reported_not_guessed(self):
        self.add_holding(exchange="UNKNOWN")
        self.arm()
        status = self.liquidator().tick()
        self.assertEqual(self.service.submitted, [])
        self.assertTrue(status["unsold"] or status["errors"])

    def test_exact_korean_venue_names_sell_all_validated_holdings_on_correct_venues(self):
        for symbol, exchange, quantity in (("NASDAQH", "나스닥", 7), ("NYSEH", "뉴욕", 4),
                                           ("AMEXH", "아멕스", 3)):
            self.add_holding(symbol=symbol, exchange=exchange, quantity=quantity)
        self.arm()
        self.liquidator().tick()
        self.assertEqual({order.symbol: (order.exchange, order.quantity) for order in self.service.submitted},
                         {"NASDAQH": ("ND", 7), "NYSEH": ("NY", 4), "AMEXH": ("NA", 3)})
        self.assertTrue(all(order.side is OrderSide.SELL for order in self.service.submitted))
        self.assertEqual(self.engine.external_policy, self.policy)
        self.assertEqual(self.store.items(), (self.watch,))

    def test_country_label_and_unknown_code_are_not_guessed_as_nasdaq(self):
        self.add_holding(symbol="COUNTRY", exchange="미국", quantity=7)
        self.add_holding(symbol="UNKNOWN", exchange="NP", quantity=4)
        self.arm()
        status = self.liquidator().tick()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts(), ())
        self.assertEqual({row["exchange"] for row in status["unsold"]}, {"미국", "NP"})
        self.assertTrue(all(row["reason"] == "UNSUPPORTED_INSTRUMENT_OR_VENUE" for row in status["unsold"]))

    def test_fresh_preflight_balance_reduces_sale_when_sellable_quantity_changes(self):
        self.add_holding(quantity=7)

        def changed_balance():
            if len(self.service.account_calls) == 2:
                self.service.positions_by_market[Market.US] = (held(quantity=7, sellable=2),)

        self.service.on_account = changed_balance
        self.arm()
        self.liquidator().tick()
        self.assertGreaterEqual(len(self.service.account_calls), 2)
        self.assertEqual([order.quantity for order in self.service.submitted], [2])

    def test_actual_lstm_engine_accepts_inactive_close_rows_but_rejects_foreign_active_watch(self):
        from dockdack.lstm30_gui import _LSTM30AutoTrader

        self.engine = _LSTM30AutoTrader(self.service, self.store, items=(self.watch,), clock=lambda: self.now)
        self.engine.external_only = True
        self.engine.external_policy = self.policy
        self.add_holding(quantity=7)
        self.arm()
        liquidator = self.liquidator()
        liquidator.tick()
        liquidator.tick()
        self.assertTrue(self.engine.orders_enabled)
        self.assertEqual(len(self.service.submitted), 1)
        self.store.save_item(WatchItem(Instrument(Market.US, "FOREIGN", "ND"), "Not approved", 31))
        self.service.positions_by_market[Market.US] = (held(symbol="NEWHELD", quantity=3),)
        status = liquidator.tick()
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(len(self.service.submitted), 1)
        self.assertTrue(status["errors"])

    def test_actual_lstm_scope_change_at_final_paced_send_blocks_liquidation(self):
        from dockdack.lstm30_gui import _LSTM30AutoTrader

        self.engine = _LSTM30AutoTrader(self.service, self.store, items=(self.watch,), clock=lambda: self.now)
        self.engine.external_only = True
        self.engine.external_policy = self.policy
        self.add_holding(quantity=7)
        self.arm()
        self.service.before_send = lambda: self.store.save_item(
            WatchItem(Instrument(Market.US, "FOREIGN", "ND"), "Not approved", 31))
        self.liquidator().tick()
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts()[0]["status"], "not_sent")

    def test_midrun_real_environment_switch_reports_error_and_disarms_without_sending(self):
        liquidator = self.liquidator()
        self.add_holding()
        self.arm()
        self.service.mode = TradingMode.REAL
        status = liquidator.tick()
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.service.account_calls, [])
        self.assertIn("ENVIRONMENT_MISMATCH", {row["reason"] for row in status["errors"]})

    def test_raw_missing_negative_or_normalization_mismatch_fail_closed(self):
        for kind in ("missing", "negative", "mismatch"):
            with self.subTest(kind=kind):
                self.service.positions_by_market[Market.US] = (held(),)

                def transform(account):
                    if kind == "missing":
                        return replace(account, raw={})
                    row = dict(account.raw["balance"][0]["result_list"][0])
                    row["poss_qty"] = "-5" if kind == "negative" else "6"
                    return replace(account, raw={"balance": [{"result_list": [row]}]})

                self.service.account_transform = transform
                self.arm()
                status = self.liquidator().tick()
                self.assertEqual(self.service.submitted, [])
                self.assertFalse(self.engine.orders_enabled)
                self.assertTrue(status["errors"])

    def test_existing_open_order_blocks_close_liquidation_without_cancelling(self):
        self.add_holding()
        self.service.open_orders = (OpenOrder(Market.US, "existing", "HELD", "Holding", "ND", "sell", "open",
                                             Decimal(1), Decimal(0), Decimal(1), Decimal(500)),)
        self.arm()
        status = self.liquidator().tick()
        self.assertEqual(self.service.submitted, [])
        self.assertTrue(status["unsold"])

    def test_pending_local_intent_blocks_close_for_that_symbol(self):
        self.add_holding(symbol="AAPL")
        item = WatchItem(Instrument(Market.US, "AAPL", "ND"), "Apple", 31)
        self.store.save_item(item)
        rule = TriggerRule.create(item, "price_ge", "sell", 1, Decimal(1000), Decimal(100))
        self.store.add_rule(rule)
        self.store.claim(rule, Decimal(500), self.now)
        self.store.finish(rule.id, "accepted", "already sent", "existing")
        self.arm()
        self.liquidator().tick()
        self.assertEqual(self.service.submitted, [])

    def test_restart_and_repeated_ticks_do_not_resubmit_an_accepted_intent(self):
        self.add_holding()
        self.arm()
        first = self.liquidator()
        first.tick()
        first.tick()
        self.liquidator().tick()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(len(self.close_attempts()), 1)

    def test_full_fill_is_reconciled_without_resubmitting_or_claiming_flatness(self):
        self.add_holding()
        self.arm()
        liquidator = self.liquidator()
        liquidator.tick()
        number = self.close_attempts()[0]["order_number"]
        self.service.fills = (OrderExecution(number, "HELD", "sell", "filled", Decimal(5), Decimal(5),
                                            Decimal(0), Decimal(500), Decimal(500), "155601"),)
        liquidator.tick()
        self.assertEqual(self.close_attempts()[0]["status"], "filled")
        self.assertEqual(len(self.service.submitted), 1)

    def test_wrong_side_or_duplicate_fill_is_not_confirmed_and_disarms(self):
        self.add_holding()
        self.arm()
        liquidator = self.liquidator()
        liquidator.tick()
        number = self.close_attempts()[0]["order_number"]
        valid = OrderExecution(number, "HELD", "sell", "filled", Decimal(5), Decimal(5),
                               Decimal(0), Decimal(500), Decimal(500), "155601")
        for fills in ((replace(valid, side="buy"),), (valid, valid)):
            with self.subTest(fills=fills):
                self.service.fills = fills
                self.arm()
                status = liquidator.tick()
                self.assertFalse(self.engine.orders_enabled)
                self.assertEqual(self.close_attempts()[0]["status"], "accepted")
                self.assertEqual(len(self.service.submitted), 1)
                self.assertTrue(status["errors"])

    def test_full_fill_after_close_reconciles_and_reports_actual_empty_account_without_new_orders(self):
        self.add_holding()
        self.arm()
        liquidator = self.liquidator()
        liquidator.tick()
        number = self.close_attempts()[0]["order_number"]
        self.service.fills = (OrderExecution(number, "HELD", "sell", "filled", Decimal(5), Decimal(5),
                                            Decimal(0), Decimal(500), Decimal(500), "155959"),)
        self.service.positions_by_market[Market.US] = ()
        self.now = self.session.closed + timedelta(seconds=1)
        status = liquidator.tick()
        self.assertEqual(self.close_attempts()[0]["status"], "filled")
        self.assertEqual(status["unsold"], [])
        self.assertEqual(len(self.service.submitted), 1)

    def test_known_rejection_still_cannot_retry_after_a_new_explicit_demo_on(self):
        self.add_holding()
        self.service.submit_error = BrokerAPIError("fake rejection", return_code=2000, status_code=200)
        self.arm()
        self.liquidator().tick()
        self.assertEqual(self.close_attempts()[0]["status"], "rejected")
        self.service.submit_error = None
        self.arm()
        self.liquidator().tick()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(len(self.close_attempts()), 1)

    def test_rejection_or_unknown_never_retries_after_restart(self):
        for failure in (BrokerAPIError("fake rejection", return_code=2000, status_code=200),
                        OrderOutcomeUnknown("fake acknowledgement missing")):
            with self.subTest(failure=type(failure).__name__):
                symbol = "REJECT" if isinstance(failure, BrokerAPIError) and not isinstance(failure, OrderOutcomeUnknown) else "UNKNOWN"
                self.service.positions_by_market[Market.US] = (held(symbol=symbol),)
                self.service.submit_error = failure
                self.arm()
                before = len(self.service.submitted)
                first = self.liquidator()
                first.tick()
                first.tick()
                self.liquidator().tick()
                self.assertEqual(len(self.service.submitted), before + 1)
                self.assertFalse(self.engine.orders_enabled)

    def test_off_or_stop_during_final_pacer_hook_prevents_send(self):
        for action in (self.engine.disarm, self.engine.stop):
            with self.subTest(action=action.__name__):
                symbol = "OFF" if action.__name__ == "disarm" else "STOP"
                self.service.positions_by_market[Market.US] = (held(symbol=symbol),)
                self.arm()
                self.service.before_send = action
                self.liquidator().tick()
                self.assertEqual(self.service.submitted, [])
                self.assertEqual(self.close_attempts()[-1]["status"], "not_sent")

    def test_final_pacer_quote_or_account_age_beyond_15_seconds_blocks_send(self):
        self.add_holding()
        self.arm()
        self.service.before_send = lambda: setattr(self, "now", self.now + timedelta(seconds=16))
        self.liquidator().tick()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts()[0]["status"], "not_sent")

    def test_sell_final_database_validation_delay_rechecks_freshness_before_send(self):
        self.add_holding()
        self.arm()
        liquidator = self.liquidator()
        state = {"sending": False, "delayed": False}
        self.service.before_send = lambda: state.update(sending=True)
        original_connection = self.store.connection

        @contextmanager
        def delayed_connection():
            with original_connection() as db:
                yield db
            if state["sending"] and not state["delayed"]:
                state["delayed"] = True
                self.now += timedelta(seconds=16)

        with patch.object(self.store, "connection", delayed_connection):
            liquidator.tick()
        self.assertTrue(state["delayed"], "delay must occur inside the installed final send guard")
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts()[0]["status"], "not_sent")

    def test_sell_off_during_last_environment_permission_prevents_send(self):
        self.add_holding()
        self.arm()
        liquidator = self.liquidator()
        state = {"sending": False, "checks": 0}
        self.service.before_send = lambda: state.update(sending=True)
        original_environment = self.engine._ensure_environment

        def off_after_last_environment(*args, **kwargs):
            original_environment(*args, **kwargs)
            if state["sending"]:
                state["checks"] += 1
                if state["checks"] == 2:
                    self.engine.disarm()

        with patch.object(self.engine, "_ensure_environment", off_after_last_environment):
            liquidator.tick()
        self.assertEqual(state["checks"], 2)
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts()[0]["status"], "not_sent")

    def test_sell_final_calendar_delay_crossing_exchange_close_prevents_send(self):
        self.add_holding()
        self.now = self.session.closed - timedelta(seconds=1)
        self.arm()
        liquidator = self.liquidator()
        state = {"sending": False, "lookups": 0}
        self.service.before_send = lambda: state.update(sending=True)

        def delayed_calendar(*args, **kwargs):
            session = session_on(*args, **kwargs)
            if state["sending"]:
                state["lookups"] += 1
                if state["lookups"] == 2:
                    self.now += timedelta(seconds=2)
            return session

        with patch("dockdack.lstm30_close.session_on", delayed_calendar):
            liquidator.tick()
        self.assertGreaterEqual(state["lookups"], 2)
        self.assertGreater(self.now, self.session.closed)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts()[0]["status"], "not_sent")

    def _poll_buy_with_final_calendar_effect(self, effect):
        """Exercise the real GUI engine, normal preflight and installed send guard."""
        from dockdack.lstm30_gui import _LSTM30AutoTrader

        item = WatchItem(Instrument(Market.US, "AAPL", "ND"), "Apple", 31)
        self.store.save_item(item)
        rule = TriggerRule.create(item, "price_ge", "buy", 1, Decimal(1000), Decimal(100))
        self.store.add_rule(rule)
        self.service.ensure_common_equity = Mock()
        self.engine = _LSTM30AutoTrader(self.service, self.store, items=self.store.items(), clock=lambda: self.now)
        self.engine.close_liquidator = self.liquidator()
        self.arm()
        state = {"sending": False, "lookups": 0, "applied": False}
        self.service.before_send = lambda: state.update(sending=True)

        def final_calendar(*args, **kwargs):
            session = session_on(*args, **kwargs)
            if state["sending"]:
                state["lookups"] += 1
                if state["lookups"] == 2:
                    state["applied"] = True
                    effect()
            return session

        with patch("dockdack.lstm30_close.session_on", final_calendar):
            self.engine.poll()
        self.assertTrue(state["applied"], "the second closing check must run after the real parent guard")
        self.assertEqual(self.service.submitted, [])
        attempts = self.store.attempts(item.id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual((attempts[0]["rule_id"], attempts[0]["status"]), (rule.id, "not_sent"))
        self.assertEqual(self.close_attempts(), ())

    def test_buy_final_calendar_delay_crossing_five_minute_cutoff_prevents_send(self):
        self.now = self.session.closed - timedelta(minutes=5, seconds=1)
        self._poll_buy_with_final_calendar_effect(lambda: setattr(self, "now", self.now + timedelta(seconds=2)))
        self.assertGreater(self.now, self.session.closed - timedelta(minutes=5))

    def test_buy_off_during_post_parent_final_calendar_prevents_send(self):
        self.now = self.session.closed - timedelta(minutes=6)
        self._poll_buy_with_final_calendar_effect(lambda: self.engine.disarm())
        self.assertFalse(self.engine.orders_enabled)

    def test_buy_final_calendar_delay_rechecks_quote_freshness_before_send(self):
        self.now = self.session.closed - timedelta(minutes=6)
        self._poll_buy_with_final_calendar_effect(lambda: setattr(self, "now", self.now + timedelta(seconds=16)))
        self.assertLess(self.now, self.session.closed - timedelta(minutes=5),
                        "freshness must block this order independently of the closing cutoff")

    def test_market_closes_during_final_pacer_and_no_order_is_sent_after_close(self):
        self.add_holding()
        self.now = self.session.closed - timedelta(seconds=1)
        self.arm()
        self.service.before_send = lambda: setattr(self, "now", self.session.closed)
        liquidator = self.liquidator()
        liquidator.tick()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.close_attempts()[0]["status"], "not_sent")
        liquidator.tick()
        self.assertEqual(self.service.submitted, [])

    def test_before_window_and_after_close_do_not_scan_or_send(self):
        self.add_holding()
        self.arm()
        liquidator = self.liquidator()
        for moment in (self.session.closed - timedelta(minutes=6), self.session.closed,
                       self.session.closed + timedelta(minutes=5)):
            self.now = moment
            liquidator.tick()
        self.assertEqual(self.service.account_calls, [])
        self.assertEqual(self.service.submitted, [])


if __name__ == "__main__":
    unittest.main()
