from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from dockdack import AccountSnapshot, BrokerAPIError, Market, OrderExecution, OrderOutcomeUnknown, OrderRequest, OrderResult, OrderSide, Position, Quote, TradingMode
from dockdack.autotrade import AutoTrader, evaluate_trigger
from dockdack.cli import identify_symbol
from dockdack.gui_service import Instrument
from dockdack.history import DailyBar, DailyHistory
from dockdack.watchlist import MarketSnapshot, TriggerRule, WatchItem, WatchStore


NOW = datetime(2026, 9, 14, 1, tzinfo=timezone.utc)


class FakeTradingService:
    def __init__(self):
        self.mode = TradingMode.DEMO
        self.prices = [Decimal("100")]
        self.submitted = []
        self.positions = ()
        self.open_orders = ()
        self.fills = ()
        self.available = Decimal("10000")
        self.submit_error = None
        self.fail_history = False
        self.quote_calls = self.history_calls = 0
        self.on_account = lambda: None
        self.on_submit = lambda: None

    def ensure_demo(self, instrument):
        if self.mode is not TradingMode.DEMO:
            raise ValueError("모의투자 전용")

    def ensure_common_equity(self, instrument):
        pass

    def resolve(self, symbol, exchange=""):
        market, symbol = identify_symbol(symbol)
        return Instrument(market, symbol, exchange or ("KRX" if market is Market.DOMESTIC else "ND"))

    def history(self, inst, days):
        self.history_calls += 1
        if self.fail_history:
            raise BrokerAPIError("차트 조회 실패")
        bars, day = [], date(2026, 9, 11)
        while len(bars) < days:
            if day.weekday() < 5:
                bars.append(DailyBar(day, Decimal(98), Decimal(102), Decimal(97), Decimal(100), Decimal(1234)))
            day -= timedelta(days=1)
        return DailyHistory(inst.market, inst.symbol, inst.exchange, inst.currency, days, tuple(reversed(bars)))

    def quote(self, inst):
        self.quote_calls += 1
        price = self.prices.pop(0) if len(self.prices) > 1 else self.prices[0]
        return Quote(inst.market, inst.symbol, "테스트 종목", inst.exchange, price, inst.currency)

    def safety_orders(self, inst):
        return self.open_orders

    def safety_account(self, inst):
        self.on_account()
        return AccountSnapshot(inst.market, inst.currency, self.positions, available_to_order=self.available)

    def executions(self, inst):
        return self.fills

    def safety_executions(self, inst):
        return self.fills

    def prepare(self, inst, side, quantity, kind, price):
        assert kind == "limit"
        return OrderRequest(inst.market, OrderSide(side), inst.symbol, quantity, inst.exchange,
                            "0" if inst.market is Market.DOMESTIC else "00", price)

    def submit(self, request):
        self.submitted.append(request)
        self.on_submit()
        if self.submit_error:
            raise self.submit_error
        return OrderResult(True, TradingMode.DEMO, request, "0000200", "접수")


def position(quantity=1, sellable=1):
    return Position(Market.DOMESTIC, "005930", "삼성전자", "KRX", "KRW", Decimal(quantity), Decimal(sellable),
                    Decimal(100), Decimal(100), Decimal(100), Decimal(0), Decimal(0))


class AutoTradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "test.sqlite3")
        self.service = FakeTradingService()
        self.item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "삼성전자", 30)
        self.store.save_item(self.item)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: NOW)

    def rule(self, **kwargs):
        values = dict(kind="price_ge", side="buy", quantity=1, max_notional=Decimal(1000), threshold=Decimal(95))
        values.update(kwargs)
        rule = TriggerRule.create(self.item, **values)
        self.store.add_rule(rule)
        return rule

    def arm(self):
        self.engine.enable_orders("DEMO_AUTOTRADE")

    def test_monitor_only_never_submits_even_when_trigger_matches(self):
        self.rule()
        self.engine.poll()
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())
        self.assertEqual(self.store.rules()[0].status, "ready")
        self.assertEqual(self.service.history_calls, 1)
        self.assertEqual(self.service.quote_calls, 2)

    def test_buy_is_limit_and_one_shot_across_polls_and_restart(self):
        self.rule()
        self.arm()
        self.engine.poll()
        self.engine.poll()
        restarted = AutoTrader(self.service, WatchStore(self.store.path), clock=lambda: NOW)
        restarted.poll()
        self.assertFalse(restarted.orders_enabled)
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.service.submitted[0].price, Decimal(100))
        self.assertEqual(self.service.submitted[0].order_type, "0")
        self.assertEqual(self.store.rules()[0].status, "accepted")

    def test_sell_requires_sellable_position_and_submits_once(self):
        self.rule(side="sell")
        self.service.positions = (position(1, 1),)
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted[0].side, OrderSide.SELL)

    def test_no_short_selling_or_additional_buy_of_owned_stock(self):
        for side, positions in (("sell", ()), ("sell", (position(1, 0),)), ("buy", (position(),))):
            with self.subTest(side=side, positions=positions):
                rule = self.rule(side=side)
                self.service.positions = positions
                self.arm()
                self.engine.poll()
                self.assertEqual(self.service.submitted, [])
                self.store.pause_rule(rule.id)

    def test_cap_and_available_funds_are_checked(self):
        for cap, available in ((Decimal(99), Decimal(10000)), (Decimal(1000), None), (Decimal(1000), Decimal(100))):
            rule = self.rule(max_notional=cap)
            self.service.available = available
            self.arm()
            self.engine.poll()
            self.assertEqual(self.service.submitted, [])
            self.store.pause_rule(rule.id)

    def test_trigger_rechecked_after_account_queries(self):
        self.rule()
        self.service.prices = [Decimal(100), Decimal(90)]
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_stop_during_preflight_prevents_submission(self):
        self.rule()
        self.arm()
        self.service.on_account = self.engine.stop
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])

    def test_ambiguous_response_or_timeout_disarms_and_blocks_restart(self):
        for error in (OrderOutcomeUnknown("missing acknowledgement"), TimeoutError("timeout")):
            with self.subTest(error=error):
                rule = self.rule()
                self.service.submit_error = error
                self.arm()
                self.engine.poll()
                self.assertFalse(self.engine.orders_enabled)
                self.assertEqual(self.store.attempts()[-1]["status"], "unknown")
                restarted = AutoTrader(self.service, self.store, clock=lambda: NOW)
                with self.assertRaises(ValueError):
                    restarted.enable_orders("DEMO_AUTOTRADE")
                before = len(self.service.submitted)
                restarted.poll()
                self.assertEqual(len(self.service.submitted), before)
                self.store.mark_reviewed(rule.id, "CHECKED_ORDER_HISTORY")

    def test_known_rejection_is_terminal_and_not_retried(self):
        self.rule()
        self.service.submit_error = BrokerAPIError("휴장", return_code=2000, status_code=200)
        self.arm()
        self.engine.poll()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.store.rules()[0].status, "rejected")

    def test_uncertain_intent_after_crash_cannot_be_claimed_again(self):
        rule = self.rule()
        self.assertTrue(self.store.claim(rule, Decimal(100), NOW))
        restarted = WatchStore(self.store.path)
        self.assertFalse(restarted.claim(rule, Decimal(100), NOW))
        with self.assertRaises(ValueError):
            self.arm()

    def test_journal_failure_before_send_fails_closed(self):
        self.rule()
        self.arm()
        with patch.object(self.store, "claim", side_effect=OSError("disk full")):
            self.engine.poll()
        self.assertEqual(self.service.submitted, [])

    def test_journal_failure_after_send_retains_intent_and_disarms(self):
        self.rule()
        self.arm()
        with patch.object(self.store, "finish", side_effect=OSError("disk full")):
            self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.store.attempts()[0]["status"], "submitting")

    def test_pending_order_blocks_other_rules_even_if_no_open_order_is_returned(self):
        self.rule()
        self.rule(threshold=Decimal(96))
        self.arm()
        self.engine.poll()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)

    def test_buy_fill_can_unlock_later_sell_rule(self):
        self.rule()
        self.rule(side="sell")
        self.arm()
        self.engine.poll()
        self.service.positions = (position(),)
        self.service.fills = (OrderExecution("0000200", "005930", "매수", "체결", Decimal(1), Decimal(1),
                                            Decimal(0), Decimal(100), Decimal(100), "100000"),)
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 2)
        self.assertEqual(self.service.submitted[-1].side, OrderSide.SELL)
        self.assertEqual(self.store.rules()[0].status, "filled")

    def test_outside_regular_hours_and_live_environment_never_order(self):
        self.rule()
        self.engine.clock = lambda: datetime(2026, 9, 14, 9, tzinfo=timezone.utc)
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.service.mode = TradingMode.REAL
        with self.assertRaises(ValueError):
            self.arm()
        self.engine.clock = lambda: NOW
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])

    def test_sma_uses_completed_days_not_today_and_handles_insufficient_data(self):
        rule = self.rule(kind="sma_ge", period=3)
        history = self.service.history(self.item.instrument, 3)
        today = DailyBar(date(2026, 9, 14), *(Decimal(999) for _ in range(5)))
        snapshot = MarketSnapshot(self.service.quote(self.item.instrument), replace(history, bars=history.bars + (today,)), NOW)
        signal = evaluate_trigger(rule, snapshot, NOW)
        self.assertEqual(signal.reference, Decimal(100))
        self.assertTrue(signal.matched)
        with self.assertRaises(ValueError):
            evaluate_trigger(replace(rule, period=4), snapshot, NOW)
        with self.assertRaises(ValueError):
            evaluate_trigger(rule, snapshot, NOW + timedelta(seconds=16))

    def test_snapshot_roundtrip_preserves_decimal_data(self):
        result = self.engine.poll()[self.item.id]
        loaded = self.store.cached_snapshot(self.item)
        self.assertEqual(loaded.history, result.history)
        self.assertEqual(loaded.quote.price, result.quote.price)
        self.assertEqual(loaded.fetched_at, NOW)

    def test_store_prevents_duplicate_items_and_preserves_attempt_history_on_removal(self):
        self.store.save_item(replace(self.item, days=60))
        self.assertEqual(len(self.store.items()), 1)
        rule = self.rule()
        self.store.claim(rule, Decimal(100), NOW)
        with self.assertRaises(ValueError):
            self.store.remove_item(self.item.id)
        self.store.mark_reviewed(rule.id, "CHECKED_ORDER_HISTORY")
        self.store.remove_item(self.item.id)
        self.assertEqual(self.store.items(), ())
        self.assertEqual(len(self.store.attempts()), 1)
        self.store.save_item(self.item)
        self.assertFalse(self.store.claim(rule, Decimal(100), NOW))

    def test_duplicate_ready_rule_is_rejected(self):
        self.rule()
        with self.assertRaisesRegex(ValueError, "동일한"):
            self.rule()

    def test_rule_validation_and_manual_review_require_explicit_values(self):
        for kwargs in ({"max_notional": Decimal(0)}, {"threshold": Decimal("NaN")}, {"quantity": True}, {"period": 1}):
            with self.assertRaises(ValueError):
                self.rule(**kwargs)
        rule = self.rule()
        self.store.claim(rule, Decimal(100), NOW)
        with self.assertRaises(ValueError):
            self.store.mark_reviewed(rule.id, "")

    def test_separate_connections_cannot_claim_the_same_rule_twice(self):
        rule = self.rule()
        second = WatchStore(self.store.path)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda store: store.claim(rule, Decimal(100), NOW), (self.store, second)))
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(len(self.store.attempts()), 1)

    def test_empty_watchlist_stays_empty_after_restart_instead_of_reseeding(self):
        seeded = WatchStore(Path(self.temp.name) / "seeded.sqlite3", seed_defaults=True)
        self.assertEqual(len(seeded.items()), 4)
        for item in seeded.items():
            seeded.remove_item(item.id)
        self.assertEqual(WatchStore(seeded.path, seed_defaults=True).items(), ())

    def test_us_auto_order_is_also_a_demo_limit_order(self):
        self.store.remove_item(self.item.id)
        self.item = WatchItem(self.service.resolve("AAPL"))
        self.store.save_item(self.item)
        self.rule()
        self.engine.clock = lambda: datetime(2026, 9, 14, 15, tzinfo=timezone.utc)
        self.arm()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.service.submitted[0].order_type, "00")
        self.assertEqual(self.service.submitted[0].market, Market.US)

    def test_price_and_sma_less_equal_include_equality(self):
        for kind in ("price_le", "sma_le"):
            rule = self.rule(kind=kind, threshold=Decimal(100))
            snapshot = MarketSnapshot(self.service.quote(self.item.instrument), self.service.history(self.item.instrument, 30), NOW)
            self.assertTrue(evaluate_trigger(rule, snapshot, NOW).matched)
            self.assertFalse(evaluate_trigger(rule, replace(snapshot, quote=replace(snapshot.quote, price=Decimal(101))), NOW).matched)


if __name__ == "__main__":
    unittest.main()
