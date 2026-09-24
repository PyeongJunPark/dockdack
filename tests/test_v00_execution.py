"""Offline v0.0 order orchestration; no transport or configured API credentials."""

import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dockdack.autotrade import AutoTrader
from dockdack.execution_policy import account_equity, allocation_quantity, holding_exit_targets
from dockdack.models import AccountSnapshot, Market, OrderSide, TradingMode, OrderResult
from dockdack.exceptions import BrokerAPIError, OrderNotSent, OrderOutcomeUnknown
from dockdack.signal_bridge import ExternalPolicy, export_charts, ingest_signals, validate_exit_conditions
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position


class V00Service(FakeTradingService):
    def __init__(self):
        super().__init__()
        self.account_calls = []
        self.cash = Decimal("40000")
        self.evaluation = Decimal("60000")
        self.available = Decimal("40000")

    def safety_account(self, inst):
        self.account_calls.append(inst.market)
        self.on_account()
        held = tuple(p for p in self.positions if p.market is inst.market)
        return AccountSnapshot(inst.market, inst.currency, held, cash=self.cash,
                               total_evaluation=self.evaluation, available_to_order=self.available)


class V00ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = WatchStore(Path(self.folder.name) / "offline.sqlite3")
        self.service = V00Service()
        self.engine = AutoTrader(self.service, self.store, clock=lambda: NOW)
        self.engine.isolated_symbol_errors = True
        self.item = WatchItem(self.service.resolve("005930"))
        self.store.save_item(self.item)
        self.sessions = patch("dockdack.autotrade.regular_session", side_effect=lambda market, now: market is Market.DOMESTIC)
        self.sessions.start()
        self.addCleanup(self.sessions.stop)

    def rule(self, item=None, **changes):
        item = item or self.item
        fields = dict(kind="price_ge", side="buy", quantity=1, max_notional=Decimal("1000000"), threshold=Decimal("95"))
        fields.update(changes)
        rule = TriggerRule.create(item, **fields)
        self.store.add_rule(rule)
        return rule

    def arm(self):
        self.engine.enable_orders("DEMO_AUTOTRADE")

    def use_us(self):
        self.sessions.stop()
        self.sessions = patch("dockdack.autotrade.regular_session", side_effect=lambda market, now: market is Market.US)
        self.sessions.start()
        self.addCleanup(self.sessions.stop)
        self.item = WatchItem(self.service.resolve("AAPL"))
        self.store.save_item(self.item)

    def external(self, *, source="model-a", **values):
        self.engine.snapshot(self.item)
        chart = export_charts(self.store, Path(self.folder.name) / "chart.json", now=NOW)
        policy = ExternalPolicy(source, 999999999, Decimal("1000000"), Decimal("1000000"))
        entry = dict(signal_id=source + "-1", export_id=chart["export_id"], market=self.item.instrument.market.value,
                     symbol=self.item.instrument.symbol, exchange=self.item.instrument.exchange, action="buy",
                     quantity=1, max_notional="1000000", generated_at=NOW.isoformat(), expires_at=(NOW + timedelta(minutes=2)).isoformat())
        entry.update(values)
        ingest_signals(self.store, dict(schema_version=1, source_id=source, signals=[entry]), policy, now=NOW)
        self.engine.external_only = True
        self.engine.external_policy = policy
        return policy

    def test_closed_market_makes_no_quote_history_or_account_requests(self):
        self.sessions.stop()
        with patch("dockdack.autotrade.regular_session", return_value=False):
            self.engine.enable_holdings_exits = True
            self.assertEqual(self.engine.poll(), {})
        self.assertEqual((self.service.quote_calls, self.service.history_calls, self.service.account_calls), (0, 0, []))

    def test_only_open_market_is_polled(self):
        self.store.save_item(WatchItem(self.service.resolve("AAPL")))
        result = self.engine.poll()
        self.assertEqual(tuple(result), (self.item.id,))
        self.assertEqual(self.service.quote_calls, 1)

    def test_holdings_absent_from_watchlist_sell_without_chart_request(self):
        self.store.remove_item(self.item.id)
        self.engine.enable_holdings_exits = True
        self.service.positions = (position(3, 3),)
        self.service.prices = [Decimal("102")]
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.history_calls, 0)
        self.assertEqual(self.store.items(), ())
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.service.submitted[0].side, OrderSide.SELL)
        self.assertEqual(self.service.submitted[0].quantity, 3)
        self.assertEqual(self.store.attempts()[0]["status"], "accepted")

    def test_independent_holdings_stop_loss(self):
        self.engine.enable_holdings_exits = True
        self.service.positions = (position(),)
        self.service.prices = [Decimal("99")]
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted[0].side, OrderSide.SELL)

    def test_holding_target_fresh_recheck_prevents_sell(self):
        self.store.remove_item(self.item.id)
        self.engine.enable_holdings_exits = True
        self.service.positions = (position(),)
        self.service.prices = [Decimal("102"), Decimal("100")]
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_holdings_pass_runs_when_watch_history_fails(self):
        self.engine.enable_holdings_exits = True
        self.service.fail_history = True
        self.service.positions = (position(),)
        self.service.prices = [Decimal("102")]
        self.arm()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertTrue(self.engine.orders_enabled)

    def test_independent_sell_pass_precedes_new_buy_scan(self):
        self.engine.enable_holdings_exits = True
        other = replace(position(), symbol="000660")
        self.service.positions = (other,)
        self.service.prices = [Decimal("102")]
        self.rule()
        self.arm()
        self.engine.poll()
        self.assertEqual([order.side for order in self.service.submitted], [OrderSide.SELL, OrderSide.BUY])

    def test_monitor_only_checks_holdings_without_submit(self):
        self.engine.enable_holdings_exits = True
        self.service.positions = (position(),)
        self.service.prices = [Decimal("102")]
        self.engine.poll()
        self.assertGreater(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_percentage_sizes_actual_durable_order(self):
        self.rule()
        self.engine.equity_buy_percent = Decimal("10")
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted[0].quantity, 100)
        self.assertEqual(self.store.rules()[0].quantity, 100)

    def test_percentage_external_quantity_one_is_not_fixed_order_size(self):
        self.external()
        self.engine.equity_buy_percent = Decimal("10")
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted[0].quantity, 100)

    def test_missing_equity_blocks_only_buy(self):
        self.rule()
        self.service.cash = None
        self.engine.equity_buy_percent = Decimal("10")
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertTrue(self.engine.orders_enabled)

    def test_size_respects_available_cash_and_fee_headroom(self):
        self.rule()
        self.service.available = Decimal("500")
        self.engine.equity_buy_percent = Decimal("10")
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted[0].quantity, 4)

    def test_size_respects_order_cap(self):
        self.rule(max_notional=Decimal("550"))
        self.engine.equity_buy_percent = Decimal("10")
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted[0].quantity, 5)

    def test_signal_targets_persist_after_acceptance_and_survive_restart(self):
        self.external(take_profit_price="110", stop_loss_price="90")
        self.arm()
        self.engine.poll()
        reopened = WatchStore(self.store.path)
        targets = holding_exit_targets(reopened, position())
        self.assertEqual(targets["take_profit_price"], Decimal("110"))
        self.assertEqual(targets["stop_loss_price"], Decimal("90"))
        self.assertEqual(targets["source"], "model-a")

    def test_rejected_buy_does_not_store_targets(self):
        self.external(take_profit_price="110", stop_loss_price="90")
        self.service.submit_error = BrokerAPIError("denied", status_code=400, return_code=1)
        self.arm()
        self.engine.poll()
        self.assertIsNone(self.store.exit_targets(self.item.id))

    def test_buy_outside_target_bracket_is_blocked(self):
        self.external(take_profit_price="99", stop_loss_price="90")
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])

    def test_fallback_exact_targets(self):
        result = holding_exit_targets(self.store, position())
        self.assertEqual((result["take_profit_price"], result["stop_loss_price"]), (Decimal("101"), Decimal("99.2")))

    def test_malformed_or_incomplete_bracket_rejected(self):
        for values in (dict(take_profit_price="110"), dict(take_profit_price="90", stop_loss_price="110"),
                       dict(take_profit_price="NaN", stop_loss_price="90")):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_exit_conditions(dict(action="buy", **values))

    def test_source_failure_does_not_stop_other_producer(self):
        good = self.external(source="good")
        bad = ExternalPolicy("bad", 100, Decimal("1000000"), Decimal("1000000"))
        def fail():
            raise ValueError("invalid JSON")
        self.engine.configure_external_sources([(good, lambda: None), (bad, fail)])
        self.arm()
        self.engine.poll()
        self.assertTrue(self.engine.orders_enabled)
        self.assertEqual(len(self.service.submitted), 1)
        self.assertIn("bad", self.engine.external_source_errors)

    def test_failing_source_cannot_execute_queued_signal(self):
        policy = self.external()
        self.engine.configure_external_sources([(policy, lambda: (_ for _ in ()).throw(ValueError("bad")))])
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertTrue(self.engine.orders_enabled)

    def test_multiple_sources_same_symbol_never_duplicate_order(self):
        first = self.external(source="a")
        second = self.external(source="b")
        self.engine.configure_external_sources([(first, lambda: None), (second, lambda: None)])
        self.arm()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)

    def test_unknown_order_quarantines_symbol_without_global_stop(self):
        self.rule()
        self.service.submit_error = OrderOutcomeUnknown("timeout")
        self.arm()
        self.engine.poll()
        self.engine.poll()
        self.assertTrue(self.engine.orders_enabled)
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.store.attempts()[0]["status"], "unknown")

    def test_us_known_rejections_stop_after_three_total_attempts(self):
        self.use_us()
        self.rule()
        self.service.submit_error = BrokerAPIError("denied", status_code=400, return_code=1)
        self.arm()
        self.engine.poll()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 3)
        self.assertEqual(len(self.store.attempts()), 3)

    def test_us_retry_rechecks_trigger(self):
        self.use_us()
        self.rule()
        self.service.submit_error = BrokerAPIError("denied", status_code=400, return_code=1)
        self.service.on_submit = lambda: setattr(self.service, "prices", [Decimal("90")])
        self.arm()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)

    def test_us_accepted_unfilled_not_retried(self):
        self.use_us()
        self.rule()
        self.arm()
        self.engine.poll()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertTrue(any("체결 확인 대기" in row["message"] for row in self.store.events(category="order")))

    def test_us_ambiguous_order_never_retries(self):
        self.use_us()
        self.rule()
        self.service.submit_error = TimeoutError("ambiguous")
        self.arm()
        self.engine.poll()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)

    def test_us_retry_external_provenance_preserved(self):
        self.use_us()
        self.external(take_profit_price="110", stop_loss_price="90")
        outcomes = [BrokerAPIError("denied", status_code=400, return_code=1), None]
        self.service.on_submit = lambda: setattr(self.service, "submit_error", outcomes.pop(0))
        self.arm()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 2)
        self.assertEqual(self.store.attempts()[-1]["status"], "accepted")
        self.assertEqual(self.store.exit_targets(self.item.id)["source"], "model-a")

    def test_size_pure_function_does_not_use_order_cap_as_equity(self):
        account = AccountSnapshot(Market.DOMESTIC, "KRW", (), cash=Decimal(1000), total_evaluation=Decimal(0), available_to_order=Decimal(1000))
        self.assertEqual(account_equity(account), Decimal(1000))
        self.assertEqual(allocation_quantity(account, Decimal(100), Decimal(10), Decimal(10000000)), 1)

    def test_config_change_while_armed_is_not_allowed(self):
        self.rule()
        self.arm()
        with self.assertRaises(ValueError):
            self.engine.configure_external_sources([])

    def test_holdings_reports_quote_targets_and_calls_checkpoint(self):
        self.store.remove_item(self.item.id)
        self.engine.enable_holdings_exits = True
        self.service.positions = (position(),)
        updates, checks = [], []
        self.engine.poll(progress=updates.append, checkpoint=lambda: checks.append(True))
        quote_update = next(data for kind, data in updates if kind == "holding_quote")
        self.assertEqual(quote_update["watch_id"], self.item.id)
        self.assertEqual(quote_update["targets"]["take_profit_price"], Decimal("101"))
        self.assertTrue(checks)

    def test_us_retry_checks_available_funds_again(self):
        self.use_us()
        self.rule()
        self.service.submit_error = BrokerAPIError("denied", status_code=400, return_code=1)
        self.service.on_submit = lambda: setattr(self.service, "available", Decimal(0))
        self.arm()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)

    def test_us_retry_stops_immediately_after_off(self):
        self.use_us()
        self.rule()
        self.service.submit_error = OrderNotSent("stopped before transport")
        self.service.on_submit = self.engine.disarm
        self.arm()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(len(self.store.attempts()), 1)

    def test_local_not_sent_guard_denial_is_not_immediately_retried(self):
        self.use_us()
        self.rule()
        self.service.submit_error = OrderNotSent("stale quote after paced wait")
        self.arm()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(len(self.store.attempts()), 1)
        self.assertTrue(self.engine.orders_enabled)

    def test_us_accepted_partial_fill_does_not_retry(self):
        from dockdack.models import OrderExecution
        self.use_us()
        self.rule(quantity=3)
        self.arm()
        self.engine.poll()
        self.service.fills = (OrderExecution("200", "AAPL", "매수", "체결", Decimal(3), Decimal(1), Decimal(2), Decimal(100), Decimal(100), "100000"),)
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.store.attempts()[0]["status"], "accepted")
        self.assertEqual(self.store.order_history()[0]["filled_quantity"], "1")
        self.assertTrue(any("부분체결" in row["message"] and "잔량 2주" in row["message"] for row in self.store.events(category="order")))

    def test_us_reconcile_normalizes_leading_zero_order_numbers(self):
        from dockdack.models import OrderExecution
        self.use_us()
        self.rule()
        self.arm()
        self.engine.poll()
        self.service.fills = (OrderExecution("200", "AAPL", "매수", "체결", Decimal(1), Decimal(1), Decimal(0), Decimal(100), Decimal(101), "100000"),)
        self.engine.poll()
        self.assertEqual(self.store.attempts()[0]["status"], "filled")
        self.assertEqual(self.store.order_history()[0]["fill_price"], "101")

    def test_reconcile_wrong_side_never_marks_filled(self):
        from dockdack.models import OrderExecution
        self.rule()
        self.arm()
        self.engine.poll()
        self.service.fills = (OrderExecution("200", "005930", "매도", "체결", Decimal(1), Decimal(1), Decimal(0), Decimal(100), Decimal(101), "100000"),)
        self.engine.poll()
        self.assertEqual(self.store.attempts()[0]["status"], "accepted")
        self.assertTrue(self.engine.orders_enabled)

    def test_message_dedup_cache_is_bounded(self):
        with patch.object(self.store, "event"):
            for number in range(5000):
                self.engine._message(str(number), "SYSTEM", "test")
        self.assertLessEqual(len(self.engine._messages), 4096)

    def test_rules_status_filter_and_display_limit_preserve_full_history(self):
        first = self.rule()
        self.store.pause_rule(first.id)
        second = self.rule(threshold=Decimal("96"))
        third = self.rule(threshold=Decimal("97"))
        self.assertEqual([rule.id for rule in self.store.rules(statuses=("ready",))], [second.id, third.id])
        self.assertEqual([rule.id for rule in self.store.rules(limit=1)], [third.id])
        self.assertEqual(len(self.store.rules()), 3)

    def test_poll_uses_only_ready_rules_despite_large_history(self):
        for number in range(20):
            old = self.rule(threshold=Decimal(95 + number))
            self.store.pause_rule(old.id)
        self.rule()
        self.arm()
        with patch.object(self.store, "rules", wraps=self.store.rules) as queries:
            self.engine.poll()
        self.assertTrue(all(call.kwargs.get("statuses") is not None for call in queries.call_args_list))
        self.assertEqual(len(self.service.submitted), 1)

    def test_us_rejected_holding_does_not_start_new_chain_every_poll(self):
        self.use_us()
        self.engine.enable_holdings_exits = True
        self.service.positions = (replace(position(), market=Market.US, symbol="AAPL", exchange="ND", currency="USD"),)
        self.service.prices = [Decimal("102")]
        self.service.submit_error = BrokerAPIError("denied", status_code=400, return_code=1)
        self.arm()
        for _ in range(4):
            self.engine.poll()
        self.assertEqual(len(self.service.submitted), 3)
        reopened = AutoTrader(self.service, WatchStore(self.store.path), clock=lambda: NOW)
        reopened.enable_holdings_exits = True
        reopened.enable_orders("DEMO_AUTOTRADE")
        reopened.poll()
        self.assertEqual(len(self.service.submitted), 3)

    def test_us_new_chain_rechecks_after_cooldown_elapsed(self):
        self.use_us()
        self.rule()
        self.service.submit_error = BrokerAPIError("denied", status_code=400, return_code=1)
        self.arm()
        self.engine.poll()
        self.rule(threshold=Decimal("96"))
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 3)
        self.engine.clock = lambda: NOW + timedelta(seconds=301)
        self.service.submit_error = None
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 4)

    def test_v00_external_sell_cannot_bypass_holding_bracket(self):
        self.external(action="sell")
        self.service.positions = (position(),)
        self.engine.enable_holdings_exits = True
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])

    def test_v00_external_sell_at_target_is_allowed_once(self):
        self.external(action="sell")
        self.service.positions = (position(),)
        self.service.prices = [Decimal("102")]
        self.engine.enable_holdings_exits = True
        self.arm()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.service.submitted[0].side, OrderSide.SELL)

    def test_v00_manual_sell_rule_remains_explicit_separate_instruction(self):
        self.rule(side="sell")
        self.service.positions = (position(),)
        self.engine.enable_holdings_exits = True
        self.arm()
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)

    def test_fallback_targets_recomputed_from_fresh_preflight_account(self):
        self.store.remove_item(self.item.id)
        self.engine.enable_holdings_exits = True
        self.service.positions = (position(),)
        self.service.prices = [Decimal("102")]
        def account_changed():
            if len(self.service.account_calls) >= 2:
                self.service.positions = (replace(position(), average_price=Decimal("101")),)
        self.service.on_account = account_changed
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])

    def test_real_mode_percentage_flow_uses_only_injected_fake_submission(self):
        # This service has no HTTP client/keys; it records Python objects only.
        self.service.mode = TradingMode.REAL
        self.service.storage_scope = "a" * 64
        self.service.live_risk_acknowledged = True
        self.service.ensure_environment = lambda instrument: None
        self.service.ensure_order_permission = lambda instrument: None
        def fake_submit(request):
            self.service.submitted.append(request)
            return OrderResult(True, TradingMode.REAL, request, "123", "offline fake acceptance")
        self.service.submit = fake_submit
        self.store = WatchStore(Path(self.folder.name) / "real-offline.sqlite3", mode=TradingMode.REAL, storage_scope="a" * 64)
        self.store.save_item(self.item)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: NOW)
        self.engine.equity_buy_percent = Decimal("10")
        self.rule()
        self.engine.enable_orders("REAL_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(self.service.submitted[0].quantity, 100)
        self.assertEqual(self.store.attempts()[0]["status"], "accepted")
