"""The continuous demo bridge uses real parsing/execution with a fake broker only."""

from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dockdack import BrokerAPIError, Market, OrderOutcomeUnknown, OrderSide, TradingMode
from dockdack.gui_service import Instrument
from dockdack.lstm30_adapter import atomic_json, read_json
from dockdack.lstm30_runtime import DemoLSTMRuntime, request_stop
from dockdack.watchlist import WatchItem
from test_autotrade import FakeTradingService, NOW, position


class DemoLSTMRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "runtime"
        self.now = NOW
        self.service = FakeTradingService()
        self.item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "Samsung", 31)
        self.predictor = SimpleNamespace(
            metadata={"market": "domestic"},
            predict=Mock(return_value={"probability_ge_1pct": 0.7, "buy_threshold": 0.5,
                                       "predicts_gain": True}),
        )

    def position_snapshot(self, stock):
        matching = [p for p in self.service.positions if p.symbol == stock["symbol"]]
        quantity = sum((p.quantity for p in matching), Decimal(0))
        sellable = sum((p.sellable_quantity for p in matching), Decimal(0))
        cost = sum((p.quantity * p.average_price for p in matching), Decimal(0))
        return {
            **{key: stock[key] for key in ("market", "symbol", "exchange", "currency")},
            "quantity": str(quantity), "sellable_quantity": str(sellable),
            "average_price": str(cost / quantity) if quantity else None,
            "fetched_at": self.now.isoformat(),
        }

    def runtime(self, **overrides):
        options = dict(predictors={"domestic": self.predictor}, quantity=1,
                       max_krw="1000", max_usd="1000", service=self.service,
                       items=[self.item], position_provider=self.position_snapshot,
                       interval_seconds=30, clock=lambda: self.now)
        options.update(overrides)
        runtime = DemoLSTMRuntime(self.path, **options)
        self.addCleanup(runtime.close)
        return runtime

    def next_poll(self, runtime):
        self.now += timedelta(seconds=31)
        return runtime.poll_once()

    def test_no_confirmation_is_monitoring_only_despite_buy_signal(self):
        runtime = self.runtime()
        status = runtime.start()
        self.assertEqual(status["phase"], "monitoring")
        self.assertFalse(status["orders_enabled"])
        self.assertEqual(self.service.submitted, [])
        self.next_poll(runtime)
        self.assertEqual(self.service.submitted, [])
        self.assertFalse(runtime.engine.orders_enabled)

    def test_invalid_or_real_confirmation_rejected_before_broker_queries(self):
        for confirmation in ("", "REAL_AUTOTRADE", "LIVE_ORDER", True):
            with self.subTest(confirmation=confirmation):
                runtime = self.runtime()
                with self.assertRaises(ValueError):
                    runtime.start(confirmation)
                self.assertEqual(self.service.quote_calls, 0)
                self.assertEqual(self.service.history_calls, 0)
                self.assertEqual(self.service.submitted, [])
                runtime.close()

    def test_real_service_cannot_start_even_with_demo_confirmation(self):
        self.service.mode = TradingMode.REAL
        with self.assertRaises(ValueError):
            runtime = self.runtime()
            runtime.start("DEMO_AUTOTRADE")
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_valid_warmup_does_not_order_then_poll_submits_one_limit_buy(self):
        runtime = self.runtime()
        status = runtime.start("DEMO_AUTOTRADE")
        self.assertEqual(status["phase"], "running")
        self.assertTrue(status["orders_enabled"])
        self.assertEqual(self.service.submitted, [])
        status = self.next_poll(runtime)
        self.assertEqual(len(self.service.submitted), 1)
        request = self.service.submitted[0]
        self.assertEqual(request.side, OrderSide.BUY)
        self.assertEqual(request.quantity, 1)
        self.assertEqual(request.order_type, "0")
        self.assertEqual(request.price, Decimal(100))
        self.assertEqual(status["session_order_counts"].get("accepted"), 1)
        self.assertEqual(runtime.store.attempts()[0]["status"], "accepted")

    def test_take_profit_and_stop_loss_submit_sells_without_prediction(self):
        for price, field in (("101", "cost_profit_pct"), ("99.2", "cost_loss_pct")):
            with self.subTest(price=price):
                self.path = Path(self.temp.name) / field
                self.service = FakeTradingService()
                self.service.positions = (position(),)
                self.service.prices = [Decimal(price)]
                self.predictor.predict.reset_mock()
                runtime = self.runtime()
                status = runtime.start("DEMO_AUTOTRADE")
                self.assertTrue(status["orders_enabled"])
                self.assertEqual(self.service.submitted, [])
                self.next_poll(runtime)
                self.assertEqual(len(self.service.submitted), 1)
                self.assertEqual(self.service.submitted[0].side, OrderSide.SELL)
                self.assertEqual(self.service.submitted[0].quantity, 1)
                self.predictor.predict.assert_not_called()
                runtime.close()

    def test_us_regular_session_uses_us_model_and_limit_order_contract(self):
        self.now = NOW + timedelta(hours=13)
        item = WatchItem(Instrument(Market.US, "AAPL", "ND"), "Apple", 31)
        predictor = SimpleNamespace(metadata={"market": "us"}, predict=Mock(return_value={
            "probability_ge_1pct": 0.7, "buy_threshold": 0.5, "predicts_gain": True}))
        runtime = self.runtime(items=[item], predictors={"us": predictor})
        runtime.start("DEMO_AUTOTRADE")
        self.next_poll(runtime)
        self.assertEqual(len(self.service.submitted), 1)
        request = self.service.submitted[0]
        self.assertEqual(request.market, Market.US)
        self.assertEqual(request.symbol, "AAPL")
        self.assertEqual(request.exchange, "ND")
        self.assertEqual(request.order_type, "00")
        predictor.predict.assert_called()

    def test_existing_position_inside_exit_bounds_never_adds_to_position(self):
        self.service.positions = (position(),)
        runtime = self.runtime()
        runtime.start("DEMO_AUTOTRADE")
        self.next_poll(runtime)
        self.assertEqual(self.service.submitted, [])
        self.predictor.predict.assert_not_called()

    def test_low_probability_holds_without_test_order(self):
        self.predictor.predict.return_value.update(probability_ge_1pct=0.2, predicts_gain=False)
        runtime = self.runtime()
        runtime.start("DEMO_AUTOTRADE")
        status = self.next_poll(runtime)
        self.assertTrue(status["orders_enabled"])
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(status["diagnostics"][self.item.id]["action"], "hold")

    def test_failed_history_warmup_stays_off_after_connection_recovers(self):
        self.service.fail_history = True
        runtime = self.runtime()
        status = runtime.start("DEMO_AUTOTRADE")
        self.assertEqual(status["phase"], "blocked")
        self.assertFalse(status["orders_enabled"])
        self.assertTrue(status["errors"])
        self.service.fail_history = False
        self.next_poll(runtime)
        self.assertFalse(runtime.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_one_bad_position_in_multi_symbol_warmup_blocks_all_orders(self):
        second = WatchItem(Instrument(Market.DOMESTIC, "000660", "KRX"), "SK Hynix", 31)

        def provider(stock):
            if stock["symbol"] == second.instrument.symbol:
                raise BrokerAPIError("fake balance unavailable")
            return self.position_snapshot(stock)

        runtime = self.runtime(items=[self.item, second], position_provider=provider)
        status = runtime.start("DEMO_AUTOTRADE")
        self.assertEqual(status["phase"], "blocked")
        self.assertFalse(status["orders_enabled"])
        self.assertEqual(self.service.submitted, [])
        self.next_poll(runtime)
        self.assertEqual(self.service.submitted, [])

    def test_missing_matching_checkpoint_cannot_arm_flat_buy(self):
        with self.assertRaises(ValueError):
            self.runtime(predictors={})
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_unknown_order_response_disarms_and_does_not_self_rearm(self):
        runtime = self.runtime()
        runtime.start("DEMO_AUTOTRADE")
        self.service.submit_error = OrderOutcomeUnknown("fake missing acknowledgement")
        status = self.next_poll(runtime)
        self.assertFalse(status["orders_enabled"])
        self.assertEqual(len(self.service.submitted), 1)
        attempt = runtime.store.attempts()[0]
        self.assertEqual(attempt["status"], "unknown")
        runtime.store.mark_reviewed(attempt["rule_id"], "CHECKED_ORDER_HISTORY")
        self.service.submit_error = None
        self.next_poll(runtime)
        self.assertFalse(runtime.engine.orders_enabled)
        self.assertEqual(len(self.service.submitted), 1)

    def test_known_rejection_disarms_instead_of_retrying_continuously(self):
        runtime = self.runtime()
        runtime.start("DEMO_AUTOTRADE")
        self.service.submit_error = BrokerAPIError("fake rejected", return_code=2000, status_code=200)
        status = self.next_poll(runtime)
        self.assertFalse(status["orders_enabled"])
        self.assertEqual(runtime.store.attempts()[0]["status"], "rejected")
        self.service.submit_error = None
        self.next_poll(runtime)
        self.assertFalse(runtime.engine.orders_enabled)
        self.assertEqual(len(self.service.submitted), 1)

    def test_quote_failure_during_running_session_disarms_stickily(self):
        runtime = self.runtime()
        runtime.start("DEMO_AUTOTRADE")
        with patch.object(self.service, "quote", side_effect=BrokerAPIError("fake quote failure")):
            status = self.next_poll(runtime)
        self.assertFalse(status["orders_enabled"])
        self.next_poll(runtime)
        self.assertFalse(runtime.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_first_symbol_quote_failure_blocks_later_symbols_in_same_poll(self):
        second = WatchItem(Instrument(Market.DOMESTIC, "000660", "KRX"), "SK Hynix", 31)
        runtime = self.runtime(items=[self.item, second])
        runtime.start("DEMO_AUTOTRADE")
        first_polled_symbol = runtime.store.items()[0].instrument.symbol
        original_quote = self.service.quote

        def quote(instrument):
            if instrument.symbol == first_polled_symbol:
                raise BrokerAPIError("fake first-symbol quote failure")
            return original_quote(instrument)

        with patch.object(self.service, "quote", side_effect=quote):
            status = self.next_poll(runtime)
        self.assertFalse(status["orders_enabled"])
        self.assertEqual(self.service.submitted, [])

    def test_same_symbol_same_local_day_has_only_one_buy_attempt(self):
        runtime = self.runtime()
        runtime.start("DEMO_AUTOTRADE")
        self.next_poll(runtime)
        attempt = runtime.store.attempts()[0]
        runtime.store.finish(attempt["rule_id"], "filled", "fake fill confirmed", "0000200")
        # The fake account remains flat, as if the earlier position was subsequently closed.
        status = self.next_poll(runtime)
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(status["diagnostics"][self.item.id]["execution_gate"], "DAILY_BUY_ATTEMPT_LIMIT")
        self.assertTrue(status["orders_enabled"])

    def test_buy_latch_is_durable_across_runner_restart(self):
        first = self.runtime()
        first.start("DEMO_AUTOTRADE")
        self.next_poll(first)
        attempt = first.store.attempts()[0]
        first.store.finish(attempt["rule_id"], "filled", "fake fill confirmed", "0000200")
        first.close()
        second = self.runtime()
        second.start("DEMO_AUTOTRADE")
        status = self.next_poll(second)
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(status["diagnostics"][self.item.id]["execution_gate"], "DAILY_BUY_ATTEMPT_LIMIT")

    def test_duplicate_runtime_is_locked_and_owner_can_continue(self):
        first = self.runtime()
        first.start()
        second = self.runtime()
        with self.assertRaises((RuntimeError, OSError)):
            second.start()
        status = self.next_poll(first)
        self.assertEqual(status["session_id"], first.session_id)
        first.close()
        second.start()
        self.assertEqual(second.status()["phase"], "monitoring")

    def test_stale_stop_marker_does_not_stop_current_session(self):
        runtime = self.runtime()
        runtime.start("DEMO_AUTOTRADE")
        atomic_json(self.path / "stop.json", {"session_id": "different-old-session"})
        status = self.next_poll(runtime)
        self.assertTrue(status["orders_enabled"])
        self.assertEqual(status["phase"], "running")

    def test_current_session_stop_marker_prevents_any_further_broker_calls(self):
        runtime = self.runtime()
        runtime.start("DEMO_AUTOTRADE")
        before = (self.service.quote_calls, self.service.history_calls)
        self.assertEqual(request_stop(self.path), runtime.session_id)
        status = self.next_poll(runtime)
        self.assertFalse(status["orders_enabled"])
        self.assertEqual(status["phase"], "stopped")
        self.assertEqual((self.service.quote_calls, self.service.history_calls), before)
        self.assertEqual(self.service.submitted, [])

    def test_non_object_stop_control_fails_closed(self):
        for payload in ([], None, "stop"):
            with self.subTest(payload=payload):
                self.path = Path(self.temp.name) / f"invalid-stop-{type(payload).__name__}"
                runtime = self.runtime()
                runtime.start("DEMO_AUTOTRADE")
                atomic_json(self.path / "stop.json", payload)
                status = self.next_poll(runtime)
                self.assertFalse(status["orders_enabled"])
                self.assertEqual(status["phase"], "stopped")
                self.assertEqual(self.service.submitted, [])
                runtime.close()

    def test_close_disarms_and_persists_stopped_status(self):
        runtime = self.runtime()
        runtime.start("DEMO_AUTOTRADE")
        runtime.close()
        self.assertFalse(runtime.engine.orders_enabled)
        status = read_json(self.path / "status.json")
        self.assertEqual(status["phase"], "stopped")
        self.assertFalse(status["orders_enabled"])
        runtime.close()  # Cleanup is idempotent.


if __name__ == "__main__":
    unittest.main()
