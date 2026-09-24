"""Offscreen proofs for the LSTM-connected demo GUI; broker calls are all fake."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import time
import unittest
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from dockdack.lstm30_gui import LSTM30WatchlistDialog

from dockdack import BrokerAPIError, Market, OrderSide, TradingMode
from dockdack.gui_service import Instrument
from dockdack.lstm30_adapter import previous_trading_day, read_json
from dockdack.lstm30_runtime import DemoLSTMRuntime, SessionLock, request_stop
from dockdack.market_schedule import calendar_for
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position
from test_lstm30_universe import ranked_common_stocks


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class LSTM30GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # First calendar/NumPy imports belong on the main thread, not a Qt worker.
        for market in Market:
            calendar_for(market, 2026)
            previous_trading_day(market.value, NOW)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "runtime"
        self.now = NOW
        self.service = FakeTradingService()
        self.item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "Samsung", 31)
        self.predictor = SimpleNamespace(metadata={"market": "domestic"}, predict=Mock(return_value={
            "probability_ge_1pct": 0.7, "buy_threshold": 0.5, "predicts_gain": True}))
        self.windows = []
        self._network_patch = patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden"))
        self.network = self._network_patch.start()

    def clock(self):
        # Auto-arm starts its next sweep immediately: unique timestamps model a real clock.
        self.now += timedelta(milliseconds=1)
        return self.now

    def position_snapshot(self, stock):
        matching = [p for p in self.service.positions if p.symbol == stock["symbol"]]
        quantity = sum((p.quantity for p in matching), Decimal(0))
        sellable = sum((p.sellable_quantity for p in matching), Decimal(0))
        cost = sum((p.quantity * p.average_price for p in matching), Decimal(0))
        return {**{key: stock[key] for key in ("market", "symbol", "exchange", "currency")},
                "quantity": str(quantity), "sellable_quantity": str(sellable),
                "average_price": str(cost / quantity) if quantity else None,
                "fetched_at": self.clock().isoformat()}

    def window(self, **overrides):
        options = dict(runtime_dir=self.root, predictors={"domestic": self.predictor}, quantity=1,
                       max_krw="1000", max_usd="1000", items=[self.item],
                       position_provider=self.position_snapshot, clock=self.clock)
        options.update(overrides)
        window = LSTM30WatchlistDialog(self.service, **options)
        window.percent_sizing.setChecked(False)  # This fixture explicitly requests quantity=1.
        self.windows.append(window)
        window.show()
        self.app.processEvents()
        return window

    def wait_idle(self, window):
        deadline = time.monotonic() + 30
        idle_rounds = 0
        # A completed warmup can queue the controller's immediate armed sweep.
        # Re-check after event delivery instead of returning in between the two.
        while idle_rounds < 3:
            self.app.processEvents()
            if window.worker or window._inspection_worker or window._activity_worker or window.pending_auto_arm:
                idle_rounds = 0
            else:
                idle_rounds += 1
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline, "GUI worker did not finish")

    def sweep(self, window):
        self.now += timedelta(seconds=31)
        window.refresh_all()
        self.wait_idle(window)

    def tearDown(self):
        try:
            for window in reversed(self.windows):
                window.stop_monitoring()
                self.wait_idle(window)
                window.shutdown()
                window.close()
                window.deleteLater()
            self.app.processEvents()
            self.network.assert_not_called()
        finally:
            self._network_patch.stop()
            self.temp.cleanup()

    def test_construction_is_off_and_makes_no_broker_requests(self):
        window = self.window()
        self.assertFalse(window.monitoring)
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.history_calls, 0)
        self.assertFalse(window.random_demo.isChecked())
        self.assertFalse(window.percent_sizing.isChecked())
        self.assertFalse(window.percent_sizing.isEnabled())
        self.assertFalse(window.additional_sources.isEnabled())
        self.assertFalse(window.engine.enable_holdings_exits)
        self.assertIsNone(window.engine.equity_buy_percent)
        self.assertEqual(window.engine.us_retry_attempts, 1)
        self.assertTrue(window.engine.session_only_poll)

    def test_monitor_only_then_explicit_gui_on_submits_model_limit_buy(self):
        window = self.window()
        window.start_session()
        self.wait_idle(window)
        self.assertTrue(window.monitoring)
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        with patch.object(window, "confirm_automation", return_value=True):
            window.arm_button.click()
        self.wait_idle(window)
        self.assertTrue(window.engine.orders_enabled)
        self.assertEqual(len(self.service.submitted), 1)
        request = self.service.submitted[0]
        self.assertEqual((request.side, request.quantity, request.order_type), (OrderSide.BUY, 1, "0"))
        self.assertEqual(request.price, Decimal(100))
        self.assertTrue(all(len(call.args[0]) == 30 for call in self.predictor.predict.call_args_list))

    def test_threshold_point_four_boundary_uses_effective_setting_and_original_metadata(self):
        for score, expected_orders in ((0.3999, 0), (0.4, 1), (0.44, 1)):
            with self.subTest(score=score):
                self.root = Path(self.temp.name) / f"threshold-{score}"
                self.service = FakeTradingService()
                predictor = SimpleNamespace(
                    metadata={"market": "domestic", "buy_threshold": 0.5},
                    buy_threshold=0.4, checkpoint_buy_threshold=0.5,
                    predict=Mock(return_value={"probability_ge_1pct": score, "buy_threshold": 0.4,
                                               "predicts_gain": score >= 0.4}))
                window = self.window(predictors={"domestic": predictor})
                window.start_session("DEMO_AUTOTRADE")
                self.wait_idle(window)
                self.assertTrue(window.engine.orders_enabled)
                self.assertEqual(len(self.service.submitted), expected_orders)
                self.assertEqual(window.engine.external_policy.max_quantity, 1)
                self.assertEqual(window.engine.external_policy.max_krw, Decimal(1000))
                self.assertEqual(window.engine.external_policy.max_usd, Decimal(1000))
                if expected_orders:
                    self.assertEqual(self.service.submitted[0].side, OrderSide.BUY)
                    self.assertEqual(self.service.submitted[0].quantity, 1)
                window.session_controller.report()
                status = read_json(self.root / "status.json")
                self.assertEqual(status["buy_thresholds"]["domestic"], 0.4)
                self.assertEqual(status["checkpoint_buy_thresholds"]["domestic"], 0.5)
                self.assertEqual(predictor.metadata["buy_threshold"], 0.5)
                window.stop_monitoring()
                self.wait_idle(window)
                window.shutdown()

    def test_authorized_launcher_warms_with_orders_off_before_auto_on(self):
        window = self.window()
        seen_permissions = []
        original_history = self.service.history

        def history(instrument, days):
            seen_permissions.append(window.engine.orders_enabled)
            return original_history(instrument, days)

        with patch.object(self.service, "history", side_effect=history):
            controller = window.start_session("DEMO_AUTOTRADE")
            self.wait_idle(window)
        self.assertIs(controller, window.session_controller)
        self.assertTrue(seen_permissions)
        self.assertFalse(seen_permissions[0])
        self.assertTrue(window.engine.orders_enabled)
        self.assertEqual(len(self.service.submitted), 1,
                         f"diagnostics={window.lstm_bridge.diagnostics}; events={window.store.events(12)}")

    def test_take_profit_and_stop_loss_are_actual_gui_engine_sell_rules(self):
        for price in ("101", "99.2"):
            with self.subTest(price=price):
                self.root = Path(self.temp.name) / f"sell-{price}"
                self.service = FakeTradingService()
                self.service.positions = (position(),)
                self.service.prices = [Decimal(price)]
                self.predictor.predict.reset_mock()
                window = self.window()
                window.start_session("DEMO_AUTOTRADE")
                self.wait_idle(window)
                self.assertTrue(window.engine.orders_enabled)
                self.assertEqual(len(self.service.submitted), 1)
                request = self.service.submitted[0]
                self.assertEqual((request.side, request.quantity), (OrderSide.SELL, 1))
                self.predictor.predict.assert_not_called()
                window.stop_monitoring()
                self.wait_idle(window)
                window.shutdown()

    def test_us_gui_uses_us_model_and_us_limit_order_contract(self):
        self.now = NOW + timedelta(hours=13)
        item = WatchItem(Instrument(Market.US, "AAPL", "ND"), "Apple", 31)
        predictor = SimpleNamespace(metadata={"market": "us"}, predict=Mock(return_value={
            "probability_ge_1pct": 0.7, "buy_threshold": 0.5, "predicts_gain": True}))
        window = self.window(items=[item], predictors={"us": predictor})
        window.start_session("DEMO_AUTOTRADE")
        self.wait_idle(window)
        self.assertEqual(len(self.service.submitted), 1)
        request = self.service.submitted[0]
        self.assertEqual((request.market, request.symbol, request.exchange, request.order_type),
                         (Market.US, "AAPL", "ND", "00"))

    def test_failed_warmup_never_auto_rearms_when_data_recovers(self):
        self.service.fail_history = True
        window = self.window()
        window.start_session("DEMO_AUTOTRADE")
        self.wait_idle(window)
        self.assertFalse(window.engine.orders_enabled)
        self.assertFalse(window.pending_auto_arm)
        self.assertEqual(self.service.submitted, [])
        self.service.fail_history = False
        self.sweep(window)
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_gui_off_blocks_new_buys_while_monitoring_continues(self):
        self.predictor.predict.return_value.update(probability_ge_1pct=0.2, predicts_gain=False)
        window = self.window()
        window.start_session("DEMO_AUTOTRADE")
        self.wait_idle(window)
        self.assertTrue(window.engine.orders_enabled)
        window.disarm_button.click()
        self.assertFalse(window.engine.orders_enabled)
        self.predictor.predict.return_value.update(probability_ge_1pct=0.7, predicts_gain=True)
        self.sweep(window)
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_ordinary_gui_off_allows_a_new_explicit_gui_on_confirmation(self):
        self.predictor.predict.return_value.update(probability_ge_1pct=0.2, predicts_gain=False)
        window = self.window()
        window.start_session("DEMO_AUTOTRADE")
        self.wait_idle(window)
        window.disarm_button.click()
        self.assertFalse(window.engine.orders_enabled)
        # An unchanged mark0 model/30-bar input reuses its immutable cache.
        # Replace the fixture model explicitly instead of mutating the object
        # returned by an earlier predict call behind that cache's back.
        original_predictor = self.predictor
        self.predictor = SimpleNamespace(metadata=dict(original_predictor.metadata), predict=Mock(return_value={
            "probability_ge_1pct": 0.7, "buy_threshold": 0.5, "predicts_gain": True}))
        window.lstm_bridge.producer.predictors["domestic"] = self.predictor
        with patch.object(window, "confirm_automation", return_value=True):
            window.arm_button.click()
        self.wait_idle(window)
        self.assertTrue(window.engine.orders_enabled)
        self.assertEqual(len(self.service.submitted), 1)
        original_predictor.predict.assert_called_once()
        self.predictor.predict.assert_called_once()

    def test_cli_stop_disarms_and_cannot_be_rearmed_by_gui_on(self):
        self.predictor.predict.return_value.update(probability_ge_1pct=0.2, predicts_gain=False)
        window = self.window()
        controller = window.start_session("DEMO_AUTOTRADE")
        self.wait_idle(window)
        self.assertTrue(window.engine.orders_enabled)
        self.assertEqual(request_stop(self.root), controller.session_id)
        deadline = time.monotonic() + 5
        while window.engine.orders_enabled or window.monitoring:
            QTest.qWait(20)
            self.assertLess(time.monotonic(), deadline, "CLI stop did not reach GUI engine")
        self.assertTrue(window.engine.external_stop.is_set())
        self.predictor.predict.return_value.update(probability_ge_1pct=0.7, predicts_gain=True)
        with patch.object(window, "confirm_automation", return_value=True):
            window.enable_auto_orders()
        self.wait_idle(window)
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_off_during_startup_cancels_pending_auto_arm(self):
        window = self.window()
        window.start_session("DEMO_AUTOTRADE")
        window.disable_auto_orders()
        self.wait_idle(window)
        self.assertFalse(window.pending_auto_arm)
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_same_day_buy_latch_survives_gui_restart(self):
        first = self.window()
        first.start_session("DEMO_AUTOTRADE")
        self.wait_idle(first)
        self.assertEqual(len(self.service.submitted), 1)
        attempt = first.store.attempts()[0]
        first.store.finish(attempt["rule_id"], "filled", "fake fill confirmed", "0000200")
        self.sweep(first)
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(first.lstm_bridge.diagnostics[self.item.id]["execution_gate"], "DAILY_BUY_ATTEMPT_LIMIT")
        first.stop_monitoring()
        self.wait_idle(first)
        self.assertTrue(first.shutdown())
        second = self.window()
        second.start_session("DEMO_AUTOTRADE")
        self.wait_idle(second)
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(second.store.path, first.store.path)

    def test_confirmed_reject_keeps_gui_on_for_other_symbols_and_reports_quarantine(self):
        other = WatchItem(Instrument(Market.DOMESTIC, "035420", "KRX"), "Naver", 31)
        original_submit = self.service.submit

        def reject_only_first(request):
            if request.symbol == self.item.instrument.symbol:
                self.service.submitted.append(request)
                raise BrokerAPIError("confirmed mock rejection", return_code=2000, status_code=200)
            return original_submit(request)

        self.service.submit = reject_only_first
        window = self.window(items=[self.item, other])
        window.start_session("DEMO_AUTOTRADE")
        self.wait_idle(window)
        self.assertTrue(window.engine.orders_enabled)
        self.assertEqual({order.symbol for order in self.service.submitted}, {"005930", "035420"})
        self.assertEqual({row["status"] for row in window.store.attempts()}, {"rejected", "accepted"})
        self.sweep(window)
        self.assertTrue(window.engine.orders_enabled)
        self.assertEqual(len(self.service.submitted), 2)
        self.assertEqual(window.lstm_bridge.diagnostics[self.item.id]["execution_gate"], "REJECTED_TODAY")
        self.assertEqual(len(window.store.attempts(self.item.id)), 1)

    def test_rejected_holding_blocks_new_sell_after_gui_restart_without_blocking_explicit_on(self):
        from test_lstm30_rejections import seed_attempt

        self.service.positions = (position(),)
        self.service.prices = [Decimal(101)]
        first = self.window()
        seed_attempt(first.store, self.item, self.now)
        first.start_session("DEMO_AUTOTRADE")
        self.wait_idle(first)
        self.assertTrue(first.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(first.lstm_bridge.diagnostics[self.item.id]["execution_gate"], "REJECTED_TODAY")
        first.disarm_button.click()
        self.sweep(first)
        self.assertFalse(first.engine.orders_enabled)
        with patch.object(first, "confirm_automation", return_value=True):
            first.arm_button.click()
        self.wait_idle(first)
        self.assertTrue(first.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        first.stop_monitoring()
        self.wait_idle(first)
        self.assertTrue(first.shutdown())
        second = self.window()
        second.start_session("DEMO_AUTOTRADE")
        self.wait_idle(second)
        self.assertTrue(second.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(second.lstm_bridge.diagnostics[self.item.id]["execution_gate"], "REJECTED_TODAY")
        self.assertFalse(any(rule.status == "ready" and rule.side is OrderSide.SELL for rule in second.store.rules()))

    def test_malformed_rejected_timestamp_blocks_gui_warmup_auto_on(self):
        from test_lstm30_rejections import seed_attempt

        window = self.window()
        # Naive ISO is renderable but unsafe for market-local-day decisions.
        # Literal corrupt strings are covered by the helper and engine tests.
        seed_attempt(window.store, self.item, self.now.replace(tzinfo=None).isoformat())
        window.start_session("DEMO_AUTOTRADE")
        self.wait_idle(window)
        self.assertFalse(window.engine.orders_enabled)
        self.assertFalse(window.pending_auto_arm)
        self.assertEqual(self.service.submitted, [])

    def test_existing_runtime_lock_prevents_gui_construction(self):
        lock = SessionLock(self.root / "session.lock")
        lock.acquire()
        try:
            with self.assertRaises(RuntimeError):
                self.window()
        finally:
            lock.release()
        self.assertEqual(self.service.quote_calls, 0)

    def test_gui_holds_same_lock_against_headless_runner_until_shutdown(self):
        window = self.window()
        runtime = DemoLSTMRuntime(self.root, predictors={"domestic": self.predictor}, quantity=1,
                                 max_krw="1000", max_usd="1000", service=self.service,
                                 items=[self.item], position_provider=self.position_snapshot, clock=self.clock)
        self.addCleanup(runtime.close)
        with self.assertRaises(RuntimeError):
            runtime.start()
        self.assertTrue(window.shutdown())
        runtime.start()
        self.assertFalse(runtime.engine.orders_enabled)
        runtime.close()

    def test_real_service_and_real_start_confirmation_are_rejected(self):
        self.service.mode = TradingMode.REAL
        with self.assertRaises(ValueError):
            self.window()
        self.service.mode = TradingMode.DEMO
        window = self.window()
        with self.assertRaises(ValueError):
            window.start_session("REAL_AUTOTRADE")
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.quote_calls, 0)

    def test_real_environment_switch_cannot_create_a_real_service(self):
        window = self.window()
        with patch("dockdack.watch_gui.TradingService", side_effect=AssertionError("REAL construction forbidden")) as factory:
            try:
                window.request_environment(TradingMode.REAL)
            except ValueError:
                pass  # Refusal can be raised or rendered in the dedicated GUI.
        factory.assert_not_called()
        self.assertEqual(window.service.mode, TradingMode.DEMO)
        self.assertFalse(window.engine.orders_enabled)

    def test_random_signal_override_is_rejected_or_reverted(self):
        window = self.window()
        self.assertFalse(window.random_demo.isEnabled())
        window.random_demo.setChecked(True)
        if window.random_demo.isChecked():
            with self.assertRaises(ValueError):
                window.configure_external()
        else:
            window.configure_external()
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_source_or_caps_override_cannot_silently_reconfigure_lstm_session(self):
        window = self.window()
        window.external_source.setText("random-demo")
        with self.assertRaises(ValueError):
            window.configure_external()
        self.assertFalse(window.engine.orders_enabled)

    def test_closing_idle_window_disarms_and_releases_lock(self):
        window = self.window()
        self.predictor.predict.return_value.update(probability_ge_1pct=0.2, predicts_gain=False)
        window.start_session("DEMO_AUTOTRADE")
        self.wait_idle(window)
        self.assertTrue(window.engine.orders_enabled)
        window.close()
        self.app.processEvents()
        self.assertFalse(window.engine.orders_enabled)
        self.assertFalse(window.isVisible())
        lock = SessionLock(self.root / "session.lock")
        lock.acquire()
        lock.release()

    def test_close_while_worker_active_stops_orders_but_retains_lock_until_idle(self):
        window = self.window()
        window.start_session()
        self.assertIsNotNone(window.worker)
        window.close()
        self.assertFalse(window.engine.orders_enabled)
        self.assertFalse(window.shutdown())
        lock = SessionLock(self.root / "session.lock")
        with self.assertRaises(RuntimeError):
            lock.acquire()
        self.wait_idle(window)
        window.close()
        self.app.processEvents()
        lock.acquire()
        lock.release()
        self.assertEqual(self.service.submitted, [])

    def test_top100_initial_ranking_failure_keeps_prior_list_and_orders_off(self):
        self.now = NOW + timedelta(hours=13)
        us_model = SimpleNamespace(metadata={"market": "us"}, predict=Mock(return_value={
            "probability_ge_1pct": 0.7, "buy_threshold": 0.5, "predicts_gain": True}))
        self.service.top_volume = Mock(side_effect=BrokerAPIError("fake common-stock ranking unavailable"))
        self.service.protected_symbols = Mock(return_value=set())
        window = self.window(ranked_markets=(Market.US,),
                             predictors={"domestic": self.predictor, "us": us_model})
        previous = window.store.items()
        window.start_session("DEMO_AUTOTRADE")
        self.wait_idle(window)
        self.assertEqual(window.store.items(), previous)
        self.assertFalse(window.engine.orders_enabled)
        self.assertFalse(window.pending_auto_arm)
        self.assertEqual(self.service.submitted, [])

    def test_top100_authorized_rotation_updates_engine_guard_but_foreign_edit_is_blocked(self):
        self.now = NOW + timedelta(hours=13)
        us_model = SimpleNamespace(metadata={"market": "us"}, predict=Mock(return_value={
            "probability_ge_1pct": 0.7, "buy_threshold": 0.5, "predicts_gain": True}))
        self.service.top_volume = Mock(return_value=ranked_common_stocks())
        self.service.protected_symbols = Mock(return_value=set())
        window = self.window(ranked_markets=(Market.US,),
                             predictors={"domestic": self.predictor, "us": us_model})
        window.lstm_universe.bootstrap()
        window.engine._ensure_environment()
        self.assertEqual(sum(item.instrument.market is Market.US for item in window.store.items()), 100)
        self.assertTrue(all(item.days >= 31 for item in window.store.items()))
        self.service.top_volume.return_value = ranked_common_stocks(101)
        window.lstm_universe.refresh(Market.US)
        window.engine._ensure_environment()
        window.store.save_item(WatchItem(Instrument(Market.US, "FOREIGN", "ND"), "Not approved", 31))
        with self.assertRaises(ValueError):
            window.engine._ensure_environment(orders=True)
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_top100_scheduler_ranking_error_disarms_actual_gui_engine(self):
        self.now = NOW + timedelta(hours=13)
        item = WatchItem(Instrument(Market.US, "AAPL", "ND"), "Apple", 31)
        model = SimpleNamespace(metadata={"market": "us"}, predict=Mock(return_value={
            "probability_ge_1pct": 0.2, "buy_threshold": 0.5, "predicts_gain": False}))
        self.service.top_volume = Mock(return_value=ranked_common_stocks())
        self.service.protected_symbols = Mock(return_value=set())
        window = self.window(items=[item], ranked_markets=(Market.US,), predictors={"us": model})
        window.lstm_universe.bootstrap()
        prior = window.store.items()
        # This isolates the scheduler's actual order-permission callback. No quote
        # sweep or order submission is run; full GUI warmup is covered separately.
        window.engine.enable_orders("DEMO_AUTOTRADE")
        self.assertTrue(window.engine.orders_enabled)
        self.service.top_volume.side_effect = BrokerAPIError("fake ranking refresh failure")
        self.now += timedelta(hours=1)
        window.scheduler.start()
        self.assertFalse(window.scheduler.tick())
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(window.engine.safety_reason, "TOP100_REFRESH_FAILED")
        self.assertEqual(window.store.items(), prior)
        self.assertEqual(self.service.submitted, [])

    def test_top100_restart_preserves_managed_membership_and_filled_buy_history(self):
        self.now = NOW + timedelta(hours=13)
        store = WatchStore(self.root / "watchlist.sqlite3")
        store.save_item(self.item)
        store.add_ranked(ranked_common_stocks(), days=31)
        old = next(item for item in store.items() if item.instrument.symbol == "S2")
        rule = TriggerRule.create(old, "price_ge", "buy", 1, Decimal(1000), Decimal(100))
        store.add_rule(rule)
        store.claim(rule, Decimal(100), self.now)
        store.finish(rule.id, "accepted", "fake accepted", "previous-filled")
        store.finish(rule.id, "filled", "fake filled", "previous-filled")
        attempts = store.attempts(old.id)
        with store.connection() as db:
            managed_before = {row[0] for row in db.execute("SELECT watch_id FROM managed_watchlist")}
        model = SimpleNamespace(metadata={"market": "us"}, predict=Mock(return_value={
            "probability_ge_1pct": 0.2, "buy_threshold": 0.5, "predicts_gain": False}))
        self.service.top_volume = Mock(return_value=ranked_common_stocks(101))
        self.service.protected_symbols = Mock(return_value=set())
        window = self.window(ranked_markets=(Market.US,),
                             predictors={"domestic": self.predictor, "us": model})
        self.assertEqual(window.store.attempts(old.id), attempts)
        with window.store.connection() as db:
            managed_after = {row[0] for row in db.execute("SELECT watch_id FROM managed_watchlist")}
        self.assertEqual(managed_after, managed_before)
        self.assertFalse(window.engine.orders_enabled)
        window.lstm_universe.bootstrap()
        self.assertNotIn(old.id, {item.id for item in window.store.items()})
        self.assertEqual(window.store.attempts(old.id), attempts)
        self.assertEqual(self.service.submitted, [])

    def test_full_warmup_requires_valid_model_inputs_for_every_symbol_before_on(self):
        items = [self.item, WatchItem(Instrument(Market.DOMESTIC, "000660", "KRX"), "SK Hynix", 31),
                 WatchItem(Instrument(Market.DOMESTIC, "035420", "KRX"), "Naver", 31)]

        def provider(stock):
            if stock["symbol"] == "035420":
                raise BrokerAPIError("fake one-symbol position unavailable")
            return self.position_snapshot(stock)

        window = self.window(items=items, position_provider=provider)
        window.start_session("DEMO_AUTOTRADE")
        self.wait_idle(window)
        self.assertEqual(self.service.history_calls, len(items))
        self.assertFalse(window.engine.orders_enabled)
        self.assertFalse(window.pending_auto_arm)
        self.assertEqual(self.service.submitted, [])
        self.assertIn("035420", str(window.lstm_bridge.diagnostics))

    def test_closing_window_pauses_new_buy_without_disarming_liquidation_authority(self):
        window = self.window()
        closer = SimpleNamespace(tick=Mock(), buy_blocked=Mock(return_value=True))
        window.engine.close_liquidator = closer
        window.start_session("DEMO_AUTOTRADE")
        self.wait_idle(window)
        self.assertTrue(window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(window.store.attempts(), ())
        self.assertTrue(closer.tick.called)
        self.assertTrue(closer.buy_blocked.called)
        self.assertEqual(window.lstm_policy.max_quantity, 1)
        self.assertEqual(window.lstm_policy.max_krw, Decimal(1000))

    def test_final_buy_guard_rechecks_closing_boundary_after_parent_guard(self):
        from dockdack.autotrade import AutoTrader
        from dockdack.exceptions import OrderNotSent
        window = self.window()
        closer = SimpleNamespace(buy_blocked=Mock(side_effect=(False, True)))
        window.engine.close_liquidator = closer
        rule = TriggerRule.create(self.item, "price_ge", "buy", 1, Decimal(1000), Decimal(100))
        with patch.object(AutoTrader, "_before_order_send") as parent_guard:
            with self.assertRaises(OrderNotSent):
                window.engine._before_order_send(self.item, rule, None)
        parent_guard.assert_called_once()
        self.assertEqual(closer.buy_blocked.call_count, 2)
        self.assertEqual(self.service.submitted, [])

    def test_close_timer_only_wakes_authorized_idle_same_worker(self):
        window = self.window()
        window.close_liquidator = SimpleNamespace(closing_markets=Mock(return_value={Market.US}))
        with patch.object(window, "refresh_all") as refresh:
            window._close_wakeup()
            refresh.assert_not_called()
            window.monitoring = True
            window.engine._armed.set()
            window._close_wakeup()
            refresh.assert_called_once()
            refresh.reset_mock()
            window.worker = object()
            try:
                window._close_wakeup()
                refresh.assert_not_called()
            finally:
                window.worker = None

    def test_close_policy_requires_both_minutes_and_explicit_confirmation(self):
        for settings in ({"close_all_before_minutes": 5},
                         {"close_all_confirmation": "DEMO_CLOSE_ALL_SELLABLE"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.window(**settings)


if __name__ == "__main__":
    unittest.main()
