from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from threading import Thread
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from dockdack.demo_session import SessionController
    from dockdack.watch_gui import WatchlistDialog

from dockdack.models import Market, Quote
from dockdack.watchlist import MarketSnapshot, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class OrderStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The badge worker uses calendars even when quote monitoring is OFF.
        # Import them here so QTest's event loop cannot starve a first import.
        from dockdack.market_schedule import calendar_for
        for market in Market:
            calendar_for(market, 2026)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.store = WatchStore(self.folder / "watch.sqlite3")
        self.service = FakeTradingService()
        self.item = WatchItem(self.service.resolve("005930"), "삼성전자")
        self.store.save_item(self.item)
        self.window = WatchlistDialog(self.service, self.store)
        self.window.engine.clock = lambda: NOW
        self.window.hourly_ranking.setChecked(False)
        self.window.external_mode.setChecked(True)
        self.window.external_krw.setValue(1000)
        self.window.configure_external()
        self.window._portfolio_payload = self.window.portfolio.refresh_due()
        instrument = self.item.instrument
        self.results = {self.item.id: MarketSnapshot(
            Quote(instrument.market, instrument.symbol, self.item.name, instrument.exchange, Decimal(100), instrument.currency),
            self.service.history(instrument, self.item.days), NOW,
        )}
        # Exercise the UI state without starting a worker or making API requests.
        self.window.monitoring = True
        self.window.update_controls()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.health_timer.stop()
        self.window.order_status_timer.stop()
        self.window.environment_timer.stop()
        self.window.worker = None
        self.window.stop_monitoring()
        self.wait_local_workers()
        self.window.close()
        self.wait_local_workers()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def wait_local_workers(self):
        deadline = time.monotonic() + 10
        idle_rounds = 0
        while idle_rounds < 2:
            self.app.processEvents()
            busy = (self.window._activity_worker or self.window._activity_pending
                    or self.window._schedule_probe or self.window.activity_pool.activeThreadCount())
            idle_rounds = 0 if busy else idle_rounds + 1
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline, "Local display worker did not finish")

    def arm(self):
        # These tests isolate state reporting; the deferred ON button workflow
        # is exercised independently, including its whole-query safety gate.
        self.window.engine.enable_orders("DEMO_AUTOTRADE")
        self.window.update_controls()
        self.assertTrue(self.window.engine.orders_enabled)

    def read_status(self, controller):
        controller.report()
        return json.loads((self.folder / "demo-session-status.json").read_text(encoding="utf-8"))

    def test_worker_disarm_updates_status_without_progress_or_completion(self):
        self.arm()
        self.assertIn("자동주문 ON", self.window.mode_label.text())
        # A long-running request produces no Qt progress/completion notification.
        self.window.worker = object()
        self.window.update_controls()
        self.assertTrue(self.window.disarm_button.isEnabled())
        thread = Thread(target=self.window.engine.disarm)
        thread.start()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(self.window.order_status_timer.isActive())
        # Wait for the real timer, not a manual sync. A fixed 300 ms allowed
        # only 50 ms scheduling slack for the 250 ms timer under a full suite.
        deadline = time.monotonic() + 3
        while "자동주문 OFF" not in self.window.mode_label.text() and time.monotonic() < deadline:
            QTest.qWait(20)
        self.assertIn("자동주문 OFF", self.window.mode_label.text())
        self.assertIn("주문 차단", self.window.mode_label.text())
        self.assertTrue(self.window.arm_button.isEnabled())
        self.assertFalse(self.window.disarm_button.isEnabled())
        self.assertTrue(self.window.monitoring)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_on_and_off_buttons_keep_their_action_after_state_changes(self):
        self.assertEqual(self.window.arm_button.text(), "자동주문 켜기 (ON)")
        self.assertEqual(self.window.disarm_button.text(), "자동주문 끄기 (OFF)")
        self.window.worker = object()
        self.window.update_controls()
        with patch.object(self.window, "confirm_automation", return_value=True) as confirm:
            self.window.arm_button.click()
            self.assertFalse(self.window.engine.orders_enabled)
            self.assertTrue(self.window.pending_auto_arm)
            self.window.arm_button.click()
            self.assertFalse(self.window.engine.orders_enabled)
            confirm.assert_called_once()
        self.window.disarm_button.click()
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertTrue(self.window.monitoring)
        self.window.worker = None
        self.arm()
        self.window.disarm_button.click()
        self.assertFalse(self.window.engine.orders_enabled)
        with patch.object(self.window.engine, "enable_orders") as enable:
            self.window.disable_auto_orders()
            self.window.disarm_button.click()
            enable.assert_not_called()
        self.assertEqual(self.window.arm_button.text(), "자동주문 켜기 (ON)")
        self.assertEqual(self.window.disarm_button.text(), "자동주문 끄기 (OFF)")
        self.assertIn("자동주문 OFF", self.window.mode_label.text())

    def test_declining_confirmation_keeps_orders_and_status_off(self):
        with patch.object(self.window, "confirm_automation", return_value=False):
            self.window.arm_button.click()
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertIn("자동주문 OFF", self.window.mode_label.text())
        self.assertTrue(self.window.arm_button.isEnabled())
        self.assertEqual(self.window.automation_status()["state"], "orders_off")

    def test_off_during_busy_warmup_cancels_auto_arm_without_stopping_monitoring(self):
        controller = SessionController(self.window, self.folder, auto_arm=True)
        self.window.worker = object()
        self.window.update_controls()
        self.assertTrue(controller.pending_arm)
        self.assertTrue(self.window.disarm_button.isEnabled())
        self.assertIn("자동주문 OFF", self.window.mode_label.text())
        self.assertEqual(self.window.automation_status()["state"], "warmup_orders_off")
        self.window.disarm_button.click()
        self.assertFalse(controller.pending_arm)
        self.assertTrue(self.window.monitoring)
        self.window.worker = None
        self.window.fresh_ids.add(self.item.id)
        with patch.object(self.window.engine, "enable_orders") as enable:
            controller.warmup_finished({}, None)
            enable.assert_not_called()
        status = self.read_status(controller)
        self.assertFalse(status["pending_arm"])
        self.assertFalse(status["orders_enabled"])
        self.assertTrue(status["monitoring"])
        self.assertEqual(status["state"], "orders_off")

    def test_report_uses_current_order_state_after_failure_off_and_manual_rearm(self):
        controller = SessionController(self.window, self.folder, auto_arm=True)
        controller.warmup_finished({}, ValueError("Initial request failed"))
        status = self.read_status(controller)
        self.assertFalse(status["orders_enabled"])
        self.assertFalse(status["pending_arm"])
        self.assertEqual(status["state"], "orders_off")
        self.arm()
        status = self.read_status(controller)
        self.assertTrue(status["orders_enabled"])
        self.assertEqual(status["state"], "armed")
        self.window.disable_auto_orders()
        status = self.read_status(controller)
        self.assertFalse(status["orders_enabled"])
        self.assertEqual(status["state"], "orders_off")
        self.arm()
        status = self.read_status(controller)
        self.assertTrue(status["orders_enabled"])
        self.assertEqual(status["state"], "armed")
        self.window.stop_monitoring()
        status = self.read_status(controller)
        self.assertFalse(status["monitoring"])
        self.assertFalse(status["orders_enabled"])
        self.assertEqual(status["state"], "stopped")
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_successful_warmup_arms_once_and_late_callback_cannot_rearm_after_off(self):
        controller = SessionController(self.window, self.folder, auto_arm=True)
        self.window.fresh_ids.add(self.item.id)
        with patch.object(self.window, "refresh_all") as refresh:
            with patch.object(self.window.engine, "enable_orders", wraps=self.window.engine.enable_orders) as enable:
                controller.warmup_finished(self.results, None)
                enable.assert_called_once_with("DEMO_AUTOTRADE")
                refresh.assert_called_once_with()
                self.assertTrue(self.window.engine.orders_enabled)
                self.assertFalse(controller.pending_arm)
                self.assertEqual(self.read_status(controller)["state"], "armed")
                self.window.disable_auto_orders()
                controller.warmup_finished(self.results, None)
                enable.assert_called_once_with("DEMO_AUTOTRADE")
                refresh.assert_called_once_with()
        status = self.read_status(controller)
        self.assertFalse(status["orders_enabled"])
        self.assertFalse(status["pending_arm"])
        self.assertTrue(status["monitoring"])
        self.assertEqual(status["state"], "orders_off")
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_warmup_item_failure_cancels_pending_activation_and_preserves_reason(self):
        controller = SessionController(self.window, self.folder, auto_arm=True)
        self.window.fresh_ids.add(self.item.id)
        self.window.errors[self.item.id] = "Broker rate limit exceeded"
        with patch.object(self.window.engine, "enable_orders") as enable:
            with patch.object(self.window, "refresh_all") as refresh:
                controller.warmup_finished({self.item.id: ValueError("Broker rate limit exceeded")}, None)
                enable.assert_not_called()
                refresh.assert_not_called()
        status = self.read_status(controller)
        self.assertFalse(status["orders_enabled"])
        self.assertFalse(status["pending_arm"])
        self.assertEqual(status["state"], "orders_off")
        self.assertIn("Broker rate limit exceeded", status["reason"])
        self.assertIn("자동주문 OFF", self.window.mode_label.text())
        self.assertTrue(self.window.monitoring)

    def test_monitor_only_controller_never_schedules_or_arms_orders_on_completion(self):
        controller = SessionController(self.window, self.folder, auto_arm=False)
        self.assertFalse(controller.pending_arm)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertEqual(self.read_status(controller)["state"], "orders_off")
        self.window.fresh_ids.add(self.item.id)
        with patch.object(self.window.engine, "enable_orders") as enable:
            with patch.object(self.window, "refresh_all") as refresh:
                controller.warmup_finished({}, None)
                enable.assert_not_called()
                refresh.assert_not_called()
        status = self.read_status(controller)
        self.assertFalse(status["orders_enabled"])
        self.assertFalse(status["pending_arm"])
        self.assertTrue(status["monitoring"])
        self.assertEqual(status["state"], "orders_off")

    def test_launcher_cannot_arm_from_prior_fresh_ids_and_empty_current_results(self):
        controller = SessionController(self.window, self.folder, auto_arm=True)
        self.window.fresh_ids.add(self.item.id)
        with patch.object(self.window.engine, "enable_orders") as enable:
            controller.warmup_finished({}, None)
            enable.assert_not_called()
        status = self.read_status(controller)
        self.assertFalse(status["orders_enabled"])
        self.assertFalse(status["pending_arm"])
        self.assertIn("완료되지", status["reason"])

    def test_launcher_cannot_arm_with_stale_account_even_after_full_quote_results(self):
        controller = SessionController(self.window, self.folder, auto_arm=True)
        market = self.item.instrument.market
        state = self.window._portfolio_payload[market]
        self.window._portfolio_payload[market] = replace(
            state, fetched_at=datetime.now(timezone.utc) - timedelta(minutes=3),
        )
        with patch.object(self.window.engine, "enable_orders") as enable:
            controller.warmup_finished(self.results, None)
            enable.assert_not_called()
        status = self.read_status(controller)
        self.assertFalse(status["orders_enabled"])
        self.assertFalse(status["pending_arm"])
        self.assertIn("stale", status["reason"])


if __name__ == "__main__":
    unittest.main()
