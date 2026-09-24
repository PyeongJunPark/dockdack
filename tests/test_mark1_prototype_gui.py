"""Offscreen actual reader flow; only fake quotes/positions, never a broker."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from datetime import timedelta
from decimal import Decimal
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from dockdack.mark1_prototype_gui import Mark1PrototypeWatchlistDialog

from dockdack.exceptions import OrderNotSent
from dockdack.gui_service import Instrument
from dockdack.lstm30_adapter import atomic_json, read_json
from dockdack.mark1_prototype_adapter import SOURCE_ID, STRATEGY_ID
from dockdack.market_schedule import calendar_for
from dockdack.models import Market
from dockdack.watchlist import WatchItem
from test_autotrade import NOW, position
from test_mark1_gui import CalendarFakeService
from test_mark1_prototype_adapter import predictor


@unittest.skipUnless(HAS_QT, "Install gui extra")
class PrototypeGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for market in Market:
            calendar_for(market, 2026)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "prototype"
        self.now = NOW
        self.service = CalendarFakeService()
        self.model = predictor()
        self.item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "Samsung", 31)
        self.windows = []
        self.network_patch = patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden"))
        self.network = self.network_patch.start()

    def clock(self):
        self.now += timedelta(milliseconds=1)
        return self.now

    def positions(self, stock):
        found = [row for row in self.service.positions if row.symbol == stock["symbol"]]
        quantity = sum((row.quantity for row in found), Decimal(0))
        sellable = sum((row.sellable_quantity for row in found), Decimal(0))
        cost = sum((row.quantity * row.average_price for row in found), Decimal(0))
        return {**{key: stock[key] for key in ("market", "symbol", "exchange", "currency")},
                "quantity": str(quantity), "sellable_quantity": str(sellable),
                "average_price": str(cost / quantity) if quantity else None, "fetched_at": self.clock().isoformat()}

    def window(self, **kwargs):
        window = Mark1PrototypeWatchlistDialog(
            self.service, runtime_dir=self.root, predictors={"domestic": self.model},
            items=[self.item], position_provider=self.positions, clock=self.clock, **kwargs)
        self.windows.append(window)
        window.show()
        self.app.processEvents()
        return window

    def idle(self, window):
        deadline = time.monotonic() + 30
        count = 0
        while count < 3:
            self.app.processEvents()
            count = 0 if (window.worker or window._inspection_worker or window._activity_worker
                          or window._schedule_probe) else count + 1
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline)

    def tearDown(self):
        try:
            for window in reversed(self.windows):
                window.stop_monitoring()
                self.idle(window)
                window.shutdown()
                self.idle(window)
                window.activity_pool.waitForDone()
                window.close()
                window.deleteLater()
            self.app.processEvents()
            self.network.assert_not_called()
        finally:
            self.network_patch.stop()
            self.temp.cleanup()

    def test_startup_off_warning_and_permanent_ui_order_block(self):
        window = self.window()
        self.assertIn("mark1 prototype", window.windowTitle())
        self.assertFalse(window.monitoring)
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual((self.service.quote_calls, self.service.history_calls), (0, 0))
        for method in (window._sync_order_controls, window.update_controls):
            method()
            self.assertFalse(window.arm_button.isEnabled())
        self.assertFalse(window.confirm_automation())
        self.assertFalse(window.enable_auto_orders())
        with self.assertRaises(ValueError):
            window.set_pending_auto_arm(True)
        with self.assertRaises(ValueError):
            window.start_session("DEMO_AUTOTRADE")
        self.assertFalse(window.monitoring)
        self.assertIn("검증 미통과", window.mark1_limitations.text())
        self.assertIn("0건", window.mark1_limitations.text())
        self.assertIn("가격단위", window.mark1_limitations.text())
        self.assertIn("-0.9%", window.environment_notice.text())
        self.assertNotIn("-0.8%", window.environment_notice.text())
        self.assertFalse(window.environment_notice.isHidden())
        self.assertEqual(read_json(self.root / "strategy.json")["source_id"], SOURCE_ID)

    def test_manual_observation_reaches_actual_reader_and_status_without_order(self):
        window = self.window()
        window.start_session()
        self.idle(window)
        window._update_mark1_table()
        self.assertTrue(window.monitoring)
        self.assertEqual(window.mark1_model_table.item(0, 2).text(), "70.00%")
        self.assertEqual(window.mark1_model_table.item(0, 3).text(), "매수")
        self.assertEqual(window.lstm_bridge.diagnostics[self.item.id]["action"], "buy")
        self.assertIn("50% 초과", window.mark1_model_table.item(0, 4).text())
        with window.store.connection() as db:
            row = db.execute("SELECT source_id,payload FROM external_signals").fetchone()
        self.assertEqual(row["source_id"], SOURCE_ID)
        self.assertEqual(json.loads(row["payload"])["action"], "buy")
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(window.store.attempts(), ())
        window.session_controller.report()
        status = read_json(self.root / "status.json")
        self.assertEqual((status["strategy"], status["source_id"]), (STRATEGY_ID, SOURCE_ID))
        self.assertTrue(status["research_only"])
        self.assertFalse(status["orders_enabled"])
        self.assertFalse(window.arm_button.isEnabled())

    def test_half_probability_is_hold_and_stop_boundary_is_point_nine(self):
        self.model.predict.return_value.update(probability_success=.5, predicts_success=False)
        window = self.window()
        window.start_session()
        self.idle(window)
        self.assertEqual(read_json(self.root / "exchange/signals.json")["signals"][0]["action"], "hold")
        window.stop_monitoring()
        self.idle(window)
        self.service.positions = (position(),)
        self.service.prices = [Decimal("99.1")]
        window.start_monitoring()
        self.idle(window)
        signal = read_json(self.root / "exchange/signals.json")["signals"][0]
        self.assertEqual((signal["action"], signal["cost_loss_pct"]), ("sell", "0.9"))
        self.assertEqual(self.service.submitted, [])

    def test_backend_and_close_liquidation_paths_cannot_write(self):
        window = self.window()
        window.engine._armed.set()
        self.assertFalse(window.engine.orders_enabled)
        for method in (window.service.submit, window.service.prepare,
                       window.close_liquidator.service.submit, window.engine._before_order_send):
            with self.assertRaises(OrderNotSent):
                method(None)
        self.assertFalse(window.close_liquidator.enabled)
        self.assertFalse(window.close_timer.isActive())
        self.assertEqual(self.service.submitted, [])

    def test_close_all_configuration_rejected_before_any_mutation(self):
        with self.assertRaisesRegex(ValueError, "마감 청산"):
            self.window(close_all_before_minutes=5, close_all_confirmation="DEMO_CLOSE_ALL_SELLABLE")
        self.assertFalse(self.root.exists())

    def test_other_strategy_unknown_runtime_and_other_bundle_rejected_unchanged(self):
        self.root.mkdir()
        old = self.root / "strategy.json"
        atomic_json(old, {"strategy": "mark1", "source_id": "mark1-daily-barrier"})
        before = old.read_bytes()
        with self.assertRaisesRegex(ValueError, "전용 폴더"):
            self.window()
        self.assertEqual(before, old.read_bytes())
        self.assertFalse((self.root / "watchlist.sqlite3").exists())
        self.root = Path(self.temp.name) / "unknown"
        self.root.mkdir()
        ledger = self.root / "watchlist.sqlite3"
        ledger.touch()
        with self.assertRaises(ValueError):
            self.window()
        self.assertEqual(ledger.stat().st_size, 0)
        self.root = Path(self.temp.name) / "valid"
        window = self.window()
        self.idle(window)
        window.shutdown()
        before = (self.root / "strategy.json").read_bytes()
        self.model.metadata["bundle_manifest_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "모델 번들"):
            self.window()
        self.assertEqual((self.root / "strategy.json").read_bytes(), before)

    def test_prototype_restart_accepts_own_marker(self):
        first = self.window()
        self.idle(first)
        self.assertTrue(first.shutdown())
        second = self.window()
        self.assertEqual(second.external_source.text(), SOURCE_ID)
        self.assertFalse(second.monitoring)
        self.assertFalse(second.engine.orders_enabled)

    def test_v00_preferences_and_additional_sources_stay_locked(self):
        window = self.window()
        self.idle(window)
        self.assertFalse(window.percent_sizing.isChecked())
        self.assertFalse(window.percent_sizing.isEnabled())
        self.assertFalse(window.additional_sources.isEnabled())
        self.assertFalse(window.engine.enable_holdings_exits)
        self.assertIsNone(window.engine.equity_buy_percent)
        self.assertEqual(window.engine.us_retry_attempts, 1)
        self.assertEqual(set(window.engine.external_sources), {SOURCE_ID})
        window.percent_sizing.setChecked(True)
        with self.assertRaisesRegex(ValueError, "고정 수량"):
            window.configure_external()
        window.percent_sizing.setChecked(False)
        window.additional_sources.table.setRowCount(1)
        with self.assertRaisesRegex(ValueError, "고정 수량"):
            window.configure_external()
        window.additional_sources.table.setRowCount(0)
        window.configure_external()
        self.assertFalse(window.engine.orders_enabled)

    def test_holdings_target_display_and_dark_localized_model_table(self):
        window = self.window()
        targets = window.engine.holding_exit_targets(position())
        window.portfolio_panel.set_exit_targets({self.item.id: targets})
        self.assertEqual(targets["stop_loss_price"], Decimal("99.1"))
        self.assertIn("-0.9%", window.portfolio_panel.heading.text())
        self.assertIn("QHeaderView::section", window.mark1_model_table.styleSheet())
        window.lstm_bridge.diagnostics[self.item.id] = {
            "prediction": {"probability_success": .45}, "action": "hold", "reason": "BELOW_OR_EQUAL_BUY_THRESHOLD"}
        window._update_mark1_table()
        self.assertEqual(window.mark1_model_table.item(0, 3).text(), "대기")
        self.assertEqual(window.mark1_model_table.item(0, 4).text(), "성공확률 50% 이하 · 대기")
        self.assertEqual(window.lstm_bridge.diagnostics[self.item.id]["reason"], "BELOW_OR_EQUAL_BUY_THRESHOLD")


if __name__ == "__main__":
    unittest.main()
