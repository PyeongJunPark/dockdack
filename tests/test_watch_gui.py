from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from dockdack.watch_gui import WatchlistDialog

from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from dockdack.models import Market
from dockdack.market_schedule import calendar_for
from dockdack.universe import RankedStock
from test_autotrade import FakeTradingService


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class WatchGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Standalone GUI runs must not perform pandas/numpy's first import in
        # a Qt worker while QTest.qWait repeatedly reacquires the GIL.
        for market in Market:
            calendar_for(market, 2026)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WatchStore(Path(self.temp.name) / "watch.sqlite3")
        self.service = FakeTradingService()
        self.item = WatchItem(self.service.resolve("005930"), "삼성전자")
        self.store.save_item(self.item)
        self.window = WatchlistDialog(self.service, self.store)
        self.window.hourly_ranking.setChecked(False)
        self.window.engine.clock = lambda: datetime(2026, 9, 14, 1, tzinfo=timezone.utc)
        self.window.show()
        self.app.processEvents()

    def wait_idle(self):
        deadline = time.monotonic() + 30
        while self.window.worker:
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline)
        self.app.processEvents()

    def tearDown(self):
        self.window.stop_monitoring()
        self.wait_idle()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_opening_dialog_does_not_start_monitoring_or_api_requests(self):
        self.assertFalse(self.window.monitoring)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.window.watch_table.rowCount(), 1)

    def test_refresh_renders_n_bars_and_price_without_orders(self):
        self.window.refresh_button.click()
        self.wait_idle()
        self.assertEqual(len(self.window.chart.bars), 30)
        self.assertEqual(self.window.bar_table.rowCount(), 30)
        self.assertIn("100 KRW", self.window.watch_table.item(0, 1).text())
        self.assertEqual(self.service.submitted, [])
        self.window.chart.grab()  # Exercise the actual painter, including constant close values.

    def test_add_and_persist_interest_then_change_n_days(self):
        self.window.symbol_input.setText("AAPL")
        self.window.add_button.click()
        self.wait_idle()
        self.assertEqual(len(self.store.items()), 2)
        self.assertEqual(self.window.watch_market, Market.US)
        self.assertEqual(self.window.watch_table.rowCount(), 1)
        self.assertEqual(self.window.selected_item().instrument.symbol, "AAPL")
        self.window.days_input.setValue(60)
        self.window.days_button.click()
        self.window.refresh_all()
        self.wait_idle()
        self.assertEqual(len(self.window.chart.bars), 60)
        self.assertEqual(self.store.items()[1].days, 60)

    def test_invalid_rule_cap_cannot_enable_orders(self):
        self.window.threshold.setValue(95)
        self.window.rule_button.click()
        self.assertEqual(self.store.rules(), ())
        self.assertIn("상한", self.window.message.text())

    def test_rule_registration_keeps_orders_off_and_monitor_requires_confirmation(self):
        self.window.threshold.setValue(95)
        self.window.max_notional.setValue(1000)
        self.window.rule_button.click()
        self.assertEqual(len(self.store.rules()), 1)
        self.assertFalse(self.window.engine.orders_enabled)
        self.window.start_button.click()
        self.wait_idle()
        with patch.object(self.window, "confirm_automation", return_value=False):
            self.window.arm_button.click()
        self.assertFalse(self.window.engine.orders_enabled)
        with patch.object(self.window, "confirm_automation", return_value=True):
            self.window.arm_button.click()
        self.wait_idle()
        self.assertTrue(self.window.engine.orders_enabled)
        self.window.refresh_all()
        self.wait_idle()
        self.assertEqual(len(self.service.submitted), 1)
        self.window.stop_button.click()
        self.assertFalse(self.window.monitoring)
        self.assertFalse(self.window.engine.orders_enabled)

    def test_request_failure_marks_old_chart_as_stale_and_blocks_order(self):
        self.window.refresh_all()
        self.wait_idle()
        self.window.engine.histories.invalidate(self.item.id)
        self.service.fail_history = True
        self.window.refresh_all()
        self.wait_idle()
        self.assertIn("조회 실패", self.window.watch_table.item(0, 3).text())
        self.assertIn("이전 조회값", self.window.chart_title.text())
        self.assertEqual(self.service.submitted, [])

    def test_edit_controls_disable_while_monitoring_but_stop_remains_available(self):
        self.window.start_monitoring()
        self.wait_idle()
        self.assertFalse(self.window.rule_button.isEnabled())
        self.assertFalse(self.window.add_button.isEnabled())
        self.assertTrue(self.window.stop_button.isEnabled())
        self.window.stop_monitoring()
        self.assertTrue(self.window.add_button.isEnabled())

    def test_switching_instrument_clears_price_and_cap_inputs(self):
        self.store.save_item(WatchItem(self.service.resolve("AAPL")))
        self.window.reload_tables()
        self.window.threshold.setValue(250000)
        self.window.max_notional.setValue(1000000)
        self.window.watch_market_tabs.setCurrentIndex(1)
        self.assertEqual(self.window.threshold.value(), 0)
        self.assertEqual(self.window.max_notional.value(), 0)

    def test_external_default_off_zero_caps_and_no_manual_external_rule(self):
        self.assertFalse(self.window.external_mode.isChecked())
        self.assertEqual(self.window.external_krw.value(), 0)
        self.assertEqual(self.window.external_usd.value(), 0)
        self.assertNotIn("external", [self.window.trigger.itemData(i) for i in range(self.window.trigger.count())])
        self.window.external_mode.setChecked(True)
        self.window.start_monitoring()
        self.wait_idle()
        with patch.object(self.window, "confirm_automation", return_value=True):
            self.window.toggle_orders()
        self.wait_idle()
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertTrue((Path(self.temp.name) / "exchange/charts.json").exists())
        self.assertFalse(self.window.external_mode.isEnabled())

    def test_export_button_writes_valid_json(self):
        self.window.refresh_all()
        self.wait_idle()
        path = str(Path(self.temp.name) / "export.json")
        with patch("dockdack.watch_gui.QFileDialog.getSaveFileName", return_value=(path, "JSON")):
            self.window.export_button.click()
        self.wait_idle()
        self.assertTrue(Path(path).exists())
        self.assertIn("내보내기 완료", self.window.message.text())
        self.assertEqual(self.service.submitted, [])

    def test_top100_button_fetches_both_markets_and_preserves_existing(self):
        def ranked(market, limit):
            self.assertEqual(limit, 100)
            if market is Market.DOMESTIC:
                return (RankedStock(market, "000660", "KRX", "SK", 1, Decimal(100), "KRW"),)
            return (RankedStock(market, "AAPL", "ND", "애플", 1, Decimal(100), "USD"),)
        self.service.top_turnover = ranked
        self.window.ranking_button.click()
        self.wait_idle()
        self.assertEqual(len(self.store.items()), 3)
        self.assertEqual(self.window.watch_tables[Market.DOMESTIC].rowCount(), 2)
        self.assertEqual(self.window.watch_tables[Market.US].rowCount(), 1)
        self.assertEqual(self.service.submitted, [])

    def test_same_signal_and_chart_path_prevents_monitor_start(self):
        self.window.signal_path.setText(self.window.chart_path.text())
        self.window.start_monitoring()
        self.assertFalse(self.window.monitoring)
        self.assertIn("다른 경로", self.window.message.text())

    def test_external_can_arm_waiting_for_future_signal_only_after_confirmation(self):
        self.window.external_mode.setChecked(True)
        self.window.external_krw.setValue(1000)
        self.window.start_monitoring()
        self.wait_idle()
        with patch.object(self.window, "confirm_automation", return_value=False):
            self.window.toggle_orders()
        self.assertFalse(self.window.engine.orders_enabled)
        with patch.object(self.window, "confirm_automation", return_value=True):
            self.window.toggle_orders()
        self.wait_idle()
        self.assertTrue(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_monitor_start_enables_hourly_scheduler_without_arming_orders(self):
        self.window.hourly_ranking.setChecked(True)
        with patch.object(self.window.scheduler,"tick",return_value=False) as tick:
            self.window.start_monitoring()
            self.wait_idle()
        self.assertTrue(self.window.schedule_timer.isActive())
        self.assertIsNotNone(self.window.scheduler.started)
        self.assertTrue(tick.called)
        self.assertFalse(self.window.engine.orders_enabled)
        self.window.stop_monitoring()
        self.assertFalse(self.window.schedule_timer.isActive())
        self.assertIsNone(self.window.scheduler.started)

    def test_hourly_wakeup_runs_during_price_poll_wait(self):
        self.window.monitoring=True
        with patch.object(self.window.scheduler,"due",return_value=True), patch.object(self.window.scheduler,"tick",return_value=True) as tick:
            self.window._schedule_wakeup()
            self.wait_idle()
        tick.assert_called_once()
        self.assertEqual(self.service.quote_calls,0)

    def test_random_toggle_uses_separate_inbox_and_restores_external_settings(self):
        original=self.window.signal_path.text()
        self.window.random_demo.setChecked(True)
        self.assertTrue(self.window.external_mode.isChecked())
        self.assertEqual(self.window.external_source.text(),"random-demo")
        self.assertNotEqual(self.window.signal_path.text(),original)
        self.window.configure_external()
        self.assertTrue(self.window.engine.external_policy.allow_market)
        self.assertIsNotNone(self.window.test_producer)
        self.assertEqual(self.window.random_us.currentData(),"blocked")
        self.window.random_demo.setChecked(False)
        self.assertEqual(self.window.signal_path.text(),original)

    def test_random_zero_cap_streams_hold_without_orders(self):
        self.window.random_demo.setChecked(True)
        self.window.start_monitoring()
        self.wait_idle()
        updates=Path(self.temp.name)/"exchange/charts_updates/domestic_KRX_005930.json"
        self.assertTrue(updates.exists())
        self.assertTrue(Path(self.window.signal_path.text()).exists())
        self.assertEqual(self.store.rules(),())
        self.assertEqual(self.service.submitted,[])

    def test_market_tables_keep_rows_and_selection_separate_across_live_reload(self):
        for symbol in ("000660", "AAPL", "MSFT"):
            self.store.save_item(WatchItem(self.service.resolve(symbol), symbol))
        self.window.reload_tables()
        tables = self.window.watch_tables
        tables[Market.DOMESTIC].selectRow(1)
        self.window.watch_market_tabs.setCurrentIndex(1)
        tables[Market.US].selectRow(1)
        self.assertEqual(self.window.selected_item().instrument.symbol, "MSFT")
        self.window.watch_market_tabs.setCurrentIndex(0)
        self.assertEqual(self.window.selected_item().instrument.symbol, "000660")
        self.window.refresh_all()
        self.wait_idle()
        self.assertEqual(self.window.watch_market, Market.DOMESTIC)
        self.assertEqual(self.window.selected_item().instrument.symbol, "000660")
        self.assertIn("000660", self.window.chart_title.text())
        for index, market in enumerate((Market.DOMESTIC, Market.US)):
            self.assertEqual(tables[market].rowCount(), 2)
            self.assertIn("(2)", self.window.watch_market_tabs.tabText(index))
            for row in range(tables[market].rowCount()):
                self.assertTrue(tables[market].item(row, 0).data(Qt.ItemDataRole.UserRole).startswith(market.value + ":"))
        self.window.watch_market_tabs.setCurrentIndex(1)
        self.assertEqual(self.window.selected_item().instrument.symbol, "MSFT")
        self.assertIn("MSFT", self.window.chart_title.text())
        self.assertEqual(self.window.chart.currency, "USD")
        self.assertEqual(self.service.submitted, [])

    def test_switching_both_market_tabs_is_display_only_even_while_orders_on(self):
        self.store.save_item(WatchItem(self.service.resolve("AAPL"), "애플"))
        self.window.reload_tables()
        account_calls = []
        self.service.on_account = lambda: account_calls.append("account")
        self.window.external_mode.setChecked(True)
        self.window.external_krw.setValue(1000)
        self.window.configure_external()
        self.window.monitoring = True
        self.window.engine.enable_orders("DEMO_AUTOTRADE")
        self.window.update_controls()
        before = self.window.automation_status()
        with patch.object(self.window, "refresh_all") as refresh:
            for index in (1, 0, 1, 0):
                self.window.watch_market_tabs.setCurrentIndex(index)
                self.window.portfolio_panel.market_tabs.setCurrentIndex(index)
                self.app.processEvents()
            refresh.assert_not_called()
        self.assertEqual(self.window.automation_status(), before)
        self.assertTrue(self.window.engine.orders_enabled)
        self.assertTrue(self.window.monitoring)
        self.assertEqual(account_calls, [])
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.history_calls, 0)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_empty_us_tab_clears_kr_chart_and_cannot_modify_kr_item_or_rule(self):
        self.window.refresh_all()
        self.wait_idle()
        self.assertEqual(len(self.window.chart.bars), 30)
        self.window.threshold.setValue(250000)
        self.window.max_notional.setValue(1000000)
        self.window.watch_market_tabs.setCurrentIndex(1)
        self.assertIsNone(self.window.selected_item())
        self.assertEqual(self.window.watch_table.rowCount(), 0)
        self.assertEqual(len(self.window.chart.bars), 0)
        self.assertEqual(self.window.bar_table.rowCount(), 0)
        self.assertIn("미국 관심종목 없음", self.window.chart_title.text())
        self.assertEqual(self.window.threshold.value(), 0)
        self.assertEqual(self.window.max_notional.value(), 0)
        self.assertEqual(self.window.threshold.suffix(), "")
        before = self.store.items()
        self.window.days_input.setValue(60)
        self.window.days_button.click()
        self.window.rule_button.click()
        with patch("dockdack.watch_gui.QMessageBox.question") as confirm:
            self.window.remove_button.click()
            confirm.assert_not_called()
        self.assertEqual(self.store.items(), before)
        self.assertEqual(self.store.rules(), ())
        self.assertEqual(self.service.submitted, [])

    def test_selected_market_does_not_limit_both_market_poll_and_chart_export(self):
        us = WatchItem(self.service.resolve("AAPL"), "애플")
        self.store.save_item(us)
        self.window.reload_tables()
        self.window.watch_market_tabs.setCurrentIndex(1)
        self.window.external_mode.setChecked(True)
        self.window.start_monitoring()
        self.wait_idle()
        self.assertEqual(self.service.quote_calls, 2)
        self.assertEqual(self.window.fresh_ids, {self.item.id, us.id})
        payload = json.loads(Path(self.window.chart_path.text()).read_text(encoding="utf-8"))
        self.assertEqual({row["market"] for row in payload["stocks"]}, {"domestic", "us"})
        self.assertEqual({row["symbol"] for row in payload["stocks"]}, {"005930", "AAPL"})
        updates = Path(self.temp.name) / "exchange/charts_updates"
        self.assertTrue((updates / "domestic_KRX_005930.json").exists())
        self.assertTrue((updates / "us_ND_AAPL.json").exists())
        self.assertEqual(self.window.watch_market, Market.US)
        self.assertEqual(self.window.chart.currency, "USD")
        self.assertTrue(self.window.monitoring)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])


if __name__ == "__main__":
    unittest.main()
