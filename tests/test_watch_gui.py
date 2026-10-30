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
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QMessageBox
    from dockdack.watch_gui import WatchlistDialog

from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from dockdack.models import Market, TradingMode
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
        # Legacy dialogs without a session lock must not mutate a shared
        # account's persisted ranking merely by opening a window.
        with patch.object(self.store, "prune_ranked_extras", side_effect=AssertionError("needs lock")):
            self.window = WatchlistDialog(self.service, self.store)
        # These legacy interaction tests use fixed one-share rules. Percentage
        # sizing is covered with complete valuation fixtures in v0.0 tests.
        self.window.percent_sizing.setChecked(False)
        self.window.hourly_ranking.setChecked(False)
        self.window.engine.clock = lambda: datetime(2026, 9, 14, 1, tzinfo=timezone.utc)
        self.window.show()
        self.app.processEvents()

    def wait_idle(self):
        deadline = time.monotonic() + 30
        while self.window.worker or self.window._activity_worker or self.window._schedule_probe:
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

    def test_short_window_collapses_holdings_account_details(self):
        self.window.resize(980, 620)
        self.window.workspace_tabs.setCurrentWidget(self.window.portfolio_panel)
        self.app.processEvents()
        self.assertTrue(self.window.portfolio_panel._compact)
        self.assertFalse(self.window.portfolio_panel.market_cards[Market.DOMESTIC].isVisible())
        self.assertTrue(self.window.portfolio_panel.detail_buttons[Market.DOMESTIC].isVisible())
        self.window.resize(1360, 900)
        self.app.processEvents()
        self.assertFalse(self.window.portfolio_panel._compact)
        self.assertTrue(self.window.portfolio_panel.market_cards[Market.DOMESTIC].isVisible())

    def test_refresh_renders_n_bars_and_price_without_orders(self):
        self.window.refresh_button.click()
        self.wait_idle()
        self.assertEqual(len(self.window.chart.bars), 30)
        self.assertEqual(self.window.bar_table.rowCount(), 30)
        price_cell = self.window.watch_table.item(0, 1)
        self.assertEqual(price_cell.text(), '100')
        self.assertIn('100 KRW', price_cell.toolTip())
        self.assertEqual(price_cell.textAlignment() & Qt.AlignmentFlag.AlignRight,
                         Qt.AlignmentFlag.AlignRight)
        self.assertNotIn('\n', price_cell.text())
        self.assertEqual(self.window.chart_title.text(), self.item.instrument.symbol)
        self.assertIn('30/30 거래일', self.window.chart_title.toolTip())
        self.assertEqual(self.service.submitted, [])
        image = self.window.chart.grab().toImage()  # Painter still draws bars and OHLCV hover details.
        backdrop = image.pixelColor(20, 10)
        self.assertEqual(backdrop.name(), '#101724')  # No overlay prose inside the chart.
        self.assertEqual(image.pixelColor(80, 10), backdrop)

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
        self.window.engine.clock = lambda: datetime(2026, 9, 14, 14, tzinfo=timezone.utc)
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

    def test_confirmation_is_one_question_with_yes_no_default_no_and_no_side_effects(self):
        self.store.add_rule(TriggerRule.create(
            self.item, kind='price_ge', side='buy', quantity=1,
            max_notional=Decimal(1000), threshold=Decimal(95)))
        self.window.external_source.setText('external-test')
        self.service.ensure_order_permission = Mock()
        for mode, name in ((TradingMode.DEMO, '모의투자'), (TradingMode.REAL, '실전투자')):
            self.service.mode = mode
            for external in (False, True):
                self.window.external_mode.setChecked(external)
                for answer in (QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes):
                    with self.subTest(mode=mode, external=external, answer=answer):
                        self.service.ensure_order_permission.reset_mock()
                        with patch('dockdack.watch_gui.QMessageBox.question', return_value=answer) as confirm:
                            self.assertEqual(self.window.confirm_automation(),
                                             answer == QMessageBox.StandardButton.Yes)
                        confirm.assert_called_once_with(
                            self.window, f'{name} 자동매매 확인', '자동매매를 켜시겠습니까?',
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No)
                        if mode is TradingMode.REAL:
                            self.service.ensure_order_permission.assert_called_once_with(self.item.instrument)
                        else:
                            self.service.ensure_order_permission.assert_not_called()
                        self.assertFalse(self.window.engine.orders_enabled)
                        self.assertFalse(self.window.pending_auto_arm)
                        self.assertFalse(self.window.monitoring)
                        self.assertEqual(self.service.submitted, [])
                        self.assertEqual((self.service.quote_calls, self.service.history_calls), (0, 0))
        self.service.mode = TradingMode.DEMO

    def test_concise_confirmation_still_rejects_invalid_external_settings(self):
        self.window.external_mode.setChecked(True)
        self.window.external_source.setText('')
        with patch('dockdack.watch_gui.QMessageBox.question') as confirm:
            with self.assertRaisesRegex(ValueError, 'source_id'):
                self.window.confirm_automation()
            confirm.assert_not_called()
        self.window.external_source.setText('external-test')
        self.window.additional_sources.add_row(source='incomplete-extra')
        with patch('dockdack.watch_gui.QMessageBox.question') as confirm:
            with self.assertRaisesRegex(ValueError, '파일 경로를 모두'):
                self.window.confirm_automation()
            confirm.assert_not_called()
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        self.assertEqual(self.service.submitted, [])

    def test_concise_confirmation_does_not_bypass_real_permissions_or_random_signal_gate(self):
        self.service.mode = TradingMode.REAL
        self.service.ensure_order_permission = Mock(side_effect=ValueError('실전 주문 권한 없음'))
        with patch('dockdack.watch_gui.QMessageBox.question') as confirm:
            with self.assertRaisesRegex(ValueError, '실전 주문 권한 없음'):
                self.window.confirm_automation()
            confirm.assert_not_called()
        self.window.external_mode.setChecked(True)
        self.window.external_source.setText('random-demo')
        self.service.ensure_order_permission.reset_mock()
        with patch('dockdack.watch_gui.QMessageBox.question') as confirm:
            with self.assertRaisesRegex(ValueError, '랜덤 모의 신호기'):
                self.window.confirm_automation()
            confirm.assert_not_called()
        self.service.ensure_order_permission.assert_not_called()
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        self.assertEqual(self.service.submitted, [])
        self.service.mode = TradingMode.DEMO

    def test_concise_manual_confirmation_still_requires_a_ready_rule(self):
        with patch('dockdack.watch_gui.QMessageBox.question') as confirm:
            self.assertFalse(self.window.confirm_automation())
            confirm.assert_not_called()
        self.assertIn('규칙을 먼저 등록', self.window.message.text())
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        self.assertEqual(self.service.submitted, [])

    def test_request_failure_marks_old_chart_as_stale_and_blocks_order(self):
        self.window.refresh_all()
        self.wait_idle()
        self.window.engine.histories.invalidate(self.item.id)
        self.service.fail_history = True
        self.window.refresh_all()
        self.wait_idle()
        self.assertIn("조회 실패", self.window.watch_table.item(0, 2).text())
        self.assertEqual(self.window.chart_title.text(), self.item.instrument.symbol)
        self.assertIn("이전 조회값", self.window.chart_title.toolTip())
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

    def test_top100_button_fetches_only_eligible_market_and_preserves_existing(self):
        calls = []
        start = 100000
        def ranked(market, limit, *, record=True):
            if record:
                calls.append(market)
            self.assertEqual(limit, 100)
            if market is Market.DOMESTIC:
                return tuple(RankedStock(market, str(start+i), "KRX", f"Common {i}", i+1,
                                         Decimal(100), "KRW", 1000-i, "volume") for i in range(100))
            return tuple(RankedStock(market, f"S{i}", "ND", f"Common {i}", i+1,
                                     Decimal(100), "USD", 1000-i, "volume") for i in range(100))
        self.service.top_volume = ranked
        self.service.top_watchlist = lambda market, limit: ranked(market, limit, record=False)
        def reject_account_lookup(market):
            raise AssertionError("TOP100 selection must not request account protection")
        self.service.protected_symbols = reject_account_lookup
        self.window.ranking_button.click()
        self.wait_idle()
        self.assertEqual(calls, [Market.DOMESTIC])
        self.assertEqual(len(self.store.items()), 100)
        self.assertEqual(self.window.watch_tables[Market.US].rowCount(), 0)
        start = 200000
        self.window.ranking_button.click()
        self.wait_idle()
        self.assertEqual(len(self.store.items()), 100)
        self.assertNotIn("100000", {item.instrument.symbol for item in self.store.items()})
        self.assertNotIn("005930", {item.instrument.symbol for item in self.store.items()})
        self.window.engine.clock = lambda: datetime(2026, 9, 14, 14, tzinfo=timezone.utc)
        self.window.ranking_button.click()
        self.wait_idle()
        self.assertEqual(calls, [Market.DOMESTIC, Market.DOMESTIC, Market.US])
        self.assertEqual(len(self.store.items()), 200)
        self.assertEqual(self.window.watch_tables[Market.DOMESTIC].rowCount(), 100)
        self.assertEqual(self.window.watch_tables[Market.US].rowCount(), 100)
        self.assertEqual(self.service.submitted, [])

    def test_top100_button_closed_market_or_close_during_fetch_keeps_previous(self):
        self.window.engine.clock = lambda: datetime(2026, 9, 19, 1, tzinfo=timezone.utc)
        previous = self.store.items()
        with patch.object(self.service, "top_volume", create=True) as ranking:
            self.window.ranking_button.click()
            self.wait_idle()
            ranking.assert_not_called()
        def closed(market, limit):
            self.window.engine.clock = lambda: datetime(2026, 9, 14, 7, tzinfo=timezone.utc)
            return ()
        self.service.top_volume = closed
        self.window.engine.clock = lambda: datetime(2026, 9, 14, 1, tzinfo=timezone.utc)
        self.window.ranking_button.click()
        self.wait_idle()
        self.assertEqual(self.store.items(), previous)
        self.assertIn("종료", self.window.message.text())

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

    def test_visible_order_badge_stops_monitoring_when_turned_off(self):
        self.window.external_mode.setChecked(True)
        self.window.external_krw.setValue(1000)
        self.window.start_monitoring()
        self.wait_idle()
        with patch.object(self.window, 'confirm_automation', return_value=True):
            self.window.mode_label.click()
        self.wait_idle()
        self.assertTrue(self.window.monitoring)
        self.assertTrue(self.window.engine.orders_enabled)
        self.assertEqual(self.window.mode_label.text(), '자동주문 ON')
        self.window.mode_label.click()
        self.assertFalse(self.window.monitoring)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertEqual(self.window.mode_label.text(), '자동주문 OFF')
        self.assertEqual(self.service.submitted, [])

    def test_programmatic_disarm_keeps_read_only_monitoring(self):
        self.window.external_mode.setChecked(True)
        self.window.external_krw.setValue(1000)
        self.window.start_monitoring()
        self.wait_idle()
        with patch.object(self.window, 'confirm_automation', return_value=True):
            self.window.toggle_orders()
        self.wait_idle()
        self.window.disable_auto_orders()
        self.assertTrue(self.window.monitoring)
        self.assertFalse(self.window.engine.orders_enabled)

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
        with patch.object(self.window.scheduler,"due",return_value=True), patch.object(self.window,"refresh_all") as refresh:
            self.window._schedule_wakeup()
            self.wait_idle()
        refresh.assert_called_once()
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
        self.assertEqual(self.window.chart.bars, ())
        self.assertEqual(self.window.chart.currency, "")
        self.assertEqual(self.service.submitted, [])

    def test_market_tab_shows_only_ranked_watch_items(self):
        self.store.save_item(WatchItem(self.service.resolve("000660"), "SK하이닉스"))
        ranks = (RankedStock(Market.DOMESTIC, "005930", "KRX", "삼성전자", 1,
                             Decimal(100), "KRW", 1000, "volume"),)
        ranks += tuple(RankedStock(Market.DOMESTIC, f"{100000 + index}", "KRX", f"종목 {index}", index + 2,
                                   Decimal(100), "KRW", 999 - index, "volume") for index in range(99))
        self.store.replace_ranked(Market.DOMESTIC, ranks, set(), volume_rankings=ranks)
        self.window.reload_tables()
        self.assertIn("(100)", self.window.watch_market_tabs.tabText(0))
        self.assertNotIn("기타", self.window.watch_market_tabs.tabText(0))
        self.assertIn("장전 모델 후보는 선정 순위만", self.window.watch_market_tabs.tabToolTip(0))
        self.assertEqual(self.window.watch_tables[Market.DOMESTIC].rowCount(), 100)

    def test_market_tab_does_not_mislabel_legacy_turnover_as_volume_top100(self):
        ranks = tuple(RankedStock(Market.DOMESTIC, f"{100000 + index}", "KRX", f"종목 {index}", index + 1,
                                  Decimal(100), "KRW") for index in range(100))
        self.store.replace_ranked(Market.DOMESTIC, ranks, set())
        self.window.reload_tables()
        self.assertNotIn("순위 100", self.window.watch_market_tabs.tabText(0))
        self.assertIn("TOP100 미확정", self.window.watch_market_tabs.tabToolTip(0))

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

    def test_selected_tab_does_not_override_market_hours_for_poll_and_export(self):
        us = WatchItem(self.service.resolve("AAPL"), "애플")
        self.store.save_item(us)
        self.window.reload_tables()
        self.window.watch_market_tabs.setCurrentIndex(1)
        self.window.external_mode.setChecked(True)
        self.window.start_monitoring()
        self.wait_idle()
        self.assertEqual(self.service.quote_calls, 1)
        self.assertEqual(self.window.fresh_ids, {self.item.id})
        payload = json.loads(Path(self.window.chart_path.text()).read_text(encoding="utf-8"))
        self.assertEqual({row["market"] for row in payload["stocks"]}, {"domestic", "us"})
        self.assertEqual({row["symbol"] for row in payload["stocks"]}, {"005930", "AAPL"})
        # Export membership remains complete; the closed-market row has no fresh data.
        us_row = next(row for row in payload["stocks"] if row["market"] == "us")
        self.assertNotEqual(us_row["status"], "ok")
        updates = Path(self.temp.name) / "exchange/charts_updates"
        self.assertTrue((updates / "domestic_KRX_005930.json").exists())
        self.assertFalse((updates / "us_ND_AAPL.json").exists())
        self.assertEqual(self.window.watch_market, Market.US)
        self.assertEqual(self.window.chart.currency, "")
        self.assertTrue(self.window.monitoring)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])


if __name__ == "__main__":
    unittest.main()
