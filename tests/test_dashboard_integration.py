from __future__ import annotations

import importlib.util
import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from dockdack.watch_gui import WatchlistDialog

from dockdack.models import Market
from dockdack.watchlist import WatchItem, WatchStore
from test_autotrade import FakeTradingService
from test_portfolio import account


class DashboardFakeService(FakeTradingService):
    def __init__(self):
        super().__init__()
        self.account_calls = []
        self.accounts = {market: account(market) for market in Market}

    def safety_account(self, instrument):
        self.account_calls.append(instrument.market)
        result = self.accounts[instrument.market]
        if isinstance(result, Exception):
            raise result
        return result


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class DashboardIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WatchStore(Path(self.temp.name) / "dashboard.sqlite3")
        self.service = DashboardFakeService()
        self.store.save_item(WatchItem(self.service.resolve("005930"), "삼성전자"))
        self.window = WatchlistDialog(self.service, self.store)
        self.window.hourly_ranking.setChecked(False)
        # GUI freshness indicators use wall time; use a nearby clock rather
        # than the portfolio unit tests' fixed historical timestamp.
        self.now = datetime.now(timezone.utc)
        self.window.engine.clock = lambda: self.now
        self.window.portfolio.clock = lambda: self.now
        self.window.show()
        self.app.processEvents()

    def wait_idle(self):
        deadline = time.monotonic() + 10
        idle_rounds = 0
        while idle_rounds < 2:
            self.app.processEvents()
            busy = (self.window.worker or self.window._activity_worker or self.window._activity_pending
                    or self.window._schedule_probe or self.window.activity_pool.activeThreadCount())
            idle_rounds = 0 if busy else idle_rounds + 1
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline, "Fake account worker did not finish")
        self.app.processEvents()

    def tearDown(self):
        self.window.health_timer.stop()
        self.window.order_status_timer.stop()
        self.window.stop_monitoring()
        self.wait_idle()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_holdings_is_default_and_opening_does_not_arm_or_invent_empty_account(self):
        self.assertIs(self.window.workspace_tabs.currentWidget(), self.window.portfolio_panel)
        self.assertEqual([self.window.workspace_tabs.tabText(i) for i in range(4)],
                         ["보유종목", "실제 주문·체결", "매매일지", "서버·감시 로그"])
        self.assertIn("미확인", self.window.portfolio_panel.summary_label.text())
        self.assertEqual(self.window.portfolio_panel.table.rowCount(), 0)
        self.assertEqual(self.service.account_calls, [])
        self.assertFalse(self.window.monitoring)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_manual_holdings_refresh_is_read_only_and_preserves_stopped_off(self):
        self.window.stop_monitoring()
        with patch.object(self.window.engine, "enable_orders") as enable:
            self.window.portfolio_panel.refresh_button.click()
            self.wait_idle()
            enable.assert_not_called()
        self.assertEqual(self.service.account_calls, [Market.DOMESTIC, Market.US])
        self.assertEqual(self.window.portfolio_panel.table.rowCount(), 1)
        self.assertEqual(self.window.portfolio_panel.tables[Market.DOMESTIC].rowCount(), 1)
        self.assertEqual(self.window.portfolio_panel.tables[Market.US].rowCount(), 1)
        self.assertFalse(self.window.monitoring)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertTrue(self.window.engine._stop.is_set())
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())
        self.assertEqual(self.window.order_history_panel.table.rowCount(), 0)

    def test_account_error_retains_prior_holdings_and_successful_empty_market_is_distinct(self):
        self.window.refresh_portfolio()
        self.wait_idle()
        self.now += timedelta(seconds=61)
        self.service.accounts[Market.DOMESTIC] = ValueError("테스트 API 요청 제한")
        self.service.accounts[Market.US] = account(Market.US, positions=())
        self.window.refresh_portfolio()
        self.wait_idle()
        self.assertEqual(self.window.portfolio_panel.table.rowCount(), 1)
        labels = self.window.portfolio_panel.market_labels
        self.assertIn("이전 잔고 유지", labels[Market.DOMESTIC]["status"].text())
        self.assertEqual(labels[Market.DOMESTIC]["holdings"].text(), "1종목")
        self.assertIn("요청 제한", labels[Market.DOMESTIC]["updated"].text())
        self.assertEqual(labels[Market.US]["status"].text(), "보유종목 없음")
        self.assertIn("잔고 오류 1건", self.window.health_label.text())
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_monitoring_off_gate_survives_manual_refresh_and_log_health_updates(self):
        self.window.monitoring = True
        self.window.update_controls()
        self.window.refresh_portfolio()
        self.wait_idle()
        self.assertTrue(self.window.monitoring)
        self.assertFalse(self.window.engine.orders_enabled)
        self.store.event("SYSTEM", "서버 정상 테스트", category="system")
        self.store.event("domestic:KRX:005930", "시세 감시 테스트", category="monitor")
        self.store.event("domestic:KRX:005930", "외부 신호 HOLD · 주문 생성 없음", category="signal")
        self.window._reload_activity(force=True)
        self.wait_idle()
        self.window._update_health()
        self.assertIn("자동주문 OFF · 주문 차단", self.window.operations_panel.runtime.text())
        self.assertEqual(self.window.operations_panel.logs["monitor"].table.rowCount(), 1)
        self.assertEqual(self.window.operations_panel.logs["signal"].table.rowCount(), 1)
        self.assertEqual(self.window.order_history_panel.table.rowCount(), 0)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_log_database_error_keeps_displayed_rows_and_warns_in_health(self):
        self.store.event("SYSTEM", "서버 상태 마지막 성공 기록", category="system")
        self.window._reload_activity(force=True)
        self.wait_idle()
        logs = self.window.operations_panel.logs["system"].table
        previous_cell = logs.item(0, 2)
        previous_count = logs.rowCount()
        # Unchanged event rows are cached: fail the revision query rather than
        # expecting a redundant full-row fetch. Keep the patch through worker completion.
        with patch.object(self.store, "event_heads", side_effect=sqlite3.OperationalError("test database unavailable")):
            self.window._reload_activity(force=True)
            self.wait_idle()
            self.window._update_health()
        self.assertEqual(logs.rowCount(), previous_count)
        self.assertIs(logs.item(0, 2), previous_cell)
        self.assertIn("기록 DB 조회 실패", self.window.health_label.text())
        self.assertIn("이전 화면 유지", self.window.operations_panel.runtime.text())
        self.assertIn("#ffda91", self.window.health_label.styleSheet())
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.account_calls, [])
        self.assertEqual(self.service.submitted, [])

    def test_closing_stops_health_and_order_status_timers(self):
        self.assertTrue(self.window.health_timer.isActive())
        self.assertTrue(self.window.order_status_timer.isActive())
        self.window.close()
        self.app.processEvents()
        self.assertFalse(self.window.health_timer.isActive())
        self.assertFalse(self.window.order_status_timer.isActive())
        self.assertFalse(self.window.monitoring)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_holdings_market_tabs_show_only_their_market_card_and_rows(self):
        self.window.refresh_portfolio()
        self.wait_idle()
        panel = self.window.portfolio_panel
        before = list(self.service.account_calls)
        self.assertEqual(panel.current_market, Market.DOMESTIC)
        self.assertTrue(panel.market_cards[Market.DOMESTIC].isVisible())
        self.assertFalse(panel.market_cards[Market.US].isVisible())
        self.assertEqual(panel.table.rowCount(), 1)
        self.assertIn("삼성전자", panel.table.item(0, 1).text())
        self.assertIn("KRW", panel.table.item(0, 0).text())
        panel.market_tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(panel.current_market, Market.US)
        self.assertFalse(panel.market_cards[Market.DOMESTIC].isVisible())
        self.assertTrue(panel.market_cards[Market.US].isVisible())
        self.assertEqual(panel.table.rowCount(), 1)
        self.assertIn("AAPL", panel.table.item(0, 1).text())
        self.assertIn("USD", panel.table.item(0, 0).text())
        self.assertNotIn("KRW", panel.summary_label.text())
        self.assertEqual(self.service.account_calls, before)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])


if __name__ == "__main__":
    unittest.main()
