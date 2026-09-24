"""Offline GUI environment-isolation regressions; no credentials/network used."""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from decimal import Decimal as D
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtCore import QDate, QPoint, QTimer
    from PySide6.QtWidgets import QApplication, QLabel, QMessageBox
    from dockdack.demo_session import SessionController
    from dockdack.environment_gui import confirm_environment
    from dockdack.watch_gui import WatchlistDialog
    from dockdack.gui import TradingWindow

from dockdack import AccountSnapshot, TradingMode
from dockdack.exceptions import ConfigurationError
from dockdack.gui_service import TradingService
from dockdack.models import Market
from dockdack.portfolio import PortfolioMarketState
from dockdack.watchlist import MarketSnapshot, TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position
from test_trade_journal import order


class OfflineRealService(FakeTradingService):
    def __init__(self):
        super().__init__()
        self.mode = TradingMode.REAL
        self.storage_scope = "unconfigured"
        self.acknowledge_live_risk = Mock()
        self.revoke_live_risk = Mock()
        self.live_risk_acknowledged = True
        # Even accidental read-only calls to REAL must fail this test suite.
        for name in ("quote", "history", "safety_account", "safety_orders", "safety_executions", "submit"):
            setattr(self, name, Mock(side_effect=AssertionError(f"unexpected REAL call: {name}")))


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class EnvironmentGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WatchStore(Path(self.temp.name) / "custom-demo.sqlite3")
        self.service = FakeTradingService()
        self.service.revoke_live_risk = Mock()
        self.item = WatchItem(self.service.resolve("005930"), "삼성전자")
        self.store.save_item(self.item)
        self._network_patch = patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden"))
        self.network = self._network_patch.start()
        self.addCleanup(self._network_patch.stop)
        self._config_patch = patch("dockdack.config.KiwoomConfig.from_env", side_effect=ConfigurationError("test: no real API keys"))
        self.config = self._config_patch.start()
        self.addCleanup(self._config_patch.stop)
        self.window = WatchlistDialog(self.service, self.store)
        for timer in self.window.findChildren(QTimer):
            timer.stop()
        self.window.hourly_ranking.setChecked(False)
        self.window.engine.clock = lambda: NOW
        self.candidate = OfflineRealService()
        self.drain_activity()

    def drain_activity(self):
        for _ in range(20):
            self.window.activity_pool.waitForDone(1000)
            self.app.processEvents()
            if self.window._activity_worker is None and self.window._schedule_probe is None:
                return
        self.fail("offline activity work did not drain")

    def tearDown(self):
        self.window.worker = None
        self.window._inspection_worker = None
        self.drain_activity()
        self.window.close()
        self.window.activity_pool.waitForDone(5000)
        self.app.processEvents()
        self.window.deleteLater()
        self.app.processEvents()
        self._config_patch.stop()
        self._network_patch.stop()
        self.temp.cleanup()

    def switch(self, candidate=None):
        with patch("dockdack.watch_gui.confirm_environment", return_value=True), \
             patch("dockdack.watch_gui.TradingService", return_value=candidate or self.candidate) as factory:
            self.window.request_environment(TradingMode.REAL)
        self.drain_activity()
        factory.assert_called_once_with(mode=TradingMode.REAL)

    def add_ready_rule(self):
        rule = TriggerRule.create(self.item, "price_ge", "buy", 1, D(1000), D(95))
        self.store.add_rule(rule)
        return rule

    def snapshot(self):
        return MarketSnapshot(self.service.quote(self.item.instrument), self.service.history(self.item.instrument, 30), NOW)

    def assert_real_not_called(self):
        self.network.assert_not_called()
        for name in ("quote", "history", "safety_account", "safety_orders", "safety_executions", "submit"):
            getattr(self.candidate, name).assert_not_called()

    def test_warning_cancel_keeps_same_environment_and_stops_orders_immediately(self):
        self.add_ready_rule()
        self.window.engine.enable_orders("DEMO_AUTOTRADE")
        self.window.monitoring = True
        self.window.pending_auto_arm = True
        original_engine = self.window.engine

        def cancel(parent, mode):
            self.assertIs(mode, TradingMode.REAL)
            self.assertFalse(parent.engine.orders_enabled)
            self.assertFalse(parent.monitoring)
            self.assertFalse(parent.pending_auto_arm)
            return False

        with patch("dockdack.watch_gui.confirm_environment", side_effect=cancel), \
             patch("dockdack.watch_gui.TradingService") as factory:
            self.window.environment_selector.buttons[TradingMode.REAL].click()
        factory.assert_not_called()
        self.assertIs(self.window.service, self.service)
        self.assertIs(self.window.engine, original_engine)
        self.assertIs(self.window.store, self.store)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertIn("전환 취소", self.window.message.text())
        self.assert_real_not_called()

    def test_real_selection_with_missing_keys_is_offline_unconfigured_and_off(self):
        # Exercise the real service's missing-key path, not a fabricated token.
        candidate = TradingService(mode=TradingMode.REAL)
        self.assertEqual(candidate.storage_scope, "unconfigured")
        self.assertEqual(candidate.brokers, {})
        self.switch(candidate)
        self.assertIs(self.window.service, candidate)
        self.assertIs(self.window.store.mode, TradingMode.REAL)
        self.assertEqual(self.window.store.storage_scope, "unconfigured")
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        self.assertTrue(candidate.live_risk_acknowledged)
        self.assertEqual(candidate.brokers, {})
        self.network.assert_not_called()
        with self.assertRaises(ValueError):
            self.window.engine.enable_orders("REAL_AUTOTRADE")
        self.assertFalse(self.window.engine.orders_enabled)
        self.network.assert_not_called()

    def test_gui_constructors_reject_real_store_from_another_key_scope(self):
        candidate = OfflineRealService()
        candidate.storage_scope = 'a' * 64
        other = WatchStore(Path(self.temp.name) / 'other-real.sqlite3',
                           mode=TradingMode.REAL, storage_scope='b' * 64)
        for constructor in (lambda: WatchlistDialog(candidate, other),
                            lambda: TradingWindow(candidate, store=other)):
            with self.assertRaisesRegex(ValueError, '저장소가 다릅니다'):
                constructor()
        self.assertEqual(other.order_history(), ())
        self.network.assert_not_called()

    def test_accepted_switch_clears_demo_quotes_holdings_rules_random_and_caps(self):
        self.add_ready_rule()
        snapshot = self.snapshot()
        self.window.snapshots[self.item.id] = snapshot
        self.window.fresh_ids.add(self.item.id)
        self.window.errors[self.item.id] = "old demo error"
        account = AccountSnapshot(Market.DOMESTIC, "KRW", (position(),), cash=D(5000))
        payload = {Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC, account, NOW, NOW)}
        self.window._progress(("portfolio", payload))
        self.assertEqual(self.window.portfolio_panel.tables[Market.DOMESTIC].rowCount(), 1)
        self.window.random_demo.setChecked(True)
        self.window.external_krw.setValue(10000000)
        self.window.external_usd.setValue(10000)
        self.window.test_producer = object()
        old_engine = self.window.engine
        self.switch()
        self.assertIsNot(self.window.engine, old_engine)
        self.assertTrue(old_engine._stop.is_set())
        self.assertIs(self.window.store.mode, TradingMode.REAL)
        self.assertEqual(self.window.store.path, Path(self.temp.name) / "real" / "unconfigured" / "watchlist.sqlite3")
        self.assertNotEqual(self.window.store.path, self.store.path)
        self.assertFalse(self.window.store.rules())
        self.assertEqual(len(self.store.rules()), 1)
        self.assertFalse(self.window.snapshots)
        self.assertFalse(self.window.fresh_ids)
        self.assertFalse(self.window.errors)
        self.assertEqual(self.window.portfolio_panel.tables[Market.DOMESTIC].rowCount(), 0)
        self.assertTrue(all(state.snapshot is None for state in self.window._portfolio_payload.values()))
        self.assertFalse(self.window.random_demo.isChecked())
        self.assertFalse(self.window.random_demo.isEnabled())
        self.assertFalse(self.window.tabs.isTabEnabled(self.window.tabs.indexOf(self.window.strategy_panel)))
        self.assertIsNone(self.window.test_producer)
        self.assertFalse(self.window.external_mode.isChecked())
        self.assertEqual(self.window.external_source.text(), "external-model")
        self.assertEqual(self.window.external_krw.value(), 0)
        self.assertEqual(self.window.external_usd.value(), 0)
        self.assertIn("real", self.window.signal_path.text())
        self.assertFalse(self.window.engine.orders_enabled)
        self.candidate.acknowledge_live_risk.assert_called_once_with("REAL_TRADING_RISK_ACKNOWLEDGED")
        self.service.revoke_live_risk.assert_called_once()
        self.assert_real_not_called()

    def test_pending_quote_worker_retains_old_service_until_completion_then_clears_its_result(self):
        self.window.worker = object()
        self.window._worker_kind = "quotes"
        self.window._done_message = ""
        self.window.monitoring = True
        snapshot = self.snapshot()
        self.switch()
        self.assertIs(self.window.service, self.service)
        self.assertIs(self.window.store, self.store)
        self.assertIsNotNone(self.window._pending_environment)
        self.assertFalse(self.window.monitoring)
        self.assertIn("전환 대기", self.window.environment_selector.badge.text())
        self.window._completed({self.item.id: snapshot}, None)
        self.drain_activity()
        self.assertIs(self.window.service, self.candidate)
        self.assertIs(self.window.store.mode, TradingMode.REAL)
        self.assertIsNone(self.window._pending_environment)
        self.assertFalse(self.window.snapshots)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.timer.isActive())
        self.assert_real_not_called()

    def test_pending_signal_inspection_also_blocks_environment_replacement(self):
        self.window._inspection_worker = object()
        self.switch()
        self.assertIs(self.window.service, self.service)
        self.assertIsNotNone(self.window._pending_environment)
        self.window._finish_environment_switch()
        self.assertIs(self.window.service, self.service)
        self.window._inspection_worker = None
        self.window._finish_environment_switch()
        self.drain_activity()
        self.assertIs(self.window.service, self.candidate)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assert_real_not_called()

    def test_old_demo_launcher_warmup_callback_cannot_arm_new_real_environment(self):
        controller = SessionController(self.window, Path(self.temp.name), auto_arm=True)
        self.assertTrue(controller.pending_arm)
        self.switch()
        # Even an unexpectedly late callback and stale pending flag cannot
        # promote DEMO startup permission into REAL permission.
        controller.pending_arm = True
        with patch.object(controller, "report"), patch.object(self.window.engine, "enable_orders") as arm:
            controller.warmup_finished({}, None)
        arm.assert_not_called()
        self.assertFalse(controller.pending_arm)
        self.assertEqual(controller.phase, "demo_startup_permission_not_applicable")
        self.assertFalse(self.window.engine.orders_enabled)
        self.assert_real_not_called()

    def test_new_demo_launcher_controller_never_schedules_real_autoarm(self):
        self.switch()
        controller = SessionController(self.window, Path(self.temp.name), auto_arm=True)
        self.assertFalse(controller.pending_arm)
        self.assert_real_not_called()

    def test_real_switch_clears_old_order_performance_and_demo_account_headings(self):
        # A fresh empty REAL ledger must not retain DEMO FIFO summary values.
        panel = self.window.order_history_panel
        old_ledger = (order(1), order(2, "sell", "110"))
        with patch.object(self.store, "order_history", return_value=old_ledger):
            panel.reload(self.store)
        self.assertIn("+10 KRW", panel.performance_label.text())
        self.switch()
        self.assertFalse(panel.records)
        self.assertEqual(panel.table.rowCount(), 0)
        self.assertNotIn("+10 KRW", panel.performance_label.text())
        self.assertFalse(panel.performance["summaries"])
        for view in (self.window.portfolio_panel, panel):
            texts = [label.text() for label in view.findChildren(QLabel)]
            self.assertFalse(any("모의계좌" in text for text in texts), texts)
            self.assertTrue(any("실전" in text or "실제투자" in text for text in texts), texts)
        self.assertIn("REAL", self.window.trade_journal_panel.mode_badge.text())

    def seed_displayed_demo_logs(self):
        views = {**self.window.operations_panel.logs, "order": self.window.order_history_panel.audit}
        with patch("dockdack.watchlist.utc_now", return_value=NOW):
            for category in views:
                self.store.event("SYSTEM", f"DEMO-only {category}", category=category)
        heads = self.store.event_heads()
        for category, view in views.items():
            view.reload(self.store, head=heads[category], force=True)
            self.assertEqual(view.table.rowCount(), 1)
            self.assertIn("DEMO-only", view.table.item(0, 2).text())
        return views, heads

    def test_real_switch_clears_demo_logs_even_when_new_background_query_fails(self):
        views, _ = self.seed_displayed_demo_logs()
        with patch("dockdack.watch_gui.collect_event_logs", side_effect=RuntimeError("offline REAL log failure")) as collect:
            self.switch()
        self.assertGreaterEqual(collect.call_count, 1)
        self.assertIs(self.window.service, self.candidate)
        self.assertIs(self.window.store.mode, TradingMode.REAL)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        self.assertIn("offline REAL log failure", self.window._last_log_error)
        for view in views.values():
            self.assertEqual(view.table.rowCount(), 0)
            self.assertIsNone(view.latest)
            self.assertIsNone(view._head)
        self.assertEqual(self.window.operations_panel.flow.text(), "최근 감시 기록 — · 최근 신호 수신 —")
        self.assert_real_not_called()

    def test_real_switch_reloads_new_logs_when_category_heads_match_demo(self):
        views, demo_heads = self.seed_displayed_demo_logs()
        real_store = WatchStore(Path(self.temp.name) / "same-head-real.sqlite3",
                                mode=TradingMode.REAL, storage_scope="unconfigured")
        with patch("dockdack.watchlist.utc_now", return_value=NOW):
            for category in views:
                real_store.event("SYSTEM", f"REAL-only {category}", category=category)
        self.assertEqual(real_store.event_heads(), demo_heads)
        self.window._environment_stores[(TradingMode.REAL, "unconfigured")] = real_store
        self.switch()
        self.assertIs(self.window.store, real_store)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        for category, view in views.items():
            self.assertEqual(view.table.rowCount(), 1)
            self.assertEqual(view.table.item(0, 2).text(), f"REAL-only {category}")
            self.assertEqual(view._head, demo_heads[category])
        self.assert_real_not_called()

    def test_progress_identifies_quote_sweep_and_journal_is_third_primary_tab(self):
        progress = self.window.sweep_progress
        self.assertTrue(progress.isTextVisible())
        self.assertEqual(progress.minimumHeight(), 24)
        self.assertEqual(progress.maximumHeight(), 24)
        self.assertIn("시세·차트 조회", progress.format())
        self.assertIn("체결 표시 아님", progress.accessibleName())
        self.window._progress((self.item.id, self.snapshot(), 3, 10))
        self.assertEqual(progress.value(), 3)
        self.assertIn("3/10종목", progress.text())
        self.assertIn("30%", progress.text())
        self.assertIn("체결", progress.toolTip())
        self.assertIs(self.window.workspace_tabs.widget(2), self.window.trade_journal_panel)
        self.assertEqual(self.window.workspace_tabs.tabText(2), "매매일지")
        self.assertEqual(self.window.trade_journal_panel.market_tabs.count(), 2)

    def test_real_warning_defaults_no_and_warns_about_money_and_existing_orders(self):
        with patch("dockdack.environment_gui.QMessageBox.warning", return_value=QMessageBox.StandardButton.No) as warning:
            self.assertFalse(confirm_environment(self.window, TradingMode.REAL))
        args = warning.call_args.args
        self.assertIn("실전투자", args[1])
        self.assertIn("실제 계좌의 돈", args[2])
        self.assertIn("이미 접수된 주문은 취소되지 않습니다", args[2])
        self.assertIn("자동주문은 OFF", args[2])
        self.assertEqual(args[-1], QMessageBox.StandardButton.No)
        self.network.assert_not_called()

    def test_programmatic_random_enable_is_rejected_in_real_mode(self):
        self.switch()
        self.window.random_demo.setChecked(True)
        self.assertFalse(self.window.random_demo.isChecked())
        self.assertIn("사용할 수 없습니다", self.window.message.text())
        self.assertFalse(self.window.engine.orders_enabled)
        self.assert_real_not_called()

    def test_returning_to_demo_restores_exact_custom_ledger_but_not_permissions(self):
        self.add_ready_rule()
        self.switch()
        new_demo = FakeTradingService()
        new_demo.storage_scope = "demo"
        with patch("dockdack.watch_gui.confirm_environment", return_value=True), \
             patch("dockdack.watch_gui.TradingService", return_value=new_demo):
            self.window.request_environment(TradingMode.DEMO)
        self.drain_activity()
        self.assertIs(self.window.store, self.store)
        self.assertEqual(self.window.store.path.name, "custom-demo.sqlite3")
        self.assertEqual(len(self.window.store.rules()), 1)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertFalse(self.window.snapshots)
        self.assertEqual(new_demo.quote_calls, 0)
        self.assertEqual(new_demo.submitted, [])
        self.candidate.revoke_live_risk.assert_called_once()
        self.assert_real_not_called()

    def test_candidate_creation_failure_does_not_restore_old_order_permission(self):
        self.add_ready_rule()
        self.window.engine.enable_orders("DEMO_AUTOTRADE")
        with patch("dockdack.watch_gui.confirm_environment", return_value=True), \
             patch("dockdack.watch_gui.TradingService", side_effect=ConfigurationError("offline setup failure")):
            self.window.request_environment(TradingMode.REAL)
        self.assertIs(self.window.service, self.service)
        self.assertIs(self.window.store, self.store)
        self.assertIsNone(self.window._pending_environment)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertIn("환경 전환 실패", self.window.message.text())
        self.assert_real_not_called()

    def test_integrated_journal_keeps_readable_rows_at_desktop_and_small_window_sizes(self):
        panel = self.window.trade_journal_panel
        self.window.workspace_tabs.setCurrentWidget(panel)
        self.drain_activity()
        panel.dates["domestic"].setDate(QDate(2026, 9, 15))
        panel.refresh(records=[order(1), order(2, "sell", "110"), order(3, "sell", "100", symbol="000001")])
        self.window.show()
        for width, height in ((1360, 900), (1080, 780)):
            with self.subTest(size=(width, height)):
                self.window.resize(width, height)
                self.app.processEvents()
                view = panel.tables["domestic"]
                self.assertGreaterEqual(view.height(), 190)
                self.assertGreaterEqual(view.columnWidth(1), 180)
                for key in ("profit", "return"):
                    value = panel.values["domestic"][key]
                    self.assertGreaterEqual(value.height(), value.fontMetrics().lineSpacing() * 2)
                scroll = panel.scroll_areas["domestic"]
                scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
                self.app.processEvents()
                top = view.mapTo(scroll.viewport(), QPoint(0, 0)).y()
                self.assertGreaterEqual(top, 0)
                self.assertLessEqual(top + view.height(), scroll.viewport().height())
        self.assert_real_not_called()


if __name__ == "__main__":
    unittest.main()
