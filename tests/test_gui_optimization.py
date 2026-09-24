"""Offline audit regressions: no broker calls, models, operational DB or orders."""
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox
    from dockdack.signal_connection_gui import connection_view
    from dockdack.v00_app import V00Window, MARK1_TRIGGER, MARK11_TRIGGER, PROTOTYPE_SOURCES
from dockdack.watchlist import WatchStore, WatchItem
from test_autotrade import FakeTradingService
from test_v00_mark1_trigger import FakeExternalFeed


@unittest.skipUnless(HAS_QT, 'Install GUI extra')
class GuiOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WatchStore(Path(self.temp.name) / 'ledger.sqlite3')
        self.service = FakeTradingService()
        self.store.save_item(WatchItem(self.service.resolve('005930'), 'test'))
        self.network = patch('requests.sessions.Session.request', side_effect=AssertionError('Offline test'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.window = V00Window(self.service, self.store, builtin=False, external_feed_factory=FakeExternalFeed)
        for timer in self.window.findChildren(QTimer):
            timer.stop()
        self.drain()

    def drain(self):
        deadline = time.monotonic() + 15
        while self.window._workspace_worker or self.window._activity_worker or self.window._schedule_probe or self.window._inspection_worker:
            self.app.processEvents()
            time.sleep(.002)
            self.assertLess(time.monotonic(), deadline)

    def tearDown(self):
        self.window.close()
        self.drain()
        for name in ('activity_pool', 'inspection_pool', 'pool'):
            getattr(self.window, name).waitForDone(10000)
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_mode_wording_never_calls_real_money_demo(self):
        for mode, expected in [('demo', '모의주문'), ('real', '실전주문')]:
            for armed in (False, True):
                view = connection_view({'trading_mode': mode, 'orders_enabled': armed})
                self.assertIn(expected, view['gate'])
                if mode == 'real':
                    self.assertNotIn('모의주문', view['gate'])

    def test_selected_model_controls_status_and_inspection_path(self):
        for check in self.window.external_model_checks.values():
            check.setChecked(True)
        self.drain()
        self.window.configure_external()
        for model in (MARK1_TRIGGER, MARK11_TRIGGER):
            source = PROTOTYPE_SOURCES[model]
            self.window.select_connection_source(source)
            status = self.window.connection_status()
            self.assertEqual(status['source_id'], source)
            self.assertEqual(status['input_path'], str(self.window._prototype_output_path(model)))
            with patch('dockdack.watch_gui.inspect_signal_file', return_value={'state': 'missing', 'summary': 'fixture'}) as inspect:
                self.window.inspect_signals()
                self.drain()
            self.assertEqual(Path(inspect.call_args.args[0]), self.window._prototype_output_path(model))
        self.assertFalse(self.service.submitted)

    def test_workspace_batch_and_days_update_run_off_gui_thread(self):
        main = threading.get_ident()
        threads = []
        original = self.store.connection
        def connect():
            threads.append(threading.get_ident())
            self.assertNotEqual(threading.get_ident(), main)
            return original()
        with patch.object(self.store, 'connection', side_effect=connect), \
                patch.object(self.store, 'cached_snapshot', side_effect=AssertionError('No N+1 cache reads')), \
                patch.object(self.store, 'cached_snapshots', wraps=self.store.cached_snapshots) as batch:
            self.window.external_model_checks[MARK1_TRIGGER].setChecked(True)
            self.assertFalse(self.window.arm_button.isEnabled())
            self.drain()
        self.assertTrue(threads)
        batch.assert_called_once()
        self.assertEqual(self.store.items()[0].days, 31)
        self.assertFalse(self.window.engine.orders_enabled)

    def test_locked_or_failed_workspace_keeps_last_view_and_blocks_on(self):
        before = self.window.watch_tables[next(iter(self.window.watch_tables))].rowCount()
        with patch.object(self.store, 'cached_snapshots', side_effect=RuntimeError('locked fixture')):
            self.window.request_workspace_reload()
            self.drain()
        self.assertIn('locked fixture', self.window._workspace_error)
        self.assertFalse(self.window.arm_button.isEnabled())
        self.assertEqual(self.window.watch_tables[next(iter(self.window.watch_tables))].rowCount(), before)
        self.window.request_workspace_reload()
        self.drain()
        self.assertFalse(self.window._workspace_error)

    def test_journal_slow_read_keeps_event_loop_responsive(self):
        entered, release = threading.Event(), threading.Event()
        original = self.window._ledger_collector.collect
        def slow():
            entered.set()
            release.wait(3)
            return original()
        with patch.object(self.window._ledger_collector, 'collect', side_effect=slow):
            self.window.trade_journal_panel.refresh_button.click()
            self.assertFalse(self.window.trade_journal_panel.refresh_button.isEnabled())
            self.assertTrue(entered.wait(1))
            heartbeat = []
            QTimer.singleShot(0, lambda: heartbeat.append(True))
            self.app.processEvents()
            self.assertEqual(heartbeat, [True])
            release.set()
            self.drain()
        self.assertTrue(self.window.trade_journal_panel.refresh_button.isEnabled())
        self.assertFalse(self.service.submitted)

    def test_source_limit_accessibility_and_compact_screen_controls(self):
        sources = self.window.additional_sources
        for _ in range(17):
            sources.add_row()
        self.assertEqual(sources.table.rowCount(), 16)
        self.assertFalse(sources.add_button.isEnabled())
        self.assertIn('16', sources.limit_label.text())
        self.assertTrue(self.window.buy_percent.accessibleName())
        self.window.resize(1280, 720)
        self.window.show()
        self.app.processEvents()
        self.assertLessEqual(self.window.height(), 720)
        for control in (self.window.stop_button, self.window.disarm_button, self.window.environment_selector):
            point = control.mapTo(self.window, control.rect().bottomRight())
            self.assertLess(point.y(), self.window.height())
        self.assertEqual(self.window.tabs.indexOf(self.window.external_panel), -1)
        self.assertGreaterEqual(self.window.signal_connection_page.indexOf(self.window.external_panel), 0)

    def test_close_policy_starts_off_and_confirmation_names_all_holding_exception(self):
        self.assertTrue(self.window.close_all_at_market_end.isChecked())
        self.assertTrue(self.window.engine.close_liquidator.enabled)
        self.assertFalse(self.window.engine.orders_enabled)
        with patch('dockdack.watch_gui.QMessageBox.question', return_value=QMessageBox.StandardButton.No) as question:
            self.assertFalse(self.window.confirm_automation())
        message = question.call_args.args[2]
        for text in ('5분', '모든 국내·미국', '수량·금액 한도', '체결', '이전 보유분'):
            self.assertIn(text, message)
        self.window.close_all_at_market_end.setChecked(False)
        self.assertFalse(self.window.engine.close_liquidator.enabled)
        self.assertFalse(self.window.engine.orders_enabled)

    def test_default_feeds_share_one_account_cache(self):
        for check in self.window.external_model_checks.values():
            check.setChecked(True)
        self.drain()
        self.window._external_feed_factory = None
        with patch('dockdack.prototype_external.ExternalPrototypeFeed', side_effect=lambda *a, **kw: FakeExternalFeed(*a, bundle_root=kw['bundle_root'])) as factory:
            self.window.configure_external()
        self.assertEqual(factory.call_count, 2)
        first, second = [call.kwargs['account_snapshots'] for call in factory.call_args_list]
        self.assertIs(first, second)

    def test_legacy_migration_requires_explicit_choice_before_target_creation(self):
        from dockdack.v00_app import prepare_legacy_ledger
        from dockdack.models import TradingMode
        service = SimpleNamespace(mode=TradingMode.DEMO, storage_scope='a' * 64)
        target = Path(self.temp.name) / 'scoped' / 'ledger.sqlite3'
        for answer in (QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes):
            with patch('dockdack.v00_app.QMessageBox.question', return_value=answer) as question, \
                    patch('dockdack.persistence.account_migration.migrate_legacy_demo') as migrate:
                notice = prepare_legacy_ledger(service, self.store.path, target)
            self.assertFalse(target.exists())
            self.assertEqual(question.call_args.args[-1], QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Yes:
                migrate.assert_called_once_with(self.store.path, target, 'a' * 64, confirmation='CONFIRM_LEGACY_DEMO_OWNERSHIP')
                self.assertIn('복사', notice)
            else:
                migrate.assert_not_called()
                if answer == QMessageBox.StandardButton.Cancel:
                    self.assertIsNone(notice)
                else:
                    self.assertIn('빈', notice)

    def test_unconfigured_or_existing_ledger_is_never_migrated(self):
        from dockdack.v00_app import prepare_legacy_ledger
        from dockdack.models import TradingMode
        target = Path(self.temp.name) / 'missing.sqlite3'
        with patch('dockdack.v00_app.QMessageBox.question') as question, \
                patch('dockdack.persistence.account_migration.migrate_legacy_demo') as migrate:
            notice = prepare_legacy_ledger(SimpleNamespace(mode=TradingMode.DEMO, storage_scope='unconfigured'), self.store.path, target)
            self.assertIn('미설정', notice)
            self.assertEqual(prepare_legacy_ledger(self.service, self.store.path, self.store.path), '')
        question.assert_not_called()
        migrate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
