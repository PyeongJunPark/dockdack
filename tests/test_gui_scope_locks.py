"""Temporary-ledger GUI session-lock regressions; no account or order calls."""
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from dockdack.watch_gui import WatchlistDialog
from dockdack.environment_store import scoped_store_path
from dockdack.lstm30_runtime import SessionLock
from dockdack.models import TradingMode
from dockdack.watchlist import WatchStore
from test_autotrade import FakeTradingService
from test_environment_gui import OfflineRealService


@unittest.skipUnless(HAS_QT, 'Install GUI extra')
class GuiScopeLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WatchStore(Path(self.temp.name) / 'watchlist.sqlite3')
        self.initial = SessionLock(self.store.path.parent / 'session.lock')
        self.initial.acquire()
        self.window = WatchlistDialog(FakeTradingService(), self.store, session_lock=self.initial)
        for timer in self.window.findChildren(QTimer):
            timer.stop()
        self.candidate = OfflineRealService()
        self.candidate.storage_scope = 'b' * 64
        self.target = scoped_store_path(self.candidate, base_folder=self.window._environment_base)
        self.network = patch('requests.sessions.Session.request', side_effect=AssertionError('No network'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.drain()

    def drain(self):
        deadline = time.monotonic() + 10
        while self.window.worker or self.window._activity_worker or self.window._workspace_worker or self.window._schedule_probe:
            self.app.processEvents()
            time.sleep(.002)
            self.assertLess(time.monotonic(), deadline)

    def tearDown(self):
        self.window.worker = None
        self.window.close()
        self.drain()
        self.window.activity_pool.waitForDone(10000)
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.initial.release()
        self.temp.cleanup()

    def request(self):
        with patch('dockdack.watch_gui.confirm_environment', return_value=True), \
                patch('dockdack.watch_gui.TradingService', return_value=self.candidate):
            self.window.request_environment(TradingMode.REAL)

    def assert_available(self, path):
        lock = SessionLock(path.parent / 'session.lock')
        lock.acquire()
        lock.release()

    def assert_locked(self, path):
        lock = SessionLock(path.parent / 'session.lock')
        try:
            with self.assertRaises(RuntimeError):
                lock.acquire()
        finally:
            lock.release()

    def test_locked_target_blocks_before_database_open(self):
        occupied = SessionLock(self.target.parent / 'session.lock')
        occupied.acquire()
        try:
            with patch('dockdack.watch_gui.store_for_service', side_effect=AssertionError('Must not open locked ledger')) as open_store:
                self.request()
            open_store.assert_not_called()
            self.assertIs(self.window.store, self.store)
            self.assertIsNone(self.window._pending_environment)
            self.assertFalse(self.window.engine.orders_enabled)
            self.assertFalse(self.target.exists())
        finally:
            occupied.release()
        self.assert_locked(self.store.path)

    def test_success_transfers_lock_and_close_releases_target(self):
        self.request()
        self.drain()
        self.assertEqual(self.window.store.path, self.target)
        self.assert_available(self.store.path)
        self.assert_locked(self.target)
        self.window.close()
        self.drain()
        self.window.close()
        self.assert_available(self.target)
        self.assertFalse(self.window.engine.orders_enabled)

    def test_store_open_failure_releases_candidate_and_keeps_initial_lock(self):
        with patch('dockdack.watch_gui.store_for_service', side_effect=RuntimeError('fixture DB failure')):
            self.request()
        self.assertIs(self.window.store, self.store)
        self.assertIsNone(self.window._pending_session_lock)
        self.assert_available(self.target)
        self.assert_locked(self.store.path)

    def test_close_during_pending_worker_releases_only_candidate_until_idle(self):
        self.window.worker = Mock()
        self.request()
        self.assertIsNotNone(self.window._pending_environment)
        self.assert_locked(self.target)
        self.window.close()
        self.assertIsNone(self.window._pending_environment)
        self.assert_available(self.target)
        self.assert_locked(self.store.path)
        self.window.worker = None
        self.window.close()
        self.assert_available(self.store.path)

    def test_same_directory_uses_existing_lock_without_reacquiring(self):
        self.assertIs(self.window._acquire_environment_lock(self.store.path), self.initial)
        self.assert_locked(self.store.path)

    def test_engine_preparation_failure_does_not_swap_or_leak_lock(self):
        with patch('dockdack.watch_gui.AutoTrader', side_effect=RuntimeError('fixture engine failure')):
            self.request()
        self.assertIs(self.window.store, self.store)
        self.assertIs(self.window.service.mode, TradingMode.DEMO)
        self.assertIsNone(self.window._pending_environment)
        self.assert_available(self.target)
        self.assert_locked(self.store.path)

    def test_scheduler_database_preparation_failure_does_not_swap_or_leak_lock(self):
        with patch('dockdack.watch_gui.RankingScheduler', side_effect=RuntimeError('fixture schema failure')):
            self.request()
        self.assertIs(self.window.store, self.store)
        self.assertIs(self.window.service.mode, TradingMode.DEMO)
        self.assertIsNone(self.window._pending_environment)
        self.assert_available(self.target)
        self.assert_locked(self.store.path)

    def normal_window(self):
        from dockdack.v00_app import V00Window
        self.window.close()
        self.drain()
        self.initial.acquire()
        self.window = V00Window(FakeTradingService(), self.store, builtin=False, session_lock=self.initial)
        for timer in self.window.findChildren(QTimer):
            timer.stop()
        self.drain()
        self.window.configure_external()
        self.window.engine.resume_monitoring()
        self.window.monitoring = True
        self.window.hourly_ranking.setChecked(False)

    def test_normal_gui_has_no_account_wide_close_authority_or_confirmation(self):
        from PySide6.QtWidgets import QMessageBox
        self.normal_window()
        self.assertFalse(hasattr(self.window.engine, "close_liquidator"))
        self.assertFalse(hasattr(self.window, 'close_all_at_market_end'))
        self.assertFalse(hasattr(self.window, 'close_maintenance_timer'))
        self.assertFalse(hasattr(self.window, 'closing_confirmation_notice'))
        with patch('dockdack.watch_gui.QMessageBox.question', return_value=QMessageBox.StandardButton.No) as question:
            self.window.confirm_automation()
        self.assertNotIn('모든 국내·미국', question.call_args.args[2])
        self.assertEqual(self.window.service.submitted, [])


if __name__ == '__main__':
    unittest.main()
