"""Temporary-ledger GUI session-lock regressions; no account or order calls."""
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
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

    def test_normal_gui_approval_arms_the_configured_close_policy_only_after_warmup(self):
        from PySide6.QtWidgets import QMessageBox
        self.normal_window()
        liquidator = self.window.engine.close_liquidator
        self.assertTrue(liquidator.enabled)
        with patch.object(liquidator, 'tick') as tick, \
                patch('dockdack.watch_gui.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes) as question, \
                patch.object(self.window, '_begin_manual_warmup'), \
                patch.object(self.window, 'refresh_all'), \
                patch.object(self.window, 'activation_failure', return_value=''):
            self.window.engine.maintenance_checkpoint()
            tick.assert_not_called()
            self.window.enable_auto_orders()
            self.assertTrue(self.window.pending_auto_arm)
            self.assertFalse(self.window.engine.orders_enabled)
            self.assertIn('모든 국내·미국', question.call_args.args[2])
            self.assertIn('5분', question.call_args.args[2])
            self.window._advance_manual_activation('quotes', {}, None)
            self.assertTrue(self.window.engine.orders_enabled)
            self.window.engine.maintenance_checkpoint()
            tick.assert_called_once_with()
            self.window.disable_auto_orders()
            self.window.engine.maintenance_checkpoint()
            tick.assert_called_once_with()
        self.assertEqual(self.window.service.submitted, [])

    def test_idle_close_wakeup_runs_on_worker_without_postponing_hourly_poll(self):
        self.normal_window()
        self.window.engine.enable_orders('DEMO_AUTOTRADE')
        self.window.interval.setValue(3600)
        self.window.timer.start(3600 * 1000)
        before = self.window.timer.remainingTime()
        main_thread = threading.get_ident()
        called_from = []
        with patch.object(self.window.engine.close_liquidator, 'tick', side_effect=lambda: called_from.append(threading.get_ident())) as tick, \
                patch.object(self.window.engine, 'close_liquidation_status', return_value={'enabled': True}), \
                patch.object(self.window.timer, 'start', wraps=self.window.timer.start) as reschedule:
            self.window._close_wakeup()
            self.assertEqual(self.window._worker_kind, 'close-maintenance')
            self.drain()
        tick.assert_called_once_with()
        self.assertNotEqual(called_from[0], main_thread)
        reschedule.assert_not_called()
        self.assertTrue(self.window.timer.isActive())
        self.assertLessEqual(self.window.timer.remainingTime(), before)
        self.assertGreater(self.window.timer.remainingTime(), before - 5000)
        self.assertEqual(self.window.service.submitted, [])

    def test_poll_deadline_expiring_during_maintenance_is_run_immediately_afterward(self):
        self.normal_window()
        self.window.engine.enable_orders('DEMO_AUTOTRADE')
        entered, release = threading.Event(), threading.Event()
        def delayed():
            entered.set()
            release.wait(3)
        with patch.object(self.window.engine.close_liquidator, 'tick', side_effect=delayed), \
                patch.object(self.window.engine, 'close_liquidation_status', return_value={'enabled': True}):
            self.window._close_wakeup()
            self.assertTrue(entered.wait(1))
            self.window.timer.stop()  # A single-shot timer is inactive when its callback fires.
            self.window.refresh_all()
            self.assertTrue(self.window._poll_after_maintenance)
            with patch.object(self.window, 'refresh_all') as refresh:
                release.set()
                self.drain()
            refresh.assert_called_once_with()
        self.assertFalse(self.window._poll_after_maintenance)
        self.assertEqual(self.window.service.submitted, [])

    def test_close_wakeup_error_disarms_and_off_real_or_transition_do_not_dispatch(self):
        self.normal_window()
        with patch.object(self.window, '_run') as dispatch:
            self.window._close_wakeup()
            dispatch.assert_not_called()
        self.window.engine.enable_orders('DEMO_AUTOTRADE')
        self.window.service.mode = TradingMode.REAL
        try:
            with patch.object(self.window, '_run') as dispatch:
                self.window._close_wakeup()
                dispatch.assert_not_called()
        finally:
            self.window.service.mode = TradingMode.DEMO
        self.window._pending_environment = (self.candidate, self.store)
        try:
            with patch.object(self.window, '_run') as dispatch:
                self.window._close_wakeup()
                dispatch.assert_not_called()
        finally:
            self.window._pending_environment = None
        with patch.object(self.window.engine.close_liquidator, 'tick', side_effect=RuntimeError('fixture close failure')), \
                patch.object(self.window.timer, 'start') as reschedule:
            self.window._close_wakeup()
            self.drain()
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.pending_auto_arm)
        reschedule.assert_not_called()
        self.assertIn('fixture close failure', self.window.message.text())

    def test_separately_confirmed_on_during_maintenance_only_requests_fresh_warmup(self):
        self.normal_window()
        self.window.engine.enable_orders('DEMO_AUTOTRADE')
        entered, release = threading.Event(), threading.Event()
        def delayed():
            entered.set()
            release.wait(3)
        with patch.object(self.window.engine.close_liquidator, 'tick', side_effect=delayed), \
                patch.object(self.window.engine, 'close_liquidation_status', return_value={'enabled': True}), \
                patch.object(self.window, '_begin_manual_warmup') as warmup:
            self.window._close_wakeup()
            self.assertTrue(entered.wait(1))
            self.window.disable_auto_orders()
            # Stand in for a separately confirmed new ON request, not a timer's permission.
            self.window._manual_arm_pending = True
            self.window.pending_auto_arm = True
            release.set()
            self.drain()
            warmup.assert_called_once_with()
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.window.service.submitted, [])


if __name__ == '__main__':
    unittest.main()
