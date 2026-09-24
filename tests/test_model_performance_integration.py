"""Full-ledger worker and desktop integration, no operational account access."""
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from decimal import Decimal as D

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dockdack.activity_snapshot import LedgerCollector
from dockdack.model_performance import model_realized_performance
from dockdack.models import TradingMode
from dockdack.trading.performance import realized_performance
from dockdack.watchlist import WatchStore
from test_model_performance import order, OLD

HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from dockdack.v00_app import V00Window
    from test_autotrade import FakeTradingService


def records():
    return (order(1, 'buy', model=OLD, name='fixture', order_number='1', message=''),
            order(2, 'sell', price='103', name='fixture', order_number='2', message=''))


class ModelPerformanceCollectorTests(unittest.TestCase):
    def test_shares_one_fifo_calculation_for_entire_ledger(self):
        ledger = records() + tuple(order(n, 'buy', status='rejected') for n in range(3, 605))
        store = SimpleNamespace(mode=TradingMode.DEMO, order_history=Mock(return_value=ledger),
                                ledger_revision=Mock(return_value=1))
        with patch('dockdack.trade_journal.realized_performance', wraps=realized_performance) as fifo, \
                patch('dockdack.model_performance.realized_performance', side_effect=AssertionError('Second FIFO pass')):
            snapshot = LedgerCollector(store).collect()
        fifo.assert_called_once()
        store.order_history.assert_called_once_with(limit=None)
        self.assertEqual(snapshot.model_performance['coverage']['ledger_order_count'], len(ledger))
        row = next(r for r in snapshot.model_performance['rows'] if r['strategy_id'] == OLD and r['market'] == 'domestic')
        self.assertEqual(row['return_pct'], D(3))

    def test_unchanged_revision_does_not_recalculate_model_report(self):
        store = SimpleNamespace(mode=TradingMode.DEMO, order_history=Mock(return_value=records()),
                                ledger_revision=Mock(return_value=1))
        collector = LedgerCollector(store)
        with patch('dockdack.activity_snapshot.model_realized_performance', wraps=model_realized_performance) as calculate:
            before = collector.collect()
            self.assertIsNone(collector.collect())
            self.assertIs(collector.collect(force=True), before)
        calculate.assert_called_once()

    def test_current_real_account_is_not_labeled_as_demo_performance(self):
        store = SimpleNamespace(mode=TradingMode.REAL, order_history=Mock(return_value=()))
        report = LedgerCollector(store).collect().model_performance
        self.assertEqual(report['mode'], 'real')
        self.assertTrue(all(row['return_pct'] is None for row in report['rows']))


@unittest.skipUnless(HAS_QT, 'Install GUI extra')
class ModelPerformanceDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch('requests.sessions.Session.request', side_effect=AssertionError('No network')))
        self.store = WatchStore(Path(self.temp.name) / 'ledger.sqlite3')
        self.service = FakeTradingService()
        self.window = V00Window(self.service, self.store, builtin=False)
        self.addCleanup(self.close_window)
        for timer in self.window.findChildren(QTimer):
            timer.stop()
        self.drain()

    def drain(self):
        deadline = time.monotonic() + 15
        while (self.window._activity_worker or self.window._workspace_worker
               or self.window._schedule_probe or self.window._inspection_worker):
            self.app.processEvents()
            time.sleep(.002)
            self.assertLess(time.monotonic(), deadline)

    def close_window(self):
        self.window.close()
        self.drain()
        for pool in ('activity_pool', 'inspection_pool', 'pool'):
            getattr(self.window, pool).waitForDone(10000)
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_performance_tab_reads_locally_off_main_thread_without_selected_models(self):
        panel = self.window.model_performance_panel
        threads = []
        def history(*, limit):
            self.assertIsNone(limit)
            threads.append(threading.get_ident())
            return records()
        with patch.object(self.store, 'order_history', side_effect=history), \
                patch.object(self.store, 'ledger_revision', return_value='changed'):
            self.window.workspace_tabs.setCurrentWidget(panel)
            self.drain()
            panel.refresh_button.click()
            self.drain()
        self.assertTrue(threads)
        self.assertTrue(all(thread != threading.get_ident() for thread in threads))
        self.assertEqual(panel.tables['domestic'].item(0, 1).text(), '+3.00%')
        self.assertIn('DEMO', panel.mode_badge.text())
        self.assertFalse(self.window._chosen_external_models())
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_read_error_is_visible_and_retains_previous_evidence(self):
        panel = self.window.model_performance_panel
        report = model_realized_performance(records(), mode='demo')
        panel.apply_report(report)
        self.window._activity_completed(None, RuntimeError('fixture read error'))
        self.assertIs(panel.report, report)
        self.assertFalse(panel.refresh_error.isHidden())
        self.assertIn('갱신 실패', panel.refresh_error.text())
        self.assertEqual(panel.tables['domestic'].item(0, 1).text(), '+3.00%')

    def test_both_model_rows_fit_in_compact_desktop_before_scrolling(self):
        self.window.resize(1280, 720)
        self.window.show()
        panel = self.window.model_performance_panel
        self.window.workspace_tabs.setCurrentWidget(panel)
        self.drain()
        panel.apply_report(model_realized_performance(records(), mode='demo'))
        self.app.processEvents()
        view = panel.tables['domestic']
        second = view.visualRect(view.model().index(1, 0)).bottomRight()
        row_bottom = view.viewport().mapTo(self.window, second).y()
        viewport = self.window.workspace_scroll.viewport()
        self.assertLessEqual(row_bottom, viewport.mapTo(self.window, viewport.rect().bottomRight()).y())
        self.assertLessEqual(self.window.height(), 720)

    def test_late_different_store_result_cannot_cross_environment(self):
        other = WatchStore(Path(self.temp.name) / 'real.sqlite3', mode=TradingMode.REAL,
                           storage_scope='b' * 64)
        snapshot = LedgerCollector(SimpleNamespace(mode='demo', order_history=lambda **_: records())).collect()
        panel = self.window.model_performance_panel
        panel.apply_snapshot(snapshot)
        self.window.store = other
        panel.set_context(other)
        try:
            self.window._activity_completed((self.store, {'heads': {}, 'events': {}}, snapshot), None)
            self.assertIn('REAL', panel.mode_badge.text())
            self.assertIsNone(panel.report)
            self.assertEqual(panel.tables['domestic'].item(0, 1).text(), '—')
        finally:
            self.window.store = self.store
            panel.set_context(self.store)
        self.assertFalse(self.window.engine.orders_enabled)


if __name__ == '__main__':
    unittest.main()
