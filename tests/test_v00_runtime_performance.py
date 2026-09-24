"""Offline threading/cache regression checks for the v0.0 desktop."""
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from dockdack.v00_app import V00Window
from dockdack.models import AccountSnapshot, Market
from dockdack.portfolio import PortfolioMarketState
from dockdack.watchlist import WatchStore, WatchItem
from test_autotrade import FakeTradingService, position
from test_lstm30_adapter import NOW, chart, prediction, producer


class PredictionCacheTests(unittest.TestCase):
    def test_same_completed_bars_infer_once_but_changed_bars_invalidate(self):
        predictor = SimpleNamespace(metadata={'market': 'domestic'}, predict=Mock(return_value=prediction()))
        model = producer(predictor=predictor)
        for index in range(30):
            data = chart()
            data['export_id'] = f'export-{index}'
            data['stocks'][0]['price'] = str(100 + index)
            payload, _ = model(data)
            self.assertEqual(payload['signals'][0]['action'], 'buy')
        self.assertEqual(predictor.predict.call_count, 1)
        data = chart()
        data['export_id'] = 'changed-bars'
        data['stocks'][0]['bars'][0]['volume'] = '2000'
        model(data)
        self.assertEqual(predictor.predict.call_count, 2)

    @unittest.skipUnless(HAS_QT, 'Install gui extra')
    def test_unavailable_native_model_emits_hold_without_repeated_import_or_orders(self):
        from dockdack.v00_app import DesktopModelBridge
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            service = FakeTradingService()
            engine = SimpleNamespace(_stop=threading.Event(), clock=lambda: NOW,
                external_reader=SimpleNamespace(path=path / 'signals.json'),
                external_policy=SimpleNamespace(max_krw=Decimal(10000000), max_usd=Decimal(10000)))
            bridge = DesktopModelBridge(SimpleNamespace(engine=engine, service=service,
                store=SimpleNamespace(path=path / 'ledger.sqlite3')))
            with patch.dict('sys.modules', {'dockdack.ml30': None}):
                bridge.publish(chart())
            self.assertTrue(bridge._load_error)
            data = chart()
            data['export_id'] = 'another-export'
            bridge.publish(data)  # Does not retry importing the blocked dependency.
            payload = json.loads((path / 'signals.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['signals'][0]['action'], 'hold')
            self.assertEqual(payload['signals'][0]['export_id'], 'another-export')
            self.assertNotIn('quantity', payload['signals'][0])
            self.assertIn('실행 불가', bridge.status)
            self.assertEqual(service.quote_calls, 0)
            self.assertFalse(service.submitted)


@unittest.skipUnless(HAS_QT, 'Install gui extra')
class BackgroundUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WatchStore(Path(self.temp.name) / 'test.sqlite3')
        self.service = FakeTradingService()
        self.item = WatchItem(self.service.resolve('005930'), '삼성전자')
        self.store.save_item(self.item)
        self.window = V00Window(self.service, self.store, builtin=False)
        self.window.engine.clock = lambda: NOW
        for timer in self.window.findChildren(QTimer):
            timer.stop()
        self.drain()

    def drain(self):
        end = time.monotonic() + 10
        while self.window._activity_worker or self.window._schedule_probe or self.window._workspace_worker:
            self.window.activity_pool.waitForDone(100)
            self.app.processEvents()
            self.assertLess(time.monotonic(), end, 'background worker did not finish')

    def tearDown(self):
        self.window.close()
        self.drain()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_visible_activity_collection_has_no_gui_thread_database_reads(self):
        self.store.event('SYSTEM', 'background log', category='system')
        main_thread = threading.get_ident()
        connection = self.store.connection
        threads = []
        def checked():
            threads.append(threading.get_ident())
            self.assertNotEqual(threading.get_ident(), main_thread)
            return connection()
        with patch.object(self.store, 'connection', side_effect=checked):
            self.window._reload_activity(force=True)
            self.drain()
        self.assertTrue(threads)
        self.assertIn('background log', self.window.operations_panel.logs['system'].table.item(0, 2).text())
        self.assertIs(self.window.order_history_panel._applied_snapshot,
                      self.window.trade_journal_panel._applied_snapshot)

    def test_schedule_probe_is_background_and_once_per_minute(self):
        self.window.monitoring = True
        threads = []
        with patch.object(self.window.scheduler, 'due', side_effect=lambda: threads.append(threading.get_ident()) or False):
            self.window._schedule_wakeup()
            self.drain()
            self.window._schedule_wakeup()
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0], threading.get_ident())
        self.assertFalse(self.service.submitted)

    def test_holdings_progress_updates_quote_and_targets_without_io_or_whole_table_rebuild(self):
        held = position()
        panel = self.window.portfolio_panel
        panel.apply({Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC,
            AccountSnapshot(Market.DOMESTIC, 'KRW', (held,)), NOW, NOW)}, now=NOW)
        name_cell = panel.table.item(0, 1)
        quote = self.service.quote(self.item.instrument)
        with patch.object(self.store, 'connection', side_effect=AssertionError('No GUI database work')):
            self.window._progress(('phase', '보유종목 매도 조건 점검'))
            self.window._progress(('holding_quote', {'watch_id': self.item.id, 'instrument': self.item.instrument,
                'quote': quote, 'position': held,
                'targets': {'take_profit_price': Decimal(120), 'stop_loss_price': Decimal(95), 'source': 'alpha'}}))
        self.assertIs(panel.table.item(0, 1), name_cell)
        self.assertIn('120', panel.table.item(0, 9).text())
        self.assertIn('95', panel.table.item(0, 10).text())
        self.assertIn('독립 매도 감시', panel.table.item(0, 5).toolTip())
        self.assertFalse(self.service.submitted)


if __name__ == '__main__':
    unittest.main()
