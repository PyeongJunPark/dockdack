"""The top model summary changes only after a complete quote-level result."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from dockdack.v00_app import MARK1_TRIGGER, MARK11_TRIGGER, MARK12_TRIGGER, V00Window

from dockdack.history import DailyHistory
from dockdack.models import Quote
from dockdack.watchlist import MarketSnapshot, WatchItem, WatchStore
from test_autotrade import FakeTradingService


NOW = datetime(2026, 9, 15, 1, 0, tzinfo=timezone.utc)


@unittest.skipUnless(HAS_QT, 'Install the gui extra')
class LatestModelSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WatchStore(Path(self.temp.name) / 'ledger.sqlite3')
        self.service = FakeTradingService()
        self.item = WatchItem(self.service.resolve('005930'), '삼성전자')
        self.store.save_item(self.item)
        self.network = patch('requests.sessions.Session.request', side_effect=AssertionError('Offline test'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.window = V00Window(self.service, self.store, builtin=False)
        for timer in self.window.findChildren(QTimer):
            timer.stop()
        self.window.engine.clock = lambda: NOW
        self._drain()
        self.feeds = {}
        for model in (MARK1_TRIGGER, MARK11_TRIGGER, MARK12_TRIGGER):
            self.feeds[model] = SimpleNamespace(_ready=True, diagnostics={}, close=Mock())
        self.window._prototype_feeds = self.feeds

    def _drain(self):
        deadline = time.monotonic() + 15
        while (self.window._workspace_worker or self.window._activity_worker
               or self.window._schedule_probe or self.window._inspection_worker):
            self.app.processEvents()
            time.sleep(.002)
            self.assertLess(time.monotonic(), deadline)

    def tearDown(self):
        self.window.close()
        self._drain()
        for name in ('activity_pool', 'inspection_pool', 'pool'):
            getattr(self.window, name).waitForDone(10000)
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def _choose(self, *models):
        for model, check in self.window.external_model_checks.items():
            check.blockSignals(True)
            check.setChecked(model in models)
            check.blockSignals(False)
        self.window._refresh_model_scores()

    @staticmethod
    def _snapshot(item, price, fetched_at):
        instrument = item.instrument
        quote = Quote(instrument.market, instrument.symbol, item.name, instrument.exchange,
                      Decimal(price), instrument.currency)
        history = DailyHistory(instrument.market, instrument.symbol, instrument.exchange,
                               instrument.currency, 0, ())
        return MarketSnapshot(quote, history, fetched_at)

    def _query(self, item, price, fetched_at):
        self.window._progress((item.id, self._snapshot(item, price, fetched_at), 1, 1))

    def _score(self, model, item, price, fetched_at, probability):
        self.feeds[model].diagnostics[item.id] = {
            'watch_id': item.id,
            'reason': ('PREDICTED_DAILY_BARRIER_SUCCESS' if Decimal(probability) > Decimal('0.5')
                       else 'BELOW_OR_EQUAL_BUY_THRESHOLD'),
            'reference_price': price,
            'prediction': {'probability_success': probability},
            '_display_quote_fetched_at': fetched_at.isoformat(),
            '_display_price': price,
        }
        self.window._refresh_model_scores()

    def test_new_quote_keeps_previous_complete_result_until_all_models_arrive(self):
        self._choose(MARK1_TRIGGER, MARK11_TRIGGER)
        self._query(self.item, '100', NOW)
        self._score(MARK1_TRIGGER, self.item, '100', NOW, '0.638')
        self._score(MARK11_TRIGGER, self.item, '100', NOW, '0.412')
        previous = self.window.latest_model_summary.text()
        self.assertIn('005930', previous)
        self.assertIn('100 KRW', previous)
        self.assertIn(NOW.astimezone().strftime('%m/%d %H:%M:%S'), previous)
        self.assertIn('52.5%', previous)

        newer = NOW + timedelta(seconds=1)
        self._query(self.item, '101', newer)
        self.assertEqual(self.window.latest_model_summary.text(), previous)
        self._score(MARK1_TRIGGER, self.item, '101', newer, '0.700')
        self.assertEqual(self.window.latest_model_summary.text(), previous)
        self._score(MARK11_TRIGGER, self.item, '101', newer, '0.500')
        current = self.window.latest_model_summary.text()
        self.assertIn('101 KRW', current)
        self.assertIn(newer.astimezone().strftime('%m/%d %H:%M:%S'), current)
        self.assertIn('60.0%', current)
        self.assertNotIn('52.5%', current)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_new_symbol_keeps_previous_complete_result_until_all_models_arrive(self):
        self._choose(MARK1_TRIGGER, MARK11_TRIGGER)
        self._query(self.item, '100', NOW)
        self._score(MARK1_TRIGGER, self.item, '100', NOW, '0.638')
        self._score(MARK11_TRIGGER, self.item, '100', NOW, '0.412')
        previous = self.window.latest_model_summary.text()

        other = WatchItem(self.service.resolve('000660'), 'SK하이닉스')
        self.store.save_item(other)
        self.window.reload_tables(items=self.store.items(), rules=[])
        newer = NOW + timedelta(seconds=1)
        self._query(other, '200', newer)
        self.assertEqual(self.window.latest_model_summary.text(), previous)
        self._score(MARK1_TRIGGER, other, '200', newer, '0.700')
        self.assertEqual(self.window.latest_model_summary.text(), previous)
        self._score(MARK11_TRIGGER, other, '200', newer, '0.500')
        current = self.window.latest_model_summary.text()
        self.assertIn('000660', current)
        self.assertIn('200 KRW', current)
        self.assertIn('60.0%', current)
        self.assertNotIn('005930', current)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_model_set_change_never_reuses_an_average_from_old_set(self):
        self._choose(MARK1_TRIGGER, MARK11_TRIGGER)
        self._query(self.item, '100', NOW)
        self._score(MARK1_TRIGGER, self.item, '100', NOW, '0.638')
        self._score(MARK11_TRIGGER, self.item, '100', NOW, '0.412')
        self.assertIn('52.5%', self.window.latest_model_summary.text())

        self._choose(MARK1_TRIGGER, MARK11_TRIGGER, MARK12_TRIGGER)
        pending = self.window.latest_model_summary.text()
        self.assertIn('모델 추정확률 —', pending)
        self.assertNotIn('52.5%', pending)
        self._score(MARK12_TRIGGER, self.item, '100', NOW, '0.552')
        self.assertIn('53.4%', self.window.latest_model_summary.text())

        self._choose(MARK1_TRIGGER, MARK12_TRIGGER)
        self.assertIn('59.5%', self.window.latest_model_summary.text())
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])


if __name__ == '__main__':
    unittest.main()
