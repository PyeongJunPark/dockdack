"""Offline market badges and honest quote/holdings progress indicators."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from dockdack.market_status import market_statuses
from dockdack.models import Market
from dockdack.v00_app import V00Window
from dockdack.watchlist import WatchItem, WatchStore
from test_autotrade import FakeTradingService


class MonitorStatusGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = FakeTradingService()
        store = WatchStore(Path(self.temp.name) / 'test.sqlite3')
        store.save_item(WatchItem(self.service.resolve('005930')))
        self.window = V00Window(self.service, store, builtin=False)
        for timer in self.window.findChildren(QTimer):
            timer.stop()
        self.drain()

    def drain(self):
        for _ in range(30):
            self.window.activity_pool.waitForDone(1000)
            self.app.processEvents()
            if self.window._activity_worker is None and self.window._schedule_probe is None:
                return
        self.fail('background display worker did not finish')

    def tearDown(self):
        self.window.worker = None
        self.window.stop_monitoring()
        self.drain()
        self.window.close()
        self.window.activity_pool.waitForDone(5000)
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def emit_holding(self, phase, **extra):
        self.window._progress(('holdings_progress', dict(phase=phase, market=Market.DOMESTIC,
            completed=extra.pop('completed', 0), total=extra.pop('total', 3), **extra)))

    def test_holdings_account_and_each_symbol_show_busy_then_actual_completion(self):
        bar = self.window.sweep_progress
        self.emit_holding('account')
        self.assertEqual((bar.minimum(), bar.maximum()), (0, 0))
        self.assertIn('한국 보유종목 목록 조회 중', bar.format())
        self.emit_holding('checking', symbol='005930', completed=1)
        self.assertIn('보유종목 매도 조건 확인 1/3종목', bar.format())
        self.assertIn('005930', bar.format())
        self.assertEqual(bar.maximum(), 0)
        self.emit_holding('checked', symbol='005930', completed=2)
        self.assertEqual((bar.value(), bar.maximum()), (2, 3))
        self.assertIn('체결 성공을 의미하지 않습니다', bar.toolTip())
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_failed_holding_advances_checked_count_but_is_not_success_claim(self):
        self.emit_holding('checked', completed=1, symbol='005930', error='quote timed out')
        bar = self.window.sweep_progress
        self.assertEqual(bar.value(), 1)
        self.assertIn('확인 실패', bar.format())
        self.assertEqual(bar.toolTip(), 'quote timed out')

    def test_busy_text_is_painted_and_wait_does_not_fake_completed_percentage(self):
        bar = self.window.sweep_progress
        with patch('dockdack.v00_widgets.monotonic', return_value=100):
            self.emit_holding('checking', symbol='005930')
        with patch('dockdack.v00_widgets.monotonic', return_value=109):
            bar.refresh_wait()
        self.assertIn('응답 대기 9초', bar.format())
        self.assertEqual(bar.maximum(), 0)
        with patch('dockdack.v00_widgets.monotonic', return_value=225):
            bar.refresh_wait()
        self.assertIn('응답 지연 확인 필요', bar.format())
        self.assertFalse(bar.grab().isNull())

    def test_watch_start_and_holdings_start_are_distinct(self):
        self.window._progress(('watch_progress', dict(market=Market.US, symbol='AAPL', name='Apple', completed=2, total=10)))
        self.assertIn('미국 관심종목 시세·차트 조회 2/10종목', self.window.sweep_progress.format())
        self.emit_holding('account')
        self.assertNotIn('시세·차트 조회', self.window.sweep_progress.format())

    def test_badges_closed_market_wait_and_regular_hours_tooltips(self):
        now = datetime(2026, 9, 16, 6, 45, tzinfo=timezone.utc)
        self.window._apply_market_status(market_statuses(now))
        self.assertIn('장 마감', self.window.market_labels[Market.DOMESTIC].text())
        self.assertIn('장전', self.window.market_labels[Market.US].text())
        self.assertIn('22:30', self.window.market_labels[Market.US].toolTip())
        self.assertTrue(self.window._all_markets_closed())
        self.emit_holding('complete', total=0)
        self.assertIn('장외 대기', self.window.sweep_progress.format())
        self.assertEqual(self.window.sweep_progress.value(), 0)
        self.assertEqual(self.service.quote_calls, 0)

    def test_unknown_market_is_not_reported_as_closed(self):
        now = datetime(2026, 9, 16, 6, 45, tzinfo=timezone.utc)
        statuses = market_statuses(now)
        statuses[Market.DOMESTIC].update(is_open=None, state='unknown', text='한국 · 장 시간 확인 필요')
        self.window._apply_market_status(statuses)
        self.assertFalse(self.window._all_markets_closed())
        self.assertEqual(self.window.market_labels[Market.DOMESTIC].property('tone'), 'error')

    def test_background_badges_never_replace_activation_sweep_market_state(self):
        self.window._market_open = {Market.DOMESTIC: True, Market.US: False}
        self.window._apply_market_status(market_statuses(datetime(2026, 9, 16, 6, 45, tzinfo=timezone.utc)))
        self.assertTrue(self.window._all_markets_closed())
        self.assertTrue(self.window._market_open[Market.DOMESTIC])

    def test_off_session_calendar_refresh_does_not_call_broker_or_ranking(self):
        self.window._market_status_minute = None
        with patch.object(self.window.scheduler, 'due', side_effect=AssertionError('OFF must not select ranks')):
            self.window._schedule_wakeup()
            self.drain()
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])


if __name__ == '__main__':
    unittest.main()
