"""Temporary storage and fake accounts only: limits, recovery, runtime locks."""
import os
import subprocess
import sys
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from dockdack.models import Market, TradingMode
from dockdack.session_guard import AccountSessionLock
from dockdack.trading.exposure_limits import remaining_buy_budget
from dockdack.trading.ledger_diagnostics import reconciliation_report
from dockdack.trading.pipeline_metrics import PipelineMetrics
from dockdack.ui.model_runtime import model_runtime_state


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.key = 'domestic:KRX:005930'
        self.position = NS(market=Market.DOMESTIC, exchange='KRX', symbol='005930',
                           quantity=D(20), sellable_quantity=D(20), current_price=D(100), evaluation_amount=D(2000))
        self.account = NS(market=Market.DOMESTIC, cash=D(8000), total_evaluation=D(2000), positions=(self.position,))
        self.store = NS(prototype_lots=Mock(return_value=[]), order_history=Mock(return_value=[]),
                        prototype_rule_source=Mock(return_value='model'), prototype_inventory=Mock(return_value={'reconciled': True}))

    def test_disabled_does_not_read_ledger(self):
        self.assertIsNone(remaining_buy_budget(self.store, self.account, self.key, 'model'))
        self.store.order_history.assert_not_called()

    def test_symbol_counts_existing_holdings_and_pending_unfilled_only(self):
        self.store.order_history.return_value = [dict(rule_id='buy', market='domestic', side='buy', status='accepted',
            watch_id=self.key, quantity='10', filled_quantity='4', reference_price='100')]
        self.assertEqual(remaining_buy_budget(self.store, self.account, self.key, 'model', 30, 0), D(394))

    def test_model_counts_owned_lots_and_tighter_limit(self):
        self.store.prototype_lots.return_value = [dict(watch_id=self.key, buy_rule_id='b', quantity_remaining=D(10), issues=())]
        self.assertEqual(remaining_buy_budget(self.store, self.account, self.key, 'model', 50, 15), D(500))

    def test_pending_reservation_does_not_round_reference_price_down(self):
        self.store.order_history.return_value = [dict(rule_id='buy', market='domestic', side='buy', status='accepted',
            watch_id=self.key, quantity='10', filled_quantity='0', reference_price='100.1')]
        self.assertEqual(remaining_buy_budget(self.store, self.account, self.key, 'model', 50, 0), D('1988.99'))

    def test_unreconciled_model_blocks_without_guessing(self):
        self.store.prototype_lots.return_value = [dict(watch_id=self.key, buy_rule_id='b', quantity_remaining=D(10), issues=())]
        self.store.prototype_inventory.return_value = {'reconciled': False}
        with self.assertRaises(ValueError):
            remaining_buy_budget(self.store, self.account, self.key, 'model', 0, 15)

    def test_bad_percentage_and_unknown_reservation_fail_closed(self):
        for percent in ('NaN', '-1', '101'):
            with self.subTest(percent=percent), self.assertRaises(ValueError):
                remaining_buy_budget(self.store, self.account, self.key, 'model', percent, 0)
        self.store.order_history.return_value = [dict(rule_id='buy', market='domestic', side='buy', status='unknown',
            watch_id=self.key, quantity='10', filled_quantity=None, reference_price='0')]
        with self.assertRaises(ValueError):
            remaining_buy_budget(self.store, self.account, self.key, 'model', 30, 0)


class SessionGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'LOCALAPPDATA': str(self.root / 'local'),
                              'DOCKDACK_ACCOUNT_LOCK_ID': '', 'DOCKDACK_SHARED_LOCK_DIR': ''})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.service = NS(mode=TradingMode.DEMO, _frozen_configs={'domestic': NS(app_key='fake-key')})

    def lock(self, name, **kwargs):
        lock = AccountSessionLock(self.root / name / 'session.lock', self.service, **kwargs)
        self.addCleanup(lock.release)
        return lock

    def test_different_copy_same_key_blocked_then_released(self):
        owner, other = self.lock('a'), self.lock('b')
        self.assertIsNone(owner.stream)
        owner.acquire()
        self.assertIsNotNone(owner.stream)
        with self.assertRaises(RuntimeError):
            other.acquire()
        owner.release()
        self.assertIsNone(owner.stream)
        other.acquire()

    def test_environment_transfer_keeps_overlapping_key_locked(self):
        owner = self.lock('a')
        owner.acquire()
        successor = self.lock('b', previous=owner)
        successor.acquire()
        owner.release()
        outsider = self.lock('c')
        with self.assertRaises(RuntimeError):
            outsider.acquire()
        successor.release()
        outsider.acquire()

    def test_shared_directory_blocks_separate_local_roots(self):
        with patch.dict(os.environ, {'DOCKDACK_SHARED_LOCK_DIR': str(self.root / 'shared')}):
            owner = self.lock('a')
            owner.acquire()
            with patch.dict(os.environ, {'LOCALAPPDATA': str(self.root / 'other-pc')}):
                other = self.lock('b')
                with self.assertRaises(RuntimeError):
                    other.acquire()

    def test_unavailable_shared_path_releases_all_partial_locks(self):
        occupied = self.root / 'file'
        occupied.touch()
        with patch.dict(os.environ, {'DOCKDACK_SHARED_LOCK_DIR': str(occupied / 'locks')}):
            with self.assertRaises(RuntimeError):
                self.lock('a').acquire()
        self.lock('a').acquire()

    def test_process_exit_releases_lock_and_other_process_is_blocked_while_owned(self):
        code = """import os, sys
from pathlib import Path
from types import SimpleNamespace as NS
from dockdack.models import TradingMode
from dockdack.session_guard import AccountSessionLock
service = NS(mode=TradingMode.DEMO, _frozen_configs={'domestic': NS(app_key='fake-key')})
lock = AccountSessionLock(Path(sys.argv[1]) / 'session.lock', service)
try:
    lock.acquire()
except RuntimeError:
    sys.exit(7)
os._exit(0)
"""
        def child():
            return subprocess.run([sys.executable, '-c', code, str(self.root / 'child')],
                                  capture_output=True, timeout=20,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).returncode
        owner = self.lock('a')
        owner.acquire()
        self.assertEqual(child(), 7)
        owner.release()
        self.assertEqual(child(), 0)
        owner.acquire()


class DiagnosticTests(unittest.TestCase):
    def test_report_separates_causes_and_preserves_order_evidence(self):
        report = reconciliation_report({'issues': ('보유량 불일치', '모델별 가중평균 체결가 미확인', '매수 모델 출처 불일치'),
            'lots': [{'model_title': 'mk1', 'buy_order_number': '123', 'buy_rule_id': 'buy1', 'quantity_remaining': 2}]})
        for part in ('수량·배분 확인', '체결 가격 확인', '매수 출처 확인', '123', 'buy1'):
            self.assertIn(part, report)

    def test_bounded_timings_and_rejection_counts(self):
        metrics = PipelineMetrics(capacity=10)
        for i in range(1000):
            metrics.record('final_send_blocked', i / 100)
        stats = metrics.snapshot()
        self.assertEqual(stats['samples'], 10)
        self.assertEqual(stats['counts']['final_send_blocked'], 1000)
        self.assertEqual(stats['max_seconds'], 9.99)

    def test_research_and_unprepared_models_never_show_order_ready(self):
        self.assertIn('주문 없음', model_runtime_state(selected=True, research=True, ready=True, orders_enabled=True))
        self.assertIn('대기', model_runtime_state(selected=True, research=False, ready=False, orders_enabled=True))
        self.assertIn('OFF', model_runtime_state(selected=True, research=False, ready=True, orders_enabled=False))
