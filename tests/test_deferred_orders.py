from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from dockdack.demo_session import SessionController
    from dockdack.watch_gui import WatchlistDialog

from dockdack.models import Market, Quote
from dockdack.watchlist import MarketSnapshot, TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class DeferredOrderTests(unittest.TestCase):
    """Drive broker completion slots without broker threads; drain local log workers."""

    @classmethod
    def setUpClass(cls):
        # QTest.qWait can starve a worker's first numpy/pandas calendar import.
        # Like the other GUI interaction fixtures, warm calendars on this
        # thread before starting the display-only background badge worker.
        from dockdack.market_schedule import calendar_for
        for market in Market:
            calendar_for(market, 2026)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.store = WatchStore(self.folder / "deferred.sqlite3")
        self.service = FakeTradingService()
        self.items = [WatchItem(self.service.resolve(symbol), symbol) for symbol in ("005930", "AAPL")]
        for item in self.items:
            self.store.save_item(item)
        self.window = WatchlistDialog(self.service, self.store)
        self.window.engine.clock = lambda: NOW
        self.window.portfolio.clock = lambda: datetime.now(timezone.utc)
        self.window.hourly_ranking.setChecked(False)
        self.window.external_mode.setChecked(True)
        self.window.external_krw.setValue(1000)
        self.window.external_usd.setValue(1000)
        self.window.configure_external()
        self.window._portfolio_payload = self.window.portfolio.refresh_due()
        self.results = {
            item.id: MarketSnapshot(
                Quote(item.instrument.market, item.instrument.symbol, item.name, item.instrument.exchange,
                      Decimal(100), item.instrument.currency),
                self.service.history(item.instrument, item.days), NOW,
            ) for item in self.items
        }
        self.window.snapshots.update(self.results)
        self.window.fresh_ids.update(self.results)
        # Captured Workers still have real Qt signals and normal completion
        # handling, but cannot call either fake or real broker methods.
        self.pool_patch = patch.object(self.window.pool, "start")
        self.started = self.pool_patch.start()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.health_timer.stop()
        self.window.order_status_timer.stop()
        self.window.environment_timer.stop()
        self.window.worker = None
        self.window.stop_monitoring()
        self.wait_local_workers()
        self.window.close()
        self.wait_local_workers()
        self.window.deleteLater()
        self.app.processEvents()
        self.pool_patch.stop()
        self.temp.cleanup()

    def wait_local_workers(self):
        deadline = time.monotonic() + 10
        idle_rounds = 0
        while idle_rounds < 2:
            self.app.processEvents()
            busy = (self.window._activity_worker or self.window._activity_pending
                    or self.window._schedule_probe or self.window.activity_pool.activeThreadCount())
            idle_rounds = 0 if busy else idle_rounds + 1
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline, "Local activity worker did not finish")

    def busy(self, kind="quotes", *, monitoring=True):
        self.window.monitoring = monitoring
        self.assertTrue(self.window._run(lambda: self.results, job_kind=kind))
        return self.window.worker

    def request(self):
        with patch.object(self.window, "confirm_automation", return_value=True) as confirm:
            self.window.arm_button.click()
            confirm.assert_called_once()
        self.assertTrue(self.window.pending_auto_arm)
        self.assertTrue(self.window._manual_arm_pending)
        self.assertFalse(self.window.engine.orders_enabled)

    def complete(self, results=None, error=None):
        worker = self.window.worker
        self.assertIsNotNone(worker)
        worker.signals.completed.emit(self.results if results is None else results, error)
        self.app.processEvents()

    def assert_off_cancelled(self):
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertFalse(self.window._manual_arm_pending)
        self.assertIn("자동주문 OFF", self.window.mode_label.text())
        self.assertEqual(self.service.submitted, [])

    def test_busy_on_confirms_once_and_arms_only_after_this_complete_successful_poll(self):
        first = self.busy()
        self.assertTrue(self.window.arm_button.isEnabled())
        with patch.object(self.window, "confirm_automation", return_value=True) as confirm:
            self.window.arm_button.click()
            self.window.arm_button.click()
            self.window.enable_auto_orders()
            confirm.assert_called_once()
        self.assertIs(self.window.worker, first)
        self.assertTrue(self.window.pending_auto_arm)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertTrue(self.window.disarm_button.isEnabled())
        with patch.object(self.window.engine, "enable_orders", wraps=self.window.engine.enable_orders) as enable:
            self.complete()
            enable.assert_called_once_with("DEMO_AUTOTRADE")
        self.assertTrue(self.window.engine.orders_enabled)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertFalse(self.window._manual_arm_pending)
        self.assertIsNot(self.window.worker, first)
        self.assertEqual(self.service.submitted, [])

    def test_idle_stopped_on_starts_monitoring_and_warmup_without_enabling_early(self):
        self.window.stop_monitoring()
        self.assertTrue(self.window.arm_button.isEnabled())
        self.request()
        self.assertTrue(self.window.monitoring)
        self.assertIsNotNone(self.window.worker)
        self.assertEqual(self.started.call_count, 1)
        self.complete()
        self.assertTrue(self.window.engine.orders_enabled)
        self.assertTrue(self.window.monitoring)
        self.assertEqual(self.service.submitted, [])

    def test_incomplete_results_fail_even_with_every_item_previously_fresh(self):
        self.busy()
        self.request()
        self.complete({self.items[0].id: self.results[self.items[0].id]})
        self.assert_off_cancelled()
        self.assertIsNone(self.window.worker)
        self.busy()
        self.complete()
        self.assert_off_cancelled()

    def test_symbol_rate_limit_failure_cancels_without_retrying_activation(self):
        self.busy()
        self.request()
        results = dict(self.results)
        results[self.items[0].id] = ValueError("허용된 API 요청 개수를 초과하였습니다")
        self.complete(results)
        self.assert_off_cancelled()
        self.assertIsNone(self.window.worker)
        self.busy()
        self.complete()
        self.assert_off_cancelled()

    def test_worker_exception_cancels_pending_and_stays_off(self):
        self.busy()
        self.request()
        self.complete({}, ValueError("unexpected worker failure"))
        self.assert_off_cancelled()

    def test_current_account_error_prevents_automatic_on_after_quotes_succeed(self):
        self.busy()
        self.request()
        states = self.window.portfolio.snapshot()
        failed = replace(states[Market.DOMESTIC], error="잔고 조회 요청 제한")
        self.window._portfolio_payload[Market.DOMESTIC] = failed
        self.window.portfolio._states[Market.DOMESTIC] = failed
        self.complete()
        self.assert_off_cancelled()

    def test_current_external_file_error_prevents_automatic_on(self):
        self.busy()
        self.request()
        self.window.engine.external_error = "현재 신호 파일 읽기 실패"
        self.complete()
        self.assert_off_cancelled()

    def test_error_recovered_during_same_sweep_still_cancels_this_approval(self):
        self.busy()
        self.request()
        self.window.engine.external_error_count += 1
        self.window.engine.external_error = ""
        self.complete()
        self.assert_off_cancelled()

    def test_stale_account_prevents_automatic_on_even_with_no_api_error(self):
        self.busy()
        self.request()
        state = self.window._portfolio_payload[Market.US]
        self.window._portfolio_payload[Market.US] = replace(
            state, fetched_at=datetime.now(timezone.utc) - timedelta(minutes=3),
        )
        self.complete()
        self.assert_off_cancelled()

    def test_unknown_account_cannot_be_treated_as_successful_empty_holdings(self):
        self.busy()
        self.request()
        state = self.window._portfolio_payload[Market.US]
        self.window._portfolio_payload[Market.US] = replace(state, snapshot=None, fetched_at=None)
        self.complete()
        self.assert_off_cancelled()

    def test_old_resolved_external_error_does_not_block_new_explicit_approval(self):
        self.window.engine.external_error_count = 3
        self.window.engine.external_error = ""
        self.window.engine._messages["external:file"] = "이전 순회에서 이미 해결된 오류"
        self.busy()
        self.request()
        self.complete()
        self.assertTrue(self.window.engine.orders_enabled)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertEqual(self.service.submitted, [])

    def test_current_scheduler_error_prevents_automatic_on(self):
        self.window.hourly_ranking.setChecked(True)
        self.busy()
        self.request()
        self.window.scheduler.errors[Market.DOMESTIC.value] = "관심종목 조회 실패"
        self.complete()
        self.assert_off_cancelled()

    def test_disabled_external_mode_and_scheduler_ignore_their_old_errors(self):
        self.window.engine.external_only = False
        self.window.engine.external_error = "이전 외부 모드 읽기 오류"
        self.window.engine.external_error_count = 5
        self.window.scheduler.errors[Market.DOMESTIC.value] = "중지된 재선정의 이전 오류"
        self.assertFalse(self.window.hourly_ranking.isChecked())
        rule = TriggerRule.create(self.items[0], "price_ge", "buy", 1, Decimal(1000), Decimal(95))
        self.store.add_rule(rule)
        self.busy()
        self.request()
        self.complete()
        self.assertTrue(self.window.engine.orders_enabled)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertEqual(self.service.submitted, [])

    def test_cancelled_manual_request_is_not_revived_by_old_launcher_callback(self):
        controller = SessionController(self.window, self.folder, auto_arm=False)
        self.busy()
        self.request()
        self.window.disable_auto_orders()
        with patch.object(self.window.engine, "enable_orders") as enable:
            controller.warmup_finished(self.results, None)
            self.complete()
            enable.assert_not_called()
        self.assert_off_cancelled()

    def test_off_cancels_pending_and_late_success_cannot_rearm(self):
        self.busy()
        self.request()
        self.window.disarm_button.click()
        self.assertTrue(self.window.monitoring)
        with patch.object(self.window.engine, "enable_orders") as enable:
            self.complete()
            enable.assert_not_called()
        self.assert_off_cancelled()

    def test_stop_cancels_pending_and_late_success_cannot_restart_monitoring(self):
        self.busy()
        self.request()
        self.window.stop_button.click()
        with patch.object(self.window.engine, "enable_orders") as enable:
            self.complete()
            enable.assert_not_called()
        self.assert_off_cancelled()
        self.assertFalse(self.window.monitoring)
        self.assertIsNone(self.window.worker)

    def test_unrelated_job_completion_launches_full_poll_but_does_not_enable(self):
        first = self.busy("task", monitoring=False)
        self.request()
        self.complete({})
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertTrue(self.window.pending_auto_arm)
        self.assertIsNotNone(self.window.worker)
        self.assertIsNot(self.window.worker, first)
        self.complete()
        self.assertTrue(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_worker_finishing_during_confirmation_requires_new_post_confirmation_poll(self):
        first = self.busy()
        def confirm_after_worker_finishes():
            first.signals.completed.emit(self.results, None)
            self.assertFalse(self.window.engine.orders_enabled)
            return True
        with patch.object(self.window, "confirm_automation", side_effect=confirm_after_worker_finishes):
            self.window.arm_button.click()
        self.assertTrue(self.window.pending_auto_arm)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertIsNotNone(self.window.worker)
        self.assertIsNot(self.window.worker, first)
        self.complete()
        self.assertTrue(self.window.engine.orders_enabled)

    def test_off_or_stop_while_confirmation_is_open_overrides_late_yes(self):
        self.busy()
        def confirm_after_off():
            self.window.disable_auto_orders()
            return True
        with patch.object(self.window, "confirm_automation", side_effect=confirm_after_off):
            self.window.arm_button.click()
        self.assert_off_cancelled()
        def confirm_after_stop():
            self.window.stop_monitoring()
            return True
        with patch.object(self.window, "confirm_automation", side_effect=confirm_after_stop):
            self.window.arm_button.click()
        self.assert_off_cancelled()
        self.assertFalse(self.window.monitoring)
        self.complete()
        self.assert_off_cancelled()

    def test_session_report_and_old_launcher_callback_do_not_cancel_manual_request(self):
        controller = SessionController(self.window, self.folder, auto_arm=False)
        self.busy("task", monitoring=False)
        self.request()
        controller.report()
        controller.warmup_finished({}, None)
        self.assertTrue(self.window.pending_auto_arm)
        self.assertTrue(self.window._manual_arm_pending)
        self.assertFalse(self.window.engine.orders_enabled)
        status = json.loads((self.folder / "demo-session-status.json").read_text(encoding="utf-8"))
        self.assertTrue(status["pending_arm"])
        self.assertFalse(status["orders_enabled"])
        self.complete({})
        self.complete()
        self.assertTrue(self.window.engine.orders_enabled)


if __name__ == "__main__":
    unittest.main()
