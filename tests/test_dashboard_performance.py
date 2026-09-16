from __future__ import annotations

import importlib.util
import os
import tempfile
import time
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from dockdack.watch_gui import WatchlistDialog

from dockdack.watchlist import WatchItem, WatchStore, utc_now
from test_autotrade import FakeTradingService


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class DashboardPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WatchStore(Path(self.temp.name) / "watch.sqlite3")
        self.service = FakeTradingService()
        self.items = [WatchItem(self.service.resolve(symbol), symbol) for symbol in ("005930", "000660", "AAPL")]
        for item in self.items:
            self.store.save_item(item)
        self.window = WatchlistDialog(self.service, self.store)
        self.window.health_timer.stop()
        self.window.order_status_timer.stop()
        self.window.hourly_ranking.setChecked(False)
        self.snapshots = {item.id: self.window.engine.snapshot(item) for item in self.items}
        self.window.snapshots.update(self.snapshots)
        self.window.reload_tables()
        self.wait_activity()

    def wait_activity(self):
        deadline = time.monotonic()+10
        while self.window._activity_worker or self.window._activity_pending:
            self.app.processEvents()
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline, "local activity worker did not finish")
        self.app.processEvents()

    def tearDown(self):
        self.window.worker = None
        self.window._inspection_worker = None
        self.wait_activity()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_quote_progress_never_reloads_database_or_replaces_unrelated_cells(self):
        target = self.items[1]
        untouched = self.window.watch_table.item(0, 1)
        changed = self.window.watch_table.item(1, 1)
        snapshot = self.snapshots[target.id]
        updated = replace(snapshot, quote=replace(snapshot.quote, price=Decimal(123)), fetched_at=utc_now())
        with patch.object(self.store, "connection", side_effect=AssertionError("GUI quote progress must not read DB")), \
                patch.object(self.window.chart, "set_history") as chart:
            self.window._progress((target.id, updated, 2, 3))
            chart.assert_not_called()
        self.assertIs(self.window.watch_table.item(0, 1), untouched)
        self.assertIs(self.window.watch_table.item(1, 1), changed)
        self.assertEqual(changed.text(), "123 KRW")
        self.assertIn(target.id, self.window.fresh_ids)

    def test_hidden_logs_do_no_periodic_database_work_and_opening_updates_them(self):
        self.store.event("SYSTEM", "new server state", category="system")
        with patch.object(self.store, "connection", side_effect=AssertionError("hidden logs must not poll DB")):
            self.window._update_health()
        self.window.workspace_tabs.setCurrentWidget(self.window.operations_panel)
        self.wait_activity()
        self.assertIn("new server state", self.window.log_table.item(0, 2).text())

    def test_log_revision_avoids_loading_unchanged_500_rows(self):
        self.store.event("SYSTEM", "server", category="system")
        self.window._reload_activity(force=True)
        self.wait_activity()
        with patch.object(self.store, "events", side_effect=AssertionError("same revision must use cache")):
            self.window.operations_panel.reload(self.store, visible_only=True)

    def configure(self):
        self.window.external_mode.setChecked(True)
        self.window.external_krw.setValue(10000000)
        self.window.external_usd.setValue(10000)
        self.window.configure_external()

    def test_execution_recovery_queues_on_existing_broker_worker_without_arming(self):
        self.window.worker = object()
        with patch.object(self.window, "_run") as run:
            self.window.refresh_executions()
            run.assert_not_called()
        self.assertTrue(self.window._fill_refresh_requested.is_set())
        self.assertFalse(self.window.engine.orders_enabled)
        progress = []
        with patch.object(self.window.fill_recovery, "refresh_due", return_value={"state": "complete"}) as refresh:
            self.window._refresh_executions_worker(progress.append)
            self.assertTrue(refresh.call_args.kwargs["force"])
        self.assertFalse(self.window._fill_refresh_requested.is_set())
        self.assertEqual(progress[-1], ("executions", {"state": "complete"}))
        self.assertFalse(self.service.submitted)

    def test_signal_inspection_can_queue_while_broker_busy_without_arming_or_ingesting(self):
        self.configure()
        self.window.monitoring = True
        self.window.worker = object()
        with patch.object(self.window.inspection_pool, "start") as start, \
                patch.object(self.window.engine, "enable_orders") as arm:
            self.window.inspect_signals()
            worker = start.call_args.args[0]
            result = worker.operation()
            self.window._inspection_completed(result, None)
            arm.assert_not_called()
        self.assertEqual(self.window._inspection["state"], "missing")
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.store.rules(), ())
        self.assertEqual(self.store.attempts(), ())
        self.assertEqual(self.service.submitted, [])

    def test_inspection_for_old_settings_is_discarded(self):
        self.configure()
        with patch.object(self.window.inspection_pool, "start"):
            self.window.inspect_signals()
            self.window.signal_path.setText(str(Path(self.temp.name) / "new.json"))
            self.window._inspection_completed({"state": "format_ok", "summary": "old success"}, None)
        self.assertIsNone(self.window._inspection)
        self.assertNotIn("old success", self.window.signal_connection_panel.inspection_label.text())
        self.assertIn("설정이 변경", self.window.message.text())

    def test_stream_and_aggregate_export_success_are_distinct(self):
        self.configure()
        observed = utc_now()
        self.window._last_update_at = observed
        status = self.window.connection_status()
        self.assertIsNone(status["last_export_at"])
        self.assertEqual(status["last_update_at"], observed)
        self.assertTrue(status["update_path"].endswith("charts_updates"))
        self.window._update_connection()
        self.assertIn("전체 차트", self.window.signal_connection_panel.step_labels["export"].text())
        self.assertFalse(self.window.engine.orders_enabled)

    def test_rate_status_is_read_only_deduplicated_and_metrics_fit_small_window(self):
        state = {"scope_id": 1, "wait_remaining_seconds": .75, "cooldown_remaining_seconds": 0,
                 "effective_interval_seconds": 1.25, "retry_count": 0, "in_flight": False}
        client = SimpleNamespace(rate_status=lambda: state.copy())
        broker = SimpleNamespace(_domestic_http=client, _us_http=client)
        self.service.brokers = {"same-one": broker, "same-two": broker}
        self.assertEqual(len(self.window.api_rate_status()), 1)
        before_calls = self.service.quote_calls
        self.window._update_health()
        self.window.resize(1080, 780)
        self.window.show()
        self.app.processEvents()
        self.assertEqual(self.service.quote_calls, before_calls)
        self.assertIn("안전 간격 1.25초", self.window.health_label.text())
        for labels in self.window.portfolio_panel.market_labels.values():
            for key in ("evaluation", "profit", "cash", "available"):
                self.assertGreaterEqual(labels[key].height(), 28)
        self.assertFalse(self.window.engine.orders_enabled)


if __name__ == "__main__":
    unittest.main()
