from __future__ import annotations

import os
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dockdack.activity_snapshot import LedgerCollector, collect_event_logs
from dockdack.gui_service import Instrument
from dockdack.models import Market
from dockdack.trade_journal import daily_trade_journal
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import NOW
from test_trade_journal import order


class ActivitySnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "ledger.sqlite3")
        self.item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "삼성전자")
        self.store.save_item(self.item)
        self.rule = TriggerRule.create(self.item, "price_ge", "buy", 1, D(1000), D(90))
        self.store.add_rule(self.rule)

    def add_attempt(self):
        self.store.claim(self.rule, D(100), NOW)
        self.store.finish(self.rule.id, "accepted", "접수", "1")

    def test_signal_flood_and_unused_rules_do_not_change_ledger_revision(self):
        revision = self.store.ledger_revision()
        for index in range(30):
            self.store.event(self.item.id, f"HOLD {index}", category="signal")
            if index < 5:
                self.store.add_rule(TriggerRule.create(self.item, "price_ge", "buy", 1, D(1000), D(100 + index)))
        self.store.save_item(replace(self.item, days=20))
        self.assertEqual(self.store.ledger_revision(), revision)

    def test_revision_covers_attempt_fill_provenance_and_held_name_but_not_membership(self):
        revision = self.store.ledger_revision()
        self.add_attempt()
        self.assertGreater(self.store.ledger_revision(), revision)
        revision = self.store.ledger_revision()
        self.store.record_execution(self.rule.id, filled_quantity=D(1), remaining_quantity=D(0),
                                    fill_price=D(101), observed_at=NOW)
        self.assertGreater(self.store.ledger_revision(), revision)
        revision = self.store.ledger_revision()
        self.store.record_fill_recovery(self.rule.id, status="enriched", message="price verified", checked_at=NOW)
        self.assertGreater(self.store.ledger_revision(), revision)
        revision = self.store.ledger_revision()
        self.store.save_item(replace(self.item, name="새 이름"))
        self.assertGreater(self.store.ledger_revision(), revision)
        self.store.finish(self.rule.id, "filled", "체결 확인")
        revision = self.store.ledger_revision()
        self.store.remove_item(self.item.id)
        self.assertEqual(self.store.ledger_revision(), revision)

    def test_unchanged_collector_never_reads_full_ledger_or_recomputes_fifo(self):
        self.add_attempt()
        collector = LedgerCollector(self.store)
        with patch("dockdack.activity_snapshot.daily_trade_journal", wraps=daily_trade_journal) as calculate:
            snapshot = collector.collect()
            self.assertIsNotNone(snapshot)
            with patch.object(self.store, "order_history", side_effect=AssertionError("must use revision")):
                for _ in range(100):
                    self.assertIsNone(collector.collect())
            self.assertEqual(calculate.call_count, 1)
            self.assertIs(snapshot.performance, snapshot.journal["performance"])

    def test_fill_snapshot_alone_is_detected_without_order_event(self):
        self.add_attempt()
        collector = LedgerCollector(self.store)
        before = collector.collect()
        heads = self.store.event_heads()
        self.store.record_execution(self.rule.id, filled_quantity=D(1), remaining_quantity=D(0),
                                    fill_price=D(101), observed_at=NOW)
        after = collector.collect()
        self.assertIsNotNone(after)
        self.assertNotEqual(before.revision, after.revision)
        self.assertEqual(after.ledger[0]["fill_price"], "101")
        self.assertEqual(self.store.event_heads(), heads)

    def test_collector_fallback_and_force_reuse_precomputed_journal(self):
        fake = SimpleNamespace(order_history=Mock(return_value=(order(1),)))
        collector = LedgerCollector(fake)
        with patch("dockdack.activity_snapshot.daily_trade_journal", wraps=daily_trade_journal) as calculate:
            first = collector.collect()
            self.assertIsNone(collector.collect())
            self.assertIs(collector.collect(force=True), first)
            self.assertEqual(calculate.call_count, 1)

    def test_log_collection_uses_only_requested_changed_categories(self):
        self.store.event("SYSTEM", "running", category="system")
        first = collect_event_logs(self.store, ("system",))
        with patch.object(self.store, "events", side_effect=AssertionError("same heads")):
            second = collect_event_logs(self.store, ("system",), previous_heads=first["heads"])
        self.assertEqual(second["events"], {})
        self.store.event("SYSTEM", "HOLD", category="signal")
        third = collect_event_logs(self.store, ("system", "signal"), previous_heads=first["heads"])
        self.assertEqual(tuple(third["events"]), ("signal",))


@unittest.skipUnless(importlib.util.find_spec("PySide6") is not None, "Install the gui extra")
class ActivitySnapshotGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_shared_worker_result_limits_paint_not_totals_and_performs_no_gui_io(self):
        from PySide6.QtCore import QDate
        from dockdack.operations_gui import OrderHistoryPanel
        from dockdack.trade_journal_gui import DailyTradeJournalPanel
        records = tuple(order(index) for index in range(1, 801))
        store = SimpleNamespace(mode="demo", order_history=Mock(return_value=records))
        snapshot = LedgerCollector(store).collect()
        store.order_history.side_effect = AssertionError("GUI must not read DB")
        orders, journal = OrderHistoryPanel(), DailyTradeJournalPanel(store)
        try:
            journal.dates["domestic"].setDate(QDate(2026, 9, 15))
            with patch("dockdack.operations_gui.realized_performance", side_effect=AssertionError("GUI FIFO")), \
                    patch("dockdack.trade_journal_gui.daily_trade_journal", side_effect=AssertionError("GUI FIFO")):
                orders.apply_snapshot(snapshot)
                journal.apply_snapshot(snapshot)
            self.assertEqual(orders.table.rowCount(), 500)
            self.assertEqual(journal.tables["domestic"].rowCount(), 500)
            self.assertEqual(journal.values["domestic"]["buy"].text(), "80,000 KRW")
            self.assertIn("전체 800건", journal.summaries["domestic"].text())
            self.assertIs(orders.performance, journal.journal["performance"])
            first_cell = orders.table.item(0, 0)
            self.assertFalse(orders.apply_snapshot(snapshot))
            self.assertFalse(journal.apply_snapshot(snapshot))
            self.assertIs(orders.table.item(0, 0), first_cell)
        finally:
            for view in (orders, journal):
                view.close()
                view.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
