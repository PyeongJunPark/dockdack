"""Offline cross-layer regression for audit remediation; temporary DBs only."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dockdack.autotrade import AutoTrader
from dockdack.environment_store import demo_scope, scoped_store_path, store_for_service
from dockdack.gui_service import Instrument
from dockdack.models import Market, TradingMode
from dockdack.performance import realized_performance
from dockdack.persistence.account_migration import CONFIRM_LEGACY_DEMO, migrate_legacy_demo
from dockdack.persistence.retention import archive_monitor_events, read_monitor_archive
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW
from test_lstm30_close import CloseService, held
from test_performance import order


class OptimizationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.store = WatchStore(self.folder / "watchlist.sqlite3")
        self.item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "test", 31)
        self.store.save_item(self.item)

    def test_key_and_reset_generation_have_separate_ledgers(self):
        scopes = [demo_scope("a" * 64, "0"), demo_scope("b" * 64, "0"), demo_scope("a" * 64, "1")]
        stores = [store_for_service(SimpleNamespace(mode=TradingMode.DEMO, storage_scope=scope),
                                    base_folder=self.folder) for scope in scopes]
        self.assertEqual(len({store.path for store in stores}), 3)
        stores[0].event("SYSTEM", "private generation")
        self.assertFalse(stores[1].events())
        self.assertFalse(stores[2].events())
        with self.assertRaises(ValueError):
            WatchStore(stores[0].path, storage_scope=scopes[1])

    def test_changed_demo_scope_cannot_arm_legacy_ledger(self):
        service = FakeTradingService()
        service.storage_scope = demo_scope("a" * 64)
        engine = AutoTrader(service, self.store, clock=lambda: NOW)
        with self.assertRaises(ValueError):
            engine.enable_orders("DEMO_AUTOTRADE")
        self.assertFalse(engine.orders_enabled)
        self.assertEqual(service.submitted, [])

    def test_untagged_legacy_database_cannot_be_silently_bound_to_a_key(self):
        import sqlite3
        from contextlib import closing
        legacy = self.folder / "untagged.sqlite3"
        with closing(sqlite3.connect(legacy)) as db:
            db.execute("CREATE TABLE legacy(value TEXT)")
            db.commit()
        before = legacy.read_bytes()
        with self.assertRaisesRegex(ValueError, "기존 기록"):
            WatchStore(legacy, storage_scope=demo_scope("a" * 64))
        self.assertEqual(before, legacy.read_bytes())

    def test_explicit_migration_is_copy_only_and_preserves_pending(self):
        rule = TriggerRule.create(self.item, "price_ge", "buy", 1, D(1000), D(90))
        self.store.add_rule(rule)
        self.store.claim(rule, D(100), NOW)
        scope = demo_scope("a" * 64)
        target = scoped_store_path(SimpleNamespace(mode=TradingMode.DEMO, storage_scope=scope), base_folder=self.folder)
        before = self.store.path.read_bytes()
        migrated = migrate_legacy_demo(self.store.path, target, scope, confirmation=CONFIRM_LEGACY_DEMO)
        self.assertEqual(before, self.store.path.read_bytes())
        self.assertEqual(migrated.attempts()[0]["status"], "submitting")
        self.assertEqual(migrated.storage_scope, scope)
        with self.assertRaises(ValueError):
            migrate_legacy_demo(self.store.path, target, scope, confirmation=CONFIRM_LEGACY_DEMO)
        with self.assertRaises(ValueError):
            migrate_legacy_demo(self.store.path, self.folder / "bad.sqlite3", scope, confirmation="")

    def test_monitor_archive_dry_run_then_lossless_and_order_signal_preservation(self):
        old = NOW - timedelta(days=100)
        with self.store.connection() as db:
            for category in ("monitor", "order", "signal", "system"):
                self.store._insert_event(db, "X", "same text " + category, category=category, at=old)
            self.store._insert_event(db, "X", "recent monitor", category="monitor", at=NOW)
        before = self.store.events()
        self.assertEqual(archive_monitor_events(self.store, now=NOW)["eligible"], 1)
        self.assertEqual(before, self.store.events())
        result = archive_monitor_events(self.store, now=NOW, apply=True)
        archived = read_monitor_archive(self.store, result["archive_id"])
        self.assertEqual(archived[0]["message"], "same text monitor")
        self.assertEqual(len(self.store.events()), 4)
        self.assertEqual(archive_monitor_events(self.store, now=NOW, apply=True)["eligible"], 0)
        self.assertEqual({r["category"] for r in self.store.events()}, {"order", "signal", "system", "monitor"})

    def test_snapshot_batch_one_connection_and_independent_corrupt_member(self):
        service = FakeTradingService()
        engine = AutoTrader(service, self.store, clock=lambda: NOW)
        snapshot = engine.snapshot(self.item, ())
        second = WatchItem(Instrument(Market.DOMESTIC, "000660", "KRX"), "second", 31)
        self.store.save_item(second)
        with self.store.connection() as db:
            db.execute("INSERT INTO snapshots VALUES(?,?)", (second.id, "broken"))
        original = self.store.connection
        with patch.object(self.store, "connection", wraps=original) as connections:
            result = self.store.cached_snapshots((self.item, second))
        self.assertEqual(connections.call_count, 1)
        self.assertEqual(result[self.item.id], snapshot)
        self.assertIsInstance(result[second.id], Exception)

    def test_holdings_checked_before_watch_and_each_ten_even_on_failures(self):
        for i in range(24):
            self.store.save_item(WatchItem(Instrument(Market.DOMESTIC, f"{i + 100000:06d}", "KRX"), "test", 31))
        engine = AutoTrader(FakeTradingService(), self.store, clock=lambda: NOW)
        engine.enable_holdings_exits = True
        counts, checked = [], [0]
        def snapshot(*args):
            checked[0] += 1
            raise ValueError("fixture read failure")
        with patch.object(engine, "_holdings_pass", side_effect=lambda *a, **kw: counts.append(checked[0])), \
                patch.object(engine, "snapshot", side_effect=snapshot):
            engine.poll()
        self.assertEqual(counts, [0, 10, 20, 25])

    def test_normal_engine_close_sells_all_unwatched_holdings_over_regular_cap_once(self):
        from dockdack.market_schedule import session_on
        session = session_on(Market.DOMESTIC, NOW.date())
        now = session.closed - timedelta(minutes=5)
        service = CloseService()
        service.positions_by_market[Market.DOMESTIC] = (held(Market.DOMESTIC, "000660", quantity=1200),)
        engine = AutoTrader(service, self.store, clock=lambda: now)
        engine.enable_holdings_exits = True
        engine.configure_close_liquidation()
        engine.maintenance_checkpoint()
        self.assertFalse(service.submitted)
        engine.enable_orders("DEMO_AUTOTRADE")
        engine.maintenance_checkpoint()
        engine.maintenance_checkpoint()
        self.assertEqual(len(service.submitted), 1)
        self.assertEqual(service.submitted[0].quantity, 1200)
        self.assertEqual(service.submitted[0].symbol, "000660")
        self.assertTrue(engine.close_liquidator.buy_blocked(Market.DOMESTIC, now))
        with self.assertRaises(ValueError):
            engine.configure_close_liquidation(enabled=False)

    def test_bulk_close_performance_partial_fill_uses_declared_lot_offsets(self):
        sale = order(3, "sell", 5, "120", rule_id="close-test", status="accepted",
                     filled_quantity="3", remaining_quantity="2",
                     close_allocations=({"lot_id": "2", "quantity": "3", "fill_offset": "0"},
                                        {"lot_id": "1", "quantity": "2", "fill_offset": "3"}))
        result = realized_performance([order(1, "buy", 2, "100"), order(2, "buy", 3, "110"), sale])
        self.assertEqual(result["by_rule_id"]["close-test"]["cost_basis"], D(330))
        self.assertEqual(result["by_rule_id"]["close-test"]["realized_profit"], D(30))
        sale["close_allocations"] = ({"lot_id": "2", "quantity": "3", "fill_offset": "1"},)
        result = realized_performance([order(1, "buy", 2, "100"), order(2, "buy", 3, "110"), sale])
        self.assertEqual(result["by_rule_id"]["close-test"]["reason_code"], "invalid_lot_allocation")


if __name__ == "__main__":
    unittest.main()
