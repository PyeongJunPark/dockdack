from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dockdack.autotrade import AutoTrader
from dockdack.signal_bridge import ExternalPolicy, SignalFileReader, atomic_json, export_charts
from dockdack.watchlist import WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW


class JournalPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.store = WatchStore(self.folder / "watch.sqlite3")
        self.service = FakeTradingService()

    def test_backfill_preserves_explicit_categories_and_both_legacy_writers(self):
        path = self.folder / "legacy.sqlite3"
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript("""
                CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT,time TEXT NOT NULL,
                                    symbol TEXT NOT NULL,message TEXT NOT NULL);
                CREATE TABLE event_categories(event_id INTEGER PRIMARY KEY REFERENCES events(id), category TEXT NOT NULL);
                PRAGMA user_version=3;
            """)
            db.execute("INSERT INTO events VALUES(1,?,'SYSTEM','외부 신호 서버 시작')", (NOW.isoformat(),))
            db.execute("INSERT INTO event_categories VALUES(1,'system')")
            db.execute("INSERT INTO events VALUES(2,?,'SYSTEM','외부 신호 수신 · hold: 1')", (NOW.isoformat(),))
        upgraded = WatchStore(path)
        self.assertEqual(upgraded.events(category="system")[0]["id"], 1)
        self.assertEqual(upgraded.events(category="signal")[0]["id"], 2)
        # The previous categorized app uses a plain INSERT, not an upsert.
        with closing(sqlite3.connect(path)) as db, db:
            event_id = db.execute("INSERT INTO events VALUES(NULL,?,'SYSTEM','외부 신호 운영 점검')",
                                  (NOW.isoformat(),)).lastrowid
            db.execute("INSERT INTO event_categories VALUES(?,'system')", (event_id,))
            db.execute("INSERT INTO events VALUES(NULL,?,'SYSTEM','조회/확인 실패: 테스트')", (NOW.isoformat(),))
        self.assertEqual(len(upgraded.events(category="system")), 2)
        self.assertEqual(len(upgraded.events(category="monitor")), 1)
        self.assertEqual(upgraded.events(), WatchStore(path).events())
        with upgraded.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events").fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM event_category_index").fetchone()[0], 4)
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 3)
            # Category changes propagate without editing historical event messages.
            db.execute("UPDATE event_categories SET category='monitor' WHERE event_id=1")
        self.assertEqual(len(upgraded.events(category="monitor")), 2)

    def test_category_lookup_plan_is_indexed_and_heads_are_cheap_and_independent(self):
        self.assertTrue(all(r == {"id": 0, "time": None} for r in self.store.event_heads().values()))
        self.store.event("SYSTEM", "server", category="system")
        before = self.store.event_heads()
        self.store.event("005930", "HOLD", category="signal")
        after = self.store.event_heads()
        self.assertEqual(before["system"], after["system"])
        self.assertGreater(after["signal"]["id"], before["signal"]["id"])
        with self.store.connection() as db:
            plan = " ".join(row[3] for row in db.execute("""EXPLAIN QUERY PLAN
                SELECT e.*,c.category FROM event_category_index c INDEXED BY event_category_index_by_category
                JOIN events e ON e.id=c.event_id WHERE c.category=? ORDER BY c.event_id DESC LIMIT 500
            """, ("system",)))
        self.assertIn("event_category_index_by_category", plan)
        self.assertNotIn("SCAN e", plan)
        self.assertNotIn("TEMP B-TREE", plan)

    def test_export_is_two_connections_and_decodes_only_selected_rows(self):
        engine = AutoTrader(self.service, self.store, clock=lambda: NOW)
        items = [WatchItem(self.service.resolve(symbol), symbol) for symbol in ("005930", "000660", "AAPL")]
        for item in items:
            self.store.save_item(item)
            engine.snapshot(item)
        connection, decode = self.store.connection, self.store.decode_snapshot
        with patch.object(self.store, "connection", wraps=connection) as connections:
            with patch.object(self.store, "decode_snapshot", wraps=decode) as decoded:
                data = export_charts(self.store, self.folder / "one.json", now=NOW, watch_ids={items[0].id})
        self.assertEqual(connections.call_count, 2)
        self.assertEqual(decoded.call_count, 1)
        self.assertEqual([r["watch_id"] for r in data["stocks"]], [items[0].id])
        with patch.object(self.store, "connection", wraps=connection) as connections:
            full = export_charts(self.store, self.folder / "all.json", now=NOW)
        self.assertEqual(connections.call_count, 2)
        self.assertEqual(len(full["stocks"]), 3)
        self.assertTrue(all(r["status"] == "ok" for r in full["stocks"]))
        self.assertEqual(self.store.chart_export_rows(set()), ())
        self.assertEqual(self.service.submitted, [])

    def test_batch_export_keeps_per_stock_corrupt_cache_isolation(self):
        engine = AutoTrader(self.service, self.store, clock=lambda: NOW)
        items = [WatchItem(self.service.resolve(symbol), symbol) for symbol in ("005930", "000660")]
        for item in items:
            self.store.save_item(item)
            engine.snapshot(item)
        with self.store.connection() as db:
            db.execute("UPDATE snapshots SET data='not json' WHERE watch_id=?", (items[0].id,))
        data = export_charts(self.store, self.folder / "mixed.json", now=NOW)
        self.assertEqual([r["status"] for r in data["stocks"]], ["error", "ok"])
        with self.store.connection() as db:
            members = [r[0] for r in db.execute("SELECT watch_id FROM chart_export_members WHERE export_id=?", (data["export_id"],))]
        self.assertEqual(members, [items[1].id])

    def test_reader_telemetry_tracks_real_ingestion_without_changing_dedup_or_errors(self):
        item = WatchItem(self.service.resolve("005930"), "삼성전자")
        self.store.save_item(item)
        AutoTrader(self.service, self.store, clock=lambda: NOW).snapshot(item)
        chart = export_charts(self.store, self.folder / "charts.json", now=NOW)
        path = self.folder / "signals.json"
        current = [NOW]
        policy = ExternalPolicy("model", 1, Decimal(1000), Decimal(1000))
        reader = SignalFileReader(self.store, path, policy, lambda: current[0])
        self.assertEqual(reader.status()["reader_state"], "waiting")
        self.assertIsNone(reader())
        self.assertEqual(reader.status()["reader_state"], "missing")
        payload = dict(schema_version=1, source_id="model", signals=[dict(
            signal_id="one", export_id=chart["export_id"], market="domestic", symbol="005930", exchange="KRX",
            action="hold", generated_at=NOW.isoformat(), expires_at=(NOW + timedelta(minutes=1)).isoformat())])
        atomic_json(path, payload)
        self.assertEqual(reader()["hold"], 1)
        status = reader.status()
        self.assertEqual(status["reader_state"], "ok")
        self.assertEqual((status["last_read_at"], status["last_accepted_at"]), (NOW, NOW))
        status["received_counts"]["hold"] = 999
        self.assertEqual(reader.status()["received_counts"]["hold"], 1)
        self.assertIsNone(reader())
        self.assertEqual(reader.status()["reader_state"], "unchanged")
        current[0] += timedelta(seconds=1)
        atomic_json(path, {"schema_version": 2})
        with self.assertRaises(ValueError):
            reader()
        self.assertEqual(reader.status()["reader_state"], "error")
        self.assertTrue(reader.status()["reader_error"])
        self.assertEqual(reader.status()["last_accepted_at"], NOW)
        self.assertEqual(reader.status()["last_read_at"], current[0])
        with patch("dockdack.signal_bridge.read_signal_file", side_effect=ValueError("bad JSON")):
            with self.assertRaises(ValueError):
                reader()
        self.assertEqual(reader.status()["last_accepted_at"], NOW)
        atomic_json(path, payload)
        self.assertEqual(reader()["duplicates"], 1)
        self.assertEqual(reader.status()["reader_error"], "")
        self.assertEqual(reader.status()["last_accepted_at"], current[0])
        self.assertEqual(self.store.order_history(), ())
        self.assertEqual(self.service.submitted, [])


if __name__ == "__main__":
    unittest.main()
