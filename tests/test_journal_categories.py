from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from dockdack.autotrade import AutoTrader
from dockdack.models import OrderExecution
from dockdack.signal_bridge import ExternalPolicy, export_charts, ingest_signals
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW


class JournalCategoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.store = WatchStore(self.folder / "journal.sqlite3")
        self.service = FakeTradingService()
        self.item = WatchItem(self.service.resolve("005930"), "삼성전자")
        self.store.save_item(self.item)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: NOW)

    def rule(self, item=None, side="buy", quantity=1):
        rule = TriggerRule.create(item or self.item, "price_ge", side, quantity,
                                  Decimal("10000"), Decimal("95"))
        self.store.add_rule(rule)
        return rule

    def test_additive_schema_preserves_legacy_writer_and_filters_before_limit(self):
        legacy_path = self.folder / "legacy.sqlite3"
        with closing(sqlite3.connect(legacy_path)) as db, db:
            db.executescript("""
                CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT,time TEXT NOT NULL,
                                    symbol TEXT NOT NULL,message TEXT NOT NULL);
                PRAGMA user_version=3;
            """)
            db.execute("INSERT INTO events VALUES(NULL,?,?,?)", (NOW.isoformat(), "SYSTEM", "서버 시작"))
        upgraded = WatchStore(legacy_path)
        # Simulate a concurrently running previous app: its positional INSERT is unchanged.
        with closing(sqlite3.connect(legacy_path)) as db, db:
            db.executemany("INSERT INTO events VALUES(NULL,?,?,?)", [
                (NOW.isoformat(), "SYSTEM", "외부 신호 수신 · random-demo · {'hold': 1}")
                for _ in range(250)
            ])
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 3)
            self.assertEqual(len(db.execute("PRAGMA table_info(events)").fetchall()), 4)
        self.assertEqual(upgraded.events(1, category="system")[0]["message"], "서버 시작")
        self.assertEqual(len(upgraded.events(200, category="signal")), 200)
        self.assertEqual(upgraded.events(1)[0]["category"], "signal")
        self.assertEqual(WatchStore(legacy_path).events(1, category="system")[0]["message"], "서버 시작")

    def test_explicit_categories_override_legacy_inference_and_validate(self):
        self.store.event("SYSTEM", "외부 신호 서버 시작", category="system")
        self.store.event(self.item.id, "조회/확인 실패: 테스트")
        self.store.event(self.item.id, "차트 JSON 내보내기 · 완료")
        self.store.event(self.item.id, "체결 확인 · 주문번호 1")
        self.assertEqual(len(self.store.events(category="system")), 1)
        self.assertEqual(len(self.store.events(category="monitor")), 2)
        self.assertEqual(len(self.store.events(category="order")), 1)
        before = self.store.events()
        with self.assertRaises(ValueError):
            self.store.event("SYSTEM", "invalid", category="filled")
        with self.assertRaises(ValueError):
            self.store.events(category="unknown")
        self.assertEqual(self.store.events(), before)

    def test_monitoring_and_hold_are_visible_without_any_order_history(self):
        self.engine.poll()
        chart = export_charts(self.store, self.folder / "charts.json", now=NOW, watch_ids={self.item.id})
        signal = dict(signal_id="hold-1", export_id=chart["export_id"], market="domestic",
                      symbol="005930", exchange="KRX", action="hold", generated_at=NOW.isoformat(),
                      expires_at=(NOW + timedelta(minutes=1)).isoformat())
        payload = dict(schema_version=1, source_id="random-demo", signals=[signal])
        policy = ExternalPolicy("random-demo", 1, Decimal(1000), Decimal(1000))
        result = ingest_signals(self.store, payload, policy, now=NOW)
        self.assertEqual(result["hold"], 1)
        self.assertEqual(len(self.store.events(category="monitor")), 2)
        receipt = self.store.events(category="signal")[0]
        self.assertEqual(receipt["symbol"], self.item.id)
        self.assertIn("HOLD · 매매하지 않음", receipt["message"])
        self.assertIn("주문 생성 없음", receipt["message"])
        self.assertEqual(self.store.order_history(), ())
        self.assertEqual(self.service.submitted, [])
        self.assertFalse(self.engine.orders_enabled)
        ingest_signals(self.store, payload, policy, now=NOW)
        self.assertEqual(len(self.store.events(category="signal")), 1)

    def test_history_distinguishes_statuses_and_preserves_removed_stock(self):
        statuses = ("accepted", "filled", "cancelled", "rejected", "unknown", "not_sent", "reviewed", "submitting")
        for index, status in enumerate(statuses):
            item = WatchItem(self.service.resolve(f"{index + 1:06d}"), f"종목 {index}")
            self.store.save_item(item)
            rule = self.rule(item, side="sell" if index % 2 else "buy")
            self.store.claim(rule, Decimal("123"), NOW + timedelta(seconds=index))
            if status in {"filled", "cancelled"}:
                self.store.finish(rule.id, "accepted", "접수", str(index))
            if status != "submitting":
                self.store.finish(rule.id, status, f"결과 {status}", str(index))
            if status == "filled":
                self.store.remove_item(item.id)
        rows = self.store.order_history()
        self.assertEqual([r["status"] for r in rows], list(reversed(statuses)))
        self.assertEqual(len(self.store.order_history(2)), 2)
        self.assertEqual({r["currency"] for r in rows}, {"KRW"})
        self.assertTrue(all(r["reference_price"] == "123" and r["fill_price"] is None for r in rows))
        self.assertTrue(all(r["filled_quantity"] is None for r in rows))
        self.assertEqual(next(r for r in rows if r["status"] == "filled")["name"], "종목 1")
        self.assertEqual(rows[0]["side"], "sell")
        self.assertEqual(rows[0]["quantity"], 1)

    def test_reconcile_persists_broker_price_separately_from_reference_price(self):
        rule = self.rule(quantity=2)
        self.store.claim(rule, Decimal(100), NOW)
        self.store.finish(rule.id, "accepted", "접수", "0000200")
        self.service.fills = (OrderExecution("0000200", "005930", "매수", "체결", Decimal(2),
                                            Decimal(1), Decimal(1), Decimal(100), Decimal(99), "100000"),)
        self.engine._reconcile(self.item)
        partial = self.store.order_history()[0]
        self.assertEqual(partial["status"], "accepted")
        self.assertEqual((partial["filled_quantity"], partial["remaining_quantity"]), ("1", "1"))
        self.assertEqual((partial["reference_price"], partial["fill_price"]), ("100", "99"))
        self.assertEqual(partial["observed_at"], NOW.isoformat())
        self.service.fills = (OrderExecution("0000200", "005930", "매수", "체결", Decimal(2),
                                            Decimal(2), Decimal(0), Decimal(100), Decimal("99.5"), "100001"),)
        self.engine._reconcile(self.item)
        final = WatchStore(self.store.path).order_history()[0]
        self.assertEqual((final["status"], final["filled_quantity"], final["remaining_quantity"], final["fill_price"]),
                         ("filled", "2", "0", "99.5"))
        self.assertEqual(self.service.submitted, [])
        self.assertFalse(self.engine.orders_enabled)

    def test_unknown_fill_price_is_never_invented(self):
        rule = self.rule()
        self.store.claim(rule, Decimal(100), NOW)
        self.store.finish(rule.id, "accepted", "접수", "1")
        for value in (None, Decimal(0), Decimal("NaN")):
            self.store.record_execution(rule.id, filled_quantity=Decimal(1), remaining_quantity=Decimal(0),
                                        fill_price=value, observed_at=NOW)
            self.assertIsNone(self.store.order_history()[0]["fill_price"])
            self.assertEqual(self.store.order_history()[0]["status"], "accepted")
        us = WatchItem(self.service.resolve("AAPL"), "애플")
        self.store.save_item(us)
        us_rule = self.rule(us)
        self.store.claim(us_rule, Decimal(200), NOW)
        self.assertEqual(self.store.order_history()[0]["currency"], "USD")


if __name__ == "__main__":
    unittest.main()
