import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from dockdack.gui_service import Instrument
from dockdack.history import DailyBar, DailyHistory
from dockdack.models import Market, Quote
from dockdack.universe import RankedStock
from dockdack.watchlist import MarketSnapshot, TriggerRule, WatchItem, WatchStore


NOW = datetime(2026, 9, 15, 2, tzinfo=timezone.utc)


class CommonWatchlistTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "watch.sqlite3")

    def item(self, symbol, market=Market.DOMESTIC, exchange=None):
        item = WatchItem(Instrument(market, symbol, exchange or ("KRX" if market is Market.DOMESTIC else "ND")))
        self.store.save_item(item)
        return item

    def rule(self, item, external=False):
        rule = TriggerRule.create(item, "external" if external else "price_ge", "buy", 1,
                                  Decimal(1000), None if external else Decimal(100))
        self.store.add_rule(rule)
        if external:
            with self.store.connection() as db:
                db.execute("INSERT INTO external_signals VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                    "common-test", rule.id, "{}", rule.id, item.id, NOW.isoformat(), NOW.isoformat(),
                    "export-test", NOW.isoformat(), "buy", "queued",
                ))
        return rule

    def rule_status(self, rule):
        with self.store.connection() as db:
            return db.execute("SELECT status FROM rules WHERE id=?", (rule.id,)).fetchone()[0]

    def test_unmanaged_etf_removed_but_eligible_manual_and_other_market_preserved(self):
        common = self.item("005930")
        etf = self.item("069500")
        us = self.item("AAPL", Market.US)
        common_rule, etf_rule, us_rule = [self.rule(item) for item in (common, etf, us)]
        result = self.store.restrict_to_common(Market.DOMESTIC, {("005930", "KRX")}, set())
        self.assertEqual(result, {"eligible": 1, "removed": 1, "protected": 0, "paused_rules": 1, "paused_signals": 0})
        self.assertEqual(self.store.items(), (common, us))
        self.assertEqual(self.rule_status(etf_rule), "paused")
        self.assertEqual(self.rule_status(common_rule), "ready")
        self.assertEqual(self.rule_status(us_rule), "ready")
        self.assertFalse(self.store.claim(etf_rule, Decimal(100), NOW))
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM managed_watchlist").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT active FROM watchlist WHERE id=?", (etf.id,)).fetchone()[0], 0)
        self.assertIn("이력 보존", self.store.events()[0]["message"])

    def test_broker_protected_ineligible_retained_but_manual_and_external_rules_paused(self):
        held = self.item("069500")
        manual = self.rule(held)
        external = self.rule(held, external=True)
        result = self.store.restrict_to_common(Market.DOMESTIC, {("005930", "KRX")}, {"069500"})
        self.assertEqual(result, {"eligible": 0, "removed": 0, "protected": 1, "paused_rules": 2, "paused_signals": 1})
        self.assertEqual(self.store.items(), (held,))
        self.assertEqual(self.rule_status(manual), "paused")
        self.assertEqual(self.rule_status(external), "paused")
        self.assertEqual(self.store.external_for_rule(external.id)["status"], "paused")
        self.assertFalse(self.store.claim(manual, Decimal(100), NOW))
        again = self.store.restrict_to_common(Market.DOMESTIC, {("005930", "KRX")}, {"069500"})
        self.assertEqual(again["paused_rules"], 0)
        self.assertEqual(again["paused_signals"], 0)

    def test_all_local_pending_states_preserve_items_and_attempts_but_pause_ready_rules(self):
        pending_rules = []
        ready_rules = []
        for index, status in enumerate(("submitting", "unknown", "accepted"), 1):
            item = self.item(f"{index:06d}")
            pending = self.rule(item, external=True)
            self.assertTrue(self.store.claim(pending, Decimal(100), NOW))
            if status != "submitting":
                self.store.finish(pending.id, status, "preserve", "order-1" if status == "accepted" else "")
            pending_rules.append((pending, status))
            ready_rules.append(self.rule(item))
        before = self.store.attempts()
        result = self.store.restrict_to_common(Market.DOMESTIC, {("005930", "KRX")}, set())
        self.assertEqual(result["protected"], 3)
        self.assertEqual(result["removed"], 0)
        self.assertEqual(result["paused_rules"], 3)
        self.assertEqual(self.store.attempts(), before)
        for rule, status in pending_rules:
            self.assertEqual(self.rule_status(rule), status)
            self.assertEqual(self.store.external_for_rule(rule.id)["status"], status)
        self.assertTrue(all(self.rule_status(rule) == "paused" for rule in ready_rules))

    def test_soft_removal_preserves_snapshot_daily_cache_filled_history_and_membership(self):
        etf = self.item("069500")
        bar = DailyBar(date(2026, 9, 14), *(Decimal(100) for _ in range(5)))
        history = DailyHistory(Market.DOMESTIC, "069500", "KRX", "KRW", 30, (bar,))
        snapshot = MarketSnapshot(Quote(Market.DOMESTIC, "069500", "ETF", "KRX", Decimal(100), "KRW"), history, NOW)
        self.store.save_snapshot(etf, snapshot)
        rule = self.rule(etf, external=True)
        self.store.claim(rule, Decimal(100), NOW)
        self.store.finish(rule.id, "accepted", "accepted", "filled-order")
        self.store.finish(rule.id, "filled", "filled")
        with self.store.connection() as db:
            db.execute("INSERT INTO history_cache VALUES(?,?,?)", (etf.id, NOW.isoformat(), '{"persisted":true}'))
            db.execute("INSERT INTO chart_exports VALUES(?,?,?,?)", ("export-test", NOW.isoformat(), "test.json", "hash"))
            db.execute("INSERT INTO chart_export_members VALUES(?,?)", ("export-test", etf.id))
        before = self.store.attempts()
        result = self.store.restrict_to_common(Market.DOMESTIC, {("005930", "KRX")}, set())
        self.assertEqual(result["removed"], 1)
        self.assertEqual(self.store.items(), ())
        self.assertEqual(self.store.cached_snapshot(etf), snapshot)
        self.assertEqual(self.store.attempts(), before)
        self.assertEqual(self.store.external_for_rule(rule.id)["status"], "filled")
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT data FROM history_cache WHERE watch_id=?", (etf.id,)).fetchone()[0], '{"persisted":true}')
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chart_export_members WHERE watch_id=?", (etf.id,)).fetchone()[0], 1)

    def test_us_eligibility_matches_exact_exchange_and_preserves_class_ticker_case(self):
        correct = self.item("BRKb", Market.US, "NY")
        wrong_venue = self.item("BRKb", Market.US, "ND")
        korean = self.item("005930")
        result = self.store.restrict_to_common(Market.US, {("BRKb", "NY")}, set())
        self.assertEqual(result["eligible"], 1)
        self.assertEqual(result["removed"], 1)
        self.assertEqual(self.store.items(), (correct, korean))
        self.assertNotIn(wrong_venue, self.store.items())

    def test_invalid_or_empty_classification_never_mutates_watchlist(self):
        item = self.item("069500")
        rule = self.rule(item)
        for market, eligible, protected in (
            (Market.DOMESTIC, set(), set()),
            (Market.DOMESTIC, {("005930", "ND")}, set()),
            (Market.DOMESTIC, {"005930"}, set()),
            (Market.DOMESTIC, {("005930", "KRX")}, None),
            ("domestic", {("005930", "KRX")}, set()),
        ):
            with self.subTest(market=market, eligible=eligible), self.assertRaises(ValueError):
                self.store.restrict_to_common(market, eligible, protected)
            self.assertEqual(self.store.items(), (item,))
            self.assertEqual(self.rule_status(rule), "ready")

    def test_filter_transaction_rolls_back_all_changes_if_event_write_fails(self):
        item = self.item("069500")
        rule = self.rule(item, external=True)
        with self.store.connection() as db:
            db.execute("CREATE TRIGGER fail_filter_event BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        with self.assertRaises(sqlite3.DatabaseError):
            self.store.restrict_to_common(Market.DOMESTIC, {("005930", "KRX")}, set())
        self.assertEqual(self.store.items(), (item,))
        self.assertEqual(self.rule_status(rule), "ready")
        self.assertEqual(self.store.external_for_rule(rule.id)["status"], "queued")

    def test_adoption_is_market_scoped_idempotent_and_skips_unranked_manual_or_inactive(self):
        kr = self.item("005930")
        us = self.item("AAPL", Market.US)
        inactive = self.item("069500")
        manual = self.item("000660")
        self.store.add_ranked((
            RankedStock(Market.DOMESTIC, "005930", "KRX", "common", 1, Decimal(1000), "KRW"),
            RankedStock(Market.DOMESTIC, "069500", "KRX", "inactive", 2, Decimal(900), "KRW"),
            RankedStock(Market.US, "AAPL", "ND", "common", 1, Decimal(1000), "USD"),
        ))
        self.store.remove_item(inactive.id)
        self.assertEqual(self.store.adopt_ranked_management(Market.DOMESTIC), 1)
        self.assertEqual(self.store.adopt_ranked_management(Market.DOMESTIC), 0)
        self.assertEqual(self.store.adopt_ranked_management(), 1)
        self.assertEqual(self.store.adopt_ranked_management(), 0)
        with self.store.connection() as db:
            managed = {row[0] for row in db.execute("SELECT watch_id FROM managed_watchlist")}
        self.assertEqual(managed, {kr.id, us.id})
        self.assertNotIn(manual.id, managed)
        self.assertNotIn(inactive.id, managed)
        # A later manual edit still unpins the stock from automated rotation.
        self.store.save_item(kr)
        with self.store.connection() as db:
            self.assertIsNone(db.execute("SELECT 1 FROM managed_watchlist WHERE watch_id=?", (kr.id,)).fetchone())


if __name__ == "__main__":
    unittest.main()
