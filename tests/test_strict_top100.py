"""Exact ranked watch membership without discarding the order ledger."""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from dockdack.gui_service import Instrument
from dockdack.models import Market
from dockdack.universe import RankedStock
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore


NOW = datetime(2026, 9, 28, 1, 0, tzinfo=timezone.utc)


def ranked(market, start=1, count=100, basis="volume"):
    return tuple(
        RankedStock(market, f"{index:06d}" if market is Market.DOMESTIC else f"S{index}",
                    "KRX" if market is Market.DOMESTIC else "ND", f"Name {index}", rank,
                    Decimal(100000 - rank), "KRW" if market is Market.DOMESTIC else "USD",
                    100000 - rank, basis)
        for rank, index in enumerate(range(start, start + count), 1)
    )


def watch(market, index):
    return WatchItem(Instrument(market, f"{index:06d}" if market is Market.DOMESTIC else f"S{index}",
                                "KRX" if market is Market.DOMESTIC else "ND"), f"Name {index}", 31)


class StrictTop100Tests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = WatchStore(Path(folder.name) / "watch.sqlite3")

    def test_rotation_is_100_per_market_even_with_manual_pending_and_frozen_extras(self):
        for market in Market:
            self.store.replace_ranked(market, ranked(market), set())
            manual, pending, frozen = (watch(market, index) for index in (1, 2, 3))
            self.store.save_item(manual)
            rule = TriggerRule.create(pending, "price_ge", "buy", 1, Decimal(1000), Decimal(100))
            self.store.add_rule(rule)
            self.assertTrue(self.store.claim(rule, Decimal(100), NOW))
            self.store.finish(rule.id, "accepted", "test accepted", f"pending-{market.value}")
            self.store.replace_ranked(market, ranked(market, 101), {pending.instrument.symbol},
                                      separate_holdings=True, preserve_watch_ids=(frozen.id,))
            active = {item.id for item in self.store.items() if item.instrument.market is market}
            self.assertEqual(len(active), 100)
            self.assertTrue({manual.id, pending.id, frozen.id}.isdisjoint(active))
            self.assertEqual(self.store.attempts(pending.id)[0]["status"], "accepted")
            self.assertEqual(self.store.order_history(watch_id=pending.id)[0]["order_number"],
                             f"pending-{market.value}")
            self.assertEqual(self.store.rules(pending.id, include_inactive=True)[0].status, "accepted")
        self.assertEqual(len(self.store.items()), 200)

    def test_launch_prune_volume_and_legacy_turnover_103_to_100_keeps_history(self):
        for market, basis in ((Market.DOMESTIC, "turnover"), (Market.US, "volume")):
            with self.subTest(market=market):
                self.store.replace_ranked(market, ranked(market, basis=basis), set())
                extras = [watch(market, index) for index in (101, 102, 103)]
                with self.store.connection() as db:
                    for item in extras:
                        db.execute("INSERT INTO watchlist VALUES(?,?,?,?,?,?,1)",
                                   (item.id, market.value, item.instrument.symbol,
                                    item.instrument.exchange, item.name, item.days))
                ready = TriggerRule.create(extras[0], "price_ge", "buy", 1, Decimal(1000), Decimal(100))
                accepted = TriggerRule.create(extras[1], "price_ge", "buy", 1, Decimal(1000), Decimal(100))
                self.store.add_rule(ready)
                self.store.add_rule(accepted)
                self.assertTrue(self.store.claim(accepted, Decimal(100), NOW))
                self.store.finish(accepted.id, "accepted", "test accepted", f"existing-order-{market.value}")
                with self.store.connection() as db:
                    db.execute(
                        "INSERT INTO external_signals VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        ("historic-model", f"queued-{market.value}", "{}", ready.id,
                         extras[0].id, NOW.isoformat(), NOW.isoformat(), "old-export",
                         NOW.isoformat(), "buy", "queued"),
                    )
                    db.execute(
                        "INSERT INTO external_signals VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        ("historic-model", f"accepted-{market.value}", "{}", accepted.id,
                         extras[1].id, NOW.isoformat(), NOW.isoformat(), "old-export",
                         NOW.isoformat(), "buy", "accepted"),
                    )
                self.assertEqual(sum(item.instrument.market is market for item in self.store.items()), 103)
                self.assertEqual(self.store.prune_ranked_extras(market), 3)
                self.assertEqual(sum(item.instrument.market is market for item in self.store.items()), 100)
                self.assertEqual(self.store.prune_ranked_extras(market), 0)
                self.assertEqual(self.store.rules(extras[0].id, include_inactive=True)[0].status, "paused")
                self.assertEqual(self.store.attempts(extras[1].id)[0]["status"], "accepted")
                self.assertEqual(self.store.order_history(watch_id=extras[1].id)[0]["order_number"],
                                 f"existing-order-{market.value}")
                with self.store.connection() as db:
                    self.assertEqual(db.execute("SELECT status FROM external_signals WHERE rule_id=?",
                                                (ready.id,)).fetchone()[0], "paused")
                    self.assertEqual(db.execute("SELECT status FROM external_signals WHERE rule_id=?",
                                                (accepted.id,)).fetchone()[0], "accepted")
                with self.assertRaisesRegex(ValueError, "TOP100"):
                    self.store.save_item(extras[2])
        self.assertEqual(len(self.store.items()), 200)

    def test_no_prune_without_complete_volume_rank(self):
        extra = watch(Market.DOMESTIC, 999999)
        self.store.save_item(extra)
        self.assertEqual(self.store.prune_ranked_extras(Market.DOMESTIC), 0)
        self.store.add_ranked(ranked(Market.DOMESTIC, count=99))
        self.assertEqual(self.store.prune_ranked_extras(Market.DOMESTIC), 0)
        self.assertIn(extra.id, {item.id for item in self.store.items()})

    def test_malformed_persisted_rank_is_not_used_to_prune(self):
        self.store.replace_ranked(Market.US, ranked(Market.US), set())
        extra = watch(Market.US, 101)
        with self.store.connection() as db:
            db.execute("INSERT INTO watchlist VALUES(?,?,?,?,?,?,1)",
                       (extra.id, "us", extra.instrument.symbol, "ND", extra.name, extra.days))
            db.execute("UPDATE turnover_ranks SET ranking_basis='turnover' WHERE market='us' AND rank=1")
        self.assertEqual(self.store.prune_ranked_extras(Market.US), 0)
        self.assertEqual(len(self.store.items()), 101)

    def test_legacy_bulk_complete_rank_replaces_extras_and_partial_cannot_expand_it(self):
        extra = watch(Market.DOMESTIC, 999999)
        self.store.save_item(extra)
        self.store.add_ranked(ranked(Market.DOMESTIC))
        self.assertEqual(len(self.store.items()), 100)
        self.assertNotIn(extra.id, {item.id for item in self.store.items()})
        with self.assertRaisesRegex(ValueError, "TOP100"):
            self.store.add_ranked(ranked(Market.DOMESTIC, 101, count=1))
        self.assertEqual(len(self.store.items()), 100)


if __name__ == "__main__":
    unittest.main()
