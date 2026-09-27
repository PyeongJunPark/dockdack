from __future__ import annotations

import tempfile
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from dockdack.history import DailyBar, DailyHistory
from dockdack.mark1_4_preopen import collect_preopen_candidates
from dockdack.market_schedule import session_on
from dockdack.models import Market
from dockdack.universe import RankedStock
from dockdack.watchlist import WatchStore


class _Histories:
    def __init__(self, dates, trading_day):
        self.dates = dates
        self.trading_day = trading_day
        self.calls = []
        self.after_get = None
        self.include_current = True
        self.omit_completed = None
        self.future = False

    def get(self, item, days):
        self.calls.append((item.id, days))
        if self.after_get:
            self.after_get()
        dates = [day for day in self.dates if day != self.omit_completed]
        if self.include_current:
            dates.append(self.trading_day)
        if self.future:
            dates.append(self.trading_day + timedelta(days=1))
        bars = tuple(DailyBar(day, Decimal("100"), Decimal("102"), Decimal("99"),
                              Decimal("101"), Decimal("1000")) for day in dates)
        inst = item.instrument
        return DailyHistory(inst.market, inst.symbol, inst.exchange, inst.currency, days, bars)


class PreopenCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "watch.sqlite3")
        self.market = Market.DOMESTIC
        self.day = date(2026, 9, 14)
        session = session_on(self.market, self.day)
        self.assertIsNotNone(session)
        self.opened = session.opened
        self.ranking_at = self.opened - timedelta(minutes=5)
        self.state = {"now": self.ranking_at + timedelta(seconds=30)}
        ranked = tuple(RankedStock(self.market, f"{index:06d}", "KRX", f"종목{index}", index,
                                   Decimal("100"), "KRW", 101 - index, "volume")
                       for index in range(1, 101))
        with patch("dockdack.persistence.watchlist.utc_now", return_value=self.ranking_at):
            self.store.replace_ranked(self.market, ranked, set())
        dates = []
        cursor = self.day - timedelta(days=1)
        while len(dates) < 30:
            if session_on(self.market, cursor) is not None:
                dates.append(cursor)
            cursor -= timedelta(days=1)
        self.dates = tuple(reversed(dates))
        self.histories = _Histories(self.dates, self.day)
        self.engine = SimpleNamespace(clock=lambda: self.state["now"], _stop=Event(),
                                      histories=self.histories)

    def collect(self):
        return collect_preopen_candidates(self.store, self.engine, self.market)

    def test_current_preopen_volume_top100_produces_complete_rows_without_quotes(self):
        result = self.collect()
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(len(result.candidates), 100)
        self.assertEqual(result.ranking_fetched_at, self.ranking_at)
        self.assertEqual(result.session_open, self.opened)
        self.assertEqual(result.candidates[0]["watch_id"], "domestic:KRX:000001")
        self.assertEqual(result.candidates[0]["symbol"], "000001")
        self.assertEqual(len(result.candidates[0]["bars"]), 30)
        self.assertEqual(result.candidates[0]["last_completed_date"], self.dates[-1].isoformat())
        self.assertEqual(result.candidates[0]["dates"], tuple(day.isoformat() for day in self.dates))
        self.assertNotIn(self.day.isoformat(), result.candidates[0]["dates"])
        self.assertEqual(len(self.histories.calls), 100)
        self.assertTrue(all(days == 31 for _, days in self.histories.calls))
        self.assertFalse(hasattr(self.engine, "service"))

    def test_stale_or_wrong_basis_ranking_never_fetches_history(self):
        with self.store.connection() as db:
            db.execute("UPDATE turnover_ranks SET fetched_at=?", ((self.opened - timedelta(minutes=11)).isoformat(),))
        result = self.collect()
        self.assertFalse(result.ok)
        self.assertEqual(result.candidates, ())
        self.assertIn("ranking_not_current_preopen_slot", result.reason)
        self.assertEqual(self.histories.calls, [])

        with self.store.connection() as db:
            db.execute("UPDATE turnover_ranks SET fetched_at=?, ranking_basis='turnover'", (self.ranking_at.isoformat(),))
        result = self.collect()
        self.assertIn("ranking_basis_invalid", result.reason)
        self.assertEqual(self.histories.calls, [])

    def test_rank_gaps_or_duplicate_membership_fail_before_history_fetch(self):
        with self.store.connection() as db:
            db.execute("UPDATE turnover_ranks SET rank=101 WHERE market='domestic' AND rank=100")
        result = self.collect()
        self.assertIn("ranking_rank_invalid", result.reason)
        self.assertEqual(self.histories.calls, [])

        with self.store.connection() as db:
            db.execute("UPDATE turnover_ranks SET rank=100 WHERE market='domestic' AND rank=101")
            db.execute("UPDATE turnover_ranks SET watch_id=(SELECT watch_id FROM turnover_ranks WHERE market='domestic' AND rank=1) WHERE market='domestic' AND rank=100")
        result = self.collect()
        self.assertIn("ranking_candidates_invalid", result.reason)
        self.assertEqual(result.candidates, ())
        self.assertEqual(self.histories.calls, [])

    def test_incomplete_or_future_history_voids_the_whole_batch(self):
        self.histories.omit_completed = self.dates[-2]
        result = self.collect()
        self.assertFalse(result.ok)
        self.assertEqual(result.candidates, ())
        self.assertIn("history_missing_consecutive_sessions", result.reason)
        self.histories.omit_completed = None
        self.histories.calls.clear()
        self.histories.future = True
        result = self.collect()
        self.assertIn("history_dates_invalid_or_future", result.reason)
        self.assertEqual(result.candidates, ())

    def test_deadline_or_cancel_after_history_fetch_never_returns_partial_batch(self):
        self.histories.after_get = lambda: self.state.update(now=self.opened)
        result = self.collect()
        self.assertIn("preopen_deadline_passed", result.reason)
        self.assertEqual(result.candidates, ())
        self.assertEqual(len(self.histories.calls), 1)

        self.state["now"] = self.ranking_at + timedelta(seconds=30)
        self.histories.calls.clear()
        self.histories.after_get = self.engine._stop.set
        result = self.collect()
        self.assertIn("cancelled", result.reason)
        self.assertEqual(result.candidates, ())
        self.assertEqual(len(self.histories.calls), 1)

    def test_outside_preopen_window_returns_no_candidates(self):
        self.state["now"] = self.opened
        result = self.collect()
        self.assertIn("preopen_deadline_passed", result.reason)
        self.assertEqual(result.candidates, ())
        self.assertEqual(self.histories.calls, [])


if __name__ == "__main__":
    unittest.main()
