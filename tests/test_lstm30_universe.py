"""Offline TOP100 authorization, atomic rotation, and selected-market scheduling."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from dockdack import BrokerAPIError, Market, TradingMode
from dockdack.gui_service import Instrument
from dockdack.lstm30_universe import LSTM30Universe, ScopedRankingScheduler
from dockdack.market_schedule import session_on
from dockdack.universe import RankedStock
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore


def ranked_common_stocks(start=1, *, market=Market.US, count=100):
    return tuple(RankedStock(
        market, f"S{index}" if market is Market.US else f"{index:06d}",
        "ND" if market is Market.US else "KRX", f"Common company {index}", rank,
        Decimal(100_000 - rank), "USD" if market is Market.US else "KRW", 100000-rank, "volume",
    ) for rank, index in enumerate(range(start, start + count), 1))


class RankingService:
    """top_volume represents the broker's already-classified common-stock API."""

    mode = TradingMode.DEMO

    def __init__(self):
        self.data = {market: ranked_common_stocks(market=market) for market in Market}
        self.protected = {market: set() for market in Market}
        self.top_volume = Mock(side_effect=lambda market, limit: self.data[market])
        self.protected_symbols = Mock(side_effect=lambda market: self.protected[market])


class LSTM30UniverseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "watch.sqlite3")
        self.service = RankingService()
        self.now = session_on(Market.US, date(2026, 9, 14)).opened
        self.universe = LSTM30Universe(self.service, self.store, ranked_markets=(Market.US,),
                                      baseline_items=(), clock=lambda: self.now)

    def test_bootstrap_has_exactly_100_unique_common_us_stocks_with_31_bars(self):
        self.universe.bootstrap()
        items = self.universe.items()
        self.assertTrue(self.universe.initialized)
        self.assertEqual(len(items), 100)
        self.assertEqual(len({item.id for item in items}), 100)
        self.assertTrue(all(item.instrument.market is Market.US for item in items))
        self.assertTrue(all(item.days >= 31 for item in items))
        self.service.top_volume.assert_called_once_with(Market.US, 100)
        self.service.protected_symbols.assert_called_once_with(Market.US)
        self.universe.validate_active()

    def test_selected_us_ranking_preserves_other_market_manual_watch(self):
        domestic = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "Samsung", 31)
        self.store.save_item(domestic)
        universe = LSTM30Universe(self.service, self.store, ranked_markets=(Market.US,),
                                 baseline_items=(domestic,), clock=lambda: self.now)
        universe.bootstrap()
        self.assertIn(domestic.id, {item.id for item in universe.items()})
        self.assertEqual(sum(item.instrument.market is Market.US for item in universe.items()), 100)
        self.service.top_volume.assert_called_once_with(Market.US, 100)

    def test_rotation_preserves_manual_held_pending_and_ready_manual_rule(self):
        self.universe.bootstrap()
        old = {item.instrument.symbol: item for item in self.store.items()}
        self.store.save_item(old["S1"])  # Explicit manual pin, same authorized identity.
        pending = TriggerRule.create(old["S2"], "price_ge", "buy", 1, Decimal(1000), Decimal(100))
        self.store.add_rule(pending)
        self.assertTrue(self.store.claim(pending, Decimal(100), self.now))
        self.store.finish(pending.id, "accepted", "fake accepted", "fake-pending")
        self.service.protected[Market.US] = {"S3"}
        manual_rule = TriggerRule.create(old["S4"], "price_ge", "sell", 1, Decimal(1000), Decimal(101))
        self.store.add_rule(manual_rule)
        self.service.data[Market.US] = ranked_common_stocks(101)
        self.universe.refresh(Market.US)
        active = {item.instrument.symbol: item for item in self.universe.items()}
        self.assertEqual(len(active), 103)
        self.assertTrue({"S1", "S2", "S4"}.issubset(active))
        self.assertNotIn("S3", active)  # Holdings remain in the independent exit scan.
        self.assertNotIn("S5", active)
        self.assertTrue(all(item.days >= 31 for item in active.values()))
        self.assertEqual(self.store.attempts()[0]["status"], "accepted")
        self.assertEqual(next(rule for rule in self.store.rules() if rule.id == manual_rule.id).status, "ready")
        self.universe.validate_active()

    def test_rotation_keeps_filled_buy_history_used_by_durable_daily_cap(self):
        self.universe.bootstrap()
        item = next(item for item in self.store.items() if item.instrument.symbol == "S5")
        rule = TriggerRule.create(item, "price_ge", "buy", 1, Decimal(1000), Decimal(100))
        self.store.add_rule(rule)
        self.store.claim(rule, Decimal(100), self.now)
        self.store.finish(rule.id, "accepted", "fake accepted", "fake-filled")
        self.store.finish(rule.id, "filled", "fake filled", "fake-filled")
        before = self.store.attempts(item.id)
        self.service.data[Market.US] = ranked_common_stocks(101)
        self.universe.refresh(Market.US)
        self.assertNotIn(item.id, {stock.id for stock in self.universe.items()})
        self.assertEqual(self.store.attempts(item.id), before)
        self.service.data[Market.US] = ranked_common_stocks()
        self.universe.refresh(Market.US)
        self.assertIn(item.id, {stock.id for stock in self.universe.items()})
        self.assertEqual(self.store.attempts(item.id), before)

    def test_short_ranking_cannot_partially_replace_previous_approved_list(self):
        self.universe.bootstrap()
        before = self.store.items()
        self.service.data[Market.US] = ranked_common_stocks(101, count=99)
        with self.assertRaises((ValueError, BrokerAPIError)):
            self.universe.refresh(Market.US)
        self.assertEqual(self.store.items(), before)
        self.universe.validate_active()

    def test_duplicate_wrong_currency_or_nonfinite_ranking_preserves_prior_list(self):
        self.universe.bootstrap()
        before = self.store.items()
        valid = ranked_common_stocks(101)
        for invalid in ((valid[0],) + valid[:-1],
                        (replace(valid[0], currency="KRW"),) + valid[1:],
                        (replace(valid[0], turnover=Decimal("NaN")),) + valid[1:]):
            with self.subTest(first=invalid[0]):
                self.service.data[Market.US] = invalid
                with self.assertRaises((ValueError, BrokerAPIError)):
                    self.universe.refresh(Market.US)
                self.assertEqual(self.store.items(), before)
                self.universe.validate_active()

    def test_classification_or_account_failure_preserves_previous_approved_list(self):
        self.universe.bootstrap()
        before = self.store.items()
        for method in ("top_volume", "protected_symbols"):
            with self.subTest(method=method), patch.object(
                    self.service, method, side_effect=BrokerAPIError("fake classification/account unavailable")):
                with self.assertRaises(BrokerAPIError):
                    self.universe.refresh(Market.US)
            self.assertEqual(self.store.items(), before)
            self.universe.validate_active()

    def test_unselected_market_refresh_is_rejected_without_query(self):
        with self.assertRaises(ValueError):
            self.universe.refresh(Market.DOMESTIC)
        self.service.top_volume.assert_not_called()

    def test_authorized_rotation_passes_guard_but_foreign_database_edit_fails(self):
        self.universe.bootstrap()
        self.service.data[Market.US] = ranked_common_stocks(101)
        self.universe.refresh(Market.US)
        self.universe.validate_active()
        self.store.save_item(WatchItem(Instrument(Market.US, "FOREIGN", "ND"), "Unapproved", 31))
        with self.assertRaises(ValueError):
            self.universe.validate_active()

    def test_existing_watch_days_are_upgraded_without_resetting_history(self):
        old = WatchItem(Instrument(Market.US, "S1", "ND"), "Old manual", 30)
        self.store.save_item(old)
        universe = LSTM30Universe(self.service, self.store, ranked_markets=(Market.US,),
                                 baseline_items=(old,), clock=lambda: self.now)
        universe.bootstrap()
        self.assertTrue(all(item.days >= 31 for item in universe.items()))


class ScopedRankingSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "watch.sqlite3")
        self.service = RankingService()
        self.session = session_on(Market.US, date(2026, 9, 14))
        self.now = self.session.opened - timedelta(minutes=11)
        self.stopped = False
        self.universe = LSTM30Universe(self.service, self.store, ranked_markets=(Market.US,),
                                      baseline_items=(), clock=lambda: self.now)
        self.now = self.session.opened - timedelta(minutes=10)
        self.universe.bootstrap()
        self.now = self.session.opened - timedelta(minutes=11)
        self.service.top_volume.reset_mock()
        self.errors = Mock()
        self.scheduler = ScopedRankingScheduler(self.service, self.store, universe=self.universe,
                                               clock=lambda: self.now, stopped=lambda: self.stopped,
                                               on_error=self.errors)
        self.scheduler.start()

    def test_only_authorized_us_market_is_ranked_and_duplicate_slot_is_not_repeated(self):
        self.assertFalse(self.scheduler.tick())
        self.now = self.session.opened + timedelta(minutes=30)
        self.assertTrue(self.scheduler.due())
        self.assertTrue(self.scheduler.tick())
        self.assertFalse(self.scheduler.tick())
        self.service.top_volume.assert_called_once_with(Market.US, 100)
        self.assertTrue(all(item.days >= 31 for item in self.store.items()))
        self.errors.assert_not_called()

    def test_bootstrap_records_current_slot_without_an_immediate_second_fetch(self):
        self.now = self.session.opened
        # The earlier bootstrap completed only the pre-open slot. Opening
        # needs its own fetch; once that finishes it must not repeat.
        self.assertTrue(self.scheduler.due())
        self.universe.bootstrap()
        self.service.top_volume.assert_called_once_with(Market.US, 100)
        self.service.top_volume.reset_mock()
        self.scheduler.record_bootstrap()
        self.assertFalse(self.scheduler.due())
        self.assertFalse(self.scheduler.tick())
        self.service.top_volume.assert_not_called()
        self.now = self.session.opened + timedelta(minutes=30)
        self.assertTrue(self.scheduler.tick())
        self.service.top_volume.assert_called_once_with(Market.US, 100)

    def test_domestic_open_does_not_trigger_an_unapproved_domestic_ranking(self):
        self.now = session_on(Market.DOMESTIC, date(2026, 9, 15)).opened
        self.scheduler.start()
        self.assertFalse(self.scheduler.due())
        self.assertFalse(self.scheduler.tick())
        self.service.top_volume.assert_not_called()

    def test_ranking_error_keeps_prior_list_and_calls_fail_closed_handler(self):
        previous = self.store.items()
        self.now = self.session.opened + timedelta(minutes=30)
        self.service.top_volume.side_effect = BrokerAPIError("fake ranking unavailable")
        self.assertFalse(self.scheduler.tick())
        self.assertEqual(self.store.items(), previous)
        self.errors.assert_called()
        self.assertTrue(self.scheduler.errors)
        self.universe.validate_active()

    def test_stopped_scheduler_makes_no_ranking_requests(self):
        self.now = self.session.opened
        self.stopped = True
        self.assertFalse(self.scheduler.due())
        self.assertFalse(self.scheduler.tick())
        self.service.top_volume.assert_not_called()

    def test_market_close_during_ranking_fetch_prevents_late_rotation(self):
        previous = self.store.items()
        self.now = self.session.opened + timedelta(minutes=30)

        def ranks_at_close(market, limit):
            self.now = self.session.closed
            return ranked_common_stocks(101)

        self.service.top_volume.side_effect = ranks_at_close
        self.assertFalse(self.scheduler.tick())
        self.assertEqual(self.store.items(), previous)
        self.errors.assert_called()


if __name__ == "__main__":
    unittest.main()
