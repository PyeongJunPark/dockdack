"""Offline contract tests for exchange-scoped, durable volume TOP100 refreshes."""

from datetime import date, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from dockdack.exceptions import BrokerAPIError
from dockdack.http import APIPage
from dockdack.lstm30_universe import LSTM30Universe, ScopedRankingScheduler
from dockdack.market_schedule import RankingScheduler, is_open, ranking_allowed, ranking_slot, session_on
from dockdack.models import Market, TradingMode
from dockdack.universe import RankedStock, top_volume
from dockdack.watchlist import WatchStore


class Pages:
    def __init__(self, *pages):
        self.pages, self.calls = pages, []

    def iter_pages(self, **kwargs):
        self.calls.append(kwargs)
        for body in self.pages:
            yield APIPage(body, True, "next", "Y")


def ranks(market):
    return tuple(RankedStock(market, f"{i:06d}" if market is Market.DOMESTIC else f"S{i}",
                            "KRX" if market is Market.DOMESTIC else "ND", "Common", i,
                            Decimal(1000+i), "KRW" if market is Market.DOMESTIC else "USD",
                            10000-i, "volume") for i in range(1, 101))


class VolumeContractTests(unittest.TestCase):
    def setUp(self):
        self.classifier = patch("dockdack.universe.common_equities",
                                side_effect=lambda http, market, candidates: frozenset(candidates)).start()
        self.addCleanup(patch.stopall)

    def test_kr_uses_share_volume_not_money_or_nonexistent_rank(self):
        http = Pages({"tdy_trde_qty_upper": [
            {"stk_cd": "005930", "trde_qty": "1,000", "trde_amt": "2"},
            {"stk_cd": "000660", "trde_qty": "2,000", "trde_amt": "1"}]})
        rows = top_volume(http, Market.DOMESTIC, 2)
        self.assertEqual([row.symbol for row in rows], ["000660", "005930"])
        self.assertEqual([row.volume for row in rows], [2000, 1000])
        self.assertEqual(rows[0].turnover, Decimal(1000000))
        self.assertEqual(rows[0].ranking_basis, "volume")
        call = http.calls[0]
        self.assertEqual(call["api_id"], "ka10030")
        self.assertEqual(call["path"], "/api/dostk/rkinfo")
        self.assertEqual(call["body"]["sort_tp"], "1")
        self.assertEqual(call["body"]["stex_tp"], "1")
        self.assertEqual(call["body"]["mrkt_open_tp"], "0")

    def test_us_uses_volume_ranking_and_unscaled_shares(self):
        http = Pages({"result_list": [
            {"stk_cd": "AAPL", "stex_tp": "ND", "rank": "1", "acc_trde_qty": "30", "trde_prica": "7"},
            {"stk_cd": "MSFT", "stex_tp": "ND", "rank": "2", "acc_trde_qty": "50", "trde_prica": "2"}]})
        rows = top_volume(http, Market.US, 2)
        self.assertEqual([row.symbol for row in rows], ["MSFT", "AAPL"])
        self.assertEqual(rows[0].volume, 50)
        self.assertEqual(rows[0].turnover, Decimal(2000))
        self.assertEqual(http.calls[0]["api_id"], "usa20530")
        self.assertEqual(http.calls[0]["body"]["qry_tp"], "0")
        self.assertEqual(http.calls[0]["body"]["stk_tp"], "1")

    def test_filter_pagination_duplicates_and_otc(self):
        self.classifier.side_effect = lambda http, market, candidates: frozenset(
            key for key in candidates if key[0] != "ETF")
        def row(symbol, exchange="ND"):
            return {"stk_cd": symbol, "stex_tp": exchange, "rank": "1", "acc_trde_qty": "100"}
        http = Pages({"result_list": [row("ETF"), row("OTC", "NP"), row("AAPL"), row("AAPL")]},
                     {"result_list": [row("MSFT")]})
        self.assertEqual({r.symbol for r in top_volume(http, Market.US, 2)}, {"AAPL", "MSFT"})

    def test_invalid_or_missing_volume_never_falls_back_to_amount(self):
        for quantity in (None, "NaN", "Infinity", "-1", "1.2", ""):
            with self.subTest(quantity=quantity), self.assertRaises(BrokerAPIError):
                top_volume(Pages({"tdy_trde_qty_upper": [
                    {"stk_cd": "005930", "trde_qty": quantity, "trde_amt": "500"}]}), Market.DOMESTIC, 1)

    def test_incomplete_or_malformed_common_universe_fails(self):
        for body in ({}, {"result_list": {}}, {"result_list": [None]}, {"result_list": []}):
            with self.subTest(body=body), self.assertRaises(BrokerAPIError):
                top_volume(Pages(body), Market.US, 1)
        for count in (0, 101, True, "10"):
            with self.assertRaises(ValueError):
                top_volume(Pages(), Market.US, count)


class RankingBoundaryTests(unittest.TestCase):
    def test_market_independence_and_exact_preopen_boundary(self):
        for market in Market:
            session = session_on(market, date(2026, 9, 16))
            before = session.opened - timedelta(minutes=10)
            self.assertFalse(ranking_allowed(market, before-timedelta(microseconds=1)))
            self.assertTrue(ranking_allowed(market, before))
            self.assertFalse(is_open(market, before))
            self.assertEqual(ranking_slot(market, before), before)
            self.assertFalse(ranking_allowed(market, session.closed))
            other = Market.US if market is Market.DOMESTIC else Market.DOMESTIC
            self.assertFalse(ranking_allowed(other, session.opened))

    def test_dst_and_halfday_slots_use_market_clock(self):
        seoul = ZoneInfo("Asia/Seoul")
        summer = session_on(Market.US, date(2026, 9, 16))
        winter = session_on(Market.US, date(2026, 11, 30))
        self.assertEqual(next(summer.slots()).astimezone(seoul).strftime("%H:%M"), "22:20")
        self.assertEqual(next(winter.slots()).astimezone(seoul).strftime("%H:%M"), "23:20")
        half = session_on(Market.US, date(2026, 11, 27))
        self.assertEqual([x.strftime("%H:%M") for x in half.slots()], ["09:20", "09:30", "10:00", "11:00", "12:00"])
        self.assertEqual(ranking_slot(Market.US, half.closed-timedelta(seconds=1)).hour, 12)
        self.assertIsNone(ranking_slot(Market.US, half.closed))


class DurableScheduleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = WatchStore(Path(self.tmp.name)/"watch.sqlite3")
        self.session = session_on(Market.DOMESTIC, date(2026, 9, 16))
        self.now = self.session.opened-timedelta(minutes=10)
        self.service = Mock(mode=TradingMode.DEMO)
        self.service.top_volume.side_effect = lambda market, limit: ranks(market)
        self.service.protected_symbols.return_value = set()
        self.scheduler = RankingScheduler(self.service, self.store, clock=lambda: self.now)
        self.scheduler.start()

    def test_preopen_once_open_once_then_long_sweep_catches_latest(self):
        self.assertTrue(self.scheduler.tick())
        self.assertFalse(self.scheduler.tick())
        self.now = self.session.opened
        self.assertTrue(self.scheduler.tick())
        self.now += timedelta(hours=2, minutes=47)
        self.assertTrue(self.scheduler.tick())
        self.assertFalse(self.scheduler.tick())
        self.assertEqual(self.service.top_volume.call_count, 3)

    def test_us_preopen_done_does_not_skip_actual_open_with_dst_or_halfday(self):
        for day in (date(2026, 9, 24), date(2026, 11, 27), date(2026, 11, 30)):
            with self.subTest(day=day):
                session = session_on(Market.US, day)
                self.now = session.opened - timedelta(minutes=10)
                before = self.service.top_volume.call_count
                self.assertTrue(self.scheduler.tick())
                self.now = session.opened - timedelta(microseconds=1)
                self.assertFalse(self.scheduler.due())
                self.now = session.opened
                self.assertEqual(ranking_slot(Market.US, self.now), session.opened)
                self.assertTrue(self.scheduler.due())
                self.assertTrue(self.scheduler.tick())
                self.assertFalse(self.scheduler.tick())
                restarted = RankingScheduler(self.service, self.store, clock=lambda: self.now)
                restarted.start()
                self.assertFalse(restarted.tick())
                self.now = session.opened + timedelta(minutes=12)
                self.assertFalse(restarted.due())
                self.now = session.opened.replace(hour=10, minute=0)
                self.assertTrue(restarted.tick())
                self.assertEqual(self.service.top_volume.call_count - before, 3)
                self.assertEqual(list(session.slots()).count(session.opened), 1)

    def test_us_start_after_open_catches_open_slot_even_when_preopen_done(self):
        session = session_on(Market.US, date(2026, 9, 24))
        self.now = session.opened - timedelta(minutes=10)
        self.assertTrue(self.scheduler.tick())
        self.now = session.opened + timedelta(minutes=9)
        restarted = RankingScheduler(self.service, self.store, clock=lambda: self.now)
        restarted.start()
        self.assertTrue(restarted.due())
        self.assertTrue(restarted.tick())
        self.assertEqual(self.service.top_volume.call_count, 2)
        with self.store.connection() as db:
            rows = db.execute("SELECT slot,status FROM ranking_runs WHERE market='us' ORDER BY slot").fetchall()
        self.assertEqual([row[1] for row in rows], ['done', 'done'])
        self.assertEqual(rows[-1][0], session.opened.astimezone(timezone.utc).isoformat())

    def test_preopen_response_crossing_open_never_overwrites_newer_slot(self):
        session = session_on(Market.US, date(2026, 9, 24))
        self.now = session.opened - timedelta(minutes=1)
        def delayed(market, limit):
            self.now = session.opened + timedelta(seconds=1)
            return ranks(market)
        self.service.top_volume.side_effect = delayed
        self.assertFalse(self.scheduler.tick())
        self.assertEqual(self.store.items(), ())
        self.assertFalse(self.scheduler.errors)
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT status FROM ranking_runs WHERE market='us'").fetchone()[0], 'superseded')
        self.service.top_volume.side_effect = lambda market, limit: ranks(market)
        self.assertTrue(self.scheduler.due())
        self.assertTrue(self.scheduler.tick())
        self.assertEqual(len(self.store.items()), 100)

    def test_superseded_preopen_is_not_a_legacy_order_disarming_error(self):
        session = session_on(Market.US, date(2026, 9, 24))
        self.now = session.opened - timedelta(minutes=1)
        universe = LSTM30Universe(self.service, self.store, ranked_markets=(Market.US,),
                                  baseline_items=(), clock=lambda: self.now)
        errors = Mock()
        scheduler = ScopedRankingScheduler(self.service, self.store, universe=universe,
                                           clock=lambda: self.now, on_error=errors)
        scheduler.start()
        def delayed(market, limit):
            self.now = session.opened + timedelta(seconds=1)
            return ranks(market)
        self.service.top_volume.side_effect = delayed
        self.assertFalse(scheduler.tick())
        errors.assert_not_called()
        self.assertFalse(scheduler.errors)
        self.service.top_volume.side_effect = lambda market, limit: ranks(market)
        self.assertTrue(scheduler.due())
        self.assertTrue(scheduler.tick())
        errors.assert_not_called()

    def test_restart_retains_dedupe_and_mid_hour_recovers_missing_slot(self):
        self.assertTrue(self.scheduler.tick())
        restarted = RankingScheduler(self.service, self.store, clock=lambda: self.now)
        restarted.start()
        self.assertFalse(restarted.tick())
        self.now += timedelta(hours=2, minutes=31)
        self.assertTrue(restarted.tick())
        self.assertEqual(self.service.top_volume.call_count, 2)

    def test_crashed_running_claim_is_recovered_after_lease_not_before(self):
        self.now = self.session.opened+timedelta(hours=1)
        slot = self.scheduler._slot(Market.DOMESTIC, self.now)
        abandoned = self.scheduler._claim(Market.DOMESTIC, slot, self.now)
        restarted = RankingScheduler(self.service, self.store, clock=lambda: self.now)
        restarted.start()
        self.now += timedelta(minutes=9)
        self.assertFalse(restarted.due())
        self.assertFalse(restarted.tick())
        self.now += timedelta(minutes=1)
        self.assertTrue(restarted.due())
        self.assertTrue(restarted.tick())
        self.assertFalse(self.scheduler._owns(Market.DOMESTIC, slot, abandoned))
        with self.store.connection() as db:
            row = db.execute("SELECT status,attempts FROM ranking_runs WHERE market=? AND slot=?", (Market.DOMESTIC.value,slot)).fetchone()
        self.assertEqual(tuple(row), ("done",2))

    def test_close_during_request_retains_original(self):
        def close(market, limit):
            self.now = self.session.closed
            return ranks(market)
        self.service.top_volume.side_effect = close
        self.assertFalse(self.scheduler.tick())
        self.assertEqual(self.store.items(), ())

    def test_legacy_running_claim_recovered_with_backoff(self):
        self.now = self.session.opened+timedelta(hours=1)
        slot = self.scheduler._slot(Market.DOMESTIC,self.now)
        self.assertTrue(self.store.claim_ranking_run(Market.DOMESTIC,slot,self.now))
        self.now += timedelta(minutes=10)
        self.assertTrue(self.scheduler.tick())

    def test_weekend_makes_no_ranking_or_account_calls(self):
        self.now = self.now.replace(day=19)
        self.assertFalse(self.scheduler.due())
        self.assertFalse(self.scheduler.tick())
        self.service.top_volume.assert_not_called()
        self.service.protected_symbols.assert_not_called()

    def test_lstm_bootstrap_closed_market_defers_independently(self):
        universe = LSTM30Universe(self.service,self.store,ranked_markets=tuple(Market),
                                  baseline_items=(),clock=lambda:self.now)
        universe.bootstrap()
        self.assertEqual(self.service.top_volume.call_args_list[0].args, (Market.DOMESTIC,100))
        self.assertEqual(self.service.top_volume.call_count,1)
        self.assertFalse(universe.initialized)
        self.now = self.session.opened
        self.assertTrue(universe.ready_for_open_markets())
        us = session_on(Market.US,date(2026,9,16))
        self.now = us.opened-timedelta(minutes=10)
        schedule = ScopedRankingScheduler(self.service,self.store,universe=universe,clock=lambda:self.now)
        schedule.start()
        self.assertTrue(schedule.tick())
        self.assertTrue(universe.initialized)
        self.assertEqual(self.service.top_volume.call_count,2)

    def test_closed_bootstrap_does_not_query_or_claim(self):
        self.now -= timedelta(seconds=1)
        universe = LSTM30Universe(self.service,self.store,ranked_markets=tuple(Market),
                                  baseline_items=(),clock=lambda:self.now)
        self.assertEqual(universe.bootstrap(), ())
        self.service.top_volume.assert_not_called()
        self.assertFalse(universe.initialized)

    def test_lstm_restart_adopts_persisted_volume_list_without_duplicate_query(self):
        universe = LSTM30Universe(self.service,self.store,ranked_markets=(Market.DOMESTIC,),
                                  baseline_items=(),clock=lambda:self.now)
        universe.bootstrap()
        self.assertTrue(universe.initialized)
        restarted = LSTM30Universe(self.service,self.store,ranked_markets=(Market.DOMESTIC,),
                                  clock=lambda:self.now)
        restarted.bootstrap()
        self.assertTrue(restarted.initialized)
        self.assertEqual(self.service.top_volume.call_count,1)
        self.assertEqual(restarted.status()["markets"]["domestic"]["ranked_count"],100)
        self.assertTrue(all(row["ranking_basis"] == "volume" for row in self.store.rankings()))

    def test_old_lease_owner_cannot_apply_after_reclaim(self):
        self.now = self.session.opened+timedelta(hours=1)
        def lose_ownership(market, limit):
            self.now += timedelta(minutes=10)
            slot = self.scheduler._slot(market,self.now)
            replacement = RankingScheduler(self.service,self.store,clock=lambda:self.now)
            self.assertIsNotNone(replacement._claim(market,slot,self.now))
            return ranks(market)
        self.service.top_volume.side_effect = lose_ownership
        self.assertFalse(self.scheduler.tick())
        self.assertEqual(self.store.items(), ())

    def test_unthrottled_final_validation_catches_database_change_after_cached_check(self):
        universe = LSTM30Universe(self.service,self.store,ranked_markets=(Market.DOMESTIC,),
                                  baseline_items=(),clock=lambda:self.now)
        universe.bootstrap()
        universe.validate_active(force=False)
        with self.store.connection() as db:
            db.execute("UPDATE watchlist SET active=0 WHERE symbol='000001'")
        with self.assertRaises(ValueError):
            universe.validate_active(force=True)


if __name__ == "__main__":
    unittest.main()
