import tempfile
import unittest
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from dockdack.market_schedule import RankingScheduler, is_open, session_on
from dockdack.gui_service import Instrument
from dockdack.models import Market
from dockdack.universe import RankedStock
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore


def ranks(start=1, market=Market.DOMESTIC):
    return tuple(RankedStock(market, f"{i:06d}" if market is Market.DOMESTIC else f"S{i}",
                            "KRX" if market is Market.DOMESTIC else "ND", "rank", n, Decimal(1000-n),
                            "KRW" if market is Market.DOMESTIC else "USD") for n,i in enumerate(range(start,start+100),1))


class ScheduleTests(unittest.TestCase):
    def test_regular_domestic_and_us_slot_times(self):
        kr = session_on(Market.DOMESTIC, date(2026,9,15))
        us = session_on(Market.US, date(2026,9,15))
        self.assertEqual([s.strftime("%H:%M") for s in kr.slots()], [f"{h:02d}:00" for h in range(9,16)])
        self.assertEqual([s.strftime("%H:%M") for s in us.slots()], ["09:30"]+[f"{h:02d}:00" for h in range(10,16)])
        self.assertTrue(is_open(Market.DOMESTIC, kr.opened))
        self.assertFalse(is_open(Market.DOMESTIC, kr.closed))

    def test_holidays_dst_early_close_and_year_first_late_open(self):
        for m,d in ((Market.US,date(2026,7,3)), (Market.US,date(2026,9,7)),
                    (Market.DOMESTIC,date(2026,9,25)), (Market.DOMESTIC,date(2026,6,3)),
                    (Market.DOMESTIC,date(2026,7,17)),
                    (Market.DOMESTIC,date(2026,12,31)), (Market.DOMESTIC,date(2026,9,19))):
            self.assertIsNone(session_on(m,d))
        early = session_on(Market.US, date(2026,11,27))
        self.assertEqual([s.strftime("%H:%M") for s in early.slots()], ["09:30","10:00","11:00","12:00"])
        self.assertEqual(session_on(Market.US,date(2026,7,2)).closed.hour,16)
        self.assertEqual(session_on(Market.US,date(2026,12,24)).closed.hour,13)
        self.assertEqual(session_on(Market.DOMESTIC,date(2026,1,2)).opened.hour,10)
        seoul = ZoneInfo("Asia/Seoul")
        self.assertEqual(session_on(Market.US,date(2026,7,6)).opened.astimezone(seoul).strftime("%H:%M"),"22:30")
        self.assertEqual(session_on(Market.US,date(2026,11,30)).opened.astimezone(seoul).strftime("%H:%M"),"23:30")
        with self.assertRaises(ValueError):
            session_on(Market.DOMESTIC,date(2026,11,19))


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name)/"db.sqlite3")
        self.now = datetime(2026,9,15,8,59,tzinfo=ZoneInfo("Asia/Seoul"))
        self.calls, self.protected = [], set()
        self.data = ranks()
        self.error = None
        class Service:
            def top_turnover(inner, market, limit):
                self.calls.append((market,limit))
                if self.error:
                    raise self.error
                return self.data
            def protected_symbols(inner, market):
                return self.protected
        self.service = Service()
        self.scheduler = RankingScheduler(self.service,self.store,clock=lambda:self.now)
        self.scheduler.start()

    def test_runs_open_and_hour_once_with_restart_dedup(self):
        self.assertFalse(self.scheduler.tick())
        self.now += timedelta(minutes=1)
        self.assertTrue(self.scheduler.due())
        self.assertTrue(self.scheduler.tick())
        self.assertFalse(self.scheduler.due())
        self.assertFalse(self.scheduler.tick())
        new = RankingScheduler(self.service,self.store,clock=lambda:self.now)
        new.start()
        self.assertFalse(new.tick())
        self.now += timedelta(hours=1)
        self.assertTrue(new.tick())
        self.assertEqual(self.calls,[(Market.DOMESTIC,100)]*2)

    def test_mid_hour_start_waits_and_late_wakeup_does_not_backfill_old_slots(self):
        self.now = self.now.replace(hour=10,minute=20)
        self.scheduler.start()
        self.assertFalse(self.scheduler.tick())
        self.now = self.now.replace(hour=11,minute=8)
        self.assertFalse(self.scheduler.tick())
        self.now = self.now.replace(hour=12,minute=0,second=8)
        self.assertTrue(self.scheduler.tick())

    def test_failure_keeps_previous_list_and_bounds_retries(self):
        self.store.add_ranked(ranks())
        previous = self.store.items()
        self.now += timedelta(minutes=1)
        self.error = ValueError("API unavailable")
        self.assertFalse(self.scheduler.tick())
        self.scheduler.tick()
        self.assertEqual(len(self.calls),1)
        for _ in range(5):
            self.now += timedelta(minutes=1)
            self.scheduler.tick()
        self.assertEqual(len(self.calls),3)
        self.assertEqual(self.store.items(),previous)

    def test_rotation_preserves_manual_held_pending_and_rules_other_market(self):
        self.store.add_ranked(ranks())
        self.store.add_ranked(ranks(market=Market.US))
        original = self.store.items()[0]
        self.store.save_item(original)  # Manual pin.
        pending = self.store.items()[1]
        rule = TriggerRule.create(pending,"price_ge","buy",1,Decimal(1000),Decimal(100))
        self.store.add_rule(rule)
        self.store.claim(rule,Decimal(100),self.now)
        self.protected = {"000003"}
        self.data = ranks(101)
        self.now += timedelta(minutes=1)
        self.assertTrue(self.scheduler.tick())
        active = {i.instrument.symbol for i in self.store.items() if i.instrument.market is Market.DOMESTIC}
        self.assertEqual(len(active),103)
        self.assertTrue({"000001","000002","000003"}.issubset(active))
        self.assertNotIn("000004",active)
        self.assertEqual(sum(i.instrument.market is Market.US for i in self.store.items()),100)
        self.assertEqual(self.store.attempts()[0]["status"],"submitting")

    def test_protection_lookup_failure_rolls_back_and_close_blocks_apply(self):
        self.now += timedelta(minutes=1)
        with patch.object(self.service,"protected_symbols",side_effect=ValueError("bad account")):
            self.assertFalse(self.scheduler.tick())
        self.assertEqual(self.store.items(),())
        self.now = self.now.replace(hour=15,minute=30)
        self.assertFalse(self.scheduler.tick())

    def test_stopping_before_tick_never_requests(self):
        self.now += timedelta(minutes=1)
        self.scheduler.stop()
        self.assertFalse(self.scheduler.tick())
        self.assertEqual(self.calls,[])


if __name__ == "__main__":
    unittest.main()
