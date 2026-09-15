import tempfile
import unittest
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from dockdack.history import DailyBar, DailyHistory
from dockdack.history_cache import HistoryCache
from dockdack.gui_service import Instrument
from dockdack.models import Market
from dockdack.watchlist import WatchItem, WatchStore


class HistoryCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name)/"db.sqlite3")
        self.item = WatchItem(Instrument(Market.DOMESTIC,"005930","KRX"))
        self.store.save_item(self.item)
        self.now = datetime(2026,9,15,10,tzinfo=ZoneInfo("Asia/Seoul"))
        self.requests = []
        self.adjusted = False
        class Service:
            def history(inner,inst,days):
                self.requests.append(days)
                cursor, bars = self.now.date(), []
                while len(bars)<days:
                    if cursor.weekday()<5:
                        price=Decimal(50 if self.adjusted else 100)
                        bars.append(DailyBar(cursor,price,price,price,price,Decimal(123)))
                    cursor-=timedelta(days=1)
                return DailyHistory(inst.market,inst.symbol,inst.exchange,inst.currency,days,tuple(reversed(bars)))
        self.service=Service()
        self.cache=HistoryCache(self.store,self.service,lambda:self.now)

    def test_initial_full_history_then_only_today_after_five_minutes(self):
        first=self.cache.get(self.item,30)
        self.now+=timedelta(minutes=4)
        self.cache.get(self.item,30)
        self.assertEqual(self.requests,[30])
        self.now+=timedelta(minutes=1)
        second=self.cache.get(self.item,30)
        self.assertEqual(self.requests,[30,1])
        self.assertEqual(first.bars[:-1],second.bars[:-1])

    def test_restart_uses_durable_completed_history(self):
        self.cache.get(self.item,30)
        self.now+=timedelta(minutes=6)
        restarted=HistoryCache(WatchStore(self.store.path),self.service,lambda:self.now)
        result=restarted.get(self.item,30)
        self.assertEqual(self.requests,[30,1])
        self.assertEqual(len(result.bars),30)

    def test_next_day_backfills_overlap_and_new_day_not_all_thirty(self):
        self.cache.get(self.item,30)
        self.now=self.now.replace(hour=16)
        self.cache.get(self.item,30)
        self.now+=timedelta(days=1)
        result=self.cache.get(self.item,30)
        self.assertEqual(self.requests,[30,1,2])
        self.assertEqual(result.bars[-1].day,date(2026,9,16))
        self.assertEqual(len(result.bars),30)

    def test_missing_days_backfilled_in_one_incremental_request(self):
        self.cache.get(self.item,30)
        self.now+=timedelta(days=3)
        result=self.cache.get(self.item,30)
        self.assertEqual(self.requests,[30,5])
        self.assertEqual(result.bars[-1].day,date(2026,9,18))

    def test_completed_bar_revision_forces_full_adjusted_resync(self):
        self.now=self.now.replace(hour=16)
        self.cache.get(self.item,30)
        self.now+=timedelta(days=1)
        self.adjusted=True
        result=self.cache.get(self.item,30)
        self.assertEqual(self.requests,[30,2,30])
        self.assertTrue(all(b.close==50 for b in result.bars))

    def test_restart_after_intraday_cache_keeps_completed_overlap_for_adjustments(self):
        self.cache.get(self.item,30)
        self.now+=timedelta(days=1)
        self.adjusted=True
        restarted=HistoryCache(WatchStore(self.store.path),self.service,lambda:self.now)
        result=restarted.get(self.item,30)
        self.assertEqual(self.requests,[30,3,30])
        self.assertTrue(all(b.close==50 for b in result.bars))

    def test_widened_period_and_manual_invalidation_backfill(self):
        self.cache.get(self.item,30)
        self.cache.get(self.item,60)
        self.cache.invalidate(self.item.id)
        self.cache.get(self.item,30)
        self.assertEqual(self.requests,[30,60,30])

    def test_after_close_and_weekend_do_not_keep_requesting_finalized_bars(self):
        self.now=datetime(2026,9,18,16,tzinfo=ZoneInfo("Asia/Seoul"))
        self.cache.get(self.item,30)
        self.now+=timedelta(hours=1)
        self.cache.get(self.item,30)
        self.now+=timedelta(days=2)
        self.cache.get(self.item,30)
        self.assertEqual(self.requests,[30])


if __name__ == "__main__":
    unittest.main()
