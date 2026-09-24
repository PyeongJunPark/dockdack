"""Calendar construction is serialized; cached session infrastructure stays fast."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dockdack.market_schedule import calendar_for
from dockdack.lstm30_adapter import _market_calendar
from dockdack.models import Market


class CalendarThreadingTests(unittest.TestCase):
    def setUp(self):
        calendar_for.cache_clear()
        _market_calendar.cache_clear()
        self.addCleanup(calendar_for.cache_clear)
        self.addCleanup(_market_calendar.cache_clear)

    def test_cold_construction_never_overlaps_same_key_or_different_years(self):
        for second_market, second_year, model in ((Market.DOMESTIC, 2026, False),
                (Market.DOMESTIC, 2027, False), (Market.US, 2026, False), (Market.DOMESTIC, 2026, True)):
            with self.subTest(market=second_market, year=second_year, model=model):
                calendar_for.cache_clear()
                _market_calendar.cache_clear()
                entered = Event()
                second_started = Event()
                second_entered = Event()
                release = Event()
                counter_lock = Lock()
                state = {"calls": 0, "active": 0, "peak": 0}

                def get_calendar(name, **kwargs):
                    with counter_lock:
                        state["calls"] += 1
                        index = state["calls"]
                        state["active"] += 1
                        state["peak"] = max(state["peak"], state["active"])
                    try:
                        if index == 1:
                            entered.set()
                            if not release.wait(5):
                                raise TimeoutError("test did not release calendar constructor")
                        else:
                            second_entered.set()
                        return (name, kwargs["start"], kwargs["end"])
                    finally:
                        with counter_lock:
                            state["active"] -= 1

                def second_request():
                    second_started.set()
                    if model:
                        return _market_calendar(second_market.value, second_year)
                    return calendar_for(second_market, second_year)

                with patch.dict("sys.modules", {"exchange_calendars": SimpleNamespace(get_calendar=get_calendar)}):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        first = pool.submit(calendar_for, Market.DOMESTIC, 2026)
                        try:
                            self.assertTrue(entered.wait(5))
                            second = pool.submit(second_request)
                            self.assertTrue(second_started.wait(5))
                            self.assertFalse(second_entered.wait(.1), "Concurrent cold constructors touched shared holiday state")
                        finally:
                            release.set()
                        self.assertEqual(first.result(timeout=5), ("XKRX", "2025-12-01", "2027-01-31"))
                        expected_name = "XKRX" if second_market is Market.DOMESTIC else "XNYS"
                        expected_start = f"{second_year-1}-01-01" if model else f"{second_year-1}-12-01"
                        expected_end = f"{second_year+1}-12-31" if model else f"{second_year+1}-01-31"
                        self.assertEqual(second.result(timeout=5), (expected_name, expected_start, expected_end))
                    self.assertEqual(state["peak"], 1)

    def test_warm_cache_does_not_acquire_initialization_lock(self):
        result = object()
        provider = Mock(return_value=result)
        with patch.dict("sys.modules", {"exchange_calendars": SimpleNamespace(get_calendar=provider)}):
            self.assertIs(calendar_for(Market.DOMESTIC, 2026), result)
            with patch("dockdack.market_schedule._calendar_init_lock") as initialization:
                initialization.__enter__.side_effect = AssertionError("cache hit must not wait on initialization")
                self.assertIs(calendar_for(Market.DOMESTIC, 2026), result)
            provider.assert_called_once_with("XKRX", start="2025-12-01", end="2027-01-31")

    def test_model_warm_cache_bypasses_shared_lock_and_preserves_its_date_range(self):
        result = object()
        provider = Mock(return_value=result)
        with patch.dict("sys.modules", {"exchange_calendars": SimpleNamespace(get_calendar=provider)}):
            self.assertIs(_market_calendar("domestic", 2026), result)
            with patch("dockdack.market_schedule._calendar_init_lock") as initialization:
                initialization.__enter__.side_effect = AssertionError("model cache hit must not wait on initialization")
                self.assertIs(_market_calendar("domestic", 2026), result)
            provider.assert_called_once_with("XKRX", start="2025-01-01", end="2027-12-31")

    def test_failed_initialization_releases_lock_and_does_not_cache_failure(self):
        result = object()
        provider = Mock(side_effect=[ValueError("invalid calendar"), result])
        with patch.dict("sys.modules", {"exchange_calendars": SimpleNamespace(get_calendar=provider)}):
            with self.assertRaisesRegex(ValueError, "invalid calendar"):
                calendar_for(Market.US, 2026)
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertIs(pool.submit(calendar_for, Market.US, 2026).result(timeout=5), result)
            self.assertIs(calendar_for(Market.US, 2026), result)
            self.assertEqual(provider.call_count, 2)


if __name__ == "__main__":
    unittest.main()
