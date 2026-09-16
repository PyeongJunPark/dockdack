import unittest
from datetime import date, datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from dockdack.market_schedule import Session
from dockdack.market_status import market_status, market_statuses
from dockdack.models import Market


SEOUL = ZoneInfo("Asia/Seoul")
NEW_YORK = ZoneInfo("America/New_York")


class MarketStatusTests(unittest.TestCase):
    def setUp(self):
        self.opened = datetime(2026, 9, 15, 9, tzinfo=SEOUL)
        self.closed = self.opened.replace(hour=15, minute=30)
        self.session = Session(self.opened, self.closed)
        self.network = patch("requests.sessions.Session.request", side_effect=AssertionError("No network permitted"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def test_regular_open_and_close_boundaries(self):
        with patch("dockdack.market_status.session_on", return_value=self.session):
            for now, expected in ((self.opened, "open"),
                                  (self.closed - timedelta(microseconds=1), "open"),
                                  (self.closed, "closed")):
                with self.subTest(now=now):
                    result = market_status(Market.DOMESTIC, now)
                    self.assertEqual(result["state"], expected)
                    self.assertEqual(result["is_open"], expected == "open")
                    self.assertIs(result["market"], Market.DOMESTIC)
                    self.assertEqual(result["checked_at"], now)

    def test_preopen_and_ten_minute_preparation_boundaries(self):
        with patch("dockdack.market_status.session_on", return_value=self.session):
            early = market_status(Market.DOMESTIC, self.opened - timedelta(minutes=10, seconds=1))
            preparing = market_status(Market.DOMESTIC, self.opened - timedelta(minutes=10))
            last_second = market_status(Market.DOMESTIC, self.opened - timedelta(seconds=1))
        self.assertEqual(early["state"], "preopen")
        self.assertEqual(early["text"], "한국 · 장전")
        for result in (preparing, last_second):
            self.assertEqual(result["state"], "preparing")
            self.assertFalse(result["is_open"])
            self.assertIn("주문 개시를 뜻하지 않습니다", result["detail"])

    def test_badge_clarifies_regular_hours_not_extended_hours_tradability(self):
        with patch("dockdack.market_status.session_on", return_value=self.session):
            result = market_status(Market.DOMESTIC, self.opened)
        self.assertEqual(result["text"], "한국 · 장중")
        self.assertIn("정규장 기준", result["detail"])
        self.assertIn("09/15 09:00~15:30 KST", result["detail"])
        self.assertIn("장전·시간외 거래 가능 여부를 뜻하지 않습니다", result["detail"])

    def test_holiday_is_not_presented_as_a_calendar_error(self):
        with patch("dockdack.market_status.session_on", return_value=None):
            result = market_status(Market.DOMESTIC, self.opened)
        self.assertEqual(result["state"], "holiday")
        self.assertEqual(result["text"], "한국 · 휴장")
        self.assertFalse(result["is_open"])

    def test_calendar_error_keeps_status_unknown(self):
        with patch("dockdack.market_status.session_on", side_effect=ValueError("특별 개장시간 확인 필요")):
            result = market_status(Market.DOMESTIC, self.opened)
        self.assertEqual(result["state"], "unknown")
        self.assertIsNone(result["is_open"])
        self.assertIn("특별 개장시간 확인 필요", result["detail"])

    def test_naive_datetime_is_unknown_not_interpreted_as_local_time(self):
        with patch("dockdack.market_status.session_on") as calendar:
            result = market_status(Market.DOMESTIC, self.opened.replace(tzinfo=None))
        calendar.assert_not_called()
        self.assertEqual(result["state"], "unknown")
        self.assertIsNone(result["is_open"])

    def test_invalid_calendar_session_is_unknown(self):
        with patch("dockdack.market_status.session_on", return_value=Session(self.opened, self.opened)):
            result = market_status(Market.DOMESTIC, self.opened)
        self.assertEqual(result["state"], "unknown")
        self.assertIsNone(result["is_open"])

    def test_one_market_failure_does_not_hide_other_status(self):
        us_open = datetime(2026, 9, 15, 9, 30, tzinfo=NEW_YORK)

        def calendar(market, day):
            if market is Market.DOMESTIC:
                raise ValueError("KR calendar unavailable")
            return Session(us_open, us_open.replace(hour=16, minute=0))

        with patch("dockdack.market_status.session_on", side_effect=calendar):
            result = market_statuses(us_open)
        self.assertIsNone(result[Market.DOMESTIC]["is_open"])
        self.assertTrue(result[Market.US]["is_open"])

    def test_us_uses_new_york_date_even_on_next_korean_calendar_day(self):
        us_open = datetime(2026, 9, 15, 9, 30, tzinfo=NEW_YORK)
        us_close = us_open.replace(hour=16, minute=0)
        now = datetime(2026, 9, 16, 1, tzinfo=SEOUL)
        with patch("dockdack.market_status.session_on", return_value=Session(us_open, us_close)) as calendar:
            result = market_status(Market.US, now)
        calendar.assert_called_once_with(Market.US, date(2026, 9, 15))
        self.assertEqual(result["state"], "open")
        self.assertIn("뉴욕 09/15 09:30~16:00 EDT", result["detail"])
        self.assertIn("한국 09/15 22:30~09/16 05:00 KST", result["detail"])

    def test_real_calendars_use_summer_and_winter_us_hours(self):
        summer = market_status(Market.US, datetime(2026, 7, 6, 10, tzinfo=NEW_YORK))
        winter = market_status(Market.US, datetime(2026, 11, 30, 10, tzinfo=NEW_YORK))
        self.assertTrue(summer["is_open"])
        self.assertTrue(winter["is_open"])
        self.assertIn("22:30~07/07 05:00 KST", summer["detail"])
        self.assertIn("23:30~12/01 06:00 KST", winter["detail"])
        self.assertIn("EST", winter["detail"])

    def test_real_us_early_close_is_not_shown_open_until_16(self):
        before = market_status(Market.US, datetime(2026, 11, 27, 12, 59, tzinfo=NEW_YORK))
        closed = market_status(Market.US, datetime(2026, 11, 27, 13, tzinfo=NEW_YORK))
        self.assertTrue(before["is_open"])
        self.assertEqual(closed["state"], "closed")
        self.assertIn("09:30~13:00 EST", closed["detail"])

    def test_real_holiday_and_unverified_special_session(self):
        holiday = market_status(Market.US, datetime(2026, 9, 7, 10, tzinfo=NEW_YORK))
        unknown = market_status(Market.DOMESTIC, datetime(2026, 11, 19, 10, tzinfo=SEOUL))
        self.assertEqual(holiday["state"], "holiday")
        self.assertFalse(holiday["is_open"])
        self.assertEqual(unknown["state"], "unknown")
        self.assertIsNone(unknown["is_open"])


if __name__ == "__main__":
    unittest.main()
