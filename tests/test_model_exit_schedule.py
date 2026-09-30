"""Time-based model exits apply to one confirmed lot, never the whole account."""
import unittest
from datetime import date, datetime, timedelta
from unittest.mock import patch

from dockdack.market_schedule import session_on
from dockdack.models import Market
from dockdack.trading.model_exit_schedule import planned_model_exit, timed_exit_due


class ModelExitScheduleTests(unittest.TestCase):
    def test_daily_model_waits_for_preclose_and_observed_fill(self):
        market = Market.DOMESTIC
        session = session_on(market, datetime(2026, 9, 28).date())
        self.assertIsNotNone(session)
        lot = {"strategy_id": "mark1-4-prototype",
               "buy_fill_observed_at": (session.opened + timedelta(minutes=20)).isoformat()}
        self.assertFalse(timed_exit_due(lot, market, session.opened + timedelta(hours=1)))
        self.assertFalse(timed_exit_due({**lot, "buy_fill_observed_at": None}, market,
                                        session.closed - timedelta(minutes=4)))
        self.assertTrue(timed_exit_due(lot, market, session.closed - timedelta(minutes=4)))
        self.assertFalse(timed_exit_due(lot, market, session.closed))

    def test_unknown_strategy_and_future_fill_never_sell(self):
        market = Market.US
        session = session_on(market, datetime(2026, 9, 25).date())
        self.assertIsNotNone(session)
        now = session.closed - timedelta(minutes=3)
        self.assertFalse(timed_exit_due({"strategy_id": "manual",
                                         "buy_fill_observed_at": session.opened.isoformat()}, market, now))
        self.assertFalse(timed_exit_due({"strategy_id": "mark1-4-prototype",
                                         "buy_fill_observed_at": (now + timedelta(minutes=1)).isoformat()}, market, now))
        with patch("dockdack.trading.model_exit_schedule.session_on",
                   side_effect=AssertionError("Future fills must not probe exit sessions")):
            self.assertFalse(timed_exit_due({"strategy_id": "mark1-4-prototype",
                                             "buy_fill_observed_at": (now + timedelta(minutes=1)).isoformat()},
                                            market, now))

    def test_planned_days_use_each_exchange_session_and_observed_fill(self):
        domestic = Market.DOMESTIC
        fill = session_on(domestic, date(2026, 9, 18)).opened + timedelta(minutes=10)
        daily = {"strategy_id": "mark1-4-prototype", "buy_fill_observed_at": fill.isoformat()}
        h3 = {**daily, "strategy_id": "mark1-11-prototype"}
        h5 = {**daily, "strategy_id": "mark1-12-prototype"}
        self.assertEqual((planned_model_exit(daily, domestic).day,
                          planned_model_exit(daily, domestic).timing), (date(2026, 9, 18), "preclose"))
        self.assertEqual((planned_model_exit(h3, domestic).day,
                          planned_model_exit(h3, domestic).timing), (date(2026, 9, 22), "elapsed"))
        # The exchange calendar skips the September 24–25 Chuseok closure.
        self.assertEqual(planned_model_exit(h5, domestic).day, date(2026, 9, 28))
        before = session_on(domestic, date(2026, 9, 21)).closed - timedelta(minutes=2)
        due = session_on(domestic, date(2026, 9, 22)).opened + timedelta(minutes=1)
        self.assertFalse(timed_exit_due(h3, domestic, before))
        self.assertTrue(timed_exit_due(h3, domestic, due))
        us_fill = session_on(Market.US, date(2026, 9, 4)).opened + timedelta(minutes=10)
        us_h3 = {"strategy_id": "mark1-11-prototype", "buy_fill_observed_at": us_fill.isoformat()}
        # U.S. Labor Day is not counted as a trading session.
        self.assertEqual(planned_model_exit(us_h3, Market.US).day, date(2026, 9, 9))

    def test_missing_or_untrusted_fill_has_no_displayed_plan(self):
        market = Market.US
        valid = session_on(market, date(2026, 9, 18)).opened.isoformat()
        for lot in ({"strategy_id": "unknown", "buy_fill_observed_at": valid},
                    {"strategy_id": "mark1-11-prototype", "buy_fill_observed_at": None},
                    {"strategy_id": "mark1-11-prototype", "buy_fill_observed_at": "not-a-date"},
                    {"strategy_id": "mark1-11-prototype", "buy_fill_observed_at": "2026-09-18T09:30:00"}):
            with self.subTest(lot=lot):
                self.assertIsNone(planned_model_exit(lot, market))

    def test_minute_prototype_exits_own_observed_fill_after_horizon_or_preclose(self):
        market = Market.DOMESTIC
        session = session_on(market, date(2026, 9, 29))
        self.assertIsNotNone(session)
        for model, minutes in ((29, 15), (30, 15), (31, 15),
                               (32, 30), (33, 30), (34, 30),
                               (35, 60), (36, 60), (37, 60)):
            with self.subTest(model=model):
                fill = session.opened + timedelta(minutes=20)
                lot = {"strategy_id": f"mark1-{model}-prototype",
                       "buy_fill_observed_at": fill.isoformat()}
                self.assertFalse(timed_exit_due(lot, market, fill + timedelta(minutes=minutes - 1)))
                self.assertTrue(timed_exit_due(lot, market, fill + timedelta(minutes=minutes)))
                self.assertFalse(timed_exit_due(lot, market, session.closed))
                self.assertFalse(timed_exit_due({**lot, "buy_fill_observed_at": None}, market,
                                                session.closed - timedelta(minutes=4)))
        late_fill = session.closed - timedelta(minutes=10)
        late_lot = {"strategy_id": "mark1-37-prototype",
                    "buy_fill_observed_at": late_fill.isoformat()}
        self.assertTrue(timed_exit_due(late_lot, market, session.closed - timedelta(minutes=4)))
        self.assertTrue(timed_exit_due(late_lot, market,
                                       session_on(market, date(2026, 9, 30)).opened))
        self.assertTrue(timed_exit_due({**late_lot,
                                        "buy_fill_observed_at": session.closed.isoformat()},
                                       market, session_on(market, date(2026, 9, 30)).opened))
        self.assertFalse(timed_exit_due({**late_lot, "strategy_id": "mark1-38-prototype"},
                                        market, session.closed - timedelta(minutes=4)))

    def test_after_close_fill_observation_waits_for_next_session_and_full_horizon(self):
        market = Market.DOMESTIC
        session = session_on(market, date(2026, 9, 29))
        following = session_on(market, date(2026, 9, 30))
        self.assertIsNotNone(session)
        self.assertIsNotNone(following)
        observed = session.closed + timedelta(minutes=1)
        for model in (29, 32, 35):
            with self.subTest(model=model):
                lot = {"strategy_id": f"mark1-{model}-prototype",
                       "buy_fill_observed_at": observed.isoformat()}
                self.assertFalse(timed_exit_due(lot, market, session.closed - timedelta(seconds=1)))
                self.assertFalse(timed_exit_due(lot, market,
                                                observed + timedelta(hours=1)))
                self.assertFalse(timed_exit_due(lot, market,
                                                following.opened - timedelta(seconds=1)))
                self.assertTrue(timed_exit_due(lot, market, following.opened))
                planned = planned_model_exit(lot, market)
                self.assertEqual((planned.day, planned.at),
                                 (following.opened.date(), following.opened))
                self.assertTrue(timed_exit_due(
                    {**lot, "buy_fill_observed_at":
                     (session.opened - timedelta(minutes=1)).isoformat()},
                    market, following.opened))

    def test_preopen_confirmed_fill_observation_waits_its_full_horizon(self):
        market = Market.DOMESTIC
        session = session_on(market, date(2026, 9, 29))
        observed = session.opened - timedelta(minutes=1)
        lot = {"strategy_id": "mark1-29-prototype",
               "buy_fill_observed_at": observed.isoformat()}
        self.assertFalse(timed_exit_due(lot, market, session.opened))
        self.assertFalse(timed_exit_due(lot, market, observed + timedelta(minutes=14)))
        self.assertTrue(timed_exit_due(lot, market, observed + timedelta(minutes=15)))
        self.assertEqual(planned_model_exit(lot, market).at,
                         observed + timedelta(minutes=15))
        holiday_lot = {**lot, "buy_fill_observed_at": "2026-09-27T10:00:00+09:00"}
        following = session_on(market, date(2026, 9, 28))
        self.assertTrue(timed_exit_due(holiday_lot, market, following.opened))


if __name__ == "__main__":
    unittest.main()
