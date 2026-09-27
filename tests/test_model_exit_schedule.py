"""Time-based model exits apply to one confirmed lot, never the whole account."""
import unittest
from datetime import datetime, timedelta

from dockdack.market_schedule import session_on
from dockdack.models import Market
from dockdack.trading.model_exit_schedule import timed_exit_due


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


if __name__ == "__main__":
    unittest.main()
