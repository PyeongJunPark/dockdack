from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

from dockdack.models import AccountSnapshot, Market, Position
from dockdack.portfolio import PortfolioCache, PortfolioMarketState


NOW = datetime(2026, 9, 15, 6, 0, tzinfo=timezone.utc)


def position(market=Market.DOMESTIC, *, symbol=None, quantity="2", profit="200"):
    return Position(
        market, symbol or ("005930" if market is Market.DOMESTIC else "AAPL"),
        "삼성전자" if market is Market.DOMESTIC else "애플",
        "KRX" if market is Market.DOMESTIC else "ND", "KRW" if market is Market.DOMESTIC else "USD",
        D(quantity), D(quantity), D("100"), D("200"), D(quantity) * 200, D(profit), D("100"),
    )


def account(market=Market.DOMESTIC, *, positions=None):
    return AccountSnapshot(
        market, "KRW" if market is Market.DOMESTIC else "USD",
        (position(market),) if positions is None else tuple(positions),
        cash=D("5000"), available_to_order=D("4500"),
    )


class FakeService:
    def __init__(self):
        self.calls = []
        self.responses = {market: account(market) for market in Market}

    def safety_account(self, instrument):
        self.calls.append(instrument)
        result = self.responses[instrument.market]
        if isinstance(result, Exception):
            raise result
        return result

    def submit(self, request):
        raise AssertionError("A portfolio refresh must never place an order")


class PortfolioCacheTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.now = NOW
        self.cache = PortfolioCache(self.service, clock=lambda: self.now)

    def test_initial_state_unknown_and_first_refresh_fetches_entire_markets(self):
        self.assertTrue(all(state.status(NOW) == "unknown" for state in self.cache.snapshot().values()))
        payload = self.cache.refresh_due()
        self.assertEqual([item.market for item in self.service.calls], [Market.DOMESTIC, Market.US])
        self.assertEqual([item.symbol for item in self.service.calls], ["005930", "AAPL"])
        self.assertTrue(all(state.status(NOW) == "ok" for state in payload.values()))
        self.assertEqual(payload[Market.DOMESTIC].snapshot.cash, D("5000"))
        self.assertEqual(payload[Market.DOMESTIC].snapshot.available_to_order, D("4500"))

    def test_automatic_and_manual_refresh_never_bypass_sixty_seconds(self):
        self.cache.refresh_due()
        self.now += timedelta(seconds=59)
        self.cache.refresh_due()
        self.cache.refresh_due(force=True)
        self.assertEqual(len(self.service.calls), 2)
        self.now += timedelta(seconds=1)
        self.cache.refresh_due()
        self.assertEqual(len(self.service.calls), 4)

    def test_force_may_shorten_configured_interval_but_not_minimum(self):
        self.cache = PortfolioCache(self.service, clock=lambda: self.now, refresh_seconds=300)
        self.cache.refresh_due()
        self.now += timedelta(seconds=60)
        self.cache.refresh_due()
        self.assertEqual(len(self.service.calls), 2)
        self.cache.refresh_due(force=True)
        self.assertEqual(len(self.service.calls), 4)

    def test_market_error_retains_its_success_and_other_market_keeps_refreshing(self):
        previous = self.cache.refresh_due()
        self.now += timedelta(seconds=60)
        self.service.responses[Market.DOMESTIC] = ValueError("잔고 요청 제한")
        self.service.responses[Market.US] = account(Market.US, positions=())
        payload = self.cache.refresh_due()
        domestic = payload[Market.DOMESTIC]
        self.assertEqual(domestic.snapshot, previous[Market.DOMESTIC].snapshot)
        self.assertEqual(domestic.fetched_at, NOW)
        self.assertEqual(domestic.last_attempt, self.now)
        self.assertEqual(domestic.status(self.now), "error")
        self.assertEqual(payload[Market.US].status(self.now), "empty")
        self.assertEqual(payload[Market.US].fetched_at, self.now)
        self.cache.refresh_due(force=True)
        self.assertEqual(len(self.service.calls), 4)
        self.now += timedelta(seconds=60)
        self.service.responses[Market.DOMESTIC] = account()
        self.assertEqual(self.cache.refresh_due()[Market.DOMESTIC].error, "")

    def test_failed_first_query_is_unknown_holdings_not_successful_empty(self):
        self.service.responses[Market.DOMESTIC] = RuntimeError("접속 오류")
        state = self.cache.refresh_due()[Market.DOMESTIC]
        self.assertIsNone(state.snapshot)
        self.assertEqual(state.status(NOW), "error")
        self.assertIsNone(state.fetched_at)

    def test_stop_prevents_requests_and_is_checked_between_markets(self):
        self.cache.refresh_due(stopped=lambda: True)
        self.assertEqual(self.service.calls, [])
        payload = self.cache.refresh_due(stopped=lambda: bool(self.service.calls))
        self.assertEqual(len(self.service.calls), 1)
        self.assertEqual(payload[Market.US].status(NOW), "unknown")

    def test_zero_positions_filtered_and_snapshot_ages_without_network(self):
        self.service.responses[Market.DOMESTIC] = account(positions=(position(quantity="0"),))
        state = self.cache.refresh_due()[Market.DOMESTIC]
        self.assertEqual(state.positions, ())
        self.assertEqual(state.status(NOW + timedelta(seconds=120)), "empty")
        self.assertEqual(state.status(NOW + timedelta(seconds=121)), "stale")
        self.assertEqual(len(self.service.calls), 2)
        payload = self.cache.snapshot()
        payload.clear()
        self.assertEqual(len(self.cache.snapshot()), 2)

    def test_wrong_market_and_nonfinite_values_are_not_shown_as_valid(self):
        self.service.responses[Market.DOMESTIC] = account(Market.US)
        self.service.responses[Market.US] = account(Market.US, positions=(replace(position(Market.US), quantity=D("NaN")),))
        self.assertTrue(all(state.status(NOW) == "error" for state in self.cache.refresh_due().values()))


if __name__ == "__main__":
    unittest.main()
