"""Offline progress contracts for the independent holdings scan; no API calls."""

import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dockdack.autotrade import AutoTrader
from dockdack.models import AccountSnapshot, Market
from dockdack.watchlist import WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position


class HoldingsService(FakeTradingService):
    def __init__(self, timeline):
        super().__init__()
        self.timeline = timeline
        self.account_errors = {}
        self.quote_errors = {}

    def safety_account(self, instrument):
        self.timeline.append(("account_call", instrument.market))
        self.on_account()
        if instrument.market in self.account_errors:
            raise self.account_errors[instrument.market]
        positions = tuple(p for p in self.positions if p.market is instrument.market)
        return AccountSnapshot(instrument.market, instrument.currency, positions)

    def quote(self, instrument):
        self.timeline.append(("quote_call", instrument.symbol))
        if instrument.symbol in self.quote_errors:
            raise self.quote_errors[instrument.symbol]
        return super().quote(instrument)


class HoldingsProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = WatchStore(Path(temporary.name) / "offline.sqlite3")
        self.timeline = []
        self.service = HoldingsService(self.timeline)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: NOW)
        self.engine.enable_holdings_exits = True
        self.events = []
        self.sessions = patch("dockdack.autotrade.regular_session", return_value=True)
        self.sessions.start()
        self.addCleanup(self.sessions.stop)

    def record(self, event):
        self.events.append(event)
        if event[0] == "holdings_progress":
            payload = event[1]
            self.timeline.append((payload["phase"], payload["symbol"] or payload["market"]))

    def structured(self, *, market=None):
        return [data for kind, data in self.events
                if kind == "holdings_progress" and (market is None or data["market"] is market)]

    def scan(self):
        self.engine._holdings_pass(set(), progress=self.record)

    def test_account_and_position_progress_precede_blocking_calls(self):
        self.service.positions = (position(), replace(position(), symbol="000660", name="SK하이닉스"))
        self.scan()
        events = self.structured(market=Market.DOMESTIC)
        self.assertEqual([p["phase"] for p in events],
                         ["account", "checking", "checked", "checking", "checked", "market_complete"])
        self.assertEqual([(p["completed"], p["total"]) for p in events],
                         [(0, 0), (0, 2), (1, 2), (1, 2), (2, 2), (2, 2)])
        self.assertLess(self.timeline.index(("account", Market.DOMESTIC)),
                        self.timeline.index(("account_call", Market.DOMESTIC)))
        for symbol in ("005930", "000660"):
            self.assertLess(self.timeline.index(("checking", symbol)), self.timeline.index(("quote_call", symbol)))
            self.assertLess(self.timeline.index(("quote_call", symbol)), self.timeline.index(("checked", symbol)))
        self.assertEqual(events[3]["name"], "SK하이닉스")
        self.assertEqual(self.service.quote_calls, 2)
        self.assertEqual(self.service.history_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_final_counts_combine_markets_but_item_counts_are_market_local(self):
        us = replace(position(), market=Market.US, currency="USD", exchange="ND", symbol="AAPL", name="애플")
        self.service.positions = (position(), us)
        self.scan()
        for market in (Market.DOMESTIC, Market.US):
            checked = next(p for p in self.structured(market=market) if p["phase"] == "checked")
            self.assertEqual((checked["completed"], checked["total"]), (1, 1))
        final = self.structured()[-1]
        self.assertEqual(final, dict(phase="complete", market=None, completed=2, total=2, symbol="", name=""))

    def test_empty_markets_complete_without_quotes(self):
        self.scan()
        self.assertEqual([p["phase"] for p in self.structured()],
                         ["account", "market_complete", "account", "market_complete", "complete"])
        self.assertTrue(all(p["completed"] == p["total"] == 0 for p in self.structured()))
        self.assertEqual(self.service.quote_calls, 0)

    def test_closed_markets_are_reported_without_account_requests(self):
        with patch("dockdack.autotrade.regular_session", return_value=False):
            self.scan()
        self.assertEqual([p["phase"] for p in self.structured()],
                         ["market_closed", "market_closed", "complete"])
        self.assertFalse(any(kind == "account_call" for kind, _ in self.timeline))

    def test_account_failure_is_reported_and_other_market_is_checked(self):
        self.service.account_errors[Market.DOMESTIC] = TimeoutError("account timeout")
        self.scan()
        events = self.structured()
        self.assertEqual([p["phase"] for p in events],
                         ["account", "market_error", "account", "market_complete", "complete"])
        self.assertEqual(events[1]["error"], "account timeout")
        self.assertIs(events[2]["market"], Market.US)

    def test_each_quote_error_and_zero_quantity_skip_advances_once(self):
        self.service.positions = (position(), replace(position(0, 0), symbol="000660"),
                                  replace(position(), symbol="035420"))
        self.service.quote_errors["005930"] = TimeoutError("quote timeout")
        self.scan()
        checked = [p for p in self.structured() if p["phase"] == "checked"]
        self.assertEqual([(p["completed"], p["total"]) for p in checked], [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(checked[0]["error"], "quote timeout")
        self.assertNotIn("error", checked[1])
        self.assertNotIn(("quote_call", "000660"), self.timeline)
        self.assertEqual(self.structured()[-1]["completed"], 3)

    def test_stop_before_scan_does_not_report_complete(self):
        self.engine.stop()
        self.scan()
        self.assertEqual(self.structured(), [])
        self.assertEqual(self.timeline, [])

    def test_stop_during_position_preserves_checked_but_not_false_completion(self):
        self.service.positions = (position(), replace(position(), symbol="000660"))
        original_quote = self.service.quote

        def stopping_quote(instrument):
            quote = original_quote(instrument)
            self.engine.stop()
            return quote

        self.service.quote = stopping_quote
        self.scan()
        events = self.structured()
        self.assertEqual([p["phase"] for p in events], ["account", "checking", "checked"])
        self.assertEqual((events[-1]["completed"], events[-1]["total"]), (1, 2))
        self.assertEqual(self.service.quote_calls, 1)

    def test_stop_during_empty_account_does_not_report_complete(self):
        self.service.on_account = self.engine.stop
        self.scan()
        self.assertEqual([p["phase"] for p in self.structured()], ["account"])

    def test_new_progress_callback_errors_propagate_not_as_broker_failures(self):
        self.service.positions = (position(),)
        for phase in ("account", "checking", "checked", "market_complete", "complete"):
            with self.subTest(phase=phase):
                def failing_progress(event):
                    if event[0] == "holdings_progress" and event[1]["phase"] == phase:
                        raise RuntimeError("UI callback failed")

                with self.assertRaisesRegex(RuntimeError, "UI callback failed"):
                    self.engine._holdings_pass(set(), progress=failing_progress)
        self.assertFalse(any("UI callback failed" in str(event) for event in self.store.events()))

    def test_poll_keeps_existing_phase_and_quote_events_compatible(self):
        self.service.positions = (position(),)
        self.engine.poll(progress=self.record)
        self.assertIn(("phase", "보유종목 매도 조건 점검"), self.events)
        quoted = [payload for kind, payload in self.events if kind == "holding_quote"]
        self.assertEqual(len(quoted), 1)
        self.assertEqual(quoted[0]["quote"].price, Decimal("100"))
        self.assertEqual(self.structured()[-1]["phase"], "complete")
        self.assertEqual(self.service.submitted, [])

    def test_watch_item_announces_quote_wait_before_snapshot(self):
        item = WatchItem(self.service.resolve("005930"), "삼성전자")
        self.store.save_item(item)
        self.engine.enable_holdings_exits = False
        updates = []

        def progress(event):
            updates.append(event)
            if event[0] == "watch_progress":
                self.assertEqual(self.service.quote_calls, 0)
                self.assertEqual(self.service.history_calls, 0)

        self.engine.poll(progress=progress)
        self.assertEqual(updates[0], ("watch_progress", dict(market=Market.DOMESTIC, symbol="005930",
                                                            name="삼성전자", completed=0, total=1)))
        self.assertEqual(updates[1][0], item.id)
        self.assertEqual(updates[1][2:], (1, 1))


if __name__ == "__main__":
    unittest.main()
