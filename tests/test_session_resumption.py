"""Offline proofs that closed-session monitoring is not an automatic OFF.

All broker operations use FakeTradingService. Real local exchange calendars,
temporary SQLite/JSON files and an injected clock exercise the complete
snapshot -> 10% demo signal -> ingestion -> order permission flow.
"""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from dockdack.autotrade import AutoTrader
from dockdack.market_schedule import session_on
from dockdack.models import Market, OrderSide
from dockdack.signal_bridge import ExternalPolicy, SignalFileReader, atomic_json, export_charts
from dockdack.test_strategy import RandomDemoSignals
from dockdack.watchlist import MarketSnapshot, WatchItem, WatchStore
from test_autotrade import position
from test_random_strategy import MarketService


DAY = date(2026, 9, 14)


class SessionResumptionTests(unittest.TestCase):
    def configure(self, market=Market.DOMESTIC, *, price="100", holding=False, draw=0.05):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.store = WatchStore(self.folder / "watch.sqlite3")
        self.service = MarketService()
        self.market = market
        self.item = WatchItem(self.service.resolve("005930" if market is Market.DOMESTIC else "AAPL"))
        self.store.save_item(self.item)
        self.session = session_on(market, DAY)
        self.now = self.session.opened - timedelta(seconds=1)
        self.service.prices = [Decimal(price)]
        if holding:
            self.service.positions = (replace(position(), market=market, symbol=self.item.instrument.symbol,
                                              exchange=self.item.instrument.exchange,
                                              currency=self.item.instrument.currency),)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: self.now)
        self.policy = ExternalPolicy("random-demo", 1, Decimal("1000"), Decimal("1000"), allow_market=True)
        self.engine.external_only = True
        self.engine.external_policy = self.policy
        self.inbox = self.folder / "signals.json"
        self.engine.external_reader = SignalFileReader(self.store, self.inbox, self.policy, lambda: self.now)
        self.draws = []

        def controlled_draw():
            self.draws.append(self.now)
            return draw

        self.producer = RandomDemoSignals(self.service, self.store, self.policy, self.inbox,
                                          clock=lambda: self.now, us_order_type="limit", draw=controlled_draw)
        self.payloads = []

    def publish(self, item, snapshot):
        chart = export_charts(self.store, self.folder / "chart.json", now=self.now, watch_ids={item.id})
        self.payloads.append(self.producer.publish(chart))

    def poll(self):
        return self.engine.poll(on_snapshot=self.publish)

    def assert_armed_monitor_only(self, results):
        self.assertTrue(self.engine.orders_enabled)
        self.assertFalse(self.engine._stop.is_set())
        if self.session.opened <= self.now < self.session.closed:
            self.assertIsInstance(results[self.item.id], MarketSnapshot)
            self.assertEqual(self.payloads[-1]["signals"][0]["action"], "hold")
        else:
            self.assertEqual(results, {})
            self.assertEqual(self.payloads, [])
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_closed_monitoring_stays_armed_and_buys_at_open_without_second_on(self):
        for market in Market:
            with self.subTest(market=market):
                self.configure(market)
                self.engine.enable_orders("DEMO_AUTOTRADE")
                self.assert_armed_monitor_only(self.poll())
                self.assert_armed_monitor_only(self.poll())
                self.assertEqual(self.service.quote_calls, 0)
                self.assertEqual(self.service.history_calls, 0)
                self.assertEqual(self.draws, [])  # Closed markets do not draw entries.
                self.now = self.session.opened
                self.poll()  # No second enable_orders call.
                self.assertTrue(self.engine.orders_enabled)
                self.assertEqual(len(self.service.submitted), 1)
                sent = self.service.submitted[0]
                self.assertEqual((sent.market, sent.side, sent.quantity), (market, OrderSide.BUY, 1))
                self.assertEqual(sent.order_type, "3" if market is Market.DOMESTIC else "00")
                self.assertEqual(len(self.draws), 1)
                self.assertEqual(self.store.attempts()[0]["status"], "accepted")

    def test_opening_keeps_current_ten_percent_entry_boundary(self):
        for market in Market:
            with self.subTest(market=market):
                self.configure(market, draw=0.1)
                self.engine.enable_orders("DEMO_AUTOTRADE")
                self.assert_armed_monitor_only(self.poll())
                self.now = self.session.opened
                self.assert_armed_monitor_only(self.poll())
                self.assertEqual(len(self.draws), 1)  # 0.10 is HOLD, not BUY.

    def test_domestic_preopen_order_acceptance_window_is_not_execution_permission(self):
        self.configure()
        self.engine.enable_orders("DEMO_AUTOTRADE")
        for hour, minute, second in ((8, 30, 0), (8, 45, 0), (8, 59, 59)):
            self.now = self.session.opened.replace(hour=hour, minute=minute, second=second)
            self.assert_armed_monitor_only(self.poll())
        self.assertEqual(self.draws, [])
        self.now = self.session.opened
        self.poll()
        self.assertEqual(len(self.service.submitted), 1)

    def test_existing_position_profit_and_loss_exits_resume_at_open(self):
        for market in Market:
            for price, condition, threshold in (("101", "cost_profit_pct", "1"),
                                                ("99.2", "cost_loss_pct", "0.8")):
                with self.subTest(market=market, price=price):
                    self.configure(market, price=price, holding=True)
                    self.engine.enable_orders("DEMO_AUTOTRADE")
                    self.assert_armed_monitor_only(self.poll())
                    self.now = self.session.opened
                    self.poll()
                    self.assertTrue(self.engine.orders_enabled)
                    self.assertEqual(len(self.service.submitted), 1)
                    sent = self.service.submitted[0]
                    self.assertEqual((sent.market, sent.side, sent.quantity), (market, OrderSide.SELL, 1))
                    self.assertEqual(sent.order_type, "3" if market is Market.DOMESTIC else "00")
                    self.assertEqual(self.payloads[-1]["signals"][0][condition], threshold)
                    self.assertEqual(self.draws, [])  # A holding is never an entry draw.

    def test_manual_off_and_stop_stay_off_across_opening_and_monitoring_resume(self):
        for market in Market:
            for action in ("disarm", "stop"):
                with self.subTest(market=market, action=action):
                    self.configure(market)
                    self.engine.enable_orders("DEMO_AUTOTRADE")
                    self.assert_armed_monitor_only(self.poll())
                    getattr(self.engine, action)()
                    self.now = self.session.opened
                    self.poll()
                    self.assertFalse(self.engine.orders_enabled)
                    self.assertEqual(self.service.submitted, [])
                    self.assertEqual(self.store.attempts(), ())
                    # Restarting read-only monitoring is deliberately not rearming.
                    self.engine.resume_monitoring()
                    self.now += timedelta(seconds=1)
                    results = self.poll()
                    self.assertIsInstance(results[self.item.id], MarketSnapshot)
                    self.assertFalse(self.engine.orders_enabled)
                    self.assertEqual(self.service.submitted, [])
                    self.assertEqual(self.store.attempts(), ())

    def test_exact_close_blocks_and_next_trading_day_open_resumes(self):
        for market in Market:
            with self.subTest(market=market):
                self.configure(market)
                self.now = self.session.closed
                self.engine.enable_orders("DEMO_AUTOTRADE")
                self.assert_armed_monitor_only(self.poll())
                self.now += timedelta(hours=1)
                self.assert_armed_monitor_only(self.poll())
                self.now = session_on(market, DAY + timedelta(days=1)).opened
                self.poll()
                self.assertTrue(self.engine.orders_enabled)
                self.assertEqual(len(self.service.submitted), 1)

    def test_close_reached_during_account_lookup_blocks_valid_generated_signal(self):
        for market in Market:
            with self.subTest(market=market):
                self.configure(market)
                self.now = self.session.closed - timedelta(microseconds=1)
                self.engine.enable_orders("DEMO_AUTOTRADE")
                self.service.on_account = lambda: setattr(self, "now", self.session.closed)
                results = self.poll()
                self.assertIsInstance(results[self.item.id], MarketSnapshot)
                self.assertTrue(self.engine.orders_enabled)
                self.assertEqual(self.payloads[-1]["signals"][0]["action"], "buy")
                self.assertEqual(self.service.submitted, [])  # Signal is not permission to send after close.
                self.assertEqual(self.store.attempts(), ())

    def test_both_markets_closed_then_open_independently_without_rearming(self):
        self.configure()
        us = WatchItem(self.service.resolve("AAPL"))
        self.store.save_item(us)
        self.now = datetime(2026, 9, 13, 20, tzinfo=timezone.utc)
        self.engine.enable_orders("DEMO_AUTOTRADE")
        results = self.poll()
        self.assertEqual(results, {})
        self.assertTrue(self.engine.orders_enabled)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])
        self.now = self.session.opened
        self.poll()
        self.assertEqual([order.market for order in self.service.submitted], [Market.DOMESTIC])
        self.now = session_on(Market.US, DAY).opened
        self.poll()
        self.assertTrue(self.engine.orders_enabled)
        self.assertEqual([order.market for order in self.service.submitted], [Market.DOMESTIC, Market.US])

    def test_closed_history_endpoint_is_not_called_and_open_needs_no_second_on(self):
        for market in Market:
            with self.subTest(market=market):
                self.configure(market)
                self.engine.enable_orders("DEMO_AUTOTRADE")
                self.service.fail_history = True
                results = self.poll()
                self.assertEqual(results, {})
                self.assertEqual(self.service.history_calls, 0)
                self.assertTrue(self.engine.orders_enabled)
                self.assertEqual(self.service.submitted, [])
                self.service.fail_history = False
                self.now = self.session.opened
                self.poll()
                self.assertTrue(self.engine.orders_enabled)
                self.assertEqual(len(self.service.submitted), 1)

    def test_producer_account_safety_fault_disarms_and_is_not_rearmed_by_open_market(self):
        self.configure(holding=True)
        valid_positions = self.service.positions
        self.service.positions = (replace(valid_positions[0], average_price=Decimal("NaN")),)
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.assert_armed_monitor_only(self.poll())  # Closed producer does not inspect account.
        self.now = self.session.opened
        result = self.poll()
        self.assertIsInstance(result[self.item.id], Exception)
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        self.service.positions = ()
        self.now += timedelta(seconds=1)
        result = self.poll()
        self.assertIsInstance(result[self.item.id], MarketSnapshot)
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_bad_external_file_while_closed_does_not_silently_rearm_when_file_recovers(self):
        self.configure()
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.assert_armed_monitor_only(self.poll())
        atomic_json(self.inbox, {"schema_version": 999, "source_id": "random-demo", "signals": []})
        self.now += timedelta(microseconds=1)
        self.poll()
        self.assertGreater(self.engine.external_error_count, 0)
        self.assertFalse(self.engine.orders_enabled)
        self.now = self.session.opened
        result = self.poll()
        self.assertIsInstance(result[self.item.id], MarketSnapshot)
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())


if __name__ == "__main__":
    unittest.main()
