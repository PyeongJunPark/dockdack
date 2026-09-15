"""No-network tests for demo loss exits and their final safety checks."""

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from dockdack.autotrade import AutoTrader
from dockdack.models import Market, OrderSide
from dockdack.signal_bridge import ExternalPolicy, export_charts, ingest_signals
from dockdack.test_strategy import RandomDemoSignals
from dockdack.watchlist import WatchItem, WatchStore
from test_autotrade import NOW, position
from test_random_strategy import MarketService


class StopLossStrategyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.store = WatchStore(self.folder / "db.sqlite3")
        self.service = MarketService()
        self.item = WatchItem(self.service.resolve("005930"))
        self.store.save_item(self.item)
        self.now = NOW
        self.engine = AutoTrader(self.service, self.store, clock=lambda: self.now)
        self.policy = ExternalPolicy("random-demo", 1, Decimal(1000), Decimal(1000), allow_market=True)
        self.engine.external_only = True
        self.engine.external_policy = self.policy
        self.service.positions = (position(),)

    def chart(self, price="99.2", item=None):
        self.service.prices = [Decimal(price)]
        item = item or self.item
        self.engine.snapshot(item)
        return export_charts(self.store, self.folder / "chart.json", now=self.now, watch_ids={item.id})

    def producer(self, **kwargs):
        return RandomDemoSignals(self.service, self.store, self.policy, self.folder / "signals.json",
                                 clock=lambda: self.now, draw=lambda: 0, **kwargs)

    def signal(self, price="99.2", **kwargs):
        return self.producer(**kwargs).publish(self.chart(price))

    def test_loss_boundary_is_inclusive_and_has_no_profit_price_floor(self):
        for price, action in (("99.200001", "hold"), ("99.2", "sell"), ("95", "sell"), ("100", "hold")):
            with self.subTest(price=price):
                self.now += timedelta(seconds=1)
                entry = self.signal(price)["signals"][0]
                self.assertEqual(entry["action"], action)
                if action == "sell":
                    self.assertEqual(entry["cost_loss_pct"], "0.8")
                    self.assertEqual(entry["quantity"], 1)
                    self.assertNotIn("min_sell_price", entry)
                    self.assertNotIn("cost_profit_pct", entry)
        self.assertEqual(self.service.submitted, [])

    def test_loss_exit_uses_weighted_actual_cost(self):
        self.service.positions = (position(quantity=1), replace(position(quantity=3), average_price=Decimal(200)))
        # Weighted cost 175, not the last row's cost or the unweighted 150.
        self.assertEqual(self.signal("173.600001")["signals"][0]["action"], "hold")
        self.now += timedelta(seconds=1)
        self.assertEqual(self.signal("173.6")["signals"][0]["action"], "sell")

    def test_unsellable_position_never_sells_or_draws_a_buy(self):
        self.service.positions = (position(sellable=0),)
        producer = self.producer()
        producer.draw = lambda: self.fail("A held position must never draw an additional buy")
        self.assertEqual(producer.publish(self.chart("95"))["signals"][0]["action"], "hold")

    def test_stale_price_does_not_generate_loss_exit(self):
        chart = self.chart()
        self.now += timedelta(seconds=16)
        self.assertEqual(self.producer().publish(chart)["signals"][0]["action"], "hold")

    def test_same_export_keeps_original_loss_decision(self):
        chart = self.chart()
        producer = self.producer()
        payload = producer.publish(chart)
        self.service.positions = ()
        self.assertEqual(producer.publish(chart), payload)

    def test_loss_exit_is_market_one_share_domestic_and_off_never_submits(self):
        ingest_signals(self.store, self.signal(), self.policy, now=self.now)
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        sent = self.service.submitted[0]
        self.assertEqual((sent.side, sent.quantity, sent.order_type, sent.price), (OrderSide.SELL, 1, "3", None))
        self.assertEqual(self.store.attempts()[0]["status"], "accepted")
        self.assertTrue(any("손절 -0.8%" in row["message"] for row in self.store.events(category="signal")))

    def test_recovered_fresh_price_blocks_loss_sell_before_claim(self):
        ingest_signals(self.store, self.signal(), self.policy, now=self.now)
        self.service.prices = [Decimal("99.2"), Decimal("99.200001")]
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())
        self.assertTrue(any("손절 조건이 성립하지 않습니다" in row["message"]
                            for row in self.store.events(category="signal")))

    def test_changed_actual_average_cost_blocks_loss_sell(self):
        ingest_signals(self.store, self.signal(), self.policy, now=self.now)
        self.service.positions = (replace(position(), average_price=Decimal("90")),)
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_missing_average_cost_or_sellable_quantity_blocks_loss_sell(self):
        ingest_signals(self.store, self.signal(), self.policy, now=self.now)
        for held in (replace(position(), average_price=Decimal("NaN")), position(sellable=0)):
            with self.subTest(held=held):
                self.service.positions = (held,)
                self.engine.enable_orders("DEMO_AUTOTRADE")
                self.engine.poll()
                self.assertEqual(self.service.submitted, [])
                self.assertEqual(self.store.attempts(), ())

    def test_us_loss_exit_is_current_price_limit_not_market(self):
        item = WatchItem(self.service.resolve("AAPL"))
        self.store.save_item(item)
        self.store.remove_item(self.item.id)
        self.now = NOW.replace(hour=15)
        self.service.positions = (replace(position(), market=Market.US, symbol="AAPL", exchange="ND", currency="USD"),)
        payload = self.producer(us_order_type="limit").publish(self.chart(item=item))
        ingest_signals(self.store, payload, self.policy, now=self.now)
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        sent = self.service.submitted[0]
        self.assertEqual((sent.market, sent.side, sent.quantity, sent.order_type, sent.price),
                         (Market.US, OrderSide.SELL, 1, "00", Decimal("99.2")))

    def test_invalid_loss_metadata_is_atomic_and_never_creates_rules(self):
        payload = self.signal()
        invalid = [None, 0.8, True, "0", "-0.8", "100", "101", "NaN", "Infinity", "0.000000001"]
        variants = [{"cost_loss_pct": value} for value in invalid]
        variants += [{"action": "buy"}, {"action": "hold"}, {"cost_profit_pct": "1"}, {"min_sell_price": "100"}]
        for changes in variants:
            with self.subTest(changes=changes):
                candidate = json.loads(json.dumps(payload))
                candidate["signals"][0].update(changes)
                with self.assertRaises(ValueError):
                    ingest_signals(self.store, candidate, self.policy, now=self.now)
                self.assertEqual(self.store.rules(), ())
                self.assertEqual(self.store.attempts(), ())

    def test_expired_loss_signal_never_creates_order(self):
        payload = self.signal()
        self.now += timedelta(minutes=3)
        result = ingest_signals(self.store, payload, self.policy, now=self.now)
        self.assertEqual(result["expired"], 1)
        self.assertEqual(self.store.rules(), ())
        self.assertEqual(self.service.submitted, [])


if __name__ == "__main__":
    unittest.main()
