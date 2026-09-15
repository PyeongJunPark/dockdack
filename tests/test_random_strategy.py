import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from dockdack.autotrade import AutoTrader
from dockdack.models import Market, OrderRequest, OrderSide
from dockdack.signal_bridge import ExternalPolicy, export_charts, ingest_signals, SignalFileReader
from dockdack.test_strategy import RandomDemoSignals
from dockdack.watchlist import WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position


class MarketService(FakeTradingService):
    def prepare(self, inst, side, quantity, kind, price):
        if kind == "market":
            if inst.market is Market.US:
                raise ValueError("US demo market unsupported")
            return OrderRequest(inst.market,OrderSide(side),inst.symbol,quantity,inst.exchange,"3",None)
        return super().prepare(inst,side,quantity,kind,price)


class RandomStrategyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.store=WatchStore(self.folder/"db.sqlite3")
        self.service=MarketService()
        self.item=WatchItem(self.service.resolve("005930"))
        self.store.save_item(self.item)
        self.now=NOW
        self.engine=AutoTrader(self.service,self.store,clock=lambda:self.now)
        self.policy=ExternalPolicy("random-demo",1,Decimal(1000),Decimal(1000),allow_market=True)
        self.engine.external_only=True
        self.engine.external_policy=self.policy
        self.path=self.folder/"random.json"

    def chart(self, item=None):
        item=item or self.item
        self.engine.snapshot(item)
        return export_charts(self.store,self.folder/"charts.json",now=self.now,watch_ids={item.id})

    def producer(self, draw=0.05, us="blocked"):
        return RandomDemoSignals(self.service,self.store,self.policy,self.path,clock=lambda:self.now,
                                 us_order_type=us,draw=lambda:draw)

    def test_ten_percent_boundary_and_immutable_draw_for_same_export(self):
        chart=self.chart()
        producer=self.producer(0.09999)
        buy=producer.publish(chart)
        self.assertEqual(buy["signals"][0]["action"],"buy")
        self.assertEqual(buy["signals"][0]["order_type"],"market")
        producer.draw=lambda:0.99
        self.assertEqual(producer.publish(chart),buy)
        self.now+=timedelta(seconds=1)
        hold=self.producer(0.1).publish(self.chart())
        self.assertEqual(hold["signals"][0]["action"],"hold")
        self.assertEqual(self.service.submitted,[])

    def test_one_percent_exit_uses_average_cost_and_does_not_add_to_position(self):
        self.service.positions=(position(),)
        for price,action in (("100.99","hold"),("101","sell")):
            self.now+=timedelta(seconds=1)
            self.service.prices=[Decimal(price)]
            payload=self.producer(0).publish(self.chart())
            self.assertEqual(payload["signals"][0]["action"],action)
            if action=="sell":
                self.assertEqual(Decimal(payload["signals"][0]["min_sell_price"]),Decimal(101))
                self.assertEqual(payload["signals"][0]["cost_profit_pct"], "1")
                self.assertNotIn("cost_loss_pct", payload["signals"][0])

    def test_sell_threshold_is_rechecked_with_latest_order_quote(self):
        self.service.positions=(position(),)
        self.service.prices=[Decimal(101)]
        payload=self.producer().publish(self.chart())
        ingest_signals(self.store,payload,self.policy,now=self.now)
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.service.prices=[Decimal(101),Decimal(100)]
        self.engine.poll()
        self.assertEqual(self.service.submitted,[])
        self.assertEqual(self.store.attempts(),())

    def test_domestic_market_buy_and_sell_follow_signal_with_durable_intent(self):
        for action in ("buy","sell"):
            if action=="sell":
                self.store.mark_reviewed(self.store.rules()[0].id,"CHECKED_ORDER_HISTORY")
                self.service.positions=(position(),)
                self.service.prices=[Decimal(101)]
                self.now+=timedelta(seconds=1)
            payload=self.producer().publish(self.chart())
            ingest_signals(self.store,payload,self.policy,now=self.now)
            self.engine.enable_orders("DEMO_AUTOTRADE")
            self.engine.poll()
            self.assertEqual(self.service.submitted[-1].side,OrderSide(action))
            self.assertEqual(self.service.submitted[-1].order_type,"3")
            self.assertIsNone(self.service.submitted[-1].price)
        self.assertEqual(len(self.service.submitted),2)
        self.assertEqual(self.store.attempts()[0]["price"],"100")

    def test_changed_average_cost_is_rechecked_before_sell(self):
        self.service.positions=(position(),)
        self.service.prices=[Decimal(101)]
        payload=self.producer().publish(self.chart())
        ingest_signals(self.store,payload,self.policy,now=self.now)
        self.service.positions=(replace(position(),average_price=Decimal(110)),)
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(self.service.submitted,[])

    def test_zero_caps_closed_session_and_pending_orders_produce_no_buy(self):
        self.policy=ExternalPolicy("random-demo",1,Decimal(0),Decimal(0),allow_market=True)
        self.assertEqual(self.producer().publish(self.chart())["signals"][0]["action"],"hold")
        self.policy=ExternalPolicy("random-demo",1,Decimal(1000),Decimal(1000),allow_market=True)
        self.now=NOW.replace(hour=9)  # 18:00 Seoul.
        self.assertEqual(self.producer().publish(self.chart())["signals"][0]["action"],"hold")

    def test_us_blocked_by_default_or_explicit_limit_never_market(self):
        us=WatchItem(self.service.resolve("AAPL"))
        self.store.save_item(us)
        self.now=NOW.replace(hour=15)
        blocked=self.producer().publish(self.chart(us))
        self.assertEqual(blocked["signals"][0]["action"],"hold")
        self.now+=timedelta(seconds=1)
        enabled=self.producer(us="limit").publish(self.chart(us))
        self.assertEqual(enabled["signals"][0]["order_type"],"limit")
        ingest_signals(self.store,enabled,self.policy,now=self.now)
        self.engine.enable_orders("DEMO_AUTOTRADE")
        self.engine.poll()
        self.assertEqual(self.service.submitted[0].order_type,"00")

    def test_market_signal_requires_explicit_policy_and_us_market_is_rejected(self):
        payload=self.producer().publish(self.chart())
        policy=replace(self.policy,allow_market=False)
        with self.assertRaises(ValueError):
            ingest_signals(self.store,payload,policy,now=self.now)
        payload["signals"][0].update(market="us",symbol="AAPL",exchange="ND")
        with self.assertRaises(ValueError):
            ingest_signals(self.store,payload,self.policy,now=self.now)

    def test_streaming_callback_publishes_and_executes_without_waiting_full_sweep(self):
        producer=self.producer()
        self.engine.external_reader=SignalFileReader(self.store,self.path,self.policy,lambda:self.now)
        self.engine.enable_orders("DEMO_AUTOTRADE")
        def publish(item,snapshot):
            data=export_charts(self.store,self.folder/"update.json",now=self.now,watch_ids={item.id})
            producer.publish(data)
        self.engine.poll(on_snapshot=publish)
        self.assertEqual(len(self.service.submitted),1)
        self.assertEqual(len(self.store.rules()),1)

    def test_bad_account_or_publisher_failure_disarms_without_order(self):
        self.service.positions=(replace(position(),average_price=Decimal("NaN")),)
        self.engine.enable_orders("DEMO_AUTOTRADE")
        producer=self.producer()
        self.engine.poll(on_snapshot=lambda item,snapshot:producer.publish(export_charts(self.store,self.folder/"update.json",now=self.now)))
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted,[])


if __name__ == "__main__":
    unittest.main()
