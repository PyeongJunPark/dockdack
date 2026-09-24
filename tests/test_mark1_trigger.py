"""Normal mark1 trigger tests: no network, no GUI start and no native model load."""
import copy
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dockdack.autotrade import AutoTrader
from dockdack.gui_service import Instrument
from dockdack.history import DailyBar, DailyHistory
from dockdack.lstm30_adapter import read_json
from dockdack.mark1_trigger import Mark1DemoSignalProducer, Mark1TriggerBridge, SOURCE_ID
from dockdack.models import AccountSnapshot, Market, OrderSide, Quote, TradingMode
from dockdack.signal_bridge import ExternalPolicy, export_charts, ingest_signals
from dockdack.watchlist import MarketSnapshot, TriggerKind, TriggerRule, WatchItem, WatchStore
from test_lstm30_adapter import NOW, position
from test_mark1_adapter import mark1_chart, prediction


def predictor(market="domestic", probability=.7):
    return SimpleNamespace(metadata={"market": market, "model_name": "fixture", "research_only": True,
                                     "deployment_allowed": False},
                           predict=Mock(return_value=prediction(probability)))


def snapshot(charts):
    stock = charts["stocks"][0]
    market = Market(stock["market"])
    bars = tuple(DailyBar(date.fromisoformat(row["date"]),
                         *(Decimal(row[key]) for key in ("open", "high", "low", "close", "volume")))
                 for row in stock["bars"])
    return MarketSnapshot(Quote(market, stock["symbol"], "fixture", stock["exchange"],
                                Decimal(stock["price"]), stock["currency"]),
                          DailyHistory(market, stock["symbol"], stock["exchange"], stock["currency"],
                                       len(bars), bars), NOW)


class ProducerTests(unittest.TestCase):
    def producer(self, probability=.7, provider=position, **kwargs):
        model = predictor(probability=probability)
        result = Mark1DemoSignalProducer({"domestic": model}, position_provider=provider,
                                        quantity=1, max_krw="10000", max_usd="1000", clock=lambda: NOW,
                                        **kwargs)
        return result, model

    def test_strict_threshold_brackets_and_risk_flags(self):
        for value, expected in ((.5, "hold"), (.50001, "buy")):
            with self.subTest(value=value):
                producer, model = self.producer(value)
                payload, diagnostics = producer(mark1_chart())
                signal = payload["signals"][0]
                self.assertEqual(signal["action"], expected)
                self.assertEqual(payload["source_id"], SOURCE_ID)
                self.assertEqual(payload["trading_mode"], "demo")
                self.assertTrue(signal["signal_id"].startswith("mark1-prototype:"))
                self.assertTrue(diagnostics[0]["research_only"])
                self.assertFalse(diagnostics[0]["deployment_allowed"])
                self.assertEqual(model.predict.call_args.kwargs["current_price"], Decimal(100))
                if expected == "buy":
                    self.assertEqual(signal["take_profit_price"], "101")
                    self.assertEqual(signal["stop_loss_price"], "99.1")

    def test_changed_current_price_is_not_cached(self):
        producer, model = self.producer()
        first = mark1_chart()
        producer(first)
        second = copy.deepcopy(first)
        second["export_id"] = "next-export"
        second["stocks"][0]["price"] = "100.5"
        producer(second)
        self.assertEqual(model.predict.call_count, 2)
        self.assertEqual(model.predict.call_args.kwargs["current_price"], Decimal("100.5"))

    def test_same_export_replay_is_immutable_with_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "state.json"
            producer, model = self.producer(state_path=state)
            first = producer(mark1_chart())
            second, other_model = self.producer(state_path=state)
            self.assertEqual(second(mark1_chart()), first)
            other_model.predict.assert_not_called()

    def test_position_sells_are_owned_only_by_holdings_exit_pass(self):
        for value in ("101", "99.1"):
            producer, model = self.producer(provider=lambda row: position(row, quantity="2", sellable="2", average="100"))
            chart = mark1_chart()
            chart["stocks"][0]["price"] = value
            payload, diagnostics = producer(chart)
            self.assertEqual(payload["signals"][0]["action"], "hold")
            self.assertNotIn("quantity", payload["signals"][0])
            self.assertEqual(diagnostics[0]["reason"], "POSITION_EXIT_MANAGED_BY_GUI")
            model.predict.assert_not_called()

    def test_stale_quote_or_nonconsecutive_bars_hold(self):
        for invalid in ("quote", "history"):
            producer, model = self.producer()
            chart = mark1_chart(count=31)
            if invalid == "quote":
                chart["stocks"][0]["quote_fetched_at"] = (NOW - timedelta(seconds=16)).isoformat()
            else:
                chart["stocks"][0]["bars"].pop(-10)
                chart["stocks"][0]["available_days"] = 30
            self.assertEqual(producer(chart)[0]["signals"][0]["action"], "hold")
            model.predict.assert_not_called()

    def test_real_producer_rejected(self):
        with self.assertRaisesRegex(ValueError, "모의"):
            self.producer(trading_mode="real")


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = WatchStore(self.root / "watchlist.sqlite3")
        self.service = SimpleNamespace(mode=TradingMode.DEMO, safety_account=Mock(
            side_effect=lambda inst: AccountSnapshot(inst.market, inst.currency, ())),
            submit=Mock(side_effect=AssertionError("No orders")))
        self.engine = SimpleNamespace(clock=lambda: NOW, _stop=Event(),
                                      _validate_snapshot=AutoTrader._validate_snapshot,
                                      external_policy=ExternalPolicy(SOURCE_ID, 3, Decimal(10000), Decimal(1000)),
                                      external_reader=SimpleNamespace(path=self.root / "signals.json"),
                                      enable_orders=Mock(side_effect=AssertionError("No arm")))
        self.window = SimpleNamespace(service=self.service, store=self.store, engine=self.engine)
        self.model = predictor()
        self.bridge = Mark1TriggerBridge(self.window, predictors={"domestic": self.model})
        self.addCleanup(patch.stopall)
        patch("requests.sessions.Session.request", side_effect=AssertionError("Network disabled")).start()

    def payload(self):
        return read_json(self.engine.external_reader.path)

    def test_constructor_is_lazy_and_publish_is_only_data(self):
        self.service.safety_account.assert_not_called()
        self.model.predict.assert_not_called()
        self.assertFalse(self.engine.external_reader.path.exists())
        self.bridge.publish(mark1_chart())
        self.assertEqual(self.payload()["signals"][0]["action"], "buy")
        self.service.submit.assert_not_called()
        self.engine.enable_orders.assert_not_called()

    def test_actual_normal_wire_ingests_without_starting_or_ordering(self):
        item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), days=30)
        self.store.save_item(item)
        self.store.save_snapshot(item, snapshot(mark1_chart()))
        chart = export_charts(self.store, self.root / "chart.json", now=NOW)
        self.bridge.publish(chart)
        result = ingest_signals(self.store, self.payload(), self.engine.external_policy, now=NOW)
        self.assertEqual(result["queued"], 1)
        self.assertEqual(self.store.rules()[0].side, OrderSide.BUY)
        self.assertEqual(self.store.attempts(), ())
        self.service.submit.assert_not_called()

    def test_market_failure_does_not_disable_other_market(self):
        chart = mark1_chart()
        chart["stocks"].extend(mark1_chart(market="us")["stocks"])
        self.bridge.publish(chart)
        self.assertEqual([row["action"] for row in self.payload()["signals"]], ["buy", "hold"])
        self.assertIn("us", self.bridge._load_error)

    def test_lazy_native_loads_only_requested_market_once(self):
        bridge = Mark1TriggerBridge(self.window)
        module = SimpleNamespace(PrototypePredictor=Mock(return_value=self.model))
        with patch.dict("sys.modules", {"dockdack.mark1_prototype_inference": module}):
            bridge.publish(mark1_chart())
            chart = mark1_chart()
            chart["export_id"] = "second"
            bridge.publish(chart)
        module.PrototypePredictor.assert_called_once_with(bridge.bundle_root, "domestic")

    def test_load_failure_clears_previous_action_and_is_not_retried_each_stock(self):
        bridge = Mark1TriggerBridge(self.window)
        module = SimpleNamespace(PrototypePredictor=Mock(side_effect=ValueError("missing artifact")))
        with patch.dict("sys.modules", {"dockdack.mark1_prototype_inference": module}):
            bridge.publish(mark1_chart())
            chart = mark1_chart()
            chart["export_id"] = "second"
            bridge.publish(chart)
        self.assertEqual(self.payload()["signals"][0]["action"], "hold")
        self.assertEqual(module.PrototypePredictor.call_count, 1)
        self.assertIn("missing artifact", bridge._load_error)

    def test_invalid_chart_or_real_mode_never_reads_account_or_predicts(self):
        for invalid in ("source", "real"):
            chart = mark1_chart()
            if invalid == "source":
                chart["source"] = "arbitrary"
            else:
                self.service.mode = TradingMode.REAL
            self.bridge.publish(chart)
            self.assertEqual(self.payload()["signals"][0]["action"], "hold")
            self.service.safety_account.assert_not_called()
            self.model.predict.assert_not_called()

    def test_position_unavailable_is_not_treated_as_flat(self):
        self.service.safety_account.side_effect = ValueError("unavailable")
        self.bridge.publish(mark1_chart())
        self.assertEqual(self.payload()["signals"][0]["action"], "hold")
        self.model.predict.assert_not_called()

    def test_stop_does_not_publish_or_query(self):
        self.engine._stop.set()
        self.bridge.publish(mark1_chart())
        self.assertFalse(self.engine.external_reader.path.exists())
        self.service.safety_account.assert_not_called()

    def execution_args(self):
        fresh = snapshot(mark1_chart())
        item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), days=30)
        rule = TriggerRule.create(item, TriggerKind.EXTERNAL.value, OrderSide.BUY.value, 1, Decimal(1000))
        return item, rule, fresh

    def test_execution_reinfers_quote_and_rounded_limit_without_account_call(self):
        item, rule, fresh = self.execution_args()
        for stage in ("preflight", "final_send"):
            self.bridge.validate_execution(item, rule, fresh, Decimal("100.1"), stage=stage)
        self.assertEqual([row.kwargs["current_price"] for row in self.model.predict.call_args_list],
                         [Decimal(100), Decimal("100.1"), Decimal(100), Decimal("100.1")])
        self.service.safety_account.assert_not_called()

    def test_execution_rejects_threshold_limit_market_and_stale(self):
        item, rule, fresh = self.execution_args()
        self.model.predict.side_effect = lambda bars, current_price: prediction(.5 if current_price > 100 else .7)
        with self.assertRaisesRegex(ValueError, "50%"):
            self.bridge.validate_execution(item, rule, fresh, Decimal("100.1"))
        with self.assertRaisesRegex(ValueError, "지정가"):
            self.bridge.validate_execution(item, rule, fresh, None)
        with self.assertRaisesRegex(ValueError, "오래"):
            self.bridge.validate_execution(item, rule, replace(fresh, fetched_at=NOW-timedelta(seconds=16)), Decimal(100))
        self.service.mode = TradingMode.REAL
        with self.assertRaisesRegex(ValueError, "실전"):
            self.bridge.validate_execution(item, rule, fresh, Decimal(100))

    def test_execution_does_not_take_over_held_sell(self):
        item, rule, fresh = self.execution_args()
        with self.assertRaisesRegex(ValueError, "매수만"):
            self.bridge.validate_execution(item, replace(rule, side=OrderSide.SELL), fresh, Decimal(100))


if __name__ == "__main__":
    unittest.main()
