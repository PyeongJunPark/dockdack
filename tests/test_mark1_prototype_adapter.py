"""Read-only prototype signals: no torch, Qt, broker, or network required."""
import copy
from decimal import Decimal
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from dockdack.exceptions import OrderNotSent
from dockdack.lstm30_adapter import atomic_json
from dockdack.mark1_adapter import decide_position
from dockdack.mark1_prototype_adapter import (
    PrototypeAutoTrader, PrototypeReadOnlyService, PrototypeSignalProducer, SOURCE_ID,
)
from dockdack.models import TradingMode
from dockdack.watchlist import WatchItem, WatchStore
from dockdack.gui_service import Instrument
from dockdack.models import Market
from test_autotrade import FakeTradingService, NOW, position as held_position
from test_lstm30_adapter import position
from test_mark1_adapter import mark1_chart, prediction


def predictor(probability=.7, market="domestic"):
    architecture = "cat_joint6" if market == "domestic" else "cat_binary8"
    return SimpleNamespace(
        metadata={"market": market, "model_name": architecture, "architecture": architecture,
                  "research_only": True, "deployment_allowed": False, "buy_threshold": .5,
                  "bundle_manifest_sha256": "a" * 64},
        buy_threshold=.5,
        predict=Mock(return_value={**prediction(probability), "probability_stop": .2,
                                   "policy_threshold": .5, "stop_probability_cap": 1.}),
    )


def producer(probability=.7, provider=position, state_path=None, market="domestic"):
    model = predictor(probability, market)
    from test_lstm30_adapter import NOW as CHART_NOW
    return PrototypeSignalProducer({market: model}, position_provider=provider, quantity=1,
                                   max_krw="10000", max_usd="10000", state_path=state_path,
                                   clock=lambda: CHART_NOW), model


class PrototypeAdapterTests(unittest.TestCase):
    def test_strict_boundary_and_research_flags(self):
        for probability, action in ((.5, "hold"), (.500001, "buy"), (.1, "hold")):
            with self.subTest(probability=probability):
                adapter, model = producer(probability)
                payload, diagnostics = adapter(mark1_chart())
                self.assertEqual(payload["source_id"], SOURCE_ID)
                self.assertEqual(payload["signals"][0]["action"], action)
                self.assertEqual(diagnostics[0]["input_features"], 184)
                self.assertIsNone(diagnostics[0]["input_tokens"])
                self.assertTrue(diagnostics[0]["research_only"])
                self.assertFalse(diagnostics[0]["deployment_allowed"])
                self.assertFalse(diagnostics[0]["orders_permitted"])
                self.assertEqual(diagnostics[0]["prediction"]["probability_success"], probability)
                self.assertEqual(model.predict.call_args.kwargs["current_price"], Decimal("100"))

    def test_exact_sell_boundaries_remain_research_signals(self):
        for price, action in (("101", "sell"), ("99.1", "sell"), ("99.2", "hold")):
            with self.subTest(price=price):
                charts = mark1_chart()
                charts["stocks"][0]["price"] = price
                adapter, model = producer(provider=lambda stock: position(stock, quantity="2", sellable="1", average="100"))
                payload, diagnostics = adapter(charts)
                self.assertEqual(payload["signals"][0]["action"], action)
                if price == "99.1":
                    self.assertEqual(payload["signals"][0]["cost_loss_pct"], "0.9")
                model.predict.assert_not_called()
                self.assertFalse(diagnostics[0]["orders_permitted"])

    def test_incomplete_or_gapped_data_holds_without_inference(self):
        for change in ("short", "gap"):
            charts = mark1_chart()
            charts["stocks"][0]["bars"].pop(-1 if change == "short" else -10)
            charts["stocks"][0]["available_days"] -= 1
            adapter, model = producer()
            payload, _ = adapter(charts)
            self.assertEqual(payload["signals"][0]["action"], "hold")
            model.predict.assert_not_called()

    def test_other_strategy_state_is_rejected_and_own_replay_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.json"
            adapter, _ = producer(state_path=path)
            first = adapter(mark1_chart())
            second, model = producer(state_path=path)
            self.assertEqual(first, second(mark1_chart()))
            model.predict.assert_not_called()
            state = copy.deepcopy(adapter.state)
            next(iter(state.values()))["payload"]["source_id"] = "mark1-daily-barrier"
            atomic_json(path, state)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                producer(state_path=path)
            self.assertEqual(path.read_bytes(), before)

    def test_unsupported_future_policy_holds_without_remapping_probability(self):
        adapter, model = producer()
        model.predict.return_value["policy_threshold"] = .65
        payload, detail = adapter(mark1_chart())
        self.assertEqual(payload["signals"][0]["action"], "hold")
        self.assertEqual(detail[0]["reason"], "PREDICTION_UNAVAILABLE")
        self.assertEqual(model.predict.return_value["probability_success"], .7)

    def test_real_chart_mode_cannot_be_opted_into(self):
        with self.assertRaises(ValueError):
            PrototypeSignalProducer({"domestic": predictor()}, position_provider=position,
                                    quantity=1, max_krw="1000", max_usd="1000", trading_mode="real")

    def test_new_price_export_is_not_cached_by_completed_bars_only(self):
        adapter, model = producer()
        charts = mark1_chart()
        adapter(charts)
        changed = copy.deepcopy(charts)
        changed["export_id"] = "export-new-price"
        changed["stocks"][0]["price"] = "99.5"
        adapter(changed)
        self.assertEqual(model.predict.call_count, 2)
        self.assertEqual(model.predict.call_args.kwargs["current_price"], Decimal("99.5"))


class PrototypeNoOrderTests(unittest.TestCase):
    def test_readonly_service_and_broker_block_all_writes_and_environment_change(self):
        broker = SimpleNamespace(mode=TradingMode.DEMO, account_domestic=Mock(return_value="balance"), place_order=Mock())
        raw = SimpleNamespace(mode=TradingMode.DEMO, quote=Mock(return_value="quote"), broker=Mock(return_value=broker), submit=Mock())
        service = PrototypeReadOnlyService(raw)
        self.assertEqual(service.quote(None), "quote")
        safe_broker = service.broker(Market.DOMESTIC)
        self.assertEqual(safe_broker.account_domestic(), "balance")
        for method in (service.prepare, service.submit, service.ensure_order_permission,
                       safe_broker.place_order, safe_broker.cancel_order, safe_broker.buy):
            with self.assertRaises(OrderNotSent):
                method(None)
        raw.submit.assert_not_called()
        broker.place_order.assert_not_called()
        cached_read = service.quote
        raw.mode = TradingMode.REAL
        with self.assertRaises(ValueError):
            cached_read(None)
        broker.mode = TradingMode.REAL
        with self.assertRaises(ValueError):
            safe_broker.account_domestic()
        for name in ("factory", "_http_for", "_domestic_http"):
            with self.assertRaises(AttributeError):
                getattr(service, name)

    def test_backend_cannot_be_armed_or_send_even_with_parent_event_forced(self):
        with tempfile.TemporaryDirectory() as folder:
            store = WatchStore(Path(folder) / "watch.sqlite3")
            item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), days=31)
            store.save_item(item)
            raw = FakeTradingService()
            engine = PrototypeAutoTrader(raw, store, items=[item], clock=lambda: NOW)
            with self.assertRaises(ValueError):
                engine.enable_orders("DEMO_AUTOTRADE")
            engine._armed.set()
            self.assertFalse(engine.orders_enabled)
            for method in (engine._preflight, engine._execute, engine._execute_once, engine._before_order_send):
                with self.assertRaises(OrderNotSent):
                    method(item, None, None)
            with self.assertRaises(OrderNotSent):
                engine._ensure_environment(orders=True)
            engine.resume_monitoring()
            engine.poll()
            self.assertEqual(raw.submitted, [])
            self.assertEqual(store.attempts(), ())

    def test_holdings_targets_ignore_shared_point_eight_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            store = WatchStore(Path(folder) / "watch.sqlite3")
            engine = PrototypeAutoTrader(FakeTradingService(), store, items=[], clock=lambda: NOW)
            targets = engine.holding_exit_targets(held_position())
            self.assertEqual(targets["take_profit_price"], Decimal("101"))
            self.assertEqual(targets["stop_loss_price"], Decimal("99.1"))
            self.assertFalse(targets["orders_permitted"])
            self.assertFalse(engine.enable_holdings_exits)
            self.assertEqual(engine.us_retry_attempts, 1)


if __name__ == "__main__":
    unittest.main()
