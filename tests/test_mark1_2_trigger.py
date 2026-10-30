"""Mark1.2 DEMO neural bridge: fake data only, no broker or GUI start."""
from decimal import Decimal
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from dockdack.mark1_2_inference import SEMANTICS
from dockdack.signals.mark1_2_trigger import (
    Mark12DemoSignalProducer, Mark12PrototypePredictor, Mark12TriggerBridge,
    SOURCE_ID, STRATEGY_ID, TITLE,
)
from dockdack.signals.prototype_external import PrototypeWorker
from test_lstm30_adapter import NOW, position
from test_mark1_adapter import mark1_chart
from test_prototype_external import request_for


def deep_predictor(probability=.7, market="domestic"):
    metadata = {"market": market, "title": TITLE, "architecture": "cnn",
                "semantics": SEMANTICS, "bundle_manifest_sha256": "a" * 64,
                "warnings": [], "research_only": True, "research_qualified": False,
                "deployment_allowed": False, "intraday_path_verified": False}
    return SimpleNamespace(metadata=metadata,
                           predict_proba=Mock(return_value=np.array([probability], dtype=np.float64)))


def adapter(probability=.7, market="domestic"):
    native = deep_predictor(probability, market)
    return Mark12PrototypePredictor(Path("unused-for-injected-test"), market, predictor=native), native


def bars():
    return [[100., 101., 99., 100., 100000.] for _ in range(30)]


class PredictorTests(unittest.TestCase):
    def test_price_aware_neural_probability_and_research_flags(self):
        model, native = adapter(.7123)
        prediction = model.predict(bars(), current_price=Decimal("100.25"))
        Mark12DemoSignalProducer.validate_prediction(prediction)
        self.assertEqual(prediction["probability_success"], .7123)
        self.assertTrue(prediction["predicts_success"])
        self.assertFalse(prediction["research_qualified"])
        self.assertFalse(prediction["deployment_allowed"])
        self.assertEqual(prediction["strategy_id"], STRATEGY_ID)
        self.assertEqual(prediction["candidate_entry_price"], 100.25)
        self.assertEqual(native.predict_proba.call_args.args[0].shape, (30, 5))
        self.assertEqual(native.predict_proba.call_args.args[1].tolist(), [100.25])

    def test_exact_half_holds_invalid_probability_and_changed_flags_fail(self):
        model, native = adapter(.5)
        prediction = model.predict(bars(), current_price=100)
        self.assertFalse(prediction["predicts_success"])
        for probability in (-.1, 1.01, float("nan")):
            native.predict_proba.return_value = np.array([probability])
            with self.assertRaises(ValueError):
                model.predict(bars(), current_price=100)
        for field, changed in (("research_qualified", True), ("deployment_allowed", True),
                               ("intraday_path_verified", True), ("strategy_id", "mark1-prototype"),
                               ("take_profit_pct", .5), ("stop_loss_pct", .4)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                Mark12DemoSignalProducer.validate_prediction({**prediction, field: changed})

    def test_bundle_identity_is_required(self):
        native = deep_predictor()
        native.metadata["deployment_allowed"] = True
        with self.assertRaisesRegex(ValueError, "연구 제한"):
            Mark12PrototypePredictor("unused", "domestic", predictor=native)


class ProducerAndWorkerTests(unittest.TestCase):
    def test_buy_hold_probability_and_source_are_distinct(self):
        for probability, action in ((.5, "hold"), (.50001, "buy")):
            model, _ = adapter(probability)
            producer = Mark12DemoSignalProducer({"domestic": model}, position_provider=position,
                quantity=1, max_krw="10000", max_usd="1000", clock=lambda: NOW)
            payload, diagnostics = producer(mark1_chart())
            self.assertEqual(payload["source_id"], SOURCE_ID)
            self.assertEqual(payload["signals"][0]["action"], action)
            self.assertTrue(payload["signals"][0]["signal_id"].startswith(STRATEGY_ID + ":"))
            self.assertEqual(diagnostics[0]["prediction"]["probability_success"], probability)
            self.assertFalse(diagnostics[0]["deployment_allowed"])
            if action == "buy":
                self.assertEqual((payload["signals"][0]["take_profit_price"],
                                  payload["signals"][0]["stop_loss_price"]), ("101", "99.1"))
                self.assertEqual(payload["signals"][0]["model_title"], TITLE)

    def test_worker_data_only_and_independent_state(self):
        model, native = adapter()
        with tempfile.TemporaryDirectory() as folder:
            worker = PrototypeWorker(STRATEGY_ID, bundle_root=Path(folder) / "injected-bundle",
                                     state_path=Path(folder) / "mark12.json",
                                     predictors={"domestic": model})
            response = worker.dispatch(request_for(STRATEGY_ID))
            self.assertEqual(response["payload"]["signals"][0]["action"], "buy")
            self.assertEqual(response["metadata"]["domestic"]["strategy_id"], STRATEGY_ID)
            self.assertEqual(response["diagnostics"][0]["prediction"]["probability_success"], .7)
            self.assertFalse(response["metadata"]["domestic"]["deployment_allowed"])
            self.assertEqual(native.predict_proba.call_count, 1)

class BridgeTests(unittest.TestCase):
    from test_mark1_trigger import BridgeTests as _OldBridgeTests
    setUp = _OldBridgeTests.setUp
    execution_args = _OldBridgeTests.execution_args

    def test_bridge_reuses_demo_only_preflight_and_final_send(self):
        from dockdack.signal_bridge import ExternalPolicy
        self.engine.external_policy = ExternalPolicy(SOURCE_ID, 1, Decimal(10000), Decimal(1000))
        model, native = adapter()
        bridge = Mark12TriggerBridge(self.window, predictors={"domestic": model},
                                     bundle_root=self.root / "injected-bundle")
        item, rule, fresh = self.execution_args()
        for stage in ("preflight", "final_send"):
            bridge.validate_execution(item, rule, fresh, Decimal("100.1"), stage=stage)
        self.assertEqual(native.predict_proba.call_count, 4)
        self.assertEqual([call.args[1].tolist() for call in native.predict_proba.call_args_list],
                         [[100.], [100.1], [100.], [100.1]])
        self.service.submit.assert_not_called()
        self.engine.enable_orders.assert_not_called()

    def test_real_mode_rejected_without_loading_model(self):
        from dockdack.models import TradingMode
        model, native = adapter()
        bridge = Mark12TriggerBridge(self.window, predictors={"domestic": model},
                                     bundle_root=self.root / "injected-bundle")
        self.service.mode = TradingMode.REAL
        with self.assertRaisesRegex(ValueError, "실전"):
            bridge._ensure_demo()
        native.predict_proba.assert_not_called()


if __name__ == "__main__":
    unittest.main()
