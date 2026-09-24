"""CPU-only synthetic checkpoint inference; never DB, broker or GUI access."""

import copy
from decimal import Decimal
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_ML = (importlib.util.find_spec("torch") is not None
          and importlib.util.find_spec("numpy") is not None)
if HAS_ML:
    import torch
    from dockdack.mark1_data import FEATURE_NAMES, TARGET, features_from_history
    from dockdack.mark1_inference import Predictor
    from dockdack.mark1_metrics import calibrated_probability
    from dockdack.mark1_models import MODEL_NAMES, build_model


@unittest.skipUnless(HAS_ML, "requires optional ml dependencies")
class Mark1InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.pt"

    @staticmethod
    def bars():
        close = torch.linspace(100., 110., 30)
        return torch.stack((close - .2, close + 1, close - 1, close,
                            torch.arange(30) * 1000 + 10000), dim=-1).tolist()

    @staticmethod
    def checkpoint(name="mlp", market="domestic"):
        config = {"input_size": 9, "sequence_length": 31, "hidden_size": 8, "dropout": .15}
        return {
            "metadata": {
                "schema_version": 1, "strategy_version": "mark_1", "market": market,
                "model_name": name, "model_config": config, "lookback": 30,
                "sequence_length": 31, "feature_names": list(FEATURE_NAMES), "target": TARGET,
                "buy_threshold": .5, "take_profit_pct": 1., "stop_loss_pct": .9,
                "calibration": {"method": "platt_monotone", "slope": .8, "bias": -.2,
                                "fit_samples": 1000},
                "intraday_path_verified": False,
            },
            "model_state_dict": build_model(name, **config).state_dict(),
        }

    def predictor(self, payload=None):
        torch.save(self.checkpoint() if payload is None else payload, self.path)
        return Predictor(self.path, device="cpu")

    def test_all_five_architectures_match_reference_and_reload_deterministically(self):
        bars, entry = self.bars(), Decimal("110.25")
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                payload = self.checkpoint(name)
                predictor = self.predictor(payload)
                reference = build_model(name, **payload["metadata"]["model_config"]).eval()
                reference.load_state_dict(payload["model_state_dict"])
                features = features_from_history(torch.tensor(bars).unsqueeze(0),
                                                 torch.tensor([float(entry)]), validate=True)
                with torch.inference_mode():
                    logit = float(reference(features).item())
                expected = float(calibrated_probability([logit], payload["metadata"]["calibration"])[0])
                result = predictor.predict(bars, entry)
                self.assertEqual(result["probability_success"], expected)
                self.assertEqual(result["predicts_success"], expected > .5)
                self.assertEqual(result["target"], TARGET)
                self.assertEqual(result["model_name"], name)
                self.assertFalse(result["intraday_path_verified"])
                self.assertEqual(result, predictor.predict(bars, entry))
                self.assertEqual(result, Predictor(self.path, device="cpu").predict(bars, entry))
                self.assertFalse(predictor.model.training)
                self.assertEqual(next(predictor.model.parameters()).device.type, "cpu")

    def test_strict_probability_threshold_is_not_greater_equal(self):
        payload = self.checkpoint()
        payload["model_state_dict"] = {key: torch.zeros_like(value)
                                       for key, value in payload["model_state_dict"].items()}
        payload["metadata"]["calibration"].update(slope=1., bias=0.)
        result = self.predictor(payload).predict(self.bars(), 110)
        self.assertEqual(result["probability_success"], .5)
        self.assertFalse(result["predicts_success"])
        payload["metadata"]["calibration"]["bias"] = 1e-6
        self.assertTrue(self.predictor(payload).predict(self.bars(), 110)["predicts_success"])

    def test_candidate_price_only_changes_query_token_and_input_is_not_mutated(self):
        predictor = self.predictor()
        bars = self.bars()
        original = copy.deepcopy(bars)
        with patch.object(predictor.model, "forward", return_value=torch.tensor([0.])) as forward:
            predictor.predict(bars, 110)
            first = forward.call_args.args[0].clone()
            predictor.predict(bars, 111)
            second = forward.call_args.args[0].clone()
        self.assertEqual(tuple(first.shape), (1, 31, 9))
        torch.testing.assert_close(first[:, :30], second[:, :30], rtol=0, atol=0)
        self.assertNotEqual(float(first[0, 30, 0]), float(second[0, 30, 0]))
        self.assertTrue(torch.equal(first[:, 30, 1:8], torch.zeros(1, 7)))
        self.assertEqual(float(first[0, 30, 8]), 1)
        self.assertEqual(bars, original)

    def test_supported_markets(self):
        for market in ("domestic", "us"):
            with self.subTest(market=market):
                predictor = self.predictor(self.checkpoint(market=market))
                self.assertEqual(predictor.market, market)
                self.assertEqual(predictor.metadata["market"], market)

    def test_invalid_checkpoint_container_or_missing_fields_rejected(self):
        for payload in (None, [], {}, {"metadata": []}, {"metadata": {}},
                        {"metadata": self.checkpoint()["metadata"]}):
            with self.subTest(payload=type(payload)), self.assertRaises(ValueError):
                torch.save(payload, self.path)
                Predictor(self.path)
        with self.assertRaises(FileNotFoundError):
            Predictor(Path(self.directory.name) / "missing.pt")

    def test_incompatible_metadata_rejected_as_value_error(self):
        changes = [
            ("schema_version", True), ("schema_version", 2),
            ("strategy_version", "mark_0"), ("market", "unknown"), ("market", []),
            ("target", "next_day_close"), ("lookback", 31), ("lookback", True),
            ("sequence_length", 30), ("sequence_length", 31.),
            ("feature_names", list(reversed(FEATURE_NAMES))),
            ("buy_threshold", .4), ("buy_threshold", True), ("buy_threshold", float("nan")),
            ("take_profit_pct", .01), ("stop_loss_pct", .8),
            ("model_name", "unknown"), ("model_config", []),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                payload = self.checkpoint()
                payload["metadata"][key] = value
                self.predictor(payload)

    def test_model_configuration_must_match_fixed_feature_contract(self):
        changes = [{"input_size": 8}, {"input_size": True}, {"sequence_length": 30},
                   {"hidden_size": 3}, {"hidden_size": 513}, {"hidden_size": 8.},
                   {"dropout": 1}, {"dropout": float("nan")}, {"dropout": True},
                   {"unexpected": 1}]
        for changeset in changes:
            with self.subTest(changes=changeset), self.assertRaises(ValueError):
                payload = self.checkpoint()
                payload["metadata"]["model_config"].update(changeset)
                self.predictor(payload)
        payload = self.checkpoint()
        del payload["metadata"]["model_config"]["dropout"]
        with self.assertRaises(ValueError):
            self.predictor(payload)

    def test_calibration_metadata_must_be_fitted_finite_and_monotone(self):
        for changes in ({"method": "identity"}, {"slope": 0}, {"slope": -1},
                        {"slope": float("inf")}, {"bias": float("nan")},
                        {"fit_samples": 0}, {"fit_samples": True}, {"fit_samples": 1.}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                payload = self.checkpoint()
                payload["metadata"]["calibration"].update(changes)
                self.predictor(payload)
        for calibration in (None, [], {}):
            with self.subTest(calibration=calibration), self.assertRaises(ValueError):
                payload = self.checkpoint()
                payload["metadata"]["calibration"] = calibration
                self.predictor(payload)

    def test_malformed_missing_extra_shape_and_nonfinite_weights_rejected(self):
        for kind in ("missing", "extra", "shape", "string", "nan", "infinity"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                payload = self.checkpoint()
                state = payload["model_state_dict"]
                key = next(iter(state))
                if kind == "missing":
                    del state[key]
                elif kind == "extra":
                    state["unknown_parameter"] = torch.tensor([1.])
                elif kind == "shape":
                    state[key] = torch.zeros(1)
                elif kind == "string":
                    state[key] = "not a tensor"
                else:
                    state[key] = torch.full_like(state[key], float("nan") if kind == "nan" else float("inf"))
                self.predictor(payload)

    def test_nonfloating_and_complex_weights_not_silently_cast(self):
        for dtype in (torch.int64, torch.bool, torch.complex64):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                payload = self.checkpoint()
                state = payload["model_state_dict"]
                key = next(iter(state))
                state[key] = state[key].to(dtype)
                self.predictor(payload)

    def test_current_price_must_be_finite_positive_float32_representable(self):
        predictor = self.predictor()
        for price in (None, True, False, 0, -1, "0", "-2", "bad", "NaN", "Infinity",
                      float("inf"), Decimal("NaN"), "1e1000", "1e-1000", "1e40", "1e-50"):
            with self.subTest(price=price), self.assertRaises(ValueError):
                predictor.predict(self.bars(), price)
        result = predictor.predict(self.bars(), "110.25")
        self.assertEqual(result, predictor.predict(self.bars(), Decimal("110.25")))

    def test_history_shape_ohlc_and_volume_validation(self):
        predictor = self.predictor()
        for bars in (None, [], self.bars()[:29], self.bars() + self.bars()[:1],
                     [[100.] * 4] * 30, [self.bars()]):
            with self.subTest(history_type=type(bars)), self.assertRaises(ValueError):
                predictor.predict(bars, 110)
        for column, value in ((0, 0), (0, -1), (0, float("nan")), (1, 1), (2, 1000),
                              (3, float("inf")), (4, -1), (4, float("nan"))):
            bars = self.bars()
            bars[0][column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                predictor.predict(bars, 110)

    def test_nonfinite_model_output_fails_closed(self):
        predictor = self.predictor()
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), patch.object(predictor.model, "forward", return_value=torch.tensor([value])):
                with self.assertRaises(ValueError):
                    predictor.predict(self.bars(), 110)


if __name__ == "__main__":
    unittest.main()
