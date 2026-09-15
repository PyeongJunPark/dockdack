import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from dockdack.lstm30_adapter import decide_position


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from dockdack.ml30 import (
        CandleLSTM, FEATURE_NAMES, LOOKBACK, Predictor, TARGET,
        _inference_backend, validate_windows, window_features,
    )


@unittest.skipUnless(HAS_TORCH, "requires optional ml dependencies")
class ThirtyBarModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def bars(self, batch=2):
        close = torch.arange(100, 130, dtype=torch.float32).expand(batch, -1)
        volume = torch.arange(LOOKBACK, dtype=torch.float32).expand(batch, -1) * 100
        return torch.stack((close - 0.5, close + 1, close - 1, close, volume), dim=-1)

    def checkpoint(self):
        torch.manual_seed(42)
        return {
            "model_state_dict": CandleLSTM(hidden_size=8).state_dict(),
            "metadata": {
                "schema_version": 1,
                "lookback": 30,
                "architecture": {"hidden_size": 8, "num_layers": 2, "dropout": 0.2},
                "feature_names": list(FEATURE_NAMES),
                "market": "domestic",
                "target": TARGET,
                "buy_threshold": 0.5,
            },
        }

    def load_checkpoint(self, directory, checkpoint=None, **predictor_options):
        path = Path(directory) / "model.pt"
        torch.save(self.checkpoint() if checkpoint is None else checkpoint, path)
        return Predictor(path, **predictor_options)

    def test_exactly_thirty_raw_bars_make_thirty_features(self):
        raw = self.bars()
        validate_windows(raw)
        actual = window_features(raw)
        self.assertEqual(tuple(actual.shape), (2, 30, 7))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual[:, 0, 3], torch.zeros(2))
        torch.testing.assert_close(actual[:, 0, 5], torch.zeros(2))
        torch.testing.assert_close(actual[:, :, 4].mean(1), torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(actual[:, 1, 5], torch.log(raw[:, 1, 3] / raw[:, 0, 3]), atol=1e-6, rtol=1e-5)

    def test_price_units_do_not_change_features(self):
        original = self.bars().double()
        scaled = original.clone()
        scaled[..., :4] *= 1500
        torch.testing.assert_close(window_features(original), window_features(scaled), atol=1e-12, rtol=1e-12)

    def test_volume_log1p_centering_removes_additive_log_offset(self):
        original = self.bars().double()
        transformed = original.clone()
        # Scaling volume+1, not raw volume, is invariant under log1p centering.
        transformed[..., 4] = (original[..., 4] + 1) * 100 - 1
        torch.testing.assert_close(window_features(original), window_features(transformed), atol=1e-12, rtol=1e-12)

    def test_zero_volume_is_supported(self):
        raw = self.bars()
        raw[..., 4] = 0
        validate_windows(raw)
        self.assertTrue(torch.equal(window_features(raw)[..., 4], torch.zeros(2, 30)))

    def test_malformed_window_shapes_are_rejected(self):
        for raw in (torch.ones(30, 5), torch.ones(1, 29, 5), torch.ones(1, 31, 5),
                    torch.ones(0, 30, 5), torch.ones(1, 30, 4), torch.ones(1, 30, 5, dtype=torch.int64)):
            with self.subTest(shape=tuple(raw.shape)), self.assertRaises(ValueError):
                window_features(raw)

    def test_invalid_market_values_are_rejected(self):
        for column, value in ((0, float("nan")), (4, float("inf")), (0, 0), (2, -1),
                              (4, -1), (1, 50), (2, 150), (0, 150), (3, 50)):
            with self.subTest(column=column, value=value):
                raw = self.bars()
                raw[0, 0, column] = value
                with self.assertRaises(ValueError):
                    validate_windows(raw)

    def test_model_dimensions_and_backward(self):
        model = CandleLSTM()
        self.assertEqual(model.lstm.input_size, 7)
        self.assertEqual(model.lstm.hidden_size, 128)
        self.assertEqual(model.lstm.num_layers, 2)
        self.assertEqual(model.lstm.dropout, 0.2)
        logits = model(self.bars())
        self.assertEqual(tuple(logits.shape), (2,))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0.0, 1.0]))
        loss.backward()
        self.assertIsNotNone(model.head.weight.grad)

    def test_invalid_architecture_is_rejected(self):
        for kwargs in ({"hidden_size": 0}, {"num_layers": 0}, {"dropout": -0.1},
                       {"dropout": 1}, {"dropout": float("nan")}, {"hidden_size": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CandleLSTM(**kwargs)

    def test_checkpoint_prediction_parity(self):
        checkpoint = self.checkpoint()
        raw = self.bars(batch=1)
        reference = CandleLSTM(hidden_size=8).eval()
        reference.load_state_dict(checkpoint["model_state_dict"])
        with torch.inference_mode():
            expected = reference(raw).sigmoid().item()
        with tempfile.TemporaryDirectory() as directory:
            predictor = self.load_checkpoint(directory, checkpoint)
            result = predictor.predict(raw[0].tolist())
            self.assertEqual(result["probability_ge_1pct"], expected)
            self.assertEqual(result["predicts_gain"], expected >= 0.5)
            self.assertEqual(result["buy_threshold"], 0.5)
            self.assertEqual(predictor.market, "domestic")
            self.assertFalse(predictor.model.training)

    def test_buy_threshold_override_preserves_checkpoint_metadata_weights_and_score(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.load_checkpoint(directory)
            path = Path(directory) / "model.pt"
            before = path.read_bytes()
            overridden = Predictor(path, buy_threshold=0.4)
            raw = self.bars(batch=1)[0]
            original_result = original.predict(raw)
            override_result = overridden.predict(raw)
            self.assertEqual(original.buy_threshold, 0.5)
            self.assertEqual(overridden.buy_threshold, 0.4)
            self.assertEqual(overridden.metadata["buy_threshold"], 0.5)
            self.assertEqual(original.metadata, overridden.metadata)
            self.assertEqual(original_result["probability_ge_1pct"], override_result["probability_ge_1pct"])
            self.assertEqual(override_result["buy_threshold"], 0.4)
            for name, value in original.model.state_dict().items():
                torch.testing.assert_close(value, overridden.model.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(torch.load(path, weights_only=True)["metadata"]["buy_threshold"], 0.5)

    def test_none_override_uses_checkpoint_threshold(self):
        checkpoint = self.checkpoint()
        checkpoint["metadata"]["buy_threshold"] = 0.6
        with tempfile.TemporaryDirectory() as directory:
            predictor = self.load_checkpoint(directory, checkpoint, buy_threshold=None)
            self.assertEqual(predictor.buy_threshold, 0.6)
            self.assertEqual(predictor.predict(self.bars(batch=1)[0])["buy_threshold"], 0.6)

    def test_buy_threshold_override_must_be_finite_and_strictly_between_zero_and_one(self):
        with tempfile.TemporaryDirectory() as directory:
            self.load_checkpoint(directory)
            path = Path(directory) / "model.pt"
            for threshold in (0, 1, -0.1, 1.1, float("nan"), float("inf"), -float("inf"),
                              True, False, "0.4", [], {}):
                with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                    Predictor(path, buy_threshold=threshold)

    def test_valid_override_does_not_hide_invalid_checkpoint_threshold(self):
        checkpoint = self.checkpoint()
        checkpoint["metadata"]["buy_threshold"] = float("nan")
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            self.load_checkpoint(directory, checkpoint, buy_threshold=0.4)

    def test_point_four_threshold_is_inclusive_through_prediction_and_adapter(self):
        for market in ("domestic", "us"):
            checkpoint = self.checkpoint()
            checkpoint["metadata"]["market"] = market
            with self.subTest(market=market), tempfile.TemporaryDirectory() as directory:
                predictor = self.load_checkpoint(directory, checkpoint, buy_threshold=0.4)
                for score, expected in ((0.3999, "hold"), (0.4, "buy"), (0.44, "buy")):
                    with self.subTest(score=score):
                        logits = Mock()
                        logits.sigmoid.return_value.item.return_value = score
                        with patch.object(predictor.model, "forward", return_value=logits):
                            prediction = predictor.predict(self.bars(batch=1)[0])
                        self.assertEqual(prediction["probability_ge_1pct"], score)
                        self.assertEqual(prediction["buy_threshold"], 0.4)
                        self.assertEqual(prediction["predicts_gain"], expected == "buy")
                        decision = decide_position(current_price=100, quantity=0, sellable_quantity=0,
                                                   prediction=prediction)
                        self.assertEqual(decision["action"], expected)

    def test_lower_entry_threshold_does_not_change_held_profit_loss_or_hold_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            predictor = self.load_checkpoint(directory, buy_threshold=0.4)
            for score in (0.3999, 0.4, 0.44):
                logits = Mock()
                logits.sigmoid.return_value.item.return_value = score
                with patch.object(predictor.model, "forward", return_value=logits):
                    prediction = predictor.predict(self.bars(batch=1)[0])
                for price, expected, field in (("101", "sell", "cost_profit_pct"),
                                               ("99.2", "sell", "cost_loss_pct"),
                                               ("100", "hold", None)):
                    with self.subTest(score=score, price=price):
                        decision = decide_position(current_price=price, quantity=1, sellable_quantity=1,
                                                   average_price=100, prediction=prediction)
                        self.assertEqual(decision["action"], expected)
                        if field:
                            self.assertEqual(decision[field], "1" if field == "cost_profit_pct" else "0.8")

    def test_inference_rejects_invalid_values_and_thirty_first_bar(self):
        with tempfile.TemporaryDirectory() as directory:
            predictor = self.load_checkpoint(directory)
            inputs = [torch.ones(29, 5), torch.ones(31, 5), [["bad"] * 5] * 30]
            invalid = self.bars(batch=1)[0]
            invalid[0, 4] = -1
            inputs.append(invalid)
            for bars in inputs:
                with self.subTest(), self.assertRaises(ValueError):
                    predictor.predict(bars)

    def test_checkpoint_contract_mismatch_is_rejected(self):
        for key, value in (("schema_version", 2), ("schema_version", True), ("lookback", 60),
                           ("feature_names", ["close"]), ("target", "next_day_up"),
                           ("market", "crypto"), ("buy_threshold", 0),
                           ("buy_threshold", 1), ("buy_threshold", float("nan")),
                           ("architecture", {})):
            checkpoint = self.checkpoint()
            checkpoint["metadata"][key] = value
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.load_checkpoint(directory, checkpoint)

    def test_bad_weights_or_nonfinite_output_are_rejected(self):
        checkpoint = self.checkpoint()
        missing = copy.deepcopy(checkpoint)
        missing["model_state_dict"].pop("head.bias")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.load_checkpoint(directory, missing)
            checkpoint["model_state_dict"]["head.bias"].fill_(float("nan"))
            predictor = self.load_checkpoint(directory, checkpoint)
            with self.assertRaises(ValueError):
                predictor.predict(self.bars(batch=1)[0])

    def test_samples_do_not_leak_into_other_batch_elements(self):
        raw = self.bars()
        original = window_features(raw)[0].clone()
        raw[1] *= 100
        torch.testing.assert_close(original, window_features(raw)[0], atol=0, rtol=0)

    def test_future_rows_outside_input_cannot_affect_prediction(self):
        history = torch.cat((self.bars(batch=1)[0], self.bars(batch=1)[0]), dim=0)
        with tempfile.TemporaryDirectory() as directory:
            predictor = self.load_checkpoint(directory)
            expected = predictor.predict(history[:30])
            history[30:, :4] *= 10
            history[30:, 4] = 0
            self.assertEqual(predictor.predict(history[:30]), expected)

    def test_windows_cuda_backend_restores_flags_even_on_failure(self):
        for initial in (True, False):
            with self.subTest(initial=initial), torch.backends.cudnn.flags(enabled=initial):
                with patch("dockdack.ml30.sys.platform", "win32"):
                    with _inference_backend("cuda:0"):
                        self.assertFalse(torch.backends.cudnn.enabled)
                    self.assertEqual(torch.backends.cudnn.enabled, initial)
                    with self.assertRaisesRegex(RuntimeError, "inference failed"):
                        with _inference_backend("cuda"):
                            self.assertFalse(torch.backends.cudnn.enabled)
                            raise RuntimeError("inference failed")
                    self.assertEqual(torch.backends.cudnn.enabled, initial)

    def test_other_platforms_and_cpu_do_not_override_backend_flags(self):
        for platform, device in (("win32", "cpu"), ("linux", "cuda"), ("darwin", "cpu")):
            with self.subTest(platform=platform, device=device), torch.backends.cudnn.flags(enabled=True):
                with patch("dockdack.ml30.sys.platform", platform), _inference_backend(device):
                    self.assertTrue(torch.backends.cudnn.enabled)
                self.assertTrue(torch.backends.cudnn.enabled)

    def test_predictor_uses_scoped_backend_context(self):
        with tempfile.TemporaryDirectory() as directory:
            predictor = self.load_checkpoint(directory)
            with patch("dockdack.ml30._inference_backend", wraps=_inference_backend) as backend:
                predictor.predict(self.bars(batch=1)[0])
            backend.assert_called_once_with(torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
