"""Offline architecture parity checks; no datasets, CUDA training or broker access."""

import importlib.util
import io
import unittest
from unittest.mock import patch


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from dockdack.mark1_models import (
        MODEL_NAMES, _rnn_backend, build_model, parameter_count, validate_features,
    )


@unittest.skipUnless(HAS_TORCH, "requires optional ml dependencies")
class Mark1ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_all_models_logits_and_backprop(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name, hidden_size=8, dropout=0)
                features = torch.randn(3, 31, 9, requires_grad=True)
                validate_features(features)
                logits = model(features)
                self.assertEqual(logits.shape, (3,))
                self.assertTrue(torch.isfinite(logits).all())
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.tensor([0., 1., 0.])
                ).backward()
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertGreater(features.grad[:, -1].abs().sum().item(), 0)
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(parameter_count(model), 0)

    def test_all_architectures_use_earliest_history_and_query(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name, hidden_size=8, dropout=0).double().eval()
                features = torch.randn(2, 31, 9, dtype=torch.float64, requires_grad=True)
                model(features).sum().backward()
                self.assertGreater(features.grad[:, 0].abs().sum().item(), 0)
                self.assertGreater(features.grad[:, -1].abs().sum().item(), 0)

    def test_checkpoint_round_trip_and_single_sample_output(self):
        features = torch.randn(1, 31, 9)
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                original = build_model(name, hidden_size=8).eval()
                buffer = io.BytesIO()
                torch.save(original.state_dict(), buffer)
                buffer.seek(0)
                restored = build_model(name, hidden_size=8).eval()
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                with torch.inference_mode():
                    expected = original(features)
                    actual = restored(features)
                self.assertEqual(actual.shape, (1,))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_custom_feature_sequence_and_odd_hidden_dimensions(self):
        for name in MODEL_NAMES:
            for width in (1, 7, 12):
                with self.subTest(model=name, hidden_size=width):
                    model = build_model(name, input_size=4, sequence_length=5,
                                        hidden_size=width, dropout=0)
                    self.assertEqual(model(torch.randn(2, 5, 4)).shape, (2,))
        tcn = build_model("tcn", sequence_length=63, hidden_size=8)
        self.assertGreaterEqual(tcn.receptive_field, 63)
        self.assertEqual(tcn(torch.randn(2, 63, 9)).shape, (2,))

    def test_invalid_hyperparameters_all_models(self):
        bad_options = [
            {"input_size": 0}, {"input_size": True}, {"input_size": 9.0},
            {"sequence_length": -1}, {"sequence_length": False},
            {"sequence_length": 31.0}, {"hidden_size": 0},
            {"hidden_size": True}, {"hidden_size": float("inf")},
            {"dropout": -0.1}, {"dropout": 1}, {"dropout": float("nan")},
            {"dropout": float("inf")}, {"dropout": True}, {"dropout": "0.1"},
        ]
        for name in MODEL_NAMES:
            for options in bad_options:
                with self.subTest(model=name, options=options), self.assertRaises(ValueError):
                    build_model(name, **options)
        for name in (None, 1, [], "LSTM", "unknown"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_model(name)

    def test_invalid_shapes_and_dtypes_all_models(self):
        invalid = [None, [], torch.randn(31, 9), torch.randn(0, 31, 9),
                   torch.randn(1, 30, 9), torch.randn(1, 31, 8),
                   torch.ones(1, 31, 9, dtype=torch.int64),
                   torch.ones(1, 31, 9, dtype=torch.complex64)]
        for name in MODEL_NAMES:
            model = build_model(name, hidden_size=8)
            for features in invalid:
                with self.subTest(model=name, features=type(features)), self.assertRaises(ValueError):
                    model(features)

    def test_finite_validation_happens_at_boundary(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            features = torch.ones(1, 31, 9)
            features[0, 30, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_features(features)
        validate_features(torch.zeros(1, 31, 9))
        validate_features(torch.randn(2, 5, 4), sequence_length=5, input_size=4)
        for options in ({"input_size": True}, {"sequence_length": 0}):
            with self.assertRaises(ValueError):
                validate_features(torch.randn(1, 31, 9), **options)

    def test_forward_does_not_repeat_finite_tensor_sync(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name, hidden_size=8).train()
                with patch("dockdack.mark1_models.torch.isfinite",
                           side_effect=AssertionError("unexpected finite-value sync")):
                    self.assertEqual(model(torch.zeros(2, 31, 9)).shape, (2,))

    def test_tcn_is_causal_and_covers_full_window(self):
        model = build_model("tcn", hidden_size=8, dropout=0).eval()
        self.assertEqual(model.dilations, (1, 2, 4, 8))
        self.assertGreaterEqual(model.receptive_field, 31)
        original = torch.randn(2, 31, 9)
        changed = original.clone()
        changed[:, 20:] += 10
        before = model.blocks(model.projection(original.transpose(1, 2)))
        after = model.blocks(model.projection(changed.transpose(1, 2)))
        torch.testing.assert_close(before[:, :, :20], after[:, :, :20], rtol=0, atol=0)

    def test_transformer_layers_initialized_independently_and_positions_saved(self):
        model = build_model("transformer", hidden_size=8)
        first, second = model.encoder.layers
        self.assertFalse(torch.equal(first.self_attn.in_proj_weight,
                                     second.self_attn.in_proj_weight))
        self.assertIn("positional_encoding", model.state_dict())
        self.assertFalse(torch.equal(model.positional_encoding[:, 0],
                                     model.positional_encoding[:, 1]))

    def test_rnn_windows_cuda_safeguard_restores_flags_on_exit_and_error(self):
        for initially_enabled in (True, False):
            with self.subTest(enabled=initially_enabled):
                with torch.backends.cudnn.flags(enabled=initially_enabled):
                    with patch("dockdack.mark1_models.sys.platform", "win32"):
                        with _rnn_backend("cuda:0"):
                            self.assertFalse(torch.backends.cudnn.enabled)
                        self.assertEqual(torch.backends.cudnn.enabled, initially_enabled)
                        with self.assertRaises(RuntimeError):
                            with _rnn_backend("cuda:0"):
                                self.assertFalse(torch.backends.cudnn.enabled)
                                raise RuntimeError("simulated RNN error")
                        self.assertEqual(torch.backends.cudnn.enabled, initially_enabled)

    def test_rnn_backend_keeps_cpu_and_non_windows_cuda_unchanged(self):
        for platform, device in (("win32", "cpu"), ("linux", "cuda:0")):
            with self.subTest(platform=platform, device=device):
                with torch.backends.cudnn.flags(enabled=True):
                    with patch("dockdack.mark1_models.sys.platform", platform):
                        with _rnn_backend(device):
                            self.assertTrue(torch.backends.cudnn.enabled)
                        self.assertTrue(torch.backends.cudnn.enabled)


if __name__ == "__main__":
    unittest.main()
