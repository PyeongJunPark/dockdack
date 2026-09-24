"""Offline causal-feature, architecture and checkpoint tests; no orders or data writes."""

import importlib.util
import inspect
import io
import math
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from dockdack.mark1_deep_models import (
        CLASS_NAMES, FEATURE_NAMES, MODEL_NAMES, TARGET, build_model,
        features_from_history, parameter_count, success_logit, validate_features,
    )


@unittest.skipUnless(HAS_TORCH, "requires optional ml dependencies")
class DeepFeatureTests(unittest.TestCase):
    @staticmethod
    def history(batch=2, dtype=None):
        dtype = dtype or torch.float64
        close = torch.linspace(95, 105, 30, dtype=dtype)[None].repeat(batch, 1)
        volume = torch.linspace(1000, 1500, 30, dtype=dtype)[None].repeat(batch, 1)
        return torch.stack((close - .1, close + 1, close - 1, close, volume), dim=-1)

    def test_shape_names_and_target_contract(self):
        history = self.history()
        result = features_from_history(history, torch.tensor([105., 106.], dtype=history.dtype), validate=True)
        self.assertEqual(result.shape, (2, 31, 18))
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))
        self.assertEqual(CLASS_NAMES, ("take_only", "stop_only", "both_touch", "neither"))
        from dockdack.mark1_data import TARGET as existing_target
        self.assertEqual(TARGET, existing_target)
        self.assertTrue(torch.isfinite(result).all())
        self.assertLessEqual(result.abs().max().item(), 12)

    def test_only_completed_history_and_entry_are_accepted(self):
        self.assertEqual(tuple(inspect.signature(features_from_history).parameters),
                         ("history", "entry_prices", "validate"))
        history = self.history(1)
        entry = torch.tensor([105.], dtype=history.dtype)
        with self.assertRaises(TypeError):
            features_from_history(history, entry, target_high=torch.tensor([110.]))

    def test_inputs_unchanged_and_price_changes_only_query(self):
        history = self.history()
        entry = torch.tensor([105., 106.], dtype=history.dtype)
        history_before, entry_before = history.clone(), entry.clone()
        first = features_from_history(history, entry, validate=True)
        second = features_from_history(history, entry * 1.005, validate=True)
        torch.testing.assert_close(history, history_before, rtol=0, atol=0)
        torch.testing.assert_close(entry, entry_before, rtol=0, atol=0)
        torch.testing.assert_close(first[:, :30], second[:, :30], rtol=0, atol=0)
        self.assertTrue(torch.all(first[:, 30, 0] != second[:, 30, 0]))
        self.assertTrue(torch.all(first[:, 30, 11] != second[:, 30, 11]))
        other = [index for index in range(18) if index not in (0, 11)]
        torch.testing.assert_close(first[:, 30, other], second[:, 30, other], rtol=0, atol=0)

    def test_masks_and_query_missing_bar_fields(self):
        history = self.history()
        result = features_from_history(history, torch.tensor([105., 106.], dtype=history.dtype))
        self.assertTrue(torch.all(result[:, :30, 16] == 1))
        self.assertTrue(torch.all(result[:, :30, 17] == 0))
        self.assertTrue(torch.all(result[:, 30, 16] == 0))
        self.assertTrue(torch.all(result[:, 30, 17] == 1))
        missing = [index for index in range(18) if index not in (0, 11, 13, 14, 17)]
        self.assertTrue(torch.all(result[:, 30, missing] == 0))

    def test_normalization_matches_historical_returns(self):
        history = self.history(1)
        entry = torch.tensor([106.], dtype=history.dtype)
        result = features_from_history(history, entry)
        logs = history[0, :, 3].log()
        returns = logs[1:] - logs[:-1]
        scale = max(.003, min(.3, returns.std(unbiased=False).item()))
        self.assertAlmostEqual(result[0, 30, 0].item(), math.log(106 / 105) / scale, places=10)
        self.assertAlmostEqual(result[0, 30, 13].item(), math.log(scale) / 5, places=10)
        self.assertAlmostEqual(result[0, 30, 14].item(), returns.mean().item() / scale, places=10)
        self.assertEqual(result[0, 0, 5].item(), 0)
        self.assertEqual(result[0, 0, 15].item(), 0)

    def test_no_cross_sample_statistics(self):
        history = self.history()
        history[1, :, :4] *= 1000
        history[1, :, 4] *= 100
        entry = torch.tensor([105., 105000.], dtype=history.dtype)
        batch = features_from_history(history, entry)
        alone = features_from_history(history[:1], entry[:1])
        torch.testing.assert_close(batch[:1], alone, rtol=0, atol=0)

    def test_price_scale_only_changes_explicit_level_feature(self):
        history = self.history(1)
        entry = torch.tensor([105.], dtype=history.dtype)
        scaled = history.clone()
        scaled[:, :, :4] *= 100
        first, second = features_from_history(history, entry), features_from_history(scaled, entry * 100)
        channels = [index for index in range(18) if index != 11]
        torch.testing.assert_close(first[:, :, channels], second[:, :, channels], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(second[:, :, 11] - first[:, :, 11],
                                   torch.full((1, 31), math.log(100) / 10, dtype=history.dtype))

    def test_flat_zero_volume_and_extreme_entry_are_finite(self):
        history = torch.full((1, 30, 5), 100., dtype=torch.float64)
        history[:, :, 4] = 0
        features = features_from_history(history, torch.tensor([1e200], dtype=history.dtype), validate=True)
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(features[0, 30, 0].item(), 12)
        self.assertTrue(torch.all(features[:, :30, 7:11] == 0))
        self.assertTrue(torch.all(features[:, :30, 4] == 0))

    def test_feature_construction_is_differentiable(self):
        history = self.history(1).requires_grad_()
        entry = torch.tensor([105.], dtype=history.dtype, requires_grad=True)
        features_from_history(history, entry).sum().backward()
        self.assertTrue(torch.isfinite(history.grad).all())
        self.assertTrue(torch.isfinite(entry.grad).all())
        self.assertGreater(entry.grad.abs().sum().item(), 0)

    def test_invalid_shape_dtype_or_device(self):
        history = self.history()
        entry = torch.tensor([105., 105.], dtype=history.dtype)
        invalid = ((None, entry), (history.tolist(), entry), (history[:, :29], entry),
                   (history[:0], entry[:0]), (history.long(), entry), (history, None),
                   (history, entry[:, None]), (history, entry[:1]), (history, entry.long()),
                   (history, entry.float()))
        for bars, candidate in invalid:
            with self.subTest(shape=getattr(bars, "shape", None)):
                with self.assertRaises(ValueError):
                    features_from_history(bars, candidate)

    def test_invalid_ohlcv_or_entry_values(self):
        history = self.history(1)
        entry = torch.tensor([105.], dtype=history.dtype)
        for channel, value in ((0, float("nan")), (1, float("inf")), (2, 0),
                               (3, -1), (4, -1), (1, 1), (2, 1000)):
            invalid = history.clone()
            invalid[0, 0, channel] = value
            with self.subTest(channel=channel, value=value):
                with self.assertRaises(ValueError):
                    features_from_history(invalid, entry, validate=True)
        for value in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                features_from_history(history, torch.tensor([value], dtype=history.dtype), validate=True)

    def test_finite_feature_boundary(self):
        for bad in (torch.zeros(1, 31, 17), torch.zeros(0, 31, 18),
                    torch.zeros(1, 31, 18, dtype=torch.long), None):
            with self.assertRaises(ValueError):
                validate_features(bad)
        values = torch.zeros(1, 31, 18)
        values[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            validate_features(values)


@unittest.skipUnless(HAS_TORCH, "requires optional ml dependencies")
class DeepModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_all_models_four_logits_and_finite_backprop(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name, width=8, dropout=0)
                features = torch.randn(3, 31, 18, requires_grad=True)
                logits = model(features)
                self.assertEqual(logits.shape, (3, 4))
                self.assertTrue(torch.isfinite(logits).all())
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2, 3])).backward()
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertGreater(features.grad[:, 0].abs().sum().item(), 0)
                self.assertGreater(features.grad[:, -1].abs().sum().item(), 0)
                gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
                self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(parameter_count(model), 0)

    def test_sample_independence_in_eval_and_training_without_dropout(self):
        features = torch.randn(3, 31, 18, dtype=torch.float64)
        for name in MODEL_NAMES:
            model = build_model(name, width=8, dropout=0).double()
            for training in (False, True):
                with self.subTest(model=name, training=training):
                    model.train(training)
                    with torch.no_grad():
                        batch, alone = model(features), model(features[:1])
                    torch.testing.assert_close(batch[:1], alone, rtol=1e-8, atol=1e-8)
            self.assertFalse(any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                                 for module in model.modules()))

    def test_price_query_changes_output(self):
        features = torch.randn(2, 31, 18)
        changed = features.clone()
        changed[:, -1, 0] += 1
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name, width=8, dropout=0).eval()
                with torch.no_grad():
                    self.assertFalse(torch.allclose(model(features), model(changed)))

    def test_query_never_enters_historical_encoder(self):
        features = torch.randn(2, 31, 18)
        changed = features.clone()
        changed[:, -1] += 100
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name, width=8, dropout=0).eval()
                captured = []
                encoder = model.stem if hasattr(model, "stem") else model.history_encoder
                handle = encoder.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].clone()))
                with torch.no_grad():
                    model(features)
                    model(changed)
                handle.remove()
                self.assertEqual(captured[0].shape[1 if name == "mlp_deep" else 2], 30)
                torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)

    def test_state_dict_round_trip_single_sample(self):
        features = torch.randn(1, 31, 18)
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name, width=8).eval()
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = build_model(name, width=8).eval()
                restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(model(features), restored(features), rtol=0, atol=0)

    def test_custom_shapes_and_odd_width(self):
        for name in MODEL_NAMES:
            for width in (4, 7, 12):
                with self.subTest(model=name, width=width):
                    model = build_model(name, input_size=3, sequence_length=3, width=width, dropout=0)
                    self.assertEqual(model(torch.randn(1, 3, 3)).shape, (1, 4))

    def test_residual_block_counts_and_independent_initialization(self):
        shallow = build_model("resnet18", width=8)
        deep = build_model("resnet34", width=8)
        self.assertEqual(shallow.block_counts, (2, 2, 2, 2))
        self.assertEqual(deep.block_counts, (3, 4, 6, 3))
        self.assertEqual(len(shallow.history_encoder), 8)
        self.assertEqual(len(deep.history_encoder), 16)
        self.assertGreater(parameter_count(deep), parameter_count(shallow))
        self.assertFalse(torch.equal(deep.history_encoder[0].main[0].weight,
                                     deep.history_encoder[1].main[0].weight))

    def test_inception_six_modules_shortened_kernel_sizes(self):
        model = build_model("inception", width=8)
        modules = [module for module in model.modules() if type(module).__name__ == "_InceptionModule"]
        self.assertEqual(len(modules), 6)
        for module in modules:
            self.assertEqual([branch.kernel_size for branch in module.branches], [(3,), (7,), (15,)])
        first, second = modules[:2]
        self.assertFalse(torch.equal(first.branches[0].weight, second.branches[0].weight))

    def test_invalid_hyperparameters(self):
        invalid = ({"input_size": 0}, {"input_size": True}, {"input_size": 2.5},
                   {"sequence_length": 2}, {"sequence_length": False}, {"width": 0},
                   {"width": 3}, {"width": 8.0}, {"dropout": -1}, {"dropout": 1},
                   {"dropout": float("nan")}, {"dropout": float("inf")}, {"dropout": True})
        for name in MODEL_NAMES:
            for options in invalid:
                with self.subTest(model=name, options=options):
                    with self.assertRaises(ValueError):
                        build_model(name, **options)
        for name in (None, [], "resnet50", "ResNet18"):
            with self.assertRaises(ValueError):
                build_model(name)

    def test_forward_rejects_bad_shape_and_dtype(self):
        for name in MODEL_NAMES:
            model = build_model(name, width=8)
            for values in (None, torch.zeros(2, 30, 18), torch.zeros(0, 31, 18),
                           torch.zeros(2, 31, 17), torch.zeros(2, 31, 18, dtype=torch.long)):
                with self.subTest(model=name, shape=getattr(values, "shape", None)):
                    with self.assertRaises(ValueError):
                        model(values)

    def test_success_logit_matches_softmax_take_only_mass(self):
        logits = torch.tensor([[1., 2., 3., 4.], [1000., -1000., 0., 1.],
                               [0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
        value = success_logit(logits)
        torch.testing.assert_close(value.sigmoid(), logits.softmax(dim=1)[:, 0])
        self.assertTrue(torch.isfinite(value).all())
        value.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertAlmostEqual(value[-1].sigmoid().item(), .25)

    def test_success_logit_rejects_incompatible_shapes(self):
        for values in (None, torch.ones(4), torch.ones(2, 3), torch.ones(0, 4),
                       torch.ones(2, 4, dtype=torch.long)):
            with self.assertRaises(ValueError):
                success_logit(values)


if __name__ == "__main__":
    unittest.main()
