"""CPU-only training orchestration checks with tiny synthetic causal histories."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HAS_ML = (importlib.util.find_spec("torch") is not None
          and importlib.util.find_spec("numpy") is not None)
if HAS_ML:
    import numpy as np
    import torch
    from examples import train_mark1_deep as runner


@unittest.skipUnless(HAS_ML, "requires optional ml dependencies")
class DeepTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mark1-deep-trainer-")
        self.root = Path(self.temporary.name)
        self.model_patch = patch.object(runner, "MODEL_CONFIG", dict(input_size=18, sequence_length=31, width=4, dropout=.2))
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.addCleanup(self.temporary.cleanup)
        self.context = dict(market="domestic", fold="synthetic", source={"sha256": "synthetic"}, splits={})

    @staticmethod
    def dataset():
        count = 52
        base = np.linspace(90, 110, count, dtype=np.float64)
        paths = base[:, None] * (1 + .001 * np.sin(np.arange(30)[None, :]))
        bars = np.stack((paths - .05, paths + .5, paths - .5, paths,
                         np.broadcast_to(np.linspace(1000, 2000, 30), paths.shape)), axis=-1)
        outcome_multipliers = np.asarray([[1, 1.02, .995, 1.01], [1, 1.005, .98, .99],
                                          [1, 1.02, .98, 1], [1, 1.005, .995, 1]])
        ohlc = base[:, None] * outcome_multipliers[np.arange(count) % 4]
        dataset = SimpleNamespace(bars=bars.reshape(-1, 5).astype(np.float32),
                                  starts=np.arange(count, dtype=np.int64) * 30,
                                  target_dates=np.arange(count, dtype=np.int32) + 15000,
                                  symbol_ids=np.zeros(count, dtype=np.int32), target_ohlc=ohlc)
        splits = dict(train=np.arange(0, 12), tune=np.arange(12, 20),
                      calibration=np.arange(20, 36), selection=np.arange(36, 52))
        return dataset, splits

    def bank(self):
        dataset, splits = self.dataset()
        return runner.DeepBank(dataset, splits, "cpu")

    def train(self, folder, *, epochs=3, bank=None, seed=42):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.train_trial(bank or self.bank(), "mlp_deep", seed, folder,
                                      self.context, epochs=epochs, batch_size=6)

    def test_actual_factor_first_and_all_nonunit_factors_are_training_queries(self):
        self.assertEqual(runner.FACTORS[0], 1)
        self.assertTrue(all(factor != 1 for factor in runner.FACTORS[1:]))
        bank = self.bank()
        part = bank.parts["train"]
        torch.testing.assert_close(part["classes"][:, 0], torch.tensor([0, 1, 2, 3] * 3))
        actual = bank.features(part, torch.arange(3))
        synthetic = bank.features(part, torch.arange(3), torch.ones(3, dtype=torch.long))
        torch.testing.assert_close(actual[:, :30], synthetic[:, :30], rtol=0, atol=0)
        self.assertTrue(torch.all(actual[:, -1, 0] != synthetic[:, -1, 0]))

    def test_target_high_low_close_cannot_change_features(self):
        dataset, splits = self.dataset()
        original = runner.DeepBank(dataset, splits, "cpu")
        modified = copy.deepcopy(dataset)
        modified.target_ohlc[:, 1] *= 2
        modified.target_ohlc[:, 2] *= .5
        modified.target_ohlc[:, 3] = modified.target_ohlc[:, 0]
        changed = runner.DeepBank(modified, splits, "cpu")
        index = torch.arange(12)
        torch.testing.assert_close(original.features(original.parts["train"], index),
                                   changed.features(changed.parts["train"], index), rtol=0, atol=0)
        self.assertFalse(torch.equal(original.parts["train"]["classes"], changed.parts["train"]["classes"]))

    def test_stable_binary_loss(self):
        logits = np.asarray([-1000., 1000., 0., 1.])
        labels = np.asarray([0., 1., 1., 0.])
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(logits), torch.tensor(labels)).item()
        self.assertAlmostEqual(runner.binary_loss(logits, labels), expected, places=12)

    def test_primary_actual_open_loss_is_not_diluted_by_synthetic_batch(self):
        bank = self.bank()
        initial = torch.tensor([.1, -.2, .3, -.4])

        class ConstantClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(initial.clone())

            def forward(self, values):
                return self.value[None].expand(len(values), -1)

        count = len(bank.parts["train"]["indices"])
        output = initial[None].expand(count, -1)
        classes = bank.parts["train"]["classes"][:, 0]
        augmented_labels = (bank.parts["train"]["classes"][:, 1] == 0).float()
        expected = (torch.nn.functional.binary_cross_entropy_with_logits(
            runner.success_logit(output), (classes == 0).float())
            + .25 * torch.nn.functional.cross_entropy(output, classes)
            + .1 * torch.nn.functional.binary_cross_entropy_with_logits(
                runner.success_logit(output), augmented_labels))
        folder = self.root / "loss"
        with patch.object(runner, "build_model", return_value=ConstantClassifier()):
            with patch.object(torch, "randint", return_value=torch.ones(count, dtype=torch.long)):
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.train_trial(bank, "mlp_deep", 42, folder, self.context,
                                       epochs=1, batch_size=count)
        history = json.loads((folder / "history.json").read_text("utf-8"))
        self.assertAlmostEqual(history[0]["training_loss"], expected.item(), places=6)

    def test_tiny_trial_produces_research_only_artifacts_and_correct_indices(self):
        folder = self.root / "trial"
        result = self.train(folder, epochs=2)
        self.assertEqual(result["epochs"], 2)
        self.assertEqual(result["selection"]["count"], 16)
        self.assertFalse(result["qualification"]["qualified"])
        checkpoint = torch.load(folder / "model.pt", weights_only=False, map_location="cpu")
        self.assertTrue(checkpoint["research_only"])
        self.assertFalse(checkpoint["intraday_path_verified"])
        self.assertEqual(checkpoint["threshold"], .5)
        self.assertEqual(checkpoint["take_profit_pct"], 1.)
        self.assertEqual(checkpoint["stop_loss_pct"], .9)
        self.assertEqual(checkpoint["class_names"], runner.CLASS_NAMES)
        with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["calibration_indices"], np.arange(20, 36))
            np.testing.assert_array_equal(archive["selection_indices"], np.arange(36, 52))
            self.assertTrue(np.isfinite(archive["selection_logits"]).all())
        self.assertFalse(list(folder.glob("*.tmp")))

    def test_interrupted_epoch_resume_restores_model_optimizer_and_rng(self):
        baseline_folder, resumed_folder = self.root / "baseline", self.root / "resumed"
        expected = self.train(baseline_folder)
        original_atomic = runner.atomic_torch

        def save_then_interrupt(path, value):
            original_atomic(path, value)
            if Path(path).name == "resume.pt" and len(value["history"]) == 1:
                raise InterruptedError("Simulate interruption just after a complete epoch checkpoint")

        with patch.object(runner, "atomic_torch", side_effect=save_then_interrupt):
            with self.assertRaises(InterruptedError):
                self.train(resumed_folder)
        restored = self.train(resumed_folder)
        self.assertEqual(restored["epochs"], expected["epochs"])
        self.assertEqual(restored["best_epoch"], expected["best_epoch"])
        self.assertEqual(restored["best_tune_loss"], expected["best_tune_loss"])
        first = torch.load(baseline_folder / "resume.pt", weights_only=False, map_location="cpu")
        second = torch.load(resumed_folder / "resume.pt", weights_only=False, map_location="cpu")
        for key in first["model"]:
            torch.testing.assert_close(first["model"][key], second["model"][key], rtol=0, atol=0)
        for key, values in first["optimizer"]["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(value, second["optimizer"]["state"][key][name], rtol=0, atol=0)
        torch.testing.assert_close(first["cpu_rng"], second["cpu_rng"], rtol=0, atol=0)
        with np.load(baseline_folder / "predictions.npz") as expected_archive:
            with np.load(resumed_folder / "predictions.npz") as actual_archive:
                np.testing.assert_array_equal(expected_archive["probabilities"], actual_archive["probabilities"])

    def test_complete_trial_is_reused_without_retraining(self):
        folder = self.root / "complete"
        first = self.train(folder, epochs=1)
        with patch.object(runner, "build_model", side_effect=AssertionError("Must reuse completed trial")):
            second = self.train(folder, epochs=1)
        self.assertEqual(first, second)

    def test_contract_mismatch_rejects_reuse(self):
        folder = self.root / "complete"
        self.train(folder, epochs=1)
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.train(folder, epochs=1, seed=43)
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.train(folder, epochs=2)

    def test_no_synthetic_queries_in_logit_evaluation(self):
        bank = self.bank()
        model = runner.build_model("mlp_deep", **runner.MODEL_CONFIG)
        actual = bank.features
        calls = []

        def inspect_features(part, index, factor_indices=None):
            calls.append(factor_indices)
            return actual(part, index, factor_indices)

        with patch.object(bank, "features", side_effect=inspect_features):
            logits = bank.logits(model, "selection", batch_size=5)
        self.assertEqual(logits.shape, (16,))
        self.assertTrue(calls)
        self.assertTrue(all(value is None for value in calls))

    def test_atomic_json_preserves_previous_valid_artifact_on_bad_value(self):
        path = self.root / "result.json"
        runner.atomic_json(path, {"valid": 1})
        with self.assertRaises(ValueError):
            runner.atomic_json(path, {"invalid": float("nan")})
        self.assertEqual(json.loads(path.read_text("utf-8")), {"valid": 1})


if __name__ == "__main__":
    unittest.main()
