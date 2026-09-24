"""Small CPU-only trainer wiring tests; no broker, real DB or GPU training."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    ML_AVAILABLE = False
else:
    ML_AVAILABLE = True
    from dockdack.mark1_data import Mark1Dataset, FEATURE_NAMES, TARGET, barrier_outcomes
    from examples import train_mark1 as trainer


if ML_AVAILABLE:
    class TinyNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, features):
            return self.weight * features[:, -1, 0] + self.bias


@unittest.skipUnless(ML_AVAILABLE, "Install the ml extra")
class TrainMark1Tests(unittest.TestCase):
    def dataset(self):
        # Separate storage prevents any accidental cross-example adjacency.
        rows, target_ohlc = [], []
        for index in range(10):
            base = 100. + index * 10
            history = np.tile([base, base * 1.02, base * .98, base, 100000], (31, 1))
            # Alternating conservative positives and both-touch negatives.
            target = [base, base * 1.015, base * (.995 if index % 2 == 0 else .98), base]
            history[-1, :4] = target
            rows.append(history)
            target_ohlc.append(target)
        return Mark1Dataset(
            bars=np.concatenate(rows).astype(np.float32), starts=np.arange(10, dtype=np.int64) * 31,
            target_dates=np.asarray([18000 + i * 365 for i in range(10)], dtype=np.int32),
            symbol_ids=np.arange(10, dtype=np.int32), target_ohlc=np.asarray(target_ohlc, dtype=np.float64),
            splits={name: np.arange(i * 2, i * 2 + 2, dtype=np.int64)
                    for i, name in enumerate(trainer.SPLITS)},
            manifest={"fixture": True, "database": "fixture.sqlite3"})

    def args(self, **overrides):
        values = dict(max_train_samples=2, max_eval_samples=2, seed=42,
                      epochs=1, patience=2, batch_size=4, cost_bps=20, device="cpu")
        values.update(overrides)
        return SimpleNamespace(**values)

    def calibration(self):
        return {"method": "platt_monotone", "slope": 1.0, "bias": 0.0}

    def test_bank_features_never_include_the_target_bar_and_preserve_query_price(self):
        dataset = self.dataset()
        bank = trainer.BatchBank(dataset, "cpu")
        part = bank.parts["train"]
        indices = torch.arange(2)
        before = bank.features(part, indices).clone()
        bank.bars[part["starts"] + 30] = torch.tensor([1., 1e6, .001, 10000., 1e9])
        after = bank.features(part, indices)
        torch.testing.assert_close(before, after)
        entries = part["entries"] * 1.005
        query = bank.features(part, indices, entries)
        torch.testing.assert_close(query[:, :30], before[:, :30])
        torch.testing.assert_close(query[:, 30, 0], torch.full((2,), float(np.log(1.005))), atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.equal(query[:, 30, 1:8], torch.zeros(2, 7)))
        for name in trainer.SPLITS:
            np.testing.assert_array_equal(bank.parts[name]["outcomes"]["success"], [True, False])

    def test_controlled_augmentation_slot_mapping_and_label_agreement(self):
        bank = trainer.BatchBank(self.dataset(), "cpu")
        part = bank.parts["train"]
        entries64 = part["ohlc"][:, :1] * np.asarray(trainer.FACTORS)[None, :]
        expected = barrier_outcomes(part["ohlc"][:, 1:2], part["ohlc"][:, 2:3],
                                    part["ohlc"][:, 3:4], entries64)["success"].reshape(-1)
        slots = torch.tensor([9, 0, 5, 3, 1, 8, 4, 7, 2, 6])
        bases = torch.div(slots, len(trainer.FACTORS), rounding_mode="floor")
        entries = torch.tensor(entries64.reshape(-1), dtype=torch.float32)[slots]
        features = bank.features(part, bases, entries)
        for row, slot in enumerate(slots.tolist()):
            base, factor_index = divmod(slot, 5)
            self.assertAlmostEqual(float(features[row, -1, 0]), np.log(trainer.FACTORS[factor_index]), places=6)
            expected_one = barrier_outcomes(*part["ohlc"][base, [1, 2, 3]], entries64[base, factor_index])
            self.assertEqual(bool(expected[slot]), bool(expected_one["success"]))

    def test_checkpoint_metadata_carries_exact_strategy_contract(self):
        metadata = trainer.checkpoint_metadata("domestic", "mlp", "mlp", self.calibration(),
                                               {"sha": "fixture"}, self.args())
        self.assertEqual(metadata["strategy_version"], "mark_1")
        self.assertEqual(metadata["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(metadata["target"], TARGET)
        self.assertEqual((metadata["lookback"], metadata["sequence_length"]), (30, 31))
        self.assertEqual((metadata["buy_threshold"], metadata["take_profit_pct"], metadata["stop_loss_pct"]),
                         (.5, 1., .9))
        self.assertIs(metadata["intraday_path_verified"], False)
        self.assertIn("hypothetical", metadata["entry_context"])
        self.assertTrue(metadata["limitations"])

    def test_train_variant_only_tunes_then_calibrates_then_selects_never_tests(self):
        bank = trainer.BatchBank(self.dataset(), "cpu")
        calls, calibration_calls = [], []
        logits_by_split = {"tune": np.array([.1, -.1]), "calibration": np.array([.7, -.7]),
                           "selection": np.array([1.2, -1.2])}

        def logits(model, split, batch_size):
            calls.append(split)
            if split == "test":
                self.fail("Final test cannot participate in a training variant")
            return logits_by_split[split]

        def calibrate(logits, labels):
            calibration_calls.append((logits.copy(), labels.copy()))
            return self.calibration()

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch.object(trainer, "build_model", side_effect=lambda *a, **k: TinyNetwork()), \
                    patch.object(bank, "logits", side_effect=logits), \
                    patch.object(trainer, "fit_calibration", side_effect=calibrate), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                result = trainer.train_variant(bank, "domestic", "mlp", Path(directory) / "model", {}, self.args())
            self.assertEqual(calls, ["tune", "calibration", "selection"])
            np.testing.assert_array_equal(calibration_calls[0][0], logits_by_split["calibration"])
            np.testing.assert_array_equal(calibration_calls[0][1], bank.parts["calibration"]["outcomes"]["success"])
            self.assertEqual(result["training_presentations_per_epoch"], 10)
            self.assertEqual(result["unique_training_base_events"], 2)
            self.assertNotIn("test", result)
            payload = torch.load(Path(directory) / "model" / "model.pt", weights_only=True)
            self.assertEqual(payload["metadata"]["calibration"], self.calibration())
            with np.load(Path(directory) / "model" / "selection_predictions.npz", allow_pickle=False) as saved:
                self.assertEqual(len(saved["probabilities"]), 2)  # No evaluation price expansion.

    def test_no_augmentation_control_keeps_equal_presentations_not_new_observations(self):
        bank = trainer.BatchBank(self.dataset(), "cpu")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch.object(trainer, "build_model", side_effect=lambda *a, **k: TinyNetwork()), \
                    patch.object(trainer, "fit_calibration", return_value=self.calibration()), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                result = trainer.train_variant(bank, "us", "mlp_no_price_aug", Path(directory) / "control", {}, self.args())
        self.assertEqual(result["architecture"], "mlp")
        self.assertEqual(result["augmentation_factors"], [1.] * 5)
        self.assertEqual(result["training_presentations_per_epoch"], 10)
        self.assertEqual(result["unique_training_base_events"], 2)

    def test_actual_optimizer_batches_pair_each_synthetic_query_with_its_own_label(self):
        bank = trainer.BatchBank(self.dataset(), "cpu")
        original_features = bank.features
        current, seen = {}, []

        def recording_features(part, indices, entries=None):
            if part is bank.parts["train"] and entries is not None:
                current["bases"] = indices.cpu().numpy().copy()
                current["entries"] = entries.cpu().numpy().copy()
            return original_features(part, indices, entries)

        class RecordingLoss(torch.nn.Module):
            def forward(loss_self, logits, labels):
                bases, entries = current["bases"], current["entries"]
                original = bank.parts["train"]["ohlc"][bases]
                expected = barrier_outcomes(original[:, 1], original[:, 2], original[:, 3], entries)["success"]
                np.testing.assert_array_equal(labels.detach().cpu().numpy().astype(bool), expected)
                seen.extend(zip(bases.tolist(), entries.tolist()))
                return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch.object(trainer, "build_model", side_effect=lambda *a, **k: TinyNetwork()), \
                    patch.object(bank, "features", side_effect=recording_features), \
                    patch.object(trainer.nn, "BCEWithLogitsLoss", side_effect=RecordingLoss), \
                    patch.object(trainer, "fit_calibration", return_value=self.calibration()), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                trainer.train_variant(bank, "domestic", "mlp", Path(directory) / "paired", {}, self.args())
        self.assertEqual(len(seen), 10)
        for base in range(2):
            actual_entries = sorted(entry for source, entry in seen if source == base)
            expected_entries = bank.parts["train"]["ohlc"][base, 0] * np.asarray(trainer.FACTORS)
            np.testing.assert_allclose(actual_entries, expected_entries, rtol=1e-6)

    def test_selection_is_locked_before_test_and_does_not_follow_better_test_model(self):
        dataset = self.dataset()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            args = self.args(output_dir=folder, db_dir=folder, cache_dir=folder / "cache", models=["mlp", "gru"])

            def fake_train(bank, market, variant, output, data_config, actual_args):
                output.mkdir()
                model = TinyNetwork()
                with torch.no_grad():
                    model.bias.fill_(0 if variant == "mlp" else 1)
                torch.save({"model_state_dict": model.state_dict()}, output / "model.pt")
                return {"variant": variant, "architecture": variant, "calibration": self.calibration(),
                        "selection": {"brier": .1 if variant == "mlp" else .2, "log_loss": .5}}

            def fake_logits(bank, model, split, batch_size):
                self.assertEqual(split, "test")
                lock = folder / "domestic" / "selection_locked.json"
                self.assertTrue(lock.exists())
                self.assertEqual(json.loads(lock.read_text())["winner"], "mlp")
                return np.array([-8., 8.]) if model.bias.item() == 0 else np.array([8., -8.])

            with patch.object(trainer, "dataset_cache", return_value=(dataset, {})), \
                    patch.object(trainer, "train_variant", side_effect=fake_train), \
                    patch.object(trainer, "build_model", side_effect=lambda *a, **k: TinyNetwork()), \
                    patch.object(trainer.BatchBank, "logits", new=fake_logits), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                result = trainer.run_market("domestic", args)
            self.assertEqual(result["winner"], "mlp")
            self.assertGreater(result["results"][0]["test"]["brier"], result["results"][1]["test"]["brier"])

    def test_cache_roundtrip_is_readonly_and_binds_exact_source_path(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            source = folder / "source.sqlite3"
            source.write_bytes(b"fixture immutable bytes")
            before = source.read_bytes()
            with patch.object(trainer, "load_dataset", return_value=self.dataset()) as loader:
                original, contract = trainer.dataset_cache(source, "domestic", folder / "cache", self.args())
                loaded, second_contract = trainer.dataset_cache(source, "domestic", folder / "cache", self.args())
            loader.assert_called_once()
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(contract, second_contract)
            self.assertEqual(contract["database_path"], str(source.resolve()))
            self.assertEqual(contract["database_sha256"], trainer.file_hash(source))
            np.testing.assert_array_equal(original.bars, loaded.bars)
            np.testing.assert_array_equal(original.target_ohlc, loaded.target_ohlc)
            for name in trainer.SPLITS:
                np.testing.assert_array_equal(original.splits[name], loaded.splits[name])
            copy = folder / "copy.sqlite3"
            copy.write_bytes(before)
            with patch.object(trainer, "load_dataset", return_value=self.dataset()) as loader:
                _, copied_contract = trainer.dataset_cache(copy, "domestic", folder / "cache", self.args())
            loader.assert_called_once()
            self.assertNotEqual(contract["database_path"], copied_contract["database_path"])
            self.assertEqual(len(list((folder / "cache").glob("*.npz"))), 2)

    def test_cache_rejects_nonempty_wal_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.sqlite3"
            source.write_bytes(b"fixture")
            Path(str(source) + "-wal").write_bytes(b"pending writes")
            with patch.object(trainer, "load_dataset") as loader, self.assertRaisesRegex(ValueError, "WAL"):
                trainer.dataset_cache(source, "domestic", Path(directory) / "cache", self.args())
            loader.assert_not_called()

    def test_cache_rejects_source_change_during_fresh_load(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.sqlite3"
            source.write_bytes(b"fixture")
            with patch.object(trainer, "load_dataset", return_value=self.dataset()), \
                    patch.object(trainer, "file_hash", side_effect=["before", "after"]), \
                    self.assertRaisesRegex(RuntimeError, "changed"):
                trainer.dataset_cache(source, "domestic", Path(directory) / "cache", self.args())
            self.assertEqual(list((Path(directory) / "cache").glob("*.npz")), [])

    def test_cache_hit_rechecks_source_hash(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source = Path(directory) / "source.sqlite3"
            source.write_bytes(b"fixture")
            cache = Path(directory) / "cache"
            with patch.object(trainer, "load_dataset", return_value=self.dataset()):
                trainer.dataset_cache(source, "domestic", cache, self.args())
            actual_hash = trainer.file_hash(source)
            with patch.object(trainer, "file_hash", side_effect=[actual_hash, "changed"]), \
                    self.assertRaisesRegex(RuntimeError, "changed"):
                trainer.dataset_cache(source, "domestic", cache, self.args())

    def test_cache_contract_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source = Path(directory) / "source.sqlite3"
            source.write_bytes(b"fixture")
            cache = Path(directory) / "cache"
            with patch.object(trainer, "load_dataset", return_value=self.dataset()):
                trainer.dataset_cache(source, "domestic", cache, self.args())
            path = next(cache.glob("*.npz"))
            with np.load(path, allow_pickle=False) as data:
                changed = {key: data[key] for key in data.files}
            changed["cache_config"] = json.dumps({"wrong": True})
            np.savez(path, **changed)
            with self.assertRaisesRegex(ValueError, "contract"):
                trainer.dataset_cache(source, "domestic", cache, self.args())


if __name__ == "__main__":
    unittest.main()
