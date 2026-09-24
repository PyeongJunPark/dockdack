"""CPU-only synthetic tests: no real evaluation data or source DB is read."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    ML_AVAILABLE = False
else:
    ML_AVAILABLE = True
    from dockdack.mark1_data import Mark1Dataset, barrier_outcomes
    from examples import backtest_mark1_deep as runner


@unittest.skipUnless(ML_AVAILABLE, "requires optional ML dependencies")
class DeepBacktestRunnerTests(unittest.TestCase):
    @staticmethod
    def dataset():
        dates = np.array(["2013-06-01", "2014-06-01", "2021-06-01", "2025-02-10",
                          "2025-02-11", "2025-02-12", "2025-02-12"],
                         dtype="datetime64[D]").astype(np.int32)
        return Mark1Dataset(
            bars=np.tile(np.array([100, 102, 98, 100, 100000], np.float32), (250, 1)),
            starts=np.arange(len(dates), dtype=np.int64) * 31,
            target_dates=dates, symbol_ids=np.array([1, 0, 2, 0, 0, 0, 1], np.int32),
            target_ohlc=np.array([[100, 102, 100, 101]] * 4 +
                                 [[100, 100.5, 99.8, 100.2], [100, 100.5, 98, 99],
                                  [100, 102, 100, 101]], np.float64),
            splits={"train": np.array([2]), "tune": np.array([2]),
                    "calibration": np.array([2]), "selection": np.array([2]),
                    "test": np.array([5, 6])},
            manifest={"symbols": [{"symbol_id": index, "symbol": f"SYN{index}", "exchange": "KRX"}
                                   for index in range(3)]})

    @staticmethod
    def calendar():
        return np.arange(np.datetime64("2010-01-01", "D"), np.datetime64("2026-01-01", "D")).astype(np.int64)

    @staticmethod
    def calibration():
        return {"method": "platt_monotone", "slope": 1., "bias": 0.,
                "weighted": False, "fit_samples": 20}

    def checkpoint(self, contract):
        model = runner.build_model("mlp_deep", **runner.MODEL_CONFIG)
        return {"state_dict": model.state_dict(), "architecture": "mlp_deep",
                "model_config": dict(runner.MODEL_CONFIG), "target": runner.TARGET,
                "feature_names": runner.FEATURE_NAMES, "class_names": runner.CLASS_NAMES,
                "context": {"market": "domestic", "fold": "walk_2024", "source": contract},
                "seed": 42, "research_only": True, "intraday_path_verified": False,
                "threshold": .5, "take_profit_pct": 1., "stop_loss_pct": .9,
                "calibration": self.calibration(), "protocol": runner.PROTOCOL}

    def test_common_membership_uses_uncapped_training_universe(self):
        dataset = self.dataset()
        before = dataset.splits["test"].copy()
        indices = runner.common_test_indices(dataset, self.calendar())
        # Symbol0 was not in capped training indices, but DID exist in full2014 training.
        np.testing.assert_array_equal(indices, [3, 4, 5])
        np.testing.assert_array_equal(dataset.splits["test"], before)
        self.assertFalse(indices.flags.writeable)

    def test_original_purge_and_calendar_membership_required(self):
        dataset = self.dataset()
        dataset.target_dates[3] = int(np.datetime64("2025-01-20", "D").astype(int))
        with self.assertRaisesRegex(ValueError, "purge"):
            runner.common_test_indices(dataset, self.calendar())
        dataset = self.dataset()
        calendar = self.calendar()
        calendar = calendar[calendar != dataset.target_dates[3]]
        with self.assertRaisesRegex(ValueError, "calendar"):
            runner.common_test_indices(dataset, calendar)

    def test_duplicate_or_missing_universe_rejected(self):
        dataset = self.dataset()
        dataset.target_dates[4] = dataset.target_dates[3]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            runner.common_test_indices(dataset, self.calendar())
        dataset = self.dataset()
        dataset.target_dates[:3] = int(np.datetime64("2013-06-01", "D").astype(int))
        with self.assertRaisesRegex(ValueError, "training universe"):
            runner.common_test_indices(dataset, self.calendar())

    def test_missing_cache_is_not_rebuilt(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "dataset_cache") as mocked:
            with self.assertRaisesRegex(FileNotFoundError, "read-only"):
                runner.load_frozen_cache({"fixture": True}, "domestic", Path(temporary))
            mocked.assert_not_called()

    def test_deep_checkpoint_roundtrip_cpu(self):
        contract = {"database_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            torch.save(self.checkpoint(contract), path)
            model, metadata = runner.load_deep_checkpoint(path, market="domestic", architecture="mlp_deep",
                                                         seed=42, source_contract=contract, device="cpu")
            self.assertFalse(model.training)
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(model(torch.zeros((2, 31, 18))).shape, (2, 4))

    def test_incompatible_checkpoint_contract_rejected(self):
        contract = {"database_sha256": "a" * 64}
        payload = self.checkpoint(contract)
        invalid = [("target", "next_close"), ("feature_names", list(runner.FEATURE_NAMES[:-1])),
                   ("class_names", ["positive", "negative"]), ("seed", 43),
                   ("research_only", False), ("threshold", .4), ("stop_loss_pct", .8),
                   ("intraday_path_verified", True), ("model_config", {}), ("protocol", {}),
                   ("calibration", {**self.calibration(), "weighted": True}),
                   ("context", {"market": "us", "fold": "walk_2024", "source": contract})]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            for name, value in invalid:
                with self.subTest(field=name):
                    changed = {**payload, name: value}
                    torch.save(changed, path)
                    with self.assertRaises(ValueError):
                        runner.load_deep_checkpoint(path, market="domestic", architecture="mlp_deep",
                                                    seed=42, source_contract=contract, device="cpu")

    def test_bad_checkpoint_weights_rejected(self):
        contract = {"database_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            for mode in ("nan", "missing", "shape"):
                payload = self.checkpoint(contract)
                key = next(iter(payload["state_dict"]))
                if mode == "nan":
                    payload["state_dict"][key].flatten()[0] = float("nan")
                elif mode == "missing":
                    del payload["state_dict"][key]
                else:
                    payload["state_dict"][key] = torch.zeros(1)
                torch.save(payload, path)
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    runner.load_deep_checkpoint(path, market="domestic", architecture="mlp_deep",
                                                seed=42, source_contract=contract, device="cpu")

    def test_incomplete_training_fails_before_data_access(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "load_frozen_cache") as cache:
            args = SimpleNamespace(training_run=Path(temporary))
            with self.assertRaises(FileNotFoundError):
                runner.run_market("domestic", args)
            cache.assert_not_called()

    def test_artifact_contract_matches_lock_source_and_frozen_code(self):
        repo = Path(runner.__file__).resolve().parents[1]
        names = ("examples/train_mark1_deep.py", "dockdack/mark1_deep_data.py",
                 "dockdack/mark1_deep_models.py", "dockdack/mark1_deep_validation.py",
                 "dockdack/mark1_data.py", "dockdack/mark1_metrics.py", "examples/train_mark1.py")
        ranking = [{"architecture": "mlp_deep", "score": .1}]
        summary = {"market": "domestic", "selected": "mlp_deep", "ranking": ranking,
                   "protocol": runner.PROTOCOL, "ensemble_calibration": self.calibration()}
        source = {"market": "domestic", "target": runner.TARGET, "version": 2,
                  "purge_sessions": 30, "start": "2010-01-01", "seed": 42,
                  "max_train_samples": 200000, "max_eval_samples": 60000,
                  "database_path": "fixture.sqlite3", "database_sha256": "a" * 64}
        frozen = {"protocol": runner.PROTOCOL,
                  "code_sha256": {Path(name).name: runner.file_hash(repo / name) for name in names}}
        locked = {"market": "domestic", "selected": "mlp_deep", "ranking": ranking,
                  "selection_rule": runner.PROTOCOL["architecture_selection"]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "domestic").mkdir()
            values = {"summary.json": {"domestic": summary, "us": {"market": "us"}},
                      "protocol.json": frozen, "domestic/summary.json": summary,
                      "domestic/source.json": source, "domestic/selection_locked.json": locked}
            for name, value in values.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            actual, contract, hashes = runner.verify_training_artifacts(root, "domestic")
            self.assertEqual(actual["selected"], "mlp_deep")
            self.assertEqual(contract, source)
            self.assertEqual(len(hashes), 5)
            # An invalid lock must never become a different post-test winner.
            changed = {**locked, "selected": "resnet34"}
            (root / "domestic/selection_locked.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection mismatch"):
                runner.verify_training_artifacts(root, "domestic")
            (root / "domestic/selection_locked.json").write_text(json.dumps(locked), encoding="utf-8")
            changed = copy.deepcopy(frozen)
            changed["code_sha256"]["mark1_deep_models.py"] = "0" * 64
            (root / "protocol.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "code changed"):
                runner.verify_training_artifacts(root, "domestic")

    def test_mean_logits_then_frozen_calibration_and_original_baseline(self):
        dataset = self.dataset()
        indices = np.array([3, 4, 5])
        contract = {"database_sha256": "a" * 64}
        seed_logits = {42: np.array([-2., 2., 1.]), 43: np.array([0., 2., 3.]),
                       44: np.array([2., 2., -1.])}
        observed = []

        class FakeDeepBank:
            def __init__(self, incoming, splits, device):
                observed.append(splits["test"].copy())
                ohlc = incoming.target_ohlc[splits["test"]]
                self.parts = {"test": {"dates": incoming.target_dates[splits["test"]],
                                       "outcomes": barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], ohlc[:, 0])}}

            def logits(self, model, split, batch_size):
                return seed_logits[model]

        class FakeOldBank:
            def __init__(self, incoming, device):
                observed.append(incoming.splits["test"].copy())

            def logits(self, model, split, batch_size):
                return np.array([0., 1., -1.])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in runner.SEEDS:
                folder = root / "domestic/walk_2024" / f"mlp_deep-{seed}"
                folder.mkdir(parents=True)
                (folder / "model.pt").write_bytes(str(seed).encode())
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            baseline = baseline_dir / "domestic.pt"
            baseline.write_bytes(b"unchanged-original")
            digest = runner.file_hash(baseline)
            args = SimpleNamespace(training_run=root, baseline_dir=baseline_dir, device="cpu", batch_size=8)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(runner, "DeepBank", FakeDeepBank))
                stack.enter_context(patch.object(runner, "BatchBank", FakeOldBank))
                stack.enter_context(patch.object(runner, "load_deep_checkpoint", side_effect=lambda path, **kw: (kw["seed"], {})))
                stack.enter_context(patch.object(runner, "Predictor", return_value=SimpleNamespace(
                    market="domestic", model=object(), metadata={"variant": "mlp_no_price_aug",
                    "dataset": contract, "calibration": self.calibration()})))
                stack.enter_context(patch.dict(runner.ORIGINAL_CHECKPOINT_SHA256, {"domestic": digest}))
                results, outcomes, dates, hashes = runner.infer_common(
                    dataset, indices, "domestic", args, {"selected": "mlp_deep", "ensemble_calibration": self.calibration()}, contract)
            expected = runner.calibrated_probability(np.mean(list(seed_logits.values()), axis=0), self.calibration())
            np.testing.assert_allclose(results["deep"], expected)
            self.assertEqual(results["deep"][0], .5)
            np.testing.assert_array_equal(observed[0], observed[1])
            np.testing.assert_array_equal(dataset.splits["test"], [5, 6])
            self.assertEqual(len(hashes), 4)

    def test_full_runner_outputs_same_sample_comparison_and_all_costs(self):
        dataset = self.dataset()
        indices = np.array([3, 4, 5])
        ohlc = dataset.target_ohlc[indices]
        outcomes = barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], ohlc[:, 0])
        probabilities = {"baseline": np.array([.8, .5, .7]), "deep": np.array([.5, .9, .8])}
        calls = []
        actual_simulate = runner.simulate_portfolio
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.sqlite3"
            source.write_bytes(b"synthetic source never opened as database")
            contract = {"database_path": str(source), "database_sha256": runner.file_hash(source)}
            model = root / "frozen.pt"
            model.write_bytes(b"synthetic frozen weights")
            checkpoint_hashes = {str(model): runner.file_hash(model)}
            prices = {(0, int(dataset.target_dates[index])): tuple(dataset.target_ohlc[index]) for index in indices}
            zero_key = (0, int(dataset.target_dates[indices[-1]]) + 1)
            prices[zero_key] = (100, 200, 1, 100)
            training = {"selected": "mlp_deep", "ensemble_calibration": self.calibration(), "research_qualified": False}
            args = SimpleNamespace(training_run=root / "training", output_dir=root / "output", cache_dir=root / "cache",
                                   device="cpu", initial_krw=10_000_000, initial_usd=10_000,
                                   max_positions=20, position_fraction=.05, volume_fraction=.001, batch_size=8)

            def checked_simulate(candidates, supplied_prices, sessions, **kwargs):
                calls.append(kwargs)
                self.assertNotIn(zero_key, supplied_prices)
                return actual_simulate(candidates, supplied_prices, sessions, **kwargs)

            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(runner, "verify_training_artifacts", return_value=(training, contract, {})))
                stack.enter_context(patch.object(runner, "load_frozen_cache", return_value=dataset))
                stack.enter_context(patch.object(runner, "read_sessions", return_value=self.calendar()))
                stack.enter_context(patch.object(runner, "load_price_panel", return_value=(
                    prices, dataset.target_dates[indices].tolist(), {"zero_volume_keys": [list(zero_key)]})))
                inference = stack.enter_context(patch.object(runner, "infer_common", return_value=(
                    probabilities, outcomes, dataset.target_dates[indices], checkpoint_hashes)))
                stack.enter_context(patch.object(runner, "simulate_portfolio", side_effect=checked_simulate))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                summary = runner.run_market("domestic", args)
            np.testing.assert_array_equal(inference.call_args.args[1], indices)
            self.assertTrue(summary["completed"])
            self.assertEqual(summary["research_evaluation"], runner.EVALUATION_STATUS)
            self.assertEqual(summary["samples"], 3)
            self.assertEqual(set(summary["models"]), {"baseline", "deep"})
            self.assertEqual(len(summary["cost_sensitivity"]), 16)
            self.assertEqual(len(calls), 16)
            self.assertEqual({item["cost_bps"] for item in calls}, {0, 10, 20, 40})
            for name in ("baseline", "deep"):
                self.assertEqual(summary["models"][name]["raw_signals"], 2)
                for mode in ("carry", "eod"):
                    ledger = json.loads((args.output_dir / "domestic" / f"{name}-{mode}.json").read_text("utf-8"))
                    self.assertEqual(ledger["cost_bps"], 20)
                    self.assertEqual(ledger["research_evaluation"], runner.EVALUATION_STATUS)
                    self.assertTrue(all("symbol" in trade for trade in ledger["trades"]))
            with np.load(args.output_dir / "domestic/predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["sample_indices"], indices)
                np.testing.assert_array_equal(saved["baseline_probabilities"], probabilities["baseline"])
                self.assertEqual(saved["evaluation_status"].item(), runner.EVALUATION_STATUS)
            self.assertEqual(runner.file_hash(source), contract["database_sha256"])


if __name__ == "__main__":
    unittest.main()
