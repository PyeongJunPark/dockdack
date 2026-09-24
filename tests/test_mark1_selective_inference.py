"""Synthetic artifact-contract/CPU inference checks; no DB, GPU or orders."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch"))
if AVAILABLE:
    import numpy as np
    from dockdack import mark1_selective_inference as inference
    from dockdack.mark1_selective_models import CLASS_NAMES, OWNER, SCHEMA_VERSION, model_path
    from dockdack.mark1_selective_policy import evaluate_signals, qualification
    from examples.train_mark1_selective import CODE_FILES, PROTOCOL


@unittest.skipUnless(AVAILABLE, "optional numpy and torch required")
class SelectiveInferenceTests(unittest.TestCase):
    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    @staticmethod
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def history():
        close = np.linspace(95., 105., 30)
        return np.column_stack((close, close * 1.01, close * .995, close * 1.002,
                                np.full(30, 10000.)))

    def fixture(self, root, name="cat_binary6", threshold=.65, cap=1.):
        frozen = {
            "protocol": copy.deepcopy(PROTOCOL), "task_type": "GPU", "threads": 12,
            "versions": {"numpy": np.__version__, "catboost": "1.2.10", "lightgbm": "4.6.0"},
            "code_sha256": {file: self.sha(inference.ROOT / file) for file in CODE_FILES},
            "db_dir": str(root / "db"), "cache_dir": str(root / "cache"),
        }
        self.write(root / "protocol.json", frozen)
        calibration = {"method": "platt_monotone", "slope": 1., "bias": 0.,
                       "fit_samples": 10, "weighted": False, "iterations": 2,
                       "regularization": 1e-6, "fit_log_loss": .6}
        metrics = evaluate_signals(np.arange(10) % 2, np.full(10, .4), np.zeros(10),
                                  np.arange(10), np.arange(10), np.zeros(10, dtype=bool))
        gate = qualification(metrics)
        policy = {
            "chosen_policy": {"threshold": threshold, "stop_probability_cap": cap},
            "chosen_metrics": metrics, "qualification": gate, "calibration_qualified": False,
            "research_only": True, "deployment_allowed": False,
        }
        summaries = {}
        for market in ("domestic", "us"):
            folder = root / market
            source = {"market": market, "target": PROTOCOL["target"], "version": 2,
                      "database_path": str(root / "db" / f"{market}_daily_clean.sqlite3"),
                      "database_sha256": "a" * 64, "purge_sessions": 30}
            self.write(folder / "source.json", source)
            ranking = [{"architecture": name}]
            self.write(folder / "selection_locked.json", {
                "market": market, "selected": name, "ranking": ranking, "source": source,
                "selection_rule": PROTOCOL["selection"],
            })
            ensembles = {}
            for fold in PROTOCOL["folds"]:
                parts = ("train", "tune", "probability_calibration", "policy_calibration", "audit")
                spec = PROTOCOL["folds"][fold]
                dates = {
                    "train": (spec["train_start"], spec["train_end"]),
                    "tune": (f"{spec['tune_year']}-03-01", f"{spec['tune_year']}-12-01"),
                    "probability_calibration": (f"{spec['calibration_year']}-03-01", f"{spec['calibration_year']}-06-01"),
                    "policy_calibration": (f"{spec['calibration_year']}-09-01", f"{spec['calibration_year']}-12-01"),
                    "audit": (f"{spec['selection_year']}-03-01", f"{spec['selection_year']}-12-01"),
                }
                splits = {part: {"count": 10, "symbols": 3, "first": dates[part][0], "last": dates[part][1]}
                          for part in parts}
                self.write(folder / fold / "splits.json", splits)
                self.write(folder / fold / "features" / "features.json", {
                    "splits": splits, "features": list(inference.FEATURE_NAMES),
                    "index_sha256": {part: "b" * 64 for part in parts},
                })
                seed_results, members = [], {}
                for seed in inference.SEEDS:
                    trial = folder / fold / f"{name}-{seed}"
                    trial.mkdir(parents=True)
                    native = model_path(trial, name)
                    native.write_bytes(f"mock native {market}/{fold}/{name}/{seed}".encode())
                    # Native loading/checksums are independently covered by the
                    # backend tests. Only the loader and raw prediction are
                    # mocked here; all surrounding artifacts are real files.
                    digest = self.sha(native)
                    members[str(seed)] = digest
                    effective = "CPU" if name == "lgbm_binary" else "GPU"
                    params = {"num_threads" if name == "lgbm_binary" else "thread_count": 12}
                    versions = {"backend": frozen["versions"]["lightgbm" if name == "lgbm_binary" else "catboost"],
                                "numpy": np.__version__, "python": "3.13.5"}
                    request = {
                        "owner": OWNER, "schema_version": SCHEMA_VERSION, "model_name": name,
                        "seed": seed, "requested_task_type": "GPU", "effective_task_type": effective,
                        "max_iterations": PROTOCOL["maximum_iterations"],
                        "early_stopping": PROTOCOL["early_stopping"], "params": params,
                        "class_names": list(CLASS_NAMES), "train_shape": [10, 184], "tune_shape": [10, 184],
                        "data_sha256": {part: "c" * 64 for part in ("x_train", "y_train", "x_tune", "y_tune")},
                        "versions": versions, "wrapper_sha256": frozen["code_sha256"]["dockdack/mark1_selective_models.py"],
                    }
                    metadata = {
                        "owner": OWNER, "schema_version": SCHEMA_VERSION, "model_name": name,
                        "request": request, "params": params, "versions": versions,
                        "best_iteration": 10, "trained_iterations": 15, "feature_count": 184,
                        "model_path": str(native), "model_sha256": digest,
                        "effective_task_type": effective, "reused": False,
                        "research_only": True, "deployment_allowed": False,
                    }
                    self.write(trial / "request.json", request)
                    self.write(trial / "metadata.json", metadata)
                    result = {"architecture": name, "seed": seed, "model": metadata}
                    self.write(trial / "result.json", result)
                    seed_results.append(result)
                ensemble = {
                    "seed_results": seed_results, "member_sha256": members,
                    "calibration": {"success": calibration, "stop": calibration if name == "cat_joint6" else None},
                    "policy_selection": policy, "audit": metrics, "qualification": gate,
                }
                self.write(folder / fold / "ensemble.json", ensemble)
                ensembles[fold] = ensemble
            summary = {
                "market": market, "completed": True, "selected": name, "ranking": ranking,
                "ensembles": ensembles, "source": source, "research_qualified": False,
                "research_only": True, "deployment_allowed": False,
            }
            self.write(folder / "summary.json", summary)
            summaries[market] = summary
        self.write(root / "summary.json", summaries)
        return root / "domestic"

    def load(self, folder):
        return patch.object(inference, "load_model", side_effect=lambda name, path: int(path.parent.name.rsplit("-", 1)[1]))

    def rewrite_market(self, root, market, value):
        self.write(root / market / "summary.json", value)
        global_summary = self.read(root / "summary.json")
        global_summary[market] = value
        self.write(root / "summary.json", global_summary)

    def rewrite_ensemble(self, root, ensemble):
        self.write(root / "domestic" / inference.FOLD / "ensemble.json", ensemble)
        summary = self.read(root / "domestic" / "summary.json")
        summary["ensembles"][inference.FOLD] = ensemble
        self.rewrite_market(root, "domestic", summary)

    def test_binary_average_logits_then_calibrate_and_no_deployment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = self.fixture(root)
            hashes = {path: self.sha(path) for path in root.rglob("*") if path.is_file()}
            with self.load(folder) as loader:
                predictor = inference.SelectivePredictor(folder)
            self.assertEqual(loader.call_count, 3)
            history = self.history()
            before = history.copy()
            def predict(model, name, features):
                self.assertEqual(features.shape, (1, 184))
                return {"success_logits": np.array([model - 42.]), "stop_logits": None}
            with patch.object(inference, "predict_raw", side_effect=predict):
                result = predictor.predict(history, 105.)
            self.assertAlmostEqual(result["probability_success"], 1 / (1 + np.exp(-1.)))
            self.assertTrue(result["selected_research"])
            self.assertFalse(result["research_qualified"])
            self.assertFalse(result["deployment_allowed"])
            self.assertTrue(result["research_only"])
            self.assertFalse(result["entry_matches_evaluated_type"])
            self.assertFalse(result["intraday_path_verified"])
            self.assertIn("unvalidated", " ".join(result["notes"]))
            self.assertNotIn("BUY", json.dumps(result))
            self.assertEqual(result["take_profit_pct"], 1.)
            self.assertEqual(result["stop_loss_pct"], .9)
            np.testing.assert_array_equal(before, history)
            self.assertEqual(hashes, {path: self.sha(path) for path in root.rglob("*") if path.is_file()})

    def test_joint_stop_cap_and_open_scope_remain_research_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = self.fixture(Path(temporary), name="cat_joint6", cap=.25)
            with self.load(folder):
                predictor = inference.SelectivePredictor(folder)
            raw = {"success_logits": np.array([2.]), "stop_logits": np.array([0.])}
            with patch.object(inference, "predict_raw", return_value=raw):
                result = predictor.predict(self.history(), 105., entry_is_session_open=True)
            self.assertEqual(result["probability_stop"], .5)
            self.assertFalse(result["selected_research"])
            self.assertTrue(result["entry_matches_evaluated_type"])
            self.assertFalse(result["intraday_path_verified"])
            self.assertFalse(result["deployment_allowed"])

    def test_exact_threshold_is_not_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = self.fixture(Path(temporary), threshold=.5)
            with self.load(folder):
                predictor = inference.SelectivePredictor(folder)
            with patch.object(inference, "predict_raw", return_value={"success_logits": np.array([0.]), "stop_logits": None}):
                self.assertFalse(predictor.predict(self.history(), 105.)["selected_research"])

    def test_bad_inputs_never_call_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = self.fixture(Path(temporary))
            with self.load(folder):
                predictor = inference.SelectivePredictor(folder)
            bad_history = self.history()
            bad_history[0, 1] = 1.
            cases = [(self.history()[:29], 105., False), (bad_history, 105., False),
                     (self.history(), float("nan"), False), (self.history(), -1., False),
                     (self.history(), True, False), (self.history(), "105", False),
                     (self.history(), 105., 1)]
            with patch.object(inference, "predict_raw", side_effect=AssertionError("must validate first")):
                for bars, price, declared in cases:
                    with self.subTest(price=price, declared=declared), self.assertRaises(ValueError):
                        predictor.predict(bars, price, entry_is_session_open=declared)

    def test_incomplete_or_inconsistent_other_market_rejected(self):
        for case in ("missing_global", "missing_market", "incomplete_other", "disagree_other"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                folder = self.fixture(root)
                global_summary = self.read(root / "summary.json")
                if case == "missing_global":
                    (root / "summary.json").unlink()
                elif case == "missing_market":
                    del global_summary["us"]
                    self.write(root / "summary.json", global_summary)
                else:
                    global_summary["us"]["completed"] = False
                    self.write(root / "summary.json", global_summary)
                    if case == "incomplete_other":
                        self.write(root / "us" / "summary.json", global_summary["us"])
                with self.load(folder) as loader, self.assertRaises(ValueError):
                    inference.SelectivePredictor(folder)
                loader.assert_not_called()

    def test_protocol_target_feature_and_code_changes_rejected(self):
        for case in ("target", "feature", "code"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                folder = self.fixture(root)
                frozen = self.read(root / "protocol.json")
                if case == "target":
                    frozen["protocol"]["target"] = "other_target"
                elif case == "feature":
                    frozen["protocol"]["feature_names"][0] = "future_high"
                else:
                    frozen["code_sha256"][CODE_FILES[0]] = "f" * 64
                self.write(root / "protocol.json", frozen)
                with self.load(folder) as loader, self.assertRaises(ValueError):
                    inference.SelectivePredictor(folder)
                loader.assert_not_called()

    def test_source_feature_bank_and_selection_mismatch_rejected(self):
        for case in ("source", "feature", "selection"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                folder = self.fixture(root)
                if case == "source":
                    path = folder / "source.json"
                    value = self.read(path)
                    value["database_sha256"] = "f" * 64
                elif case == "feature":
                    path = folder / inference.FOLD / "features" / "features.json"
                    value = self.read(path)
                    value["features"][0] = "future_high"
                else:
                    path = folder / "selection_locked.json"
                    value = self.read(path)
                    value["selected"] = "cat_binary8"
                self.write(path, value)
                with self.load(folder) as loader, self.assertRaises(ValueError):
                    inference.SelectivePredictor(folder)
                loader.assert_not_called()

    def test_model_digest_metadata_and_seed_order_rejected(self):
        for case in ("hash", "metadata", "seed"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                folder = self.fixture(root)
                trial = folder / inference.FOLD / "cat_binary6-42"
                if case == "hash":
                    model_path(trial, "cat_binary6").write_bytes(b"corrupt")
                elif case == "metadata":
                    value = self.read(trial / "metadata.json")
                    value["feature_count"] = 183
                    self.write(trial / "metadata.json", value)
                else:
                    ensemble = self.read(folder / inference.FOLD / "ensemble.json")
                    ensemble["seed_results"].reverse()
                    self.rewrite_ensemble(root, ensemble)
                with self.load(folder), self.assertRaises(ValueError):
                    inference.SelectivePredictor(folder)

    def test_bad_calibration_and_out_of_grid_policy_rejected(self):
        for case in ("calibration", "count", "policy", "binary_stop"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                folder = self.fixture(root)
                ensemble = self.read(folder / inference.FOLD / "ensemble.json")
                if case == "calibration":
                    ensemble["calibration"]["success"]["slope"] = -1.
                elif case == "count":
                    ensemble["calibration"]["success"]["fit_samples"] += 1
                elif case == "policy":
                    ensemble["policy_selection"]["chosen_policy"]["threshold"] = .66
                else:
                    ensemble["calibration"]["stop"] = ensemble["calibration"]["success"]
                self.rewrite_ensemble(root, ensemble)
                with self.load(folder) as loader, self.assertRaises(ValueError):
                    inference.SelectivePredictor(folder)
                loader.assert_not_called()

    def test_false_qualification_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = self.fixture(root)
            summary = self.read(folder / "summary.json")
            summary["research_qualified"] = True
            self.rewrite_market(root, "domestic", summary)
            with self.load(folder), self.assertRaisesRegex(ValueError, "qualification"):
                inference.SelectivePredictor(folder)

    def test_nonchronological_split_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = self.fixture(root)
            split_path = folder / inference.FOLD / "splits.json"
            feature_path = folder / inference.FOLD / "features" / "features.json"
            splits = self.read(split_path)
            splits["train"]["last"] = "2025-01-01"
            features = self.read(feature_path)
            features["splits"] = splits
            self.write(split_path, splits)
            self.write(feature_path, features)
            with self.load(folder), self.assertRaisesRegex(ValueError, "chronological"):
                inference.SelectivePredictor(folder)

    def test_reused_transient_metadata_flag_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = self.fixture(root)
            ensemble = self.read(folder / inference.FOLD / "ensemble.json")
            for result in ensemble["seed_results"]:
                result["model"]["reused"] = True
                trial = folder / inference.FOLD / f"cat_binary6-{result['seed']}"
                self.write(trial / "result.json", result)
            self.rewrite_ensemble(root, ensemble)
            with self.load(folder):
                self.assertEqual(len(inference.SelectivePredictor(folder).models), 3)

    def test_no_broker_gui_order_or_mutating_imports(self):
        source = inspect.getsource(inference)
        import ast
        tree = ast.parse(source)
        imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any("broker" in (name or "") or "gui" in (name or "") for name in imported))
        self.assertNotIn(".write_text(", source)
        self.assertNotIn(".save_model(", source)


if __name__ == "__main__":
    unittest.main()
