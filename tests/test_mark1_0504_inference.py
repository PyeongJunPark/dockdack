"""The new target is independently sealed and grants no execution authority."""
from __future__ import annotations

from decimal import Decimal
import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dockdack import mark1_0504_inference as module


CALIBRATION = {"method": "platt_monotone", "slope": 1., "bias": 0., "fit_samples": 100, "weighted": False}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def fixture(root):
    root.mkdir()
    references = {}
    for market, name in module.MARKET_MODELS.items():
        folder = root / market
        folder.mkdir()
        members = []
        for seed in module.SEEDS:
            native = folder / f"seed{seed}.cbm"
            native.write_bytes(f"synthetic 0504 fixture {market} {seed}".encode())
            sidecar = native.with_name(native.name + ".json")
            write_json(sidecar, {"owner": "dockdack.mark1_selective_models", "schema_version": 1,
                                 "model_name": name, "model_sha256": module.sha256_file(native),
                                 "feature_count": 184, "class_names": list(module.CLASS_NAMES)})
            members.append({"seed": seed, "path": f"{market}/{native.name}",
                            "sha256": module.sha256_file(native), "sidecar_path": f"{market}/{sidecar.name}",
                            "sidecar_sha256": module.sha256_file(sidecar)})
        write_json(folder / "manifest.json", {
            "market": market, "model_name": name, "fold": "walk_2024", "seeds": list(module.SEEDS),
            "members": members, "risk_flags": module.RISK_FLAGS,
            "calibration": {"success": CALIBRATION, "stop": CALIBRATION if market == "domestic" else None},
            "policy": {"threshold": .5, "stop_probability_cap": 1.},
            "source": {"database_path": "Z:/never-read/never-present.sqlite3"},
        })
        references[market] = {"path": f"{market}/manifest.json", "sha256": module.sha256_file(folder / "manifest.json")}
    write_json(root / "manifest.json", {
        "owner": module.OWNER, "schema_version": module.SCHEMA_VERSION, "title": module.TITLE,
        "version": module.BUNDLE_VERSION, "completed": True, "markets": references,
        "semantics": module.SEMANTICS, "risk_flags": module.RISK_FLAGS, "warnings": module.WARNINGS,
        "feature_count": 184, "feature_names": list(module.FEATURE_NAMES),
        "runtime_code_sha256": module.runtime_code_hashes(), "training_versions": {"catboost": "1.2.10"},
    })
    seal(root)


def seal(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for market in manifest["markets"]:
        manifest["markets"][market]["sha256"] = module.sha256_file(root / market / "manifest.json")
    write_json(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(module.sha256_file(root / "manifest.json"), encoding="ascii")


def change(root, relative, mutation):
    path = root / relative
    data = json.loads(path.read_text())
    mutation(data)
    write_json(path, data)
    seal(root)


class HalfPercentInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "portable"
        fixture(self.root)
        self.versions = patch.object(module, "_versions", return_value={"python": "3.14.6", "numpy": "2.5.3", "catboost": "1.2.10"})
        self.versions.start()
        self.addCleanup(self.versions.stop)
        self.native = patch.object(module, "load_model", side_effect=lambda name, path: {"name": name, "path": str(path)})
        self.native.start()
        self.addCleanup(self.native.stop)
        self.bars = np.tile([100., 100.6, 99.8, 100.2, 100000.], (30, 1))

    def raw(self, success=.7, stop=.2):
        def result(model, name, features):
            self.assertEqual(features.shape, (1, 184))
            return {"success_logits": np.array([np.log(success / (1 - success))]),
                    "stop_logits": np.array([np.log(stop / (1 - stop))]) if name == "cat_joint6" else None}
        return patch.object(module, "predict_raw", side_effect=result)

    def test_both_markets_new_barriers_and_no_input_mutation(self):
        original = self.bars.copy()
        with self.raw():
            for market in module.MARKET_MODELS:
                predictor = module.HalfPercentPredictor(self.root, market)
                result = predictor.predict(self.bars, current_price=Decimal("100"))
                self.assertAlmostEqual(result["probability_success"], .7)
                self.assertTrue(result["predicts_success"])
                self.assertAlmostEqual(result["candidate_take_price"], 100.5)
                self.assertAlmostEqual(result["candidate_stop_price"], 99.6)
                self.assertEqual((result["take_profit_pct"], result["stop_loss_pct"]), (.5, .4))
                self.assertEqual(result["buy_threshold"], .5)
                self.assertEqual(result["inference_device"], "cpu")
                self.assertNotIn("action", result)
                self.assertNotIn("order", result)
                self.assertFalse(result["entry_matches_evaluated_type"])
                for key, value in module.RISK_FLAGS.items():
                    self.assertIs(result[key], value)
                    self.assertIs(predictor.metadata[key], value)
        np.testing.assert_array_equal(original, self.bars)

    def test_exact_half_abstains_and_numeric_price_types(self):
        with self.raw(.5):
            predictor = module.HalfPercentPredictor(self.root, "us")
            for price in (100., 100, "100.0", Decimal("100.00"), np.float64(100)):
                result = predictor.predict(self.bars.tolist(), current_price=price)
                self.assertEqual(result["probability_success"], .5)
                self.assertFalse(result["predicts_success"])

    def test_mean_logits_not_mean_probabilities(self):
        logits = iter((-4., 1., 1.))
        with patch.object(module, "predict_raw", side_effect=lambda *args: {
                "success_logits": np.array([next(logits)]), "stop_logits": None}):
            result = module.HalfPercentPredictor(self.root, "us").predict(self.bars, 100)
        self.assertAlmostEqual(result["probability_success"], 1 / (1 + np.exp(2 / 3)))
        self.assertFalse(result["predicts_success"])

    def test_new_price_recomputes_features(self):
        seen = []
        def raw(model, name, features):
            seen.append(features.copy())
            return {"success_logits": np.array([0.]), "stop_logits": None}
        with patch.object(module, "predict_raw", side_effect=raw):
            predictor = module.HalfPercentPredictor(self.root, "us")
            predictor.predict(self.bars, 100)
            predictor.predict(self.bars, 101)
        self.assertFalse(np.array_equal(seen[0], seen[3]))

    def test_training_float32_history_boundary_parity(self):
        boundary = np.tile([84700., 84700., 84361.2, 84700., 1000.], (30, 1))
        expected = module.features_from_history(boundary.astype(np.float32)[None], np.array([84700.]))
        original_precision = module.features_from_history(boundary[None], np.array([84700.]))
        self.assertFalse(np.array_equal(expected, original_precision))
        seen = []
        def raw(model, name, features):
            seen.append(features.copy())
            return {"success_logits": np.array([float(features.sum()) / 100]), "stop_logits": None}
        with patch.object(module, "predict_raw", side_effect=raw):
            predictor = module.HalfPercentPredictor(self.root, "us")
            raw_result = predictor.predict(boundary, 84700.)
            stored_result = predictor.predict(boundary.astype(np.float32), 84700.)
        self.assertEqual(raw_result["probability_success"], stored_result["probability_success"])
        for actual in seen:
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(predictor.metadata["historical_representation"],
                         "validate_original_then_float32_raw_cache_parity")

    def test_invalid_original_ohlc_not_hidden_by_float32_rounding(self):
        almost = self.bars.copy()
        almost[0] = [100., 100. - 1e-8, 99., 99.5, 1000.]
        # These values collapse to a valid candle in float32, but original
        # invalid inputs must still be rejected before conversion.
        self.assertEqual(np.float32(almost[0, 0]), np.float32(almost[0, 1]))
        with self.assertRaisesRegex(ValueError, "LOW/HIGH"):
            module.HalfPercentPredictor(self.root, "us").predict(almost, 100)

    def test_metadata_defensive_copy(self):
        predictor = module.HalfPercentPredictor(self.root, "domestic")
        changed = predictor.metadata
        changed["deployment_allowed"] = True
        changed["policy"]["threshold"] = 0
        changed["warnings"].clear()
        self.assertFalse(predictor.metadata["deployment_allowed"])
        self.assertEqual(predictor.metadata["policy"]["threshold"], .5)
        self.assertTrue(predictor.metadata["warnings"])

    def test_tampered_root_model_sidecar_rejected(self):
        for filename in ("manifest.json", "domestic/seed42.cbm", "domestic/seed43.cbm.json"):
            path = self.root / filename
            old = path.read_bytes()
            path.write_bytes(old + b" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                module.HalfPercentPredictor(self.root, "domestic")
            path.write_bytes(old)

    def test_feature_code_target_risk_contract_fail_closed(self):
        path = self.root / "manifest.json"
        original = path.read_bytes()
        mutations = [lambda x: x["feature_names"].reverse(),
                     lambda x: x["runtime_code_sha256"].update(barrier_features="0" * 64),
                     lambda x: x["semantics"].update(stop_loss_pct=.9),
                     lambda x: x["semantics"].update(take_profit_pct=1.),
                     lambda x: x["risk_flags"].update(deployment_allowed=True),
                     lambda x: x.update(completed=False)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                change(self.root, "manifest.json", mutation)
                with self.assertRaisesRegex(ValueError, "contract"):
                    module.HalfPercentPredictor(self.root, "domestic")
                path.write_bytes(original)
                seal(self.root)

    def test_old_bundle_explicitly_rejected(self):
        change(self.root, "manifest.json", lambda x: x.update(owner="dockdack.mark1_prototype"))
        with self.assertRaisesRegex(ValueError, "contract"):
            module.HalfPercentPredictor(self.root, "us")

    def test_inference_wrapper_hash_is_sealed_and_mismatch_rejected(self):
        self.assertEqual(module.runtime_code_hashes()["inference"],
                         module.sha256_file(Path(module.__file__)))
        change(self.root, "manifest.json",
               lambda value: value["runtime_code_sha256"].update(inference="0" * 64))
        with self.assertRaisesRegex(ValueError, "contract"):
            module.HalfPercentPredictor(self.root, "us")

    def test_invalid_calibration_rejected(self):
        path = self.root / "domestic/manifest.json"
        original = path.read_bytes()
        for value in (0., -1., True):
            change(self.root, "domestic/manifest.json", lambda x: x["calibration"]["success"].update(slope=value))
            with self.assertRaises(ValueError):
                module.HalfPercentPredictor(self.root, "domestic")
            path.write_bytes(original)
            seal(self.root)

    def test_nonfinite_manifest_rejected(self):
        change(self.root, "us/manifest.json", lambda x: x["calibration"]["success"].update(bias=float("nan")))
        with self.assertRaisesRegex(ValueError, "JSON"):
            module.HalfPercentPredictor(self.root, "us")

    def test_policy_mutation_rejected(self):
        change(self.root, "domestic/manifest.json", lambda x: x["policy"].update(threshold=.4))
        with self.assertRaisesRegex(ValueError, "Policy"):
            module.HalfPercentPredictor(self.root, "domestic")

    def test_missing_market_rejected(self):
        change(self.root, "manifest.json", lambda x: x["markets"].pop("us"))
        with self.assertRaisesRegex(ValueError, "Both"):
            module.HalfPercentPredictor(self.root, "domestic")

    def test_wrong_seed_or_traversal_rejected(self):
        change(self.root, "us/manifest.json", lambda x: x["members"][0].update(path="../outside.cbm"))
        with self.assertRaisesRegex(ValueError, "path"):
            module.HalfPercentPredictor(self.root, "us")

    def test_bad_input_and_nonfinite_backend_rejected(self):
        predictor = module.HalfPercentPredictor(self.root, "us")
        for price in (True, False, None, [], [100], 0, -1, "nan", "Infinity", Decimal("NaN"), "1e999"):
            with self.subTest(price=price), self.assertRaises(ValueError):
                predictor.predict(self.bars, current_price=price)
        for bars in (self.bars[:29], self.bars.astype(bool), self.bars * float("nan")):
            with self.assertRaises(ValueError):
                predictor.predict(bars, current_price=100)
        with patch.object(module, "predict_raw", return_value={"success_logits": np.array([np.nan]), "stop_logits": None}):
            with self.assertRaisesRegex(ValueError, "invalid raw"):
                predictor.predict(self.bars, current_price=100)

    def test_binary_cannot_invent_stop_head(self):
        change(self.root, "us/manifest.json", lambda x: x["calibration"].update(stop=CALIBRATION))
        with self.assertRaisesRegex(ValueError, "stop"):
            module.HalfPercentPredictor(self.root, "us")

    def test_runtime_rejects_unsupported_backend_major(self):
        self.versions.stop()
        with patch.object(module.package_metadata, "version", return_value="2.0.0"):
            with self.assertRaisesRegex(ValueError, "version"):
                module.HalfPercentPredictor(self.root, "us")

    def test_export_refuses_existing_destination_before_sources(self):
        from examples.export_mark1_0504 import export_bundle
        with self.assertRaisesRegex(FileExistsError, "replace"):
            export_bundle("deliberately-absent", self.root)
        self.assertTrue((self.root / "manifest.json").is_file())

    def test_complete_synthetic_export_verifies_24_cpu_comparisons(self):
        from examples import export_mark1_0504 as exporter
        from examples import train_mark1_0504 as training
        parent = Path(self.temporary.name)
        run, cache = parent / "training", parent / "cache"
        run.mkdir()
        cache.mkdir()
        frozen = {"protocol": training.canonical(training.PROTOCOL),
                  "code_sha256": {name: module.sha256_file(exporter.ROOT / name) for name in training.CODE_FILES},
                  "raw_cache_dir": str(cache), "old_run": str(parent / "unused_old_source"),
                  "versions": {"catboost": "synthetic-test"}}
        write_json(run / "protocol.json", frozen)
        old_artifact = parent / "untouched-old-prototype.bin"
        old_artifact.write_bytes(b"protected original synthetic artifact")
        write_json(run / "completed.json", {"no_orders": True,
                   "original_prototype_sha256": {str(old_artifact): module.sha256_file(old_artifact)}})
        summaries, datasets = {}, {}
        for market, architecture in module.MARKET_MODELS.items():
            folder = run / market
            folder.mkdir()
            database = parent / f"{market}-readonly.sqlite3"
            database.write_bytes(b"synthetic database never opened by inference")
            source = {"database_path": str(database), "database_sha256": module.sha256_file(database)}
            experiment = {"target": module.TARGET, "raw_cache_target_is_not_used_for_labels": True}
            key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
            (cache / f"{market}-{key}.npz").write_bytes(b"mocked immutable raw-window cache")
            write_json(folder / "source.json", source)
            write_json(folder / "experiment_data.json", experiment)
            hashes = {}
            for seed in module.SEEDS:
                trial = folder / exporter.FOLD / f"{architecture}-{seed}"
                trial.mkdir(parents=True)
                native = trial / f"{architecture}.cbm"
                native.write_bytes(f"synthetic native {market} {seed}".encode())
                digest = module.sha256_file(native)
                hashes[str(seed)] = digest
                write_json(native.with_name(native.name + ".json"), {
                    "feature_count": 184, "model_sha256": digest})
                request = {"seed": seed, "model_name": architecture,
                           "wrapper_sha256": frozen["code_sha256"]["dockdack/mark1_selective_models.py"]}
                write_json(trial / "request.json", request)
                write_json(trial / "metadata.json", {"model_sha256": digest, "feature_count": 184,
                    "model_name": architecture, "research_only": True, "deployment_allowed": False,
                    "request": request, "best_iteration": 1, "versions": {"catboost": "synthetic-test"}})
            metrics = {"cost_bps": 20, "block_bootstrap": {"cost_bps": 20}}
            ensemble = {"target": module.TARGET,
                "policy": {"threshold": .5, "stop_probability_cap": 1.},
                "calibration": {"success": CALIBRATION, "stop": CALIBRATION if market == "domestic" else None},
                "audit": metrics, "validation": metrics,
                "qualification": training.qualification(metrics),
                "validation_qualification": training.qualification(metrics), "member_sha256": hashes}
            for fold in training.PROTOCOL["folds"]:
                (folder / fold).mkdir(exist_ok=True)
                write_json(folder / fold / "ensemble.json", ensemble)
            summaries[market] = {"market": market, "selected": architecture, "target": module.TARGET,
                "completed": True, "research_only": True, "deployment_allowed": False,
                "intraday_path_verified": False, "source": source, "experiment_data": experiment,
                "ensembles": {fold: ensemble for fold in training.PROTOCOL["folds"]}}
            write_json(folder / "summary.json", summaries[market])
            dataset = SimpleNamespace(
                bars=np.tile(self.bars.astype(np.float32), (4, 1)), starts=np.array([0, 30, 60, 90]),
                target_dates=np.array([np.datetime64(f"2024-{month:02}-15", "D").astype(np.int64)
                                       for month in (3, 6, 9, 12)]),
                target_ohlc=np.tile([100., 101., 99., 100.], (4, 1)))
            datasets[market] = (dataset, source, experiment)
        write_json(run / "summary.json", summaries)
        def raw(model, name, features):
            return {"success_logits": np.array([float(features.sum()) / 100]),
                    "stop_logits": np.array([-.2]) if name == "cat_joint6" else None}
        destination = parent / "new_0504_bundle"
        with patch.object(training, "load_experiment_dataset", side_effect=lambda market, *args: datasets[market]), \
             patch.object(exporter, "load_model", return_value=object()), \
             patch.object(exporter, "predict_raw", side_effect=raw), \
             patch.object(module, "predict_raw", side_effect=raw):
            result = exporter.export_bundle(run, destination)
            predictor = module.HalfPercentPredictor(destination, "domestic")
            self.assertFalse(predictor.metadata["deployment_allowed"])
        self.assertEqual(result["prediction_comparisons"], 24)
        validation = json.loads((destination / "export-validation.json").read_text())
        self.assertTrue(validation["passed"])
        self.assertTrue(validation["protected_files_unchanged"])
        self.assertEqual(validation["protected_sha256_before"], validation["protected_sha256_after"])
        self.assertTrue(all(value["prediction_calls"] == 12 for value in validation["markets"].values()))


if __name__ == "__main__":
    unittest.main()
