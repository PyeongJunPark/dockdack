"""Portable inference contracts, without optional native libraries or trading."""
from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dockdack import mark1_prototype_inference as module


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
            native.write_bytes(f"synthetic native fixture {market} {seed}".encode())
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
            "source": {"database_path": "Z:/deliberately-absent-training-cache/never-read.sqlite3"},
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


class PrototypeTests(unittest.TestCase):
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
        self.bars = np.tile([100., 102., 98., 101., 100000.], (30, 1))

    def raw(self, success=.7, stop=.2):
        def result(model, name, features):
            self.assertEqual(features.shape, (1, 184))
            return {"success_logits": np.array([np.log(success / (1 - success))]),
                    "stop_logits": np.array([np.log(stop / (1 - stop))]) if name == "cat_joint6" else None}
        return patch.object(module, "predict_raw", side_effect=result)

    def test_both_markets_portable_cpu_contract_and_no_mutation(self):
        original = self.bars.copy()
        with self.raw():
            for market in module.MARKET_MODELS:
                predictor = module.PrototypePredictor(self.root, market)
                result = predictor.predict(self.bars, current_price=Decimal("100"))
                self.assertAlmostEqual(result["probability_success"], .7)
                self.assertTrue(result["predicts_success"])
                self.assertEqual(result["candidate_take_price"], 101.)
                self.assertEqual(result["candidate_stop_price"], 99.1)
                self.assertEqual(result["buy_threshold"], .5)
                for key, value in module.RISK_FLAGS.items():
                    self.assertIs(result[key], value)
                    self.assertIs(predictor.metadata[key], value)
                self.assertEqual(result["inference_device"], "cpu")
                self.assertEqual(predictor.metadata["bundle_manifest_sha256"], module.sha256_file(self.root / "manifest.json"))
                self.assertEqual(predictor.metadata["market_manifest_sha256"], module.sha256_file(self.root / market / "manifest.json"))
                self.assertNotIn("action", result)
                self.assertNotIn("order", result)
                self.assertFalse(result["entry_matches_evaluated_type"])
        np.testing.assert_array_equal(original, self.bars)

    def test_exactly_half_abstains_and_decimal_string_price(self):
        with self.raw(.5):
            predictor = module.PrototypePredictor(self.root, "us")
            for price in (100., 100, "100.0", Decimal("100.00"), np.float64(100)):
                result = predictor.predict(self.bars.tolist(), current_price=price)
                self.assertEqual(result["probability_success"], .5)
                self.assertFalse(result["predicts_success"])

    def test_metadata_is_defensive_copy(self):
        predictor = module.PrototypePredictor(self.root, "domestic")
        changed = predictor.metadata
        changed["deployment_allowed"] = True
        changed["policy"]["threshold"] = 0
        changed["warnings"].clear()
        self.assertFalse(predictor.metadata["deployment_allowed"])
        self.assertEqual(predictor.metadata["policy"]["threshold"], .5)
        self.assertTrue(predictor.metadata["warnings"])

    def test_tampered_root_rejected(self):
        with (self.root / "manifest.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            module.PrototypePredictor(self.root, "us")

    def test_tampered_model_and_sidecar_rejected(self):
        for filename in ("seed42.cbm", "seed43.cbm.json"):
            path = self.root / "domestic" / filename
            old = path.read_bytes()
            path.write_bytes(old + b" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                module.PrototypePredictor(self.root, "domestic")
            path.write_bytes(old)

    def test_feature_names_source_hash_and_semantics_fail_closed(self):
        path = self.root / "manifest.json"
        original = path.read_bytes()
        mutations = [lambda x: x["feature_names"].reverse(),
                     lambda x: x["runtime_code_sha256"].update(features="0" * 64),
                     lambda x: x["semantics"].update(stop_loss_pct=1.),
                     lambda x: x["risk_flags"].update(deployment_allowed=True),
                     lambda x: x.update(completed=False)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                change(self.root, "manifest.json", mutation)
                with self.assertRaisesRegex(ValueError, "contract"):
                    module.PrototypePredictor(self.root, "domestic")
                path.write_bytes(original)
                seal(self.root)

    def test_invalid_calibration_rejected(self):
        path = self.root / "domestic/manifest.json"
        original = path.read_bytes()
        for value in (0., -1., True):
            change(self.root, "domestic/manifest.json", lambda x: x["calibration"]["success"].update(slope=value))
            with self.assertRaises(ValueError):
                module.PrototypePredictor(self.root, "domestic")
            path.write_bytes(original)
            seal(self.root)

    def test_nonfinite_manifest_rejected(self):
        change(self.root, "us/manifest.json", lambda x: x["calibration"]["success"].update(bias=float("nan")))
        with self.assertRaisesRegex(ValueError, "JSON"):
            module.PrototypePredictor(self.root, "us")

    def test_policy_mutation_rejected(self):
        change(self.root, "domestic/manifest.json", lambda x: x["policy"].update(threshold=.6))
        with self.assertRaisesRegex(ValueError, "Policy"):
            module.PrototypePredictor(self.root, "domestic")

    def test_missing_second_market_rejected(self):
        change(self.root, "manifest.json", lambda x: x["markets"].pop("us"))
        with self.assertRaisesRegex(ValueError, "Both"):
            module.PrototypePredictor(self.root, "domestic")

    def test_member_order_and_path_rejected(self):
        change(self.root, "us/manifest.json", lambda x: x["members"][0].update(path="../outside.cbm"))
        with self.assertRaisesRegex(ValueError, "path"):
            module.PrototypePredictor(self.root, "us")

    def test_bad_input_and_nonfinite_backend_rejected(self):
        predictor = module.PrototypePredictor(self.root, "us")
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
            module.PrototypePredictor(self.root, "us")

    def test_runtime_rejects_unsupported_backend_major(self):
        self.versions.stop()
        with patch.object(module.package_metadata, "version", return_value="2.0.0"):
            with self.assertRaisesRegex(ValueError, "version"):
                module.PrototypePredictor(self.root, "us")

    def test_exporter_refuses_existing_destination_before_loading_sources(self):
        from examples.export_mark1_prototype import export_bundle
        # The exporter lazily imports the frozen verifier; Torch might be absent
        # in a minimal inference installation, so mock that module for this guard.
        import types
        import sys
        dummy = types.ModuleType("dockdack.mark1_selective_inference")
        dummy.SelectivePredictor = object
        with patch.dict(sys.modules, {"dockdack.mark1_selective_inference": dummy}):
            with self.assertRaisesRegex(FileExistsError, "replace"):
                export_bundle("deliberately-nonexistent", self.root)
        self.assertTrue((self.root / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
