"""The named mark1.1 bundle changes attribution, never its trained semantics."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dockdack import mark1_0504_inference as source_module
from dockdack import mark1_1_prototype_inference as alias_module
from examples.export_mark1_1_prototype import export_alias, snapshot
from test_mark1_0504_inference import fixture


class Mark11PrototypeInferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.alias = self.root / "original", self.root / "named"
        fixture(self.source)
        versions = patch.object(source_module, "_versions", return_value={"python": "3.13.7", "numpy": "2.2.6", "catboost": "1.2.10"})
        versions.start()
        self.addCleanup(versions.stop)
        native = patch.object(source_module, "load_model", side_effect=lambda name, path: {"name": name})
        native.start()
        self.addCleanup(native.stop)
        raw = patch.object(source_module, "predict_raw", side_effect=lambda model, name, features: {
            "success_logits": np.array([.25]),
            "stop_logits": np.array([-.25]) if name == "cat_joint6" else None,
        })
        raw.start()
        self.addCleanup(raw.stop)
        self.source_before = snapshot(self.source)
        self.export = export_alias(self.source, self.alias)
        self.bars = np.tile([100., 100.6, 99.8, 100.2, 100000.], (30, 1))

    def rewrite_alias(self, mutation):
        path = self.alias / "alias.json"
        value = json.loads(path.read_text())
        mutation(value)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        (self.alias / "alias.sha256").write_text(source_module.sha256_file(path), encoding="ascii")

    def test_identity_and_source_provenance_for_both_markets(self):
        for market in ("domestic", "us"):
            predictor = alias_module.Mark11PrototypePredictor(self.alias, market)
            result = predictor.predict(self.bars, 100)
            self.assertEqual(result["title"], "mark1.1 prototype")
            self.assertEqual(result["strategy_id"], "mark1-1-prototype")
            self.assertEqual(result["version"], "20260924-v1")
            self.assertEqual(result["source_model_version"], source_module.BUNDLE_VERSION)
            self.assertEqual(result["source_bundle_manifest_sha256"], self.source_before["manifest.json"])
            self.assertEqual(result["bundle_manifest_sha256"], source_module.sha256_file(self.alias / "alias.json"))
            self.assertEqual((result["take_profit_pct"], result["stop_loss_pct"]), (.5, .4))
            self.assertEqual(result["target"], source_module.TARGET)
            self.assertTrue(result["predicts_success"])
            for key, flag in source_module.RISK_FLAGS.items():
                self.assertIs(result[key], flag)
                self.assertIs(predictor.metadata[key], flag)

    def test_original_artifacts_are_byte_identical(self):
        self.assertEqual(snapshot(self.source), self.source_before)
        for relative, digest in self.source_before.items():
            self.assertEqual(source_module.sha256_file(self.alias / relative), digest)
        self.assertEqual(self.export["prediction_comparisons"], 24)
        self.assertTrue(self.export["source_files_unchanged"])

    def test_probabilities_and_query_outputs_are_unchanged(self):
        for market in ("domestic", "us"):
            source = source_module.HalfPercentPredictor(self.source, market)
            alias = alias_module.Mark11PrototypePredictor(self.alias, market)
            for price in (99.5, 100., 100.5):
                expected, actual = source.predict(self.bars, price), alias.predict(self.bars, price)
                for key in ("probability_success", "probability_stop", "candidate_take_price",
                            "candidate_stop_price", "selected_research", "path_limitation", "warnings"):
                    self.assertEqual(expected[key], actual[key])

    def test_exact_half_is_not_a_buy(self):
        with patch.object(source_module, "predict_raw", return_value={"success_logits": np.array([0.]), "stop_logits": None}):
            result = alias_module.Mark11PrototypePredictor(self.alias, "us").predict(self.bars, 100)
        self.assertEqual(result["probability_success"], .5)
        self.assertFalse(result["predicts_success"])

    def test_metadata_is_a_defensive_copy(self):
        predictor = alias_module.Mark11PrototypePredictor(self.alias, "domestic")
        metadata = predictor.metadata
        metadata["title"] = "different"
        metadata["deployment_allowed"] = True
        metadata["warnings"].clear()
        self.assertEqual(predictor.metadata["title"], alias_module.TITLE)
        self.assertFalse(predictor.metadata["deployment_allowed"])
        self.assertTrue(predictor.metadata["warnings"])

    def test_no_existing_bundle_can_be_replaced(self):
        before = snapshot(self.alias)
        with self.assertRaises(FileExistsError):
            export_alias(self.source, self.alias)
        self.assertEqual(snapshot(self.alias), before)

    def test_resealed_risk_relaxation_is_rejected(self):
        self.rewrite_alias(lambda data: data["contract"]["risk_flags"].update(deployment_allowed=True))
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            alias_module.Mark11PrototypePredictor(self.alias, "us")

    def test_resealed_source_version_change_is_rejected(self):
        self.rewrite_alias(lambda data: data["contract"].update(source_model_version="other"))
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            alias_module.Mark11PrototypePredictor(self.alias, "us")

    def test_source_member_tampering_is_rejected(self):
        (self.alias / "us/seed42.cbm").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            alias_module.Mark11PrototypePredictor(self.alias, "us")

    def test_validation_tampering_is_rejected(self):
        (self.alias / "alias-validation.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            alias_module.Mark11PrototypePredictor(self.alias, "us")

    def test_added_files_are_rejected(self):
        (self.alias / "surprise.txt").write_text("no", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory"):
            alias_module.Mark11PrototypePredictor(self.alias, "us")

    def test_wrapper_code_contract_is_sealed(self):
        self.rewrite_alias(lambda data: data["contract"].update(wrapper_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            alias_module.Mark11PrototypePredictor(self.alias, "us")

    def test_invalid_market_and_input_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "market must"):
            alias_module.Mark11PrototypePredictor(self.alias, "unknown")
        predictor = alias_module.Mark11PrototypePredictor(self.alias, "us")
        with self.assertRaisesRegex(ValueError, "30 numeric"):
            predictor.predict(self.bars[:29], 100)

    def test_path_traversal_is_rejected(self):
        self.rewrite_alias(lambda data: data["source_files_sha256"].update({"../escape": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "Unsafe artifact"):
            alias_module.Mark11PrototypePredictor(self.alias, "us")


if __name__ == "__main__":
    unittest.main()
