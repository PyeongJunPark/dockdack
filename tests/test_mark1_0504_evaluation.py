"""No-broker evaluation contracts, ensemble math and missing-artifact guards."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from examples import backtest_mark1_0504 as evaluation
from examples import train_mark1_0504 as training


class SmallBarrierEvaluationTests(unittest.TestCase):
    def test_both_completed_markets_required_before_model_load(self):
        for complete in ({}, {"domestic": {"completed": True}}, {"us": {"completed": True}}):
            with self.subTest(complete=complete), patch.object(evaluation, "read_json", return_value=complete), \
                    patch.object(evaluation, "load_model") as load:
                with self.assertRaisesRegex(ValueError, "Both new-target"):
                    evaluation.verify_new_training(Path("nonexistent"))
                load.assert_not_called()

    def test_wrong_protocol_rejected_before_any_model_load(self):
        complete = {"domestic": {}, "us": {}}
        with patch.object(evaluation, "read_json", side_effect=[complete, {"protocol": {}}]), \
                patch.object(evaluation, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                evaluation.verify_new_training(Path("unused"))
            load.assert_not_called()

    def test_missing_code_hashes_rejected(self):
        complete = {"domestic": {}, "us": {}}
        frozen = {"protocol": evaluation._canonical(training.PROTOCOL), "code_sha256": {}}
        with patch.object(evaluation, "read_json", side_effect=[complete, frozen]):
            with self.assertRaisesRegex(ValueError, "code hashes"):
                evaluation.verify_new_training(Path("unused"))

    def test_changed_training_code_rejected(self):
        frozen = {"protocol": evaluation._canonical(training.PROTOCOL),
                  "code_sha256": {name: "frozen" for name in training.CODE_FILES}}
        with patch.object(evaluation, "read_json", side_effect=[{"domestic": {}, "us": {}}, frozen]), \
                patch.object(evaluation, "file_hash", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "training code changed"):
                evaluation.verify_new_training(Path("unused"))

    def test_raw_logits_average_before_calibration_and_chunk_order(self):
        indices = np.arange(5)
        summary = {"selected": "cat_binary8", "ensembles": {"walk_2024": {
            "member_sha256": {str(seed): "ok" for seed in evaluation.SEEDS},
            "calibration": {"success": {"method": "platt", "slope": 1., "intercept": 0.}, "stop": None}}}}
        # The middle member's logits are not symmetric: averaging calibrated
        # probabilities would differ substantially from calibrating mean logits.
        offsets = {42: -5., 43: 2., 44: 6.}
        features_seen = []

        def features(dataset, picked):
            features_seen.extend(picked.tolist())
            return picked[:, None].astype(float)

        def load(name, path):
            return int(path.parent.name.rsplit("-", 1)[1])

        def raw(model, name, values):
            return {"success_logits": values[:, 0] + offsets[model], "stop_logits": None}

        def calibrate(raw_values, calibration):
            return 1 / (1 + np.exp(-raw_values["success_logits"])), None

        with patch.object(evaluation, "file_hash", return_value="ok"), \
                patch.object(evaluation, "load_model", side_effect=load), \
                patch.object(evaluation, "predict_raw", side_effect=raw), \
                patch.object(evaluation, "calibrated", side_effect=calibrate):
            actual, stops, hashes = evaluation.infer_ensemble(SimpleNamespace(), indices, "us",
                Path("unused"), summary, features, 2)
        np.testing.assert_allclose(actual, 1 / (1 + np.exp(-(indices + 1))))
        self.assertEqual(features_seen, indices.tolist())
        self.assertIsNone(stops)
        self.assertEqual(len(hashes), 3)

    def test_member_checksum_mismatch_prevents_prediction(self):
        summary = {"selected": "cat_binary8", "ensembles": {"walk_2024": {
            "member_sha256": {str(seed): "frozen" for seed in evaluation.SEEDS}}}}
        with patch.object(evaluation, "file_hash", return_value="changed"), \
                patch.object(evaluation, "predict_raw") as predict:
            with self.assertRaisesRegex(ValueError, "model changed"):
                evaluation.infer_ensemble(None, np.arange(1), "us", Path("unused"), summary, None, 1)
            predict.assert_not_called()

    def test_targets_and_cost_grid_are_explicit(self):
        self.assertEqual(evaluation.COMPARATORS, {"old_0109": (.01, .009), "new_0504": (.005, .004)})
        self.assertEqual(evaluation.COSTS, (0, 10, 20, 40))
        self.assertIn("reused", evaluation.EVALUATION_STATUS)


if __name__ == "__main__":
    unittest.main()
