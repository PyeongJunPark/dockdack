import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.audit_mark1_us_training import audit, feature_health, sha


class TrainingDiagnosticTests(unittest.TestCase):
    def test_feature_health_counts_without_mutating_input(self):
        names = ["query_log_gap", "query_gap_vol_scaled", "historical_return_std",
                 "w30_past_take_only_rate", "w30_past_both_touch_rate", "constant"]
        values = np.array([[0, -20, .01, .1, .2, 7],
                           [1, 0, .02, .2, .3, 7],
                           [2, 20, .03, .3, .4, 7]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npy"
            np.save(path, values, allow_pickle=False)
            before = sha(path)
            result = feature_health(path, names)
            self.assertEqual(result["shape"], [3, 6])
            self.assertEqual(result["nonfinite_values"], 0)
            self.assertEqual(result["constant_features"], ["constant"])
            self.assertEqual(result["clipped_value_count"], 2)
            self.assertEqual(result["selected_feature_quantiles"]["query_log_gap"][2], 1)
            self.assertEqual(before, sha(path))
            json.dumps(result, allow_nan=False)

    def test_invalid_schema_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npy"
            np.save(path, np.ones((2, 3)), allow_pickle=False)
            with self.assertRaises(ValueError):
                feature_health(path, ["only_one"])

    def test_empty_features_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npy"
            np.save(path, np.ones((0, 1)), allow_pickle=False)
            with self.assertRaises(ValueError):
                feature_health(path, ["empty"])

    def test_cannot_overwrite_frozen_run_or_non_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for destination in (root / "run/summary.json", root / "database.sqlite3"):
                with self.assertRaises(ValueError):
                    audit(root / "run", destination)


if __name__ == "__main__":
    unittest.main()
