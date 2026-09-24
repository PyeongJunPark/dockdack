import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
import tempfile

import numpy as np

from dockdack.mark1_data import Mark1Dataset
from examples import train_mark1_0504 as training


class HalfTrainingTests(unittest.TestCase):
    def test_cost_gate_uses_new_breakeven(self):
        metrics = {"signal_count": 100, "signal_days": 30, "symbol_count": 20, "cost_bps": 20,
                   "block_bootstrap": {"precision_lower": .64, "net_mean_lower": .001, "cost_bps": 20}}
        self.assertFalse(training.qualification(metrics)["qualified"])
        metrics["block_bootstrap"]["precision_lower"] = .7
        self.assertTrue(training.qualification(metrics)["qualified"])
        metrics["block_bootstrap"]["net_mean_lower"] = 0
        self.assertFalse(training.qualification(metrics)["qualified"])

    def test_quarantine_filters_all_splits_without_changing_raw_bars(self):
        bars = np.tile([100., 101., 99.8, 100., 1000.], (36, 1))
        ids = np.array([0, 1, 2, 3, 0, 1])
        dataset = Mark1Dataset(bars, np.arange(6), np.arange(6)+19000, ids,
            np.tile([100., 100.6, 99.9, 100.1], (6, 1)),
            {"train": np.array([0, 1, 2]), "test": np.array([3, 4, 5])},
            {"symbols": [{"symbol": name, "symbol_id": i} for i, name in enumerate(["AAPL", "FCEL", "BNED", "BBSI"])], "target": "old"})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"us/source.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"target": "old"}))
            with patch.object(training, "load_frozen_cache", return_value=dataset):
                filtered, source, audit = training.load_experiment_dataset("us", Path(folder), Path(folder))
        self.assertIs(filtered.bars, dataset.bars)
        self.assertEqual(filtered.symbol_ids.tolist(), [0, 0])
        self.assertEqual(filtered.splits["train"].tolist(), [0])
        self.assertEqual(filtered.splits["test"].tolist(), [1])
        self.assertEqual(dataset.starts.tolist(), list(range(6)))
        self.assertEqual(source["target"], "old")
        self.assertEqual(audit["target"], training.TARGET)
        self.assertEqual(audit["eligible_samples"], 2)

    def test_features_ignore_target_future_high_low_close(self):
        bars = np.tile([100., 100.6, 99.9, 100.1, 1000.], (31, 1))
        dataset = Mark1Dataset(bars, np.array([0]), np.array([20000]), np.array([0]),
                               np.array([[100., 102., 98., 101.]]), {}, {})
        values = training.feature_array(dataset, np.array([0]))
        changed = replace(dataset, target_ohlc=np.array([[100., 130., 70., 80.]]))
        np.testing.assert_array_equal(values, training.feature_array(changed, np.array([0])))
        self.assertEqual(values.shape, (1, 184))

    def test_protocol_does_not_promote_or_optimize_on_reused_test(self):
        self.assertEqual(training.PROTOCOL["policy"]["threshold"], .5)
        self.assertEqual(training.PROTOCOL["ensemble_seeds"], [42, 43, 44])
        self.assertIn("no GUI", training.PROTOCOL["deployment"])
        self.assertEqual(training.PROTOCOL["take_profit_pct"], .5)
        self.assertEqual(training.PROTOCOL["stop_loss_pct"], .4)


if __name__ == "__main__":
    unittest.main()
