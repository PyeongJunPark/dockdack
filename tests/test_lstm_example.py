from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:
    ML_AVAILABLE = False
else:
    ML_AVAILABLE = True
    from examples.train_lstm_daily import (
        DailyWindows, LSTMClassifier, load_bars, make_features, split_and_scale,
    )


@unittest.skipUnless(ML_AVAILABLE, "Install the ml extra to test the LSTM example")
class LSTMExampleTests(unittest.TestCase):
    def test_missing_database_does_not_create_an_empty_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.sqlite3"
            with self.assertRaisesRegex(ValueError, "Database not found"):
                load_bars(path, "005930", "KRX", "2010-01-01")
            self.assertFalse(path.exists())

    def test_features_use_only_current_and_previous_bars(self):
        bars = np.array([[100, 104, 99, 102, 10], [103, 108, 101, 106, 20],
                         [107, 110, 105, 109, 30]], dtype=np.float64)
        features = make_features(bars)
        expected = np.r_[np.log(bars[1, :4] / 102), np.log1p(20) - np.log1p(10)]
        np.testing.assert_allclose(features[0], expected, rtol=1e-6)
        bars[2] *= 2
        np.testing.assert_array_equal(make_features(bars)[0], features[0])

    def test_future_data_does_not_change_training_normalization(self):
        features = np.random.default_rng(1).normal(size=(150, 5)).astype(np.float32)
        normalized, splits, mean, std = split_and_scale(features, 10)
        train, validation, test = splits
        self.assertLess(train[-1], validation[0])
        self.assertLess(validation[-1], test[0])
        np.testing.assert_allclose(mean, features[:train[-1]].mean(axis=0), atol=1e-6)
        changed = features.copy()
        changed[train[-1]:] += 1000
        normalized2, _, mean2, std2 = split_and_scale(changed, 10)
        np.testing.assert_array_equal(mean, mean2)
        np.testing.assert_array_equal(std, std2)
        np.testing.assert_array_equal(normalized[:train[-1]], normalized2[:train[-1]])

    def test_window_excludes_target_day_and_label_is_next_close_direction(self):
        bars = np.array([[10, 11, 9, 10, 1], [11, 12, 10, 11, 2],
                         [9, 10, 8, 9, 3], [12, 13, 11, 12, 4]], dtype=np.float64)
        features = make_features(bars)
        labels = (features[:, 3] > 0).astype(np.float32)
        dataset = DailyWindows(features, labels, np.array([2]), lookback=2)
        inputs, label = dataset[0]
        np.testing.assert_array_equal(inputs.numpy(), features[:2])
        self.assertEqual(label.item(), int(bars[3, 3] > bars[2, 3]))
        self.assertEqual(tuple(inputs.shape), (2, 5))

    def test_checkpoint_round_trip_preserves_eval_prediction(self):
        model = LSTMClassifier(hidden_size=8, num_layers=1).eval()
        inputs = torch.randn(2, 10, 5)
        with torch.no_grad():
            expected = model(inputs)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "model.pt"
            torch.save({"state": model.state_dict(), "mean": [0.0] * 5}, checkpoint)
            loaded = torch.load(checkpoint, weights_only=True)
        restored = LSTMClassifier(hidden_size=8, num_layers=1).eval()
        restored.load_state_dict(loaded["state"])
        with torch.no_grad():
            torch.testing.assert_close(restored(inputs), expected)
        self.assertEqual(tuple(expected.shape), (2,))


if __name__ == "__main__":
    unittest.main()
