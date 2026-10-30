"""Causal and chronological contracts of the research-only MK1.2 gate."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from examples.research_mark1_2_neural_gate import (
    FEATURE_NAMES, chronological_partition, frequency_threshold, gate_features,
    sample_positions, signal_metrics,
)


class NeuralGateResearchTests(unittest.TestCase):
    def test_calendar_split_ignores_symbol_order_and_keeps_validation_later(self):
        dates = np.array([4, 1, 3, 2, 4, 1, 3, 2], dtype=np.int32)
        training, validation = chronological_partition(dates, .5)
        np.testing.assert_array_equal(np.unique(dates[training]), [1, 2])
        np.testing.assert_array_equal(np.unique(dates[validation]), [3, 4])
        self.assertLess(dates[training].max(), dates[validation].min())

    def test_frequency_uses_validation_scores_and_strict_threshold(self):
        scores = np.linspace(.1, .9, 100)
        threshold, target = frequency_threshold(scores, 40)
        self.assertEqual(target, 52)
        self.assertEqual(int((scores > threshold).sum()), 52)
        with self.assertRaises(ValueError):
            frequency_threshold(np.array([.1, np.nan]), 1)

    def test_only_prior_bars_and_actual_open_enter_gate_features(self):
        bars = np.array([[100 + i * .1, 101 + i * .1, 99 + i * .1,
                          100.4 + i * .1, 1000 + i] for i in range(32)], dtype=np.float32)
        dataset = SimpleNamespace(bars=bars, starts=np.array([0, 2]),
                                  target_ohlc=np.array([[104., 110., 99., 106.],
                                                        [105., 112., 98., 107.]], dtype=np.float64))
        indices = np.array([0, 1], dtype=np.int64)
        raw = np.array([.3, -.1], dtype=np.float32)
        first = gate_features(dataset, indices, raw)
        self.assertEqual(first.shape, (2, len(FEATURE_NAMES)))
        dataset.target_ohlc[:, 1:] = np.array([[500., .01, 400.], [500., .01, 400.]])
        np.testing.assert_array_equal(first, gate_features(dataset, indices, raw))
        changed = gate_features(dataset, indices, raw + 1)
        np.testing.assert_allclose(changed[:, 0], first[:, 0] + 1)
        dataset.target_ohlc[0, 0] += .5
        self.assertFalse(np.array_equal(first[0], gate_features(dataset, indices, raw)[0]))

    def test_label_blind_sampling_and_20bp_statistics(self):
        positions = np.arange(1000)
        sampled = sample_positions(positions, 50, 42)
        self.assertEqual(len(sampled), 50)
        np.testing.assert_array_equal(sampled, sample_positions(positions, 50, 42))
        self.assertTrue(np.all(sampled[1:] > sampled[:-1]))
        metrics = signal_metrics(np.array([True, False, True]), np.array([.01, -.009, .004]),
                                 np.array([1, 1, 2]), np.array([10, 11, 10]),
                                 np.array([True, False, True]))
        self.assertEqual(metrics["signals"], 2)
        self.assertEqual(metrics["precision_success"], 1.)
        self.assertAlmostEqual(metrics["net_mean_20bp"], .005)
        self.assertEqual(metrics["signal_days"], 2)
        self.assertEqual(metrics["symbols"], 1)


if __name__ == "__main__":
    unittest.main()
