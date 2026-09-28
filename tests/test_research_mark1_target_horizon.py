"""Small synthetic checks for the research runner's frozen-decision helpers."""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from examples.research_mark1_target_horizon import (
    _apply_normalization, _fit_torch, _qualification, _safe_output,
    _split_samples, _temperature, _thresholds,
)


def _period(model: float, *, trades: int = 40,
            unconditional: float = 0.01, momentum: float = 0.02) -> dict:
    return {
        "model": {"trades": trades, "mean_net_return": model / 10,
                  "total_return": model},
        "unconditional": {"total_return": unconditional},
        "fixed_momentum": {"total_return": momentum},
    }


class TargetHorizonRunnerTest(unittest.TestCase):
    def test_predeclared_gate_cannot_select_hit_rate_or_one_lucky_period(self):
        periods = {"tune": _period(0.03), "calibration": _period(0.04),
                   "selection": _period(0.05)}
        self.assertTrue(_qualification(periods, min_trades=30)["passed"])
        periods["selection"] = _period(0.05, trades=29)
        self.assertFalse(_qualification(periods, min_trades=30)["passed"])
        periods["selection"] = _period(0.05, momentum=0.06)
        self.assertFalse(_qualification(periods, min_trades=30)["passed"])
        periods["selection"] = _period(-0.01)
        periods["calibration"] = _period(-0.01)
        self.assertFalse(_qualification(periods, min_trades=30)["passed"])
        periods["selection"] = _period(0.05, trades=40)
        periods["calibration"] = _period(0.04)
        self.assertFalse(_qualification(periods, min_trades=50)["passed"])

    def test_sampling_stays_within_approved_splits_and_is_reproducible(self):
        dataset = SimpleNamespace(splits={
            name: np.arange(index * 30, index * 30 + 30, dtype=np.int64)
            for index, name in enumerate(("train", "tune", "calibration",
                                          "selection", "test"))})
        first = _split_samples(dataset, train_cap=12, eval_cap=7, seed=41)
        second = _split_samples(dataset, train_cap=12, eval_cap=7, seed=41)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 12 + 4 * 7)
        self.assertEqual(len(np.unique(first)), len(first))
        with self.assertRaises(ValueError):
            _split_samples(dataset, train_cap=-1, eval_cap=7, seed=41)

    def test_neural_two_task_fit_is_finite_without_future_labels_as_features(self):
        rng = np.random.default_rng(7)
        features = rng.normal(0, 0.2, size=(32, 10, 6)).astype(np.float32)
        hits = np.asarray([0, 1] * 16, dtype=bool)
        net = np.where(hits, 0.03, -0.04).astype(np.float64)
        torch.set_num_threads(1)
        fitted = _fit_torch("linear", 10, features, hits, net,
                            seed=7, epochs=2, batch_size=8, device="cpu")
        logits, returns = fitted.predict(features)
        self.assertEqual(logits.shape, (32,))
        self.assertEqual(returns.shape, (32,))
        self.assertTrue(np.isfinite(logits).all() and np.isfinite(returns).all())
        self.assertEqual(fitted.center.shape, (6,))
        normalized = _apply_normalization(features, fitted.center, fitted.scale)
        self.assertTrue(np.isfinite(normalized).all())

    def test_temperature_and_thresholds_are_finite_and_deterministic(self):
        logits = np.asarray([-2.0, -0.5, 0.5, 2.0])
        hits = np.asarray([0, 0, 1, 1])
        temp, brier = _temperature(logits, hits)
        self.assertGreater(temp, 0)
        self.assertTrue(0 <= brier <= 1)
        levels = _thresholds(np.asarray([-0.03, -0.01, 0.01, 0.04]))
        self.assertEqual(levels[0], float("-inf"))
        self.assertIn(0.0, levels)

    def test_output_guard_rejects_existing_or_outside_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                _safe_output(Path(directory) / "trial")
        with self.assertRaises(ValueError):
            _safe_output(Path(__file__).resolve().parents[1])
