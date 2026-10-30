"""Offline causal selection checks for the target/horizon research utilities."""
from __future__ import annotations

import unittest

import numpy as np

from dockdack.mark1_target_horizon_eval import (
    fit_ridge_return, nonoverlap_block_summary,
)


class RidgeReturnTests(unittest.TestCase):
    def test_fit_is_training_only_and_predicts_without_refitting(self):
        train = np.array([[-2., 1.], [-1., 0.], [0., 1.], [1., 0.], [2., 1.]])
        target = 0.01 + train[:, 0] * 0.02
        fitted = fit_ridge_return(train, target, alpha=0.001)
        original_mean = fitted.feature_mean.copy()
        forecast = fitted.predict(np.array([[100., 1.]]))
        self.assertEqual(forecast.shape, (1,))
        np.testing.assert_array_equal(fitted.feature_mean, original_mean)
        np.testing.assert_allclose(original_mean, train.mean(axis=0))
        self.assertGreater(float(fitted.predict([[1., 0.]])[0]),
                           float(fitted.predict([[-1., 0.]])[0]))

    def test_nonfinite_and_wrong_width_fail_closed(self):
        with self.assertRaises(ValueError):
            fit_ridge_return([[1.], [float("nan")]], [0.01, 0.02])
        fitted = fit_ridge_return([[1.], [2.]], [0.01, 0.02])
        with self.assertRaises(ValueError):
            fitted.predict([[1., 2.]])


class NonoverlapSummaryTests(unittest.TestCase):
    def test_horizon_spacing_top_k_and_realized_block_drawdown(self):
        result = nonoverlap_block_summary(
            scores=[0.7, 0.9, 0.8, 1.0, 0.6, 0.5],
            net_returns=[0.02, -0.1, 0.04, 0.9, -0.2, 0.1],
            target_session_ordinals=[10, 10, 11, 11, 12, 12],
            symbol_ids=[1, 2, 1, 2, 1, 2],
            hits=[True, False, True, True, False, True],
            eligible=[True] * 6, horizon=2, top_k=1, allocation=0.1,
            anchor_ordinal=10,
        )
        self.assertEqual(result["blocks"], 2)
        self.assertEqual(result["trades"], 2)
        self.assertEqual(result["selected_indices"], [1, 4])
        self.assertEqual(result["target_hits"], 0)
        self.assertAlmostEqual(result["total_return"], (1 - 0.01) * (1 - 0.02) - 1)
        self.assertGreater(result["max_realized_block_drawdown"], 0)

    def test_score_floor_leaves_cash_idle(self):
        result = nonoverlap_block_summary(
            scores=[-0.01, -0.02], net_returns=[0.02, -0.5],
            target_session_ordinals=[0, 1], symbol_ids=[1, 1],
            hits=[True, False], eligible=[True, True], horizon=2,
            score_floor=0.0,
        )
        self.assertEqual(result["blocks"], 1)
        self.assertEqual(result["trades"], 0)
        self.assertEqual(result["total_return"], 0.0)

    def test_duplicate_identity_and_overallocation_rejected(self):
        arguments = dict(scores=[0.1, 0.2], net_returns=[0.01, 0.02],
                         target_session_ordinals=[0, 0], symbol_ids=[1, 1],
                         hits=[True, True], eligible=[True, True], horizon=1)
        with self.assertRaises(ValueError):
            nonoverlap_block_summary(**arguments)
        arguments["symbol_ids"] = [1, 2]
        with self.assertRaises(ValueError):
            nonoverlap_block_summary(**arguments, top_k=2, allocation=0.6)


if __name__ == "__main__":
    unittest.main()
