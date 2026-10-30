"""Offline Mark1.4-v2 sparse neuroevolution and train-only selection checks."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import unittest

import numpy as np

from dockdack.mark1_4_evolution import GENOME_SIZE, simulate_portfolio
from dockdack.mark1_4_sparse import (derive_train_threshold, evolve_sparse,
                                     qualifies_sparse_strategy, score_sparse_genome)
from test_mark1_4_evolution import _synthetic_samples


def _splits(samples):
    dates = samples.target_ordinals
    return {
        "train": np.flatnonzero(dates <= 100),
        "validation": np.flatnonzero((dates >= 131) & (dates <= 165)),
        "test": np.flatnonzero(dates >= 196),
    }


def _all_positive_samples():
    samples, _ = _synthetic_samples()
    return replace(samples, exit_close=samples.entry_open * 1.02)


class Mark14SparseTests(unittest.TestCase):
    def test_final_score_layer_is_linear_not_tanh_saturated(self):
        samples, _ = _synthetic_samples(symbols=1, days=35)
        genome = np.zeros(GENOME_SIZE, dtype=np.float32)
        genome[-2] = 2.0  # The output bias, immediately before log(q).
        genome[-1] = math.log(.05)
        scores = score_sparse_genome(samples.windows, genome, device="cpu")
        np.testing.assert_array_equal(scores, np.full(len(scores), 2.0))

    def test_train_quantile_threshold_has_bounded_monotone_coverage(self):
        train_scores = np.linspace(-5., 5., 1001, dtype=np.float64)
        thresholds = []
        for target in (.001, .05, .20):
            threshold, provenance = derive_train_threshold(train_scores, math.log(target))
            achieved = float(np.mean(train_scores > threshold))
            self.assertEqual(provenance["source"], "train_scores_only")
            self.assertEqual(provenance["train_candidate_count"], len(train_scores))
            self.assertFalse(provenance["validation_or_test_used"])
            self.assertTrue(provenance["quantile_method"].startswith("linear"))
            self.assertAlmostEqual(provenance["numeric_score_threshold"], threshold)
            self.assertAlmostEqual(provenance["target_train_candidate_coverage"], target)
            self.assertAlmostEqual(provenance["achieved_train_candidate_coverage"], achieved)
            self.assertGreater(achieved, 0)
            self.assertLessEqual(achieved, target + 1 / len(train_scores))
            thresholds.append(threshold)
        self.assertGreater(thresholds[0], thresholds[1])
        self.assertGreater(thresholds[1], thresholds[2])
        tied_threshold, tied = derive_train_threshold(np.ones(100), math.log(.20))
        self.assertEqual(tied_threshold, 1.)
        self.assertEqual(tied["achieved_train_candidate_coverage"], 0.)
        with self.assertRaises(ValueError):
            derive_train_threshold(train_scores, math.log(.0009))
        with self.assertRaises(ValueError):
            derive_train_threshold(train_scores, math.log(.21))

    def test_activity_minimum_profit_and_missing_path_are_hard_gates(self):
        samples = _all_positive_samples()
        train = _splits(samples)["train"]
        scores = np.ones(len(samples.windows), dtype=np.float64)
        good = simulate_portfolio(samples, train, scores, cost_bps=20)
        self.assertGreaterEqual(good["executed_sessions"], 20)
        self.assertGreaterEqual(good["executed_trades"], 50)
        self.assertGreater(good["compound_net_return"], 0)
        self.assertTrue(qualifies_sparse_strategy(good))
        self.assertFalse(qualifies_sparse_strategy({**good, "executed_sessions": 19}))
        self.assertFalse(qualifies_sparse_strategy({**good, "executed_trades": 49}))
        self.assertFalse(qualifies_sparse_strategy({**good, "compound_net_return": 0.}))
        missing_open = samples.entry_open.copy()
        missing_close = samples.exit_close.copy()
        missing_open[train[0]] = np.nan
        missing_close[train[0]] = np.nan
        incomplete = simulate_portfolio(replace(samples, entry_open=missing_open,
                                                exit_close=missing_close),
                                        train, scores, cost_bps=20)
        self.assertTrue(incomplete["incomplete_data"])
        self.assertIsNone(incomplete["compound_net_return"])
        self.assertFalse(qualifies_sparse_strategy(incomplete))
        all_missing_open = samples.entry_open.copy()
        all_missing_close = samples.exit_close.copy()
        all_missing_open[train] = np.nan
        all_missing_close[train] = np.nan
        all_missing = replace(samples, entry_open=all_missing_open,
                              exit_close=all_missing_close)
        report, genome = evolve_sparse(all_missing, _splits(all_missing), seed=7,
                                       population_size=8, generations=1,
                                       device="cpu", cost_bps=20)
        self.assertIsNone(genome)
        self.assertIsNone(report["selected_strategy"])
        self.assertTrue(report["no_profitable_strategy"])
        self.assertEqual(report["search"]["profitable_activity_qualified_trials"], 0)
        json.dumps(report, allow_nan=False)

    def test_losing_trades_never_displace_cash_and_report_is_finite(self):
        samples = _all_positive_samples()
        losing = replace(samples, exit_close=samples.entry_open * .99)
        train = _splits(losing)["train"]
        buy = simulate_portfolio(losing, train, np.ones(len(losing.windows)), cost_bps=20)
        cash = simulate_portfolio(losing, train, np.full(len(losing.windows), -1.), cost_bps=20)
        self.assertLess(buy["compound_net_return"], cash["compound_net_return"])
        self.assertEqual(cash["compound_net_return"], 0.)
        self.assertFalse(qualifies_sparse_strategy(buy))
        report, genome = evolve_sparse(losing, _splits(losing), seed=29,
                                       population_size=16, generations=2,
                                       device="cpu", cost_bps=20)
        self.assertIsNone(genome)
        self.assertIsNone(report["selected_strategy"])
        self.assertTrue(report["no_profitable_strategy"])
        self.assertEqual(report["baselines"]["no_trade"]["test"]["compound_net_return"], 0.)
        self.assertEqual(sum(report["search"]["initial_q_grid_counts"].values()), 16)
        self.assertEqual(len(report["search"]["initial_q_grid_counts"]), 8)
        self.assertFalse(report["search"]["validation_or_test_used_in_selection"])
        json.dumps(report, allow_nan=False)

    def test_same_seed_and_changed_holdout_cannot_change_train_search(self):
        samples = _all_positive_samples()
        splits = _splits(samples)
        settings = dict(seed=31, population_size=16, generations=2,
                        device="cpu", cost_bps=20)
        report_a, genome_a = evolve_sparse(samples, splits, **settings)
        report_b, genome_b = evolve_sparse(samples, splits, **settings)
        modified_windows = samples.windows.copy()
        holdout = np.concatenate((splits["validation"], splits["test"]))
        modified_windows[holdout, -5:, :4] *= 1.1
        modified_close = samples.exit_close.copy()
        modified_close[holdout] *= .5
        changed = replace(samples, windows=modified_windows, exit_close=modified_close)
        report_c, genome_c = evolve_sparse(changed, splits, **settings)
        self.assertEqual(report_a["search"], report_b["search"])
        self.assertEqual(report_a["search"], report_c["search"])
        self.assertEqual(report_a["selected_strategy"] is None,
                         report_b["selected_strategy"] is None)
        self.assertEqual(report_a["selected_strategy"] is None,
                         report_c["selected_strategy"] is None)
        self.assertIsNotNone(report_a["best_active_exploratory"])
        self.assertIsNotNone(report_a["selected_strategy"])
        self.assertEqual(sum(report_a["search"]["initial_q_grid_counts"].values()), 16)
        self.assertEqual(len(report_a["search"]["initial_q_grid_counts"]), 8)
        self.assertFalse(report_a["search"]["validation_or_test_used_in_selection"])
        for label in ("best_active_exploratory", "selected_strategy"):
            original = report_a[label]
            for later in (report_b[label], report_c[label]):
                self.assertEqual(original["genome_sha256"], later["genome_sha256"])
                self.assertEqual(original["frozen_numeric_score_threshold"],
                                 later["frozen_numeric_score_threshold"])
                self.assertEqual(original["threshold_provenance"],
                                 later["threshold_provenance"])
                self.assertEqual(original["candidate_coverage_by_split"]["train"],
                                 later["candidate_coverage_by_split"]["train"])
            self.assertFalse(original["threshold_provenance"]["validation_or_test_used"])
        if genome_a is not None:
            np.testing.assert_array_equal(genome_a, genome_b)
            np.testing.assert_array_equal(genome_a, genome_c)
            train_scores = score_sparse_genome(samples.windows, genome_a, device="cpu")
            expected_threshold, _ = derive_train_threshold(train_scores[splits["train"]],
                                                           float(genome_a[-1]))
            self.assertEqual(report_a["selected_strategy"]["frozen_numeric_score_threshold"],
                             expected_threshold)
        else:
            self.assertIsNone(genome_b)
            self.assertIsNone(genome_c)
        json.dumps(report_a, allow_nan=False)
        json.dumps(report_c, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
