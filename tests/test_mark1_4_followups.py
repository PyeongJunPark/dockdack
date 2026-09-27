"""Offline timing, selection and accounting checks for Mark1.4 follow-ups.

These fixtures are synthetic. They never read the user's market database or
invoke the GUI, broker, or an order path.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dockdack.mark1_4_evolution import EvolutionSamples, simulate_portfolio
from dockdack.mark1_4_data import load_mark14_candidates
from dockdack.mark1_4_followup_models import (extract_engineered_features,
                                               fit_score_variant)
from dockdack.mark1_4_followup_selection import select_calibrated_policy
from dockdack.mark1_4_horizon import (compare_fixed_horizon, load_horizon_closes,
                                      simulate_fixed_horizon)
from dockdack.mark1_4_robust_ga import _fitness as robust_year_fitness, evolve_robust
from test_clean_daily_dataset import catalog_item, create_source, price_rows, sessions
from test_mark1_4_evolution import _synthetic_samples


def _model_splits(samples):
    ordinals = samples.target_ordinals
    return {
        "train": np.flatnonzero(ordinals <= 100),
        "validation": np.flatnonzero((ordinals >= 131) & (ordinals <= 165)),
        "test": np.flatnonzero(ordinals >= 196),
    }


def _horizon_samples(days: int = 9, symbols: int = 2) -> EvolutionSamples:
    calendar = sessions(days)
    bar = np.column_stack((np.full(30, 100.), np.full(30, 101.),
                           np.full(30, 99.), np.full(30, 100.),
                           np.full(30, 100_000.))).astype(np.float32)
    count = days * symbols
    return EvolutionSamples(
        windows=np.repeat(bar[None, :, :], count, axis=0),
        target_dates=np.repeat(np.asarray(calendar, dtype="U10"), symbols),
        target_ordinals=np.repeat(np.arange(days, dtype=np.int32), symbols),
        symbol_ids=np.tile(np.arange(symbols, dtype=np.int32), days),
        entry_open=np.full(count, 100., dtype=np.float64),
        exit_close=np.full(count, 101., dtype=np.float64),
        source={"synthetic": True, "research_only": True},
    )


def _four_year_samples():
    samples, _ = _synthetic_samples(symbols=4, days=330)
    ordinal = samples.target_ordinals
    labels = np.empty(len(ordinal), dtype="U10")
    for index, day in enumerate(ordinal):
        if day <= 249:
            year = 2018 + (int(day) - 30) // 55
            offset = (int(day) - 30) % 55
        elif day <= 294:
            year, offset = 2022, int(day) - 250
        else:
            year, offset = 2023, int(day) - 295
        labels[index] = (date(year, 1, 1) + timedelta(days=offset)).isoformat()
    samples = replace(samples, target_dates=labels,
                      exit_close=samples.entry_open * 1.02)
    splits = {
        "train": np.flatnonzero(ordinal <= 249),
        "validation": np.flatnonzero((ordinal >= 280) & (ordinal <= 294)),
        "test": np.flatnonzero(ordinal >= 325),
    }
    return samples, splits


class FollowupModelTests(unittest.TestCase):
    def test_all_variants_fit_on_train_only_and_return_same_sample_universe(self):
        samples, _ = _synthetic_samples()
        split = _model_splits(samples)
        holdout = np.concatenate((split["validation"], split["test"]))
        modified_exit = samples.exit_close.copy()
        modified_exit[holdout] *= 3.
        changed_targets = replace(samples, exit_close=modified_exit)
        for variant in ("e1_rank", "e2_uncertainty", "e5_features"):
            with self.subTest(variant=variant):
                settings = dict(seed=13, device="cpu", epochs=2,
                                batch_size=64, cost_bps=20)
                scores, artifact = fit_score_variant(samples, split["train"],
                                                     variant, **settings)
                again, repeat = fit_score_variant(samples, split["train"],
                                                  variant, **settings)
                altered, after = fit_score_variant(changed_targets,
                                                   split["train"], variant,
                                                   **settings)
                self.assertEqual(scores.shape, (len(samples.windows),))
                self.assertTrue(np.isfinite(scores).all())
                np.testing.assert_array_equal(scores, again)
                np.testing.assert_array_equal(scores, altered)
                self.assertEqual(artifact, repeat)
                self.assertEqual(artifact, after)
                self.assertEqual(artifact["variant"], variant)
                self.assertTrue(artifact["research_only"])
                self.assertFalse(artifact["deployment_allowed"])
                self.assertEqual(artifact["cost_bps"], 20)
                self.assertEqual(artifact["train_rows_requested"], len(split["train"]))
                json.dumps(artifact, allow_nan=False)
                if variant == "e1_rank":
                    self.assertGreater(artifact["train_pair_count_last_epoch"], 0)
                if variant == "e2_uncertainty":
                    self.assertEqual(artifact["score_formula"], "mean_minus_one_sigma")
                    mean = np.asarray(artifact["prediction_components"]["predicted_mean_percent"])
                    sigma = np.asarray(artifact["prediction_components"]["predicted_sigma_percent"])
                    np.testing.assert_allclose(scores, mean - sigma,
                                               rtol=1e-5, atol=1e-5)

    def test_engineered_features_use_only_completed_window(self):
        samples, _ = _synthetic_samples()
        features = extract_engineered_features(samples.windows)
        self.assertEqual(features.shape[0], len(samples.windows))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(np.isfinite(features).all())
        changed_outcomes = replace(samples, entry_open=samples.entry_open * 2,
                                   exit_close=samples.exit_close * .5)
        np.testing.assert_array_equal(features,
                                      extract_engineered_features(changed_outcomes.windows))


class FollowupHorizonTests(unittest.TestCase):
    def test_daily_and_horizon_trace_arrays_are_aligned_and_unknown_stays_null(self):
        samples = _horizon_samples()
        rows = np.arange(len(samples.windows))
        scores = np.ones(len(rows), dtype=np.float64)
        daily = simulate_portfolio(samples, rows, scores, cost_bps=20)
        self.assertEqual(daily["daily_ordinals"], list(range(9)))
        self.assertEqual(daily["daily_signals"], [2] * 9)
        self.assertEqual(daily["daily_selected_symbol_ids"], [[0, 1]] * 9)
        self.assertEqual(len(daily["daily_drawdowns"]), 9)
        self.assertTrue(all(value is not None for value in daily["daily_drawdowns"]))
        without_day_four = rows[samples.target_ordinals[rows] != 4]
        gap = simulate_portfolio(samples, without_day_four, scores, cost_bps=20)
        self.assertEqual(gap["daily_ordinals"], list(range(9)))
        self.assertEqual(gap["daily_signals"][4], 0)
        self.assertEqual(gap["daily_selected_symbol_ids"][4], [])
        self.assertIsNone(gap["daily_dates"][4])
        missing_one_day = samples.exit_close.copy()
        missing_one_day[0] = np.nan
        unknown = simulate_portfolio(replace(samples, exit_close=missing_one_day),
                                     rows, scores, cost_bps=20)
        self.assertEqual(unknown["daily_signals"], [2] * 9)
        self.assertEqual(unknown["daily_selected_symbol_ids"], [[0, 1]] * 9)
        self.assertEqual(unknown["daily_drawdowns"], [None] * 9)
        self.assertEqual(unknown["daily_returns"], [None] * 9)
        held = simulate_fixed_horizon(samples, rows, scores,
                                      np.full(len(rows), 103.), horizon=3,
                                      split_start_ordinal=0,
                                      split_end_ordinal=8)
        self.assertEqual(held["entry_ordinals"], [0, 3, 6])
        self.assertEqual(held["entry_signal_counts"], [2, 2, 2])
        self.assertEqual(held["entry_selected_symbol_ids"], [[0, 1]] * 3)
        self.assertEqual(len(held["drawdowns_at_exit_points"]), 3)
        missing_horizon_exit = np.full(len(rows), 103.)
        missing_horizon_exit[0] = np.nan
        unknown_held = simulate_fixed_horizon(samples, rows, scores,
                                              missing_horizon_exit, horizon=3,
                                              split_start_ordinal=0,
                                              split_end_ordinal=8)
        self.assertEqual(unknown_held["entry_signal_counts"], [2, 2, 2])
        self.assertEqual(unknown_held["entry_selected_symbol_ids"], [[0, 1]] * 3)
        self.assertEqual(unknown_held["drawdowns_at_exit_points"], [None] * 3)
        self.assertEqual(unknown_held["realized_block_returns"], [None] * 3)

    def test_horizon_exit_is_future_outcome_only_and_raw_db_is_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            calendar = tuple(sessions(220))
            ordinary_rows = price_rows(calendar, count=len(calendar))
            changed_rows = [dict(row) for row in ordinary_rows]
            changed_rows[207]["close"] = "101"
            base_path = Path(folder) / "base.sqlite3"
            changed_path = Path(folder) / "future_changed.sqlite3"
            create_source(base_path, ordinary_rows, [catalog_item()], "domestic")
            create_source(changed_path, changed_rows, [catalog_item()], "domestic")
            outputs = []
            for path in (base_path, changed_path):
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                samples = load_mark14_candidates(
                    path, "domestic", start=calendar[0],
                    calibration_end=calendar[95], train_end=calendar[130],
                    test_end=calendar[-1], max_symbols=1, session_dates=calendar)
                future_closes = load_horizon_closes(path, samples, calendar,
                                                     horizons=(3, 5))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
                outputs.append((samples, future_closes))
            old_samples, old_closes = outputs[0]
            new_samples, new_closes = outputs[1]
            old = np.flatnonzero(old_samples.target_dates == calendar[205])
            new = np.flatnonzero(new_samples.target_dates == calendar[205])
            self.assertEqual((len(old), len(new)), (1, 1))
            np.testing.assert_array_equal(old_samples.windows[old],
                                          new_samples.windows[new])
            self.assertEqual(old_samples.entry_open[old[0]],
                             new_samples.entry_open[new[0]])
            self.assertEqual(old_closes[3][old[0]], 100.)
            self.assertEqual(new_closes[3][new[0]], 101.)
            self.assertEqual(old_closes[5][old[0]], new_closes[5][new[0]])
            tail = np.flatnonzero(old_samples.target_dates == calendar[-2])
            self.assertEqual(len(tail), 1)
            self.assertTrue(np.isnan(old_closes[5][tail[0]]))

    def test_nonoverlapping_horizon_three_compounds_ten_percent_and_cost(self):
        samples = _horizon_samples()
        rows = np.arange(len(samples.windows))
        scores = np.ones(len(rows), dtype=np.float64)
        horizon_close = np.full(len(rows), 103., dtype=np.float64)
        result = simulate_fixed_horizon(samples, rows, scores, horizon_close,
                                        horizon=3, threshold=0, cost_bps=20,
                                        allocation=.1, max_positions=10,
                                        initial_equity=10_000_000.,
                                        split_start_ordinal=0,
                                        split_end_ordinal=8)
        self.assertEqual(result["signals"], 6)
        self.assertEqual(result["unresolved_signals"], 0)
        self.assertEqual(result["executed_trades"], 6)
        equity = 10_000_000.
        for _ in range(3):
            shares = math.floor(.1 * equity / (100 * 1.001))
            equity += 2 * shares * (103 * .999 - 100 * 1.001)
        self.assertAlmostEqual(result["final_equity"], equity, places=5)
        self.assertAlmostEqual(result["compound_net_return"], equity / 10_000_000 - 1)
        json.dumps(result, allow_nan=False)

    def test_horizon_missing_exit_never_becomes_zero_return_or_fill(self):
        samples = _horizon_samples()
        rows = np.arange(len(samples.windows))
        exit_close = np.full(len(rows), 103., dtype=np.float64)
        exit_close[0] = np.nan
        result = simulate_fixed_horizon(samples, rows, np.ones(len(rows)),
                                        exit_close, horizon=3,
                                        split_start_ordinal=0,
                                        split_end_ordinal=8)
        self.assertGreaterEqual(result["unresolved_signals"], 1)
        self.assertIsNone(result["final_equity"])
        self.assertIsNone(result["compound_net_return"])
        json.dumps(result, allow_nan=False)

    def test_five_session_sensitivity_has_nonoverlapping_entries(self):
        samples = _horizon_samples(days=10)
        rows = np.arange(len(samples.windows))
        held = simulate_fixed_horizon(samples, rows, np.ones(len(rows)),
                                      np.full(len(rows), 104.), horizon=5,
                                      cost_bps=20, split_start_ordinal=0,
                                      split_end_ordinal=9)
        self.assertEqual(held["entry_ordinals"], [0, 5])
        self.assertEqual(held["exit_ordinals"], [4, 9])
        self.assertEqual(held["entry_stride_sessions"], 5)
        self.assertEqual(held["signals"], 4)
        self.assertEqual(held["executed_trades"], 4)
        shorter_boundary = simulate_fixed_horizon(
            samples, rows[:18], np.ones(len(rows)), np.full(len(rows), 104.),
            horizon=5, split_start_ordinal=0, split_end_ordinal=8)
        self.assertEqual(shorter_boundary["entry_ordinals"], [0])
        self.assertEqual(shorter_boundary["ineligible_trailing_entry_sessions"], 1)

    def test_one_day_comparator_uses_identical_nonoverlapping_entry_cohort(self):
        samples = _horizon_samples()
        rows = np.arange(len(samples.windows))
        horizon_close = np.full(len(rows), 103., dtype=np.float64)
        report = compare_fixed_horizon(samples, rows, np.ones(len(rows)),
                                       horizon_close, horizon=3,
                                       split_start_ordinal=0,
                                       split_end_ordinal=8)
        one = report["one_day"]
        longer = report["multi_day"]
        self.assertEqual(one["signals"], longer["signals"])
        self.assertEqual(one["signals"], 6)
        self.assertGreater(longer["compound_net_return"], one["compound_net_return"])
        json.dumps(report, allow_nan=False)


class FollowupRobustGATests(unittest.TestCase):
    def test_e3_does_not_substitute_lower_qualified_genome_for_best_unqualified(self):
        samples, splits = _four_year_samples()
        calls = 0

        def staged_fitness(_year_results):
            nonlocal calls
            calls += 1
            if calls == 1:
                return 2., True, False  # Highest fitness but fails annual gate.
            if calls == 2:
                return 1., True, True  # Lower fitness, otherwise qualified.
            return (2., True, False) if calls > 8 else (-.02, False, False)

        with patch("dockdack.mark1_4_robust_ga._fitness", side_effect=staged_fitness):
            report, genome = evolve_robust(samples, splits, seed=19,
                                           population_size=8, generations=1,
                                           device="cpu")
        self.assertIsNone(genome)
        self.assertIsNone(report["selected_strategy"])
        self.assertTrue(report["no_train_qualified_strategy"])
        self.assertIsNotNone(report["best_active_exploratory"])
        self.assertIsNotNone(report["best_qualified_but_not_selected_sha256"])
        trials = report["search"]["trial_records"]
        self.assertEqual(len(trials), 8)
        self.assertEqual(trials[0]["fitness"], 2.)
        self.assertFalse(trials[0]["qualified"])
        self.assertEqual(trials[1]["fitness"], 1.)
        self.assertTrue(trials[1]["qualified"])
        self.assertEqual(report["search"]["qualified_trials"], 1)
        json.dumps(report, allow_nan=False)

    def test_worst_training_year_and_hard_activity_missing_gates(self):
        years = {
            str(2018 + index): {
                "incomplete_data": False,
                "executed_trades": 20,
                "executed_sessions": 10,
                "annualized_mean_daily_return": [.10, .20, .30, .40][index],
                "max_drawdown": [-.20, -.10, -.10, -.10][index],
                "compound_net_return": .01,
            }
            for index in range(4)
        }
        fitness, active, qualified = robust_year_fitness(years)
        self.assertAlmostEqual(fitness, .10 - .25 * .20)
        self.assertTrue(active)
        self.assertTrue(qualified)
        fewer_trades = {key: dict(value) for key, value in years.items()}
        fewer_trades["2018"]["executed_trades"] = 19
        self.assertEqual(robust_year_fitness(fewer_trades)[1:], (False, False))
        inactive_year = {key: dict(value) for key, value in years.items()}
        inactive_year["2019"]["executed_sessions"] = 9
        self.assertEqual(robust_year_fitness(inactive_year)[1:], (False, False))
        losing_year = {key: dict(value) for key, value in years.items()}
        losing_year["2020"]["compound_net_return"] = -.001
        self.assertEqual(robust_year_fitness(losing_year)[1:], (True, False))
        missing_year = {key: dict(value) for key, value in years.items()}
        missing_year["2021"]["incomplete_data"] = True
        self.assertEqual(robust_year_fitness(missing_year), (-1_000_000., False, False))

    def test_e3_threshold_and_genome_ignore_holdout_changes(self):
        samples, splits = _four_year_samples()
        settings = dict(seed=19, population_size=16, generations=2,
                        device="cpu", cost_bps_train=40,
                        cost_bps_eval=20)
        original, genome = evolve_robust(samples, splits, **settings)
        windows = samples.windows.copy()
        holdout = np.concatenate((splits["validation"], splits["test"]))
        windows[holdout, -5:, :4] *= 1.1
        exits = samples.exit_close.copy()
        exits[holdout] *= .5
        changed = replace(samples, windows=windows, exit_close=exits)
        altered, altered_genome = evolve_robust(changed, splits, **settings)
        self.assertEqual(original["search"], altered["search"])
        self.assertFalse(original["search"]["validation_or_test_used_in_selection"])
        self.assertEqual(original["search"]["train_roundtrip_cost_bps"], 40)
        self.assertEqual(original["search"]["evaluation_roundtrip_cost_bps"], 20)
        self.assertEqual(original["search"]["train_years"],
                         ["2018", "2019", "2020", "2021"])
        self.assertIsNotNone(original["best_active_exploratory"])
        self.assertIsNotNone(original["selected_strategy"])
        self.assertEqual(original["selected_strategy"] is None,
                         altered["selected_strategy"] is None)
        for label in ("best_active_exploratory", "selected_strategy"):
            first, second = original[label], altered[label]
            self.assertEqual(first is None, second is None)
            if first is not None:
                self.assertEqual(first["genome_sha256"], second["genome_sha256"])
                self.assertEqual(first["frozen_numeric_score_threshold"],
                                 second["frozen_numeric_score_threshold"])
                self.assertEqual(first["threshold_provenance"],
                                 second["threshold_provenance"])
                self.assertEqual(first["train_year_results_at_40bp"],
                                 second["train_year_results_at_40bp"])
        if genome is None:
            self.assertIsNone(altered_genome)
        else:
            np.testing.assert_array_equal(genome, altered_genome)
        json.dumps(original, allow_nan=False)
        json.dumps(altered, allow_nan=False)


class FollowupCalibrationTests(unittest.TestCase):
    def _fixture(self):
        samples, _ = _four_year_samples()
        years = np.asarray(samples.target_dates)
        train = np.flatnonzero((years >= "2018-01-01") & (years <= "2020-12-31"))
        calibration = np.flatnonzero((years >= "2021-01-01") & (years <= "2021-12-31"))
        holdout = np.flatnonzero(years >= "2022-01-01")
        scores = np.zeros(len(samples.windows), dtype=np.float32)
        scores[train] = np.linspace(-1., 1., len(train), dtype=np.float32)
        scores[calibration] = 2.0
        scores[holdout] = 3.0
        return samples, train, calibration, holdout, scores

    def test_train_threshold_and_2021_choice_ignore_later_prices_and_scores(self):
        samples, train, calibration, holdout, scores = self._fixture()
        source = select_calibrated_policy(
            samples, {13: scores}, train, calibration,
            target_coverages=(.001, .02, .05), min_executed_trades=20)
        self.assertEqual(source["decision"], "selected")
        self.assertFalse(source["later_outcomes_or_scores_used"])
        self.assertEqual(source["evaluated_seed_coverage_pairs"], 3)
        self.assertTrue(source["research_only"])
        self.assertFalse(source["deployment_allowed"])
        for row in source["candidates"]:
            q = row["target_train_candidate_coverage"]
            self.assertAlmostEqual(row["train_numeric_score_threshold"],
                                   float(np.quantile(scores[train].astype(np.float64),
                                                     1 - q, method="linear")))
            self.assertEqual(row["status"], "eligible_above_cash")
            self.assertEqual(row["calibration"]["cost_bps"], 20.)
            self.assertEqual(row["calibration"]["allocation"], .1)
            self.assertEqual(row["calibration"]["max_positions"], 10)
        changed_score = scores.copy()
        changed_score[holdout] = np.nan
        changed_exit = samples.exit_close.copy()
        changed_exit[holdout] = np.nan
        changed_samples = replace(samples, exit_close=changed_exit)
        altered = select_calibrated_policy(
            changed_samples, {13: changed_score}, train, calibration,
            target_coverages=(.001, .02, .05), min_executed_trades=20)
        self.assertEqual(source, altered)
        json.dumps(source, allow_nan=False)

    def test_best_eligible_is_selected_even_when_losing_but_invalid_paths_choose_cash(self):
        samples, train, calibration, _, scores = self._fixture()
        losing_exit = samples.exit_close.copy()
        losing_exit[calibration] = samples.entry_open[calibration] * .99
        selective_scores = scores.copy()
        selective_scores[calibration[samples.symbol_ids[calibration] >= 2]] = -2.
        losing = select_calibrated_policy(
            replace(samples, exit_close=losing_exit),
            {13: scores, 14: selective_scores},
            train, calibration, target_coverages=(.02,),
            min_executed_trades=20)
        self.assertEqual(losing["decision"], "selected")
        self.assertFalse(losing["no_eligible_strategy"])
        self.assertEqual(losing["selected"]["seed"], 14)
        self.assertLess(losing["selected"]["calibration_objective"], 0)
        self.assertEqual(losing["selected"]["calibration_objective"],
                         max(row["calibration_objective"] for row in losing["candidates"]))
        self.assertTrue(all(row["status"] == "eligible_below_or_equal_cash"
                            for row in losing["candidates"]))
        missing_exit = samples.exit_close.copy()
        missing_exit[calibration[0]] = np.nan
        missing = select_calibrated_policy(
            replace(samples, exit_close=missing_exit), {13: scores},
            train, calibration, target_coverages=(.02,),
            min_executed_trades=20)
        self.assertEqual(missing["decision"], "cash")
        self.assertEqual(missing["candidates"][0]["status"],
                         "incomplete_selected_outcome")
        too_few = select_calibrated_policy(
            samples, {13: scores}, train, calibration,
            target_coverages=(.02,), min_executed_trades=300)
        self.assertEqual(too_few["decision"], "cash")
        self.assertEqual(too_few["candidates"][0]["status"],
                         "insufficient_integer_share_fills")
        for report in (losing, missing, too_few):
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
