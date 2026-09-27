"""Synthetic, offline regression checks for Mark1.4 research.

All data in this file is generated in memory or a disposable SQLite database.
No test reads the user's market databases or starts an order runner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dockdack.mark1_4_data import load_mark14_candidates, mark14_chronological_splits
from dockdack.mark1_4_evolution import (EvolutionSamples, _training_fitness, evolve,
                                        genome_to_payload, normalize_windows,
                                        score_genome, simulate_missing_scenarios,
                                        simulate_portfolio)
from test_clean_daily_dataset import catalog_item, create_source, price_rows, sessions


def _synthetic_samples(symbols: int = 4, days: int = 210) -> tuple[EvolutionSamples, tuple[str, ...]]:
    calendar = tuple(sessions(days))
    windows, dates, ordinals, symbol_ids, entry_open, exit_close = [], [], [], [], [], []
    for day in range(30, days):
        for symbol in range(symbols):
            age = np.arange(30, dtype=np.float64)
            base = 95 + symbol * 4 + day * .025 + .4 * np.sin((day - 29 + age) / 6 + symbol)
            opened = base * (1 + .001 * np.sin(age + symbol))
            closed = base * (1 + .002 * np.cos(age / 2 + symbol))
            high = np.maximum(opened, closed) * 1.005
            low = np.minimum(opened, closed) * .995
            volume = 50_000 + 100 * day + 500 * symbol + 20 * age
            windows.append(np.column_stack((opened, high, low, closed, volume)))
            dates.append(calendar[day])
            ordinals.append(day)
            symbol_ids.append(symbol)
            entry_open.append(100. + symbol)
            exit_close.append((100. + symbol) * (1 + .006 * math.sin(day / 3 + symbol)))
    samples = EvolutionSamples(
        np.asarray(windows, dtype=np.float32), np.asarray(dates, dtype="U10"),
        np.asarray(ordinals, dtype=np.int32), np.asarray(symbol_ids, dtype=np.int32),
        np.asarray(entry_open, dtype=np.float64), np.asarray(exit_close, dtype=np.float64),
        {"synthetic": True, "research_only": True},
    )
    return samples, calendar


def _two_day_allocation_samples(*, missing_top: bool = False) -> EvolutionSamples:
    """Twelve candidates daily: ten high-ranked +1%, two low-ranked +100%."""
    base = np.linspace(100., 102., 30, dtype=np.float32)
    bar = np.column_stack((base, base * 1.01, base * .99, base, np.full(30, 80_000)))
    windows = np.repeat(bar[None, :, :], 24, axis=0)
    dates = np.repeat(np.array(["2024-01-01", "2024-01-02"], dtype="U10"), 12)
    ordinals = np.repeat(np.array([0, 1], dtype=np.int32), 12)
    ids = np.tile(np.arange(12, dtype=np.int32), 2)
    entry = np.full(24, 100., dtype=np.float64)
    exit_price = np.where(ids < 10, 101., 200.).astype(np.float64)
    if missing_top:
        entry[0] = np.nan
        exit_price[0] = np.nan
    return EvolutionSamples(windows, dates, ordinals, ids, entry, exit_price,
                            {"synthetic": True, "research_only": True})


class Mark14RawDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.calendar = tuple(sessions(220))

    def _rows(self):
        rows = price_rows(self.calendar, count=len(self.calendar))
        for index, row in enumerate(rows):
            opened = 100 + 2 * math.sin(index / 9)
            closed = opened * (1 + .003 * math.cos(index / 4))
            row["open"] = f"{opened:.8f}"
            row["close"] = f"{closed:.8f}"
            row["high"] = f"{max(opened, closed) * 1.01:.8f}"
            row["low"] = f"{min(opened, closed) * .99:.8f}"
        return rows

    def _load(self, rows, filename):
        source = self.folder / filename
        create_source(source, rows, [catalog_item()], "domestic")
        fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
        samples = load_mark14_candidates(
            source, "domestic", start=self.calendar[0],
            calibration_end=self.calendar[95], train_end=self.calendar[130],
            test_end=self.calendar[-1], max_symbols=1, session_dates=self.calendar,
        )
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), fingerprint)
        return samples

    def test_target_day_change_cannot_change_preopen_30_bars(self):
        rows = self._rows()
        changed = deepcopy(rows)
        target = 205
        new_open = float(changed[target]["open"]) * 1.01
        new_close = float(changed[target]["close"]) * 1.015
        changed[target]["open"] = f"{new_open:.8f}"
        changed[target]["close"] = f"{new_close:.8f}"
        changed[target]["high"] = f"{max(new_open, new_close) * 1.01:.8f}"
        changed[target]["low"] = f"{min(new_open, new_close) * .99:.8f}"
        base = self._load(rows, "base.sqlite3")
        altered = self._load(changed, "changed.sqlite3")
        before = np.flatnonzero(base.target_dates == self.calendar[target])
        after = np.flatnonzero(altered.target_dates == self.calendar[target])
        self.assertEqual((len(before), len(after)), (1, 1))
        self.assertEqual(base.windows.shape[1:], (30, 5))
        np.testing.assert_array_equal(base.windows[before], altered.windows[after])
        np.testing.assert_array_equal(
            normalize_windows(base.windows[before]), normalize_windows(altered.windows[after]))
        self.assertNotEqual(base.entry_open[before[0]], altered.entry_open[after[0]])
        self.assertNotEqual(base.exit_close[before[0]], altered.exit_close[after[0]])

    def test_missing_target_is_still_a_candidate_not_a_zero_return(self):
        rows = self._rows()
        rows.pop(205)
        samples = self._load(rows, "missing.sqlite3")
        missing = np.flatnonzero(samples.target_dates == self.calendar[205])
        self.assertEqual(len(missing), 1)
        self.assertEqual(samples.windows[missing].shape, (1, 30, 5))
        self.assertTrue(np.isfinite(samples.windows[missing]).all())
        self.assertTrue(np.isnan(samples.entry_open[missing[0]]))
        self.assertTrue(np.isnan(samples.exit_close[missing[0]]))

    def test_later_liquidity_cannot_rewrite_calibration_top_one(self):
        first = self._rows()
        second = price_rows(self.calendar, count=len(self.calendar), symbol="000660")
        for row in first:
            row["trade_value"] = "2000"
        for index, row in enumerate(second):
            row["trade_value"] = "1000" if index <= 95 else "100000"
        source = self.folder / "calibration.sqlite3"
        create_source(source, first + second,
                      [catalog_item(), catalog_item("000660")], "domestic")
        samples = load_mark14_candidates(
            source, "domestic", start=self.calendar[0],
            calibration_end=self.calendar[95], train_end=self.calendar[130],
            test_end=self.calendar[-1], max_symbols=1, session_dates=self.calendar,
        )
        self.assertEqual(samples.source["selected_symbols"][0]["symbol"], "005930")
        self.assertEqual(set(samples.symbol_ids.tolist()), {0})

    def test_chronological_split_keeps_30_session_embargo(self):
        samples = self._load(self._rows(), "split.sqlite3")
        splits = mark14_chronological_splits(
            samples, self.calendar, train_start=self.calendar[100],
            train_end=self.calendar[130], validation_end=self.calendar[175],
            test_end=self.calendar[-1], embargo_sessions=30,
        )
        self.assertTrue(all(len(splits[name]) for name in ("train", "validation", "test")))
        self.assertLess(samples.target_ordinals[splits["train"]].max() + 30,
                        samples.target_ordinals[splits["validation"]].min())
        self.assertLess(samples.target_ordinals[splits["validation"]].max() + 30,
                        samples.target_ordinals[splits["test"]].min())
        self.assertEqual(len(set(splits["train"]) & set(splits["validation"])), 0)
        self.assertEqual(len(set(splits["validation"]) & set(splits["test"])), 0)


class Mark14EvolutionTests(unittest.TestCase):
    def test_missing_target_scenarios_are_diagnostics_not_exact_equity(self):
        samples = _two_day_allocation_samples(missing_top=True)
        scores = (12 - samples.symbol_ids).astype(np.float64)
        rows = np.arange(len(scores))
        exact = simulate_portfolio(samples, rows, scores, cost_bps=20)
        scenarios = simulate_missing_scenarios(samples, rows, scores, cost_bps=20)
        self.assertIsNone(exact["final_equity"])
        self.assertIsNone(exact["compound_net_return"])
        self.assertEqual(scenarios["unresolved_selected"], 1)
        initial = exact["initial_equity"]
        gain_per_share = 101 * .999 - 100 * 1.001
        expected = {}
        for name in ("no_fill", "full_budget_loss"):
            capital = float(initial)
            first_shares = math.floor(capital * .1 / (100 * 1.001))
            capital += 9 * first_shares * gain_per_share
            if name == "full_budget_loss":
                capital -= .1 * initial
            second_shares = math.floor(capital * .1 / (100 * 1.001))
            capital += 10 * second_shares * gain_per_share
            expected[name] = capital
            scenario = scenarios[name]
            self.assertAlmostEqual(scenario["final_equity"], capital, places=5)
            self.assertAlmostEqual(scenario["compound_net_return"], capital / initial - 1)
            self.assertEqual(scenario["unresolved_signals"], 1)
            self.assertTrue(all(value is not None for value in scenario["daily_returns"]))
        self.assertGreater(expected["no_fill"], expected["full_budget_loss"])
        # A scenario never writes an invented value into the exact path.
        self.assertIsNone(exact["final_equity"])
        self.assertIsNone(exact["compound_net_return"])
        json.dumps(scenarios, allow_nan=False)

    def test_fully_observed_missing_scenarios_equal_exact_path(self):
        samples = _two_day_allocation_samples()
        scores = (12 - samples.symbol_ids).astype(np.float64)
        rows = np.arange(len(scores))
        exact = simulate_portfolio(samples, rows, scores, cost_bps=20)
        scenarios = simulate_missing_scenarios(samples, rows, scores, cost_bps=20)
        self.assertEqual(scenarios["unresolved_selected"], 0)
        for name in ("no_fill", "full_budget_loss"):
            scenario = scenarios[name]
            self.assertEqual(scenario["final_equity"], exact["final_equity"])
            self.assertEqual(scenario["compound_net_return"], exact["compound_net_return"])
            self.assertEqual(scenario["daily_returns"], exact["daily_returns"])
            self.assertEqual(scenario["executed_trades"], exact["executed_trades"])
        json.dumps(scenarios, allow_nan=False)

    def test_interior_exchange_session_without_candidates_is_flat_cash_day(self):
        regular = _two_day_allocation_samples()
        scores = (12 - regular.symbol_ids).astype(np.float64)
        rows = np.arange(len(scores))
        baseline = simulate_portfolio(regular, rows, scores, cost_bps=20)
        dates = regular.target_dates.copy()
        dates[12:] = "2024-01-03"
        ordinals = regular.target_ordinals.copy()
        ordinals[12:] = 2
        gap = replace(regular, target_dates=dates, target_ordinals=ordinals)
        with_gap = simulate_portfolio(gap, rows, scores, cost_bps=20)
        self.assertEqual(with_gap["sessions"], 3)
        self.assertEqual(with_gap["calendar_sessions_missing_candidate"], 1)
        self.assertEqual(with_gap["daily_dates"], ["2024-01-01", None, "2024-01-03"])
        self.assertEqual(with_gap["daily_returns"][1], 0.)
        self.assertEqual(with_gap["active_sessions"], 2)
        self.assertAlmostEqual(with_gap["final_equity"], baseline["final_equity"])
        self.assertAlmostEqual(with_gap["annualized_mean_daily_return"],
                               252 * sum(baseline["daily_returns"]) / 3)

    def test_zero_share_and_no_trade_paths_have_no_positive_fitness(self):
        samples = _two_day_allocation_samples()
        rows = np.arange(len(samples.windows))
        selected_scores = (12 - samples.symbol_ids).astype(np.float64)
        too_expensive = simulate_portfolio(samples, rows, selected_scores,
                                           initial_equity=100., cost_bps=20)
        self.assertEqual(too_expensive["signals"], 20)
        self.assertEqual(too_expensive["executed_trades"], 0)
        self.assertEqual(too_expensive["final_equity"], 100.)
        self.assertLess(_training_fitness(too_expensive), 0)
        no_trade = simulate_portfolio(samples, rows, np.full(len(rows), -1.),
                                      initial_equity=10_000_000.)
        self.assertEqual(no_trade["signals"], 0)
        self.assertEqual(no_trade["executed_trades"], 0)
        self.assertLess(_training_fitness(no_trade), 0)
        missing = _two_day_allocation_samples(missing_top=True)
        unresolved = simulate_portfolio(missing, rows, selected_scores)
        self.assertLess(_training_fitness(unresolved), 0)

    def test_top_ten_use_ten_percent_current_equity_with_cost_and_integer_shares(self):
        samples = _two_day_allocation_samples()
        scores = (12 - samples.symbol_ids).astype(np.float64)
        indices = np.arange(len(scores))
        result = simulate_portfolio(samples, indices, scores, threshold=0, cost_bps=20,
                                    allocation=.1, max_positions=10)
        self.assertEqual(result["sessions"], 2)
        self.assertEqual(result["signals"], 20)
        self.assertEqual(result["observed_signals"], 20)
        self.assertEqual(result["unresolved_signals"], 0)
        self.assertEqual(result["active_sessions"], 2)
        self.assertFalse(result["incomplete_data"])
        initial = result["initial_equity"]
        self.assertGreaterEqual(initial, 1_001)
        equity = float(initial)
        expected_daily = []
        for _ in range(2):
            shares_per_position = math.floor(equity * .1 / (100 * 1.001))
            self.assertGreater(shares_per_position, 0)
            following = equity + 10 * shares_per_position * (101 * .999 - 100 * 1.001)
            expected_daily.append(following / equity - 1)
            equity = following
        self.assertAlmostEqual(result["final_equity"], equity, places=5)
        np.testing.assert_allclose(result["daily_returns"], expected_daily, rtol=0, atol=1e-10)
        no_cost = simulate_portfolio(samples, indices, scores, threshold=0, cost_bps=0,
                                     allocation=.1, max_positions=10)
        self.assertEqual(no_cost["signals"], 20)
        self.assertGreater(no_cost["final_equity"], result["final_equity"])

    def test_missing_selected_outcome_never_becomes_a_fill_or_flat_day(self):
        samples = _two_day_allocation_samples(missing_top=True)
        scores = (12 - samples.symbol_ids).astype(np.float64)
        result = simulate_portfolio(samples, np.arange(len(scores)), scores,
                                    threshold=0, cost_bps=20, allocation=.1,
                                    max_positions=10)
        self.assertEqual(result["signals"], 20)
        self.assertEqual(result["unresolved_signals"], 1)
        self.assertEqual(result["unresolved_sessions"], 1)
        self.assertTrue(result["incomplete_data"])
        self.assertIsNone(result["final_equity"])
        self.assertIsNone(result["compound_net_return"])
        self.assertIsNone(result["daily_returns"][0])
        self.assertIsNone(result["daily_returns"][1])

    def test_same_seed_same_champion_and_holdout_labels_cannot_select_it(self):
        samples, _ = _synthetic_samples()
        splits = {
            "train": np.flatnonzero(samples.target_ordinals <= 100),
            "validation": np.flatnonzero((samples.target_ordinals >= 131) &
                                         (samples.target_ordinals <= 165)),
            "test": np.flatnonzero(samples.target_ordinals >= 196),
        }
        self.assertTrue(all(len(part) for part in splits.values()))
        self.assertLess(samples.target_ordinals[splits["train"]].max() + 30,
                        samples.target_ordinals[splits["validation"]].min())
        self.assertLess(samples.target_ordinals[splits["validation"]].max() + 30,
                        samples.target_ordinals[splits["test"]].min())
        from dockdack.broker.kiwoom import KiwoomBroker
        with patch.object(KiwoomBroker, "place_order", side_effect=AssertionError("research sent an order")):
            report_a, genome_a = evolve(samples, splits, seed=17, population_size=6,
                                        generations=2, device="cpu", cost_bps=20)
            report_b, genome_b = evolve(samples, splits, seed=17, population_size=6,
                                        generations=2, device="cpu", cost_bps=20)
            changed_exit = samples.exit_close.copy()
            changed_exit[np.concatenate((splits["validation"], splits["test"]))] *= 3.0
            changed = replace(samples, exit_close=changed_exit)
            report_c, genome_c = evolve(changed, splits, seed=17, population_size=6,
                                        generations=2, device="cpu", cost_bps=20)
            missing_open = samples.entry_open.copy()
            missing_close = samples.exit_close.copy()
            missing_open[splits["test"][0]] = np.nan
            missing_close[splits["test"][0]] = np.nan
            missing_holdout = replace(samples, entry_open=missing_open,
                                      exit_close=missing_close)
            report_d, genome_d = evolve(missing_holdout, splits, seed=17,
                                        population_size=6, generations=2,
                                        device="cpu", cost_bps=20)
        self.assertTrue(report_a["research_only"])
        self.assertFalse(report_a["deployment_allowed"])
        self.assertIn("train", report_a["results"])
        self.assertIn("validation", report_a["results"])
        self.assertIn("test", report_a["results"])
        np.testing.assert_array_equal(score_genome(samples.windows, genome_a, device="cpu"),
                                      score_genome(samples.windows, genome_b, device="cpu"))
        np.testing.assert_array_equal(score_genome(samples.windows, genome_a, device="cpu"),
                                      score_genome(samples.windows, genome_c, device="cpu"))
        np.testing.assert_array_equal(score_genome(samples.windows, genome_a, device="cpu"),
                                      score_genome(samples.windows, genome_d, device="cpu"))
        self.assertEqual(report_a["search"], report_b["search"])
        self.assertEqual(report_a["search"], report_c["search"])
        self.assertEqual(report_a["search"], report_d["search"])
        self.assertEqual(report_a["active_contender"] is None,
                         report_d["active_contender"] is None)
        if report_a["active_contender"] is not None:
            self.assertEqual(report_a["active_contender"]["genome_sha256"],
                             report_d["active_contender"]["genome_sha256"])
            self.assertEqual(report_a["active_contender"]["genome"],
                             report_d["active_contender"]["genome"])
            self.assertEqual(report_a["active_contender"]["train_fitness"],
                             report_d["active_contender"]["train_fitness"])
        missing_sensitivity = report_d["missing_target_scenarios"]
        self.assertTrue(missing_sensitivity["research_only"])
        self.assertTrue(missing_sensitivity["not_exact"])
        self.assertTrue(missing_sensitivity["not_used_for_genetic_selection"])
        always_test = missing_sensitivity["baselines"]["always_buy_top10_liquidity_rank"]["test"]
        self.assertGreaterEqual(always_test["unresolved_selected"], 1)
        self.assertIsNone(report_d["baselines"]["always_buy_top10_liquidity_rank"]
                          ["test"]["compound_net_return"])
        self.assertIsNotNone(always_test["no_fill"]["compound_net_return"])
        self.assertIsNotNone(always_test["full_budget_loss"]["compound_net_return"])
        self.assertGreater(always_test["no_fill"]["final_equity"],
                           always_test["full_budget_loss"]["final_equity"])
        self.assertEqual(set(report_a["baselines"]),
                         {"no_trade", "always_buy_top10_liquidity_rank",
                          "positive_5day_momentum_top10"})
        for part in ("train", "validation", "test"):
            no_trade = report_a["baselines"]["no_trade"][part]
            self.assertEqual(no_trade["signals"], 0)
            self.assertEqual(no_trade["executed_trades"], 0)
            self.assertEqual(no_trade["compound_net_return"], 0.)
            self.assertEqual(no_trade["final_equity"], report_a["policy"]["initial_equity"])
            self.assertGreater(report_a["baselines"]["always_buy_top10_liquidity_rank"]
                               [part]["signals"], 0)
            complete_scenario = report_a["missing_target_scenarios"]["champion"][part]
            self.assertEqual(complete_scenario["unresolved_selected"], 0)
            for assumption in ("no_fill", "full_budget_loss"):
                self.assertEqual(complete_scenario[assumption]["final_equity"],
                                 report_a["results"][part]["final_equity"])
        self.assertEqual(report_a["champion_sha256"],
                         hashlib.sha256(genome_a.tobytes()).hexdigest())
        json.dumps(report_a, allow_nan=False)
        json.dumps(report_d, allow_nan=False)
        payload = genome_to_payload(genome_a)
        self.assertTrue(bool(payload["research_only"]))
        self.assertFalse(bool(payload["deployment_allowed"]))
        np.testing.assert_array_equal(payload["genome"], genome_a)


if __name__ == "__main__":
    unittest.main()
