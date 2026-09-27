"""Research-only Mark1.4 E3: sparse GA selected by worst training year.

The experiment changes only the fitness used for genetic selection. Four
chronological training years are scored at a deliberately stressed 40 bp
roundtrip cost. Validation and reused historical test outcomes are inspected
only after the training champion and its numeric threshold are frozen.
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping

import numpy as np

from dockdack.mark1_4_evolution import (
    EvolutionSamples, normalize_windows, simulate_missing_scenarios,
    simulate_portfolio,
)
from dockdack.mark1_4_sparse import (
    INITIAL_Q_GRID, _breed, _device, _indices, _random_population, _scores,
    derive_train_threshold, score_sparse_genome,
)


def _year_groups(samples: EvolutionSamples, train: np.ndarray) -> dict[str, np.ndarray]:
    years = np.asarray(samples.target_dates)[train].astype("U10")
    groups = {year: train[np.char.startswith(years, year)]
              for year in sorted(set(day[:4] for day in years.tolist()))}
    if len(groups) != 4 or any(len(rows) == 0 for rows in groups.values()):
        raise ValueError("E3 needs four nonempty chronological training years")
    return groups


def _fitness(year_results: Mapping[str, dict]) -> tuple[float, bool, bool]:
    """Worst-year 40bp objective; return fitness, active, train-qualified."""
    results = list(year_results.values())
    if any(row["incomplete_data"] for row in results):
        return -1_000_000.0, False, False
    active = (sum(row["executed_trades"] for row in results) >= 80
              and all(row["executed_sessions"] >= 10 for row in results))
    if not active:
        return -0.02, False, False
    robust = min(float(row["annualized_mean_daily_return"])
                 - 0.25 * abs(float(row["max_drawdown"])) for row in results)
    qualified = robust > 0 and all(row["compound_net_return"] > 0 for row in results)
    return robust, True, qualified


def _summary(result: dict) -> dict:
    # Retain the realized daily path and signal trace for an auditable report.
    return {key: value for key, value in result.items()
            if key != "observed_day_proxy_returns"}


def evolve_robust(samples: EvolutionSamples, splits: Mapping[str, np.ndarray], *,
                  seed: int = 41, population_size: int = 64,
                  generations: int = 50, device: str = "auto",
                  chunk_rows: int = 2048, cost_bps_train: float = 40.0,
                  cost_bps_eval: float = 20.0, allocation: float = 0.1,
                  max_positions: int = 10,
                  initial_equity: float = 10_000_000.0
                  ) -> tuple[dict, np.ndarray | None]:
    """Freeze the train-best four-year-robust genome before any OOS scoring.

    `selected_strategy` is None when no candidate has positive stressed-cost
    fitness and positive net returns in every training year. Exploratory best
    active remains visible but is never a deployment approval.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (type(population_size) is not int or population_size < 8
            or population_size > 256 or population_size % len(INITIAL_Q_GRID)):
        raise ValueError("population_size must be a multiple of eight in [8,256]")
    if type(generations) is not int or not 1 <= generations <= 500:
        raise ValueError("generations must be in [1,500]")
    if not 0 <= cost_bps_eval <= cost_bps_train < 10_000:
        raise ValueError("Expected finite nonnegative evaluation/training cost")
    if not all(map(math.isfinite, (cost_bps_eval, cost_bps_train))):
        raise ValueError("Costs must be finite")
    selected = _indices(samples, splits)
    years = _year_groups(samples, selected["train"])
    runtime_device = _device(device)
    normalized = normalize_windows(samples.windows)
    rng = np.random.default_rng(seed)
    population = _random_population(rng, population_size)
    best_active = best_qualified = None
    best_active_fitness = best_qualified_fitness = -math.inf
    complete_trials = active_trials = qualified_trials = 0
    history: list[dict] = []
    trial_records: list[dict] = []
    total_rows = len(samples.windows)
    train_rows = selected["train"]

    for generation in range(1, generations + 1):
        train_scores = _scores(normalized[train_rows], population,
                               device=runtime_device, chunk_rows=chunk_rows)
        fitness = np.empty(population_size, dtype=np.float64)
        generation_qualified = 0
        for member in range(population_size):
            threshold, provenance = derive_train_threshold(
                train_scores[:, member], float(population[member, -1]))
            full_scores = np.full(total_rows, np.nan, dtype=np.float32)
            full_scores[train_rows] = train_scores[:, member]
            year_results = {year: simulate_portfolio(
                samples, rows, full_scores, threshold=threshold,
                cost_bps=cost_bps_train, allocation=allocation,
                max_positions=max_positions, initial_equity=initial_equity)
                for year, rows in years.items()}
            score, active, qualified = _fitness(year_results)
            fitness[member] = score
            complete = all(not row["incomplete_data"]
                           for row in year_results.values())
            complete_trials += complete
            trial_records.append({
                "generation": generation,
                "member": member,
                "genome_sha256": hashlib.sha256(population[member].tobytes()).hexdigest(),
                "numeric_score_threshold": threshold,
                "target_train_candidate_coverage": provenance["target_train_candidate_coverage"],
                "fitness": float(score),
                "complete_path": complete,
                "active": active,
                "qualified": qualified,
                "rejection_reason": (None if qualified else
                                     "missing_selected_outcome" if not complete else
                                     "activity_below_floor" if not active else
                                     "worst_year_or_annual_return_gate"),
                "yearly_40bp": {year: {
                    "compound_net_return": row["compound_net_return"],
                    "executed_trades": row["executed_trades"],
                    "executed_sessions": row["executed_sessions"],
                    "unresolved_signals": row["unresolved_signals"],
                } for year, row in year_results.items()},
            })
            if active:
                active_trials += 1
                if score > best_active_fitness:
                    best_active_fitness = float(score)
                    best_active = (population[member].copy(), threshold,
                                   provenance, {year: _summary(row)
                                                for year, row in year_results.items()})
            if qualified:
                qualified_trials += 1
                generation_qualified += 1
                if score > best_qualified_fitness:
                    best_qualified_fitness = float(score)
                    best_qualified = (population[member].copy(), threshold,
                                      provenance, {year: _summary(row)
                                                   for year, row in year_results.items()})
        rank = np.argsort(-fitness, kind="stable")
        history.append({
            "generation": generation,
            "best_train_robust_fitness": float(fitness[rank[0]]),
            "median_train_robust_fitness": float(np.median(fitness)),
            "qualified_this_generation": generation_qualified,
        })
        if generation != generations:
            population = _breed(rng, population[rank], generation)

    def evaluate(candidate):
        if candidate is None:
            return None
        genome, threshold, provenance, yearly40 = candidate
        scores = score_sparse_genome(samples.windows, genome,
                                     device=runtime_device, chunk_rows=chunk_rows)
        results = {}
        scenarios = {}
        for name in ("train", "validation", "test"):
            rows = selected[name]
            results[name] = _summary(simulate_portfolio(
                samples, rows, scores, threshold=threshold,
                cost_bps=cost_bps_eval, allocation=allocation,
                max_positions=max_positions, initial_equity=initial_equity))
            raw = simulate_missing_scenarios(
                samples, rows, scores, threshold=threshold,
                cost_bps=cost_bps_eval, allocation=allocation,
                max_positions=max_positions, initial_equity=initial_equity)
            scenarios[name] = {
                "unresolved_selected": raw["unresolved_selected"],
                "no_fill": _summary(raw["no_fill"]),
                "full_budget_loss": _summary(raw["full_budget_loss"]),
            }
        cost_grid = {
            name: {
                str(cost): _summary(simulate_portfolio(
                    samples, selected[name], scores, threshold=threshold,
                    cost_bps=cost, allocation=allocation,
                    max_positions=max_positions,
                    initial_equity=initial_equity))
                for cost in (0.0, 20.0, 40.0)
            }
            for name in ("train", "validation", "test")
        }
        return {
            "research_only": True, "deployment_allowed": False,
            "train_only_selection": True,
            "genome_sha256": hashlib.sha256(genome.tobytes()).hexdigest(),
            "genome": genome.tolist(),
            "frozen_numeric_score_threshold": threshold,
            "threshold_provenance": provenance,
            "train_year_results_at_40bp": yearly40,
            "train_robust_fitness": _fitness(yearly40)[0],
            "results_at_20bp": results,
            "results_cost_grid_bps": cost_grid,
            "missing_target_scenarios_at_20bp": scenarios,
        }

    active_report = evaluate(best_active)
    # Predeclared rule: the highest-fitness active genome itself must clear
    # every annual return and robust-fitness gate. Do not quietly substitute
    # a lower-fitness qualified genome after seeing the search results.
    selected_candidate = (best_active if best_active is not None and
                          _fitness(best_active[3])[2] else None)
    qualified_report = evaluate(selected_candidate)
    report = {
        "experiment": "mark1-4-e3-worst-year-sparse-genetic-search",
        "research_only": True, "deployment_allowed": False,
        "search": {
            "seed": seed, "device": runtime_device,
            "population_size": population_size, "generations": generations,
            "fitness_trials": population_size * generations,
            "complete_path_trials": complete_trials,
            "active_trials": active_trials,
            "qualified_trials": qualified_trials,
            "rejected_trials": population_size * generations - qualified_trials,
            "train_years": list(years),
            "train_roundtrip_cost_bps": cost_bps_train,
            "evaluation_roundtrip_cost_bps": cost_bps_eval,
            "fitness": "minimum yearly (252*mean daily net return - .25*abs(max drawdown)) at 40bp",
            "activity_gate": "80 integer-share train fills and 10 executed sessions per training year",
            "qualification_gate": "positive robust fitness and positive exact compound return in every training year",
            "validation_or_test_used_in_selection": False,
            "history": history,
            "trial_records": trial_records,
        },
        "no_train_qualified_strategy": selected_candidate is None,
        "best_active_exploratory": active_report,
        "best_qualified_but_not_selected_sha256": (
            hashlib.sha256(best_qualified[0].tobytes()).hexdigest()
            if best_qualified is not None and selected_candidate is None else None),
        "selected_strategy": qualified_report,
        "source": samples.source,
        "limitations": [
            "Historical catalog is not point-in-time and omits some delisted names.",
            "Opening/closing prices are hypothetical fills; cost is an assumed roundtrip rate.",
            "2022 and 2023-2024 were already seen in earlier research, not pristine holdouts.",
            "No GUI, broker order or deployable signal is connected.",
        ],
    }
    return report, selected_candidate[0].copy() if selected_candidate is not None else None
