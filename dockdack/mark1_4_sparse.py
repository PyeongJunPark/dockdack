"""Mark1.4 v2: research-only sparse neuroevolution with train-frozen thresholds.

This is a separate experiment from Mark1.4 v1. Individuals are arbitrary
30×OHLCV nonlinear scorers and a target candidate coverage gene. All
thresholds, mating decisions, activity gates and champion selection use only
the chronological training split. No broker, GUI or operational writer exists
in this module.
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping

import numpy as np
import torch

from dockdack.mark1_4_evolution import (
    EvolutionSamples, GENOME_SIZE, HIDDEN, INPUTS, LOOKBACK,
    normalize_windows, simulate_missing_scenarios, simulate_portfolio,
)


INITIAL_Q_GRID = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
MIN_Q = min(INITIAL_Q_GRID)
MAX_Q = max(INITIAL_Q_GRID)


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return requested


def _scores(normalized: np.ndarray, genomes: np.ndarray, *, device: str,
            chunk_rows: int) -> np.ndarray:
    """Vectorized population scorer; hidden tanh, final *linear* score."""
    population = np.asarray(genomes, dtype=np.float32)
    if population.ndim != 2 or population.shape[1] != GENOME_SIZE:
        raise ValueError(f"genomes must have shape [population,{GENOME_SIZE}]")
    if not np.isfinite(population).all() or chunk_rows < 1:
        raise ValueError("Genomes must be finite and chunk_rows positive")
    flattened = np.asarray(normalized, dtype=np.float32).reshape(-1, INPUTS)
    target = _device(device)
    weights = torch.as_tensor(population, device=target)
    first_end = INPUTS * HIDDEN
    bias_end = first_end + HIDDEN
    second_end = bias_end + HIDDEN
    first = weights[:, :first_end].reshape(-1, INPUTS, HIDDEN)
    bias = weights[:, first_end:bias_end]
    second = weights[:, bias_end:second_end]
    out_bias = weights[:, second_end]
    out = np.empty((len(flattened), len(population)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(flattened), chunk_rows):
            x = torch.as_tensor(flattened[start:start + chunk_rows], device=target)
            hidden = torch.tanh(torch.einsum("ni,pih->nph", x, first) + bias)
            score = (hidden * second).sum(dim=2) + out_bias
            out[start:start + len(x)] = score.cpu().numpy()
    if not np.isfinite(out).all():
        raise ValueError("Sparse model produced nonfinite scores")
    return out


def score_sparse_genome(windows: np.ndarray, genome: np.ndarray, *,
                        device: str = "auto", chunk_rows: int = 2048) -> np.ndarray:
    individual = np.asarray(genome, dtype=np.float32)
    if individual.shape != (GENOME_SIZE,):
        raise ValueError(f"genome must have {GENOME_SIZE} genes")
    return _scores(normalize_windows(windows), individual[None, :],
                   device=device, chunk_rows=chunk_rows)[:, 0]


def derive_train_threshold(train_scores: np.ndarray, log_q: float) -> tuple[float, dict]:
    """Freeze one numeric threshold from training candidate scores only."""
    values = np.asarray(train_scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Training scores must be a nonempty finite vector")
    if not math.isfinite(log_q):
        raise ValueError("log_q must be finite")
    q = math.exp(float(log_q))
    if not MIN_Q * (1 - 1e-6) <= q <= MAX_Q * (1 + 1e-6):
        raise ValueError("Target coverage q must remain in [0.001,0.20]")
    threshold = float(np.quantile(values, 1 - q, method="linear"))
    provenance = {
        "source": "train_scores_only", "train_candidate_count": len(values),
        "target_train_candidate_coverage": q,
        "achieved_train_candidate_coverage": float(np.mean(values > threshold)),
        "numeric_score_threshold": threshold,
        "quantile_method": "linear; strict score > threshold, ties abstain",
        "validation_or_test_used": False,
    }
    return threshold, provenance


def _fitness(result: Mapping) -> float:
    """Strict train-only integer-share objective; cash benchmark is zero."""
    if result["incomplete_data"]:
        return -1_000_000.0 - result["unresolved_signals"] / max(1, result["signals"])
    if result["executed_trades"] == 0:
        return -0.02
    daily = np.asarray(result["daily_returns"], dtype=np.float64)
    mean_annual = 252 * float(daily.mean())
    downside = float(np.sqrt(np.mean(np.square(np.minimum(daily, 0)))))
    return (mean_annual - 0.5 * math.sqrt(252) * downside
            - 0.25 * abs(result["max_drawdown"]))


def _active(result: Mapping) -> bool:
    return (not result["incomplete_data"]
            and result["executed_sessions"] >= 20
            and result["executed_trades"] >= 50)


def qualifies_sparse_strategy(result: Mapping) -> bool:
    """Predeclared activity and cash-beating gates for a selected strategy."""
    return (_active(result) and result["compound_net_return"] > 0
            and _fitness(result) > 0)


def _random_population(rng: np.random.Generator, size: int) -> np.ndarray:
    if size < len(INITIAL_Q_GRID) or size % len(INITIAL_Q_GRID):
        raise ValueError("population_size must be a positive multiple of eight")
    out = np.empty((size, GENOME_SIZE), dtype=np.float32)
    first_end = INPUTS * HIDDEN
    bias_end = first_end + HIDDEN
    second_end = bias_end + HIDDEN
    out[:, :first_end] = rng.normal(0, 0.5, (size, first_end)).astype(np.float32)
    out[:, first_end:bias_end] = rng.normal(0, 0.15, (size, HIDDEN)).astype(np.float32)
    out[:, bias_end:second_end] = rng.normal(0, 0.45, (size, HIDDEN)).astype(np.float32)
    out[:, second_end] = rng.normal(0, 0.15, size).astype(np.float32)
    q = np.repeat(np.asarray(INITIAL_Q_GRID), size // len(INITIAL_Q_GRID))
    rng.shuffle(q)
    out[:, -1] = np.log(q).astype(np.float32)
    return out


def _breed(rng: np.random.Generator, ranked: np.ndarray,
           generation: int) -> np.ndarray:
    size = len(ranked)
    elite_count = max(2, size // 8)
    elites = ranked[:elite_count]
    children = [individual.copy() for individual in elites]
    # Sparse niches get fresh random explorers; this prevents collapse of the
    # initial coverage grid while mating remains governed by train fitness.
    immigrants = min(len(INITIAL_Q_GRID), size // 8)
    if immigrants:
        fresh = _random_population(rng, len(INITIAL_Q_GRID))
        for offset in range(immigrants):
            fresh[offset, -1] = math.log(INITIAL_Q_GRID[(generation + offset) % 8])
            children.append(fresh[offset])
    while len(children) < size:
        left = elites[int(rng.integers(elite_count))]
        right = elites[int(rng.integers(elite_count))]
        mask = rng.random(GENOME_SIZE) < 0.5
        child = np.where(mask, left, right).astype(np.float32)
        changes = rng.random(GENOME_SIZE - 1) < 0.04
        child[:-1][changes] += rng.normal(0, 0.12, int(changes.sum())).astype(np.float32)
        if rng.random() < 0.5:
            child[-1] += float(rng.normal(0, 0.3))
        child[-1] = np.clip(child[-1], math.log(MIN_Q), math.log(MAX_Q))
        children.append(child)
    return np.stack(children)


def _indices(samples: EvolutionSamples, splits: Mapping[str, np.ndarray]) -> dict:
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("splits must contain train, validation, test")
    normalized = {}
    for name in ("train", "validation", "test"):
        values = np.asarray(splits[name], dtype=np.int64)
        if (values.ndim != 1 or len(values) == 0 or np.any(values < 0)
                or np.any(values >= len(samples.windows))
                or len(np.unique(values)) != len(values)):
            raise ValueError(f"Invalid {name} split indices")
        normalized[name] = values
    parts = list(normalized.values())
    if len(np.unique(np.concatenate(parts))) != sum(map(len, parts)):
        raise ValueError("Split indices overlap")
    ordinals = np.asarray(samples.target_ordinals)
    train, validation, test = (ordinals[normalized[name]] for name in
                               ("train", "validation", "test"))
    if not (train.max() + LOOKBACK < validation.min()
            and validation.max() + LOOKBACK < test.min()):
        raise ValueError("Chronological splits require a 30-session embargo")
    return normalized


def _summary(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in {
        "daily_dates", "daily_returns", "observed_day_proxy_returns"
    }}


def genome_to_payload(genome: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    individual = np.asarray(genome, dtype=np.float32)
    if individual.shape != (GENOME_SIZE,) or not np.isfinite(individual).all():
        raise ValueError("Invalid sparse genome")
    if not math.isfinite(threshold):
        raise ValueError("Frozen threshold must be finite")
    first_end = INPUTS * HIDDEN
    bias_end = first_end + HIDDEN
    second_end = bias_end + HIDDEN
    return {
        "genome": individual.copy(),
        "first_weights": individual[:first_end].reshape(INPUTS, HIDDEN).copy(),
        "first_bias": individual[first_end:bias_end].copy(),
        "second_weights": individual[bias_end:second_end].copy(),
        "second_bias": np.asarray(individual[second_end], dtype=np.float32),
        "target_candidate_coverage": np.asarray(math.exp(float(individual[-1])), dtype=np.float32),
        "frozen_train_numeric_threshold": np.asarray(threshold, dtype=np.float64),
        "research_only": np.asarray(True),
        "deployment_allowed": np.asarray(False),
    }


def evolve_sparse(samples: EvolutionSamples, splits: Mapping[str, np.ndarray], *,
                  seed: int = 42, population_size: int = 64, generations: int = 50,
                  device: str = "auto", chunk_rows: int = 2048,
                  cost_bps: float = 20.0, allocation: float = 0.1,
                  max_positions: int = 10,
                  initial_equity: float = 10_000_000.0,
                  cost_grid_bps: tuple[float, ...] = (0, 10, 20, 40, 80)
                  ) -> tuple[dict, np.ndarray | None]:
    """Evolve sparse scorers; return only a train-qualified profitable genome.

    The search never reads validation/test scores or outcomes until the train
    champion is frozen. The separate best active candidate is exploratory and
    can lose money; its existence never forces a deployment recommendation.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(generations) is not int or not 1 <= generations <= 500:
        raise ValueError("generations must be in [1,500]")
    if type(population_size) is not int or not 8 <= population_size <= 256 or population_size % 8:
        raise ValueError("population_size must be in [8,256] and divisible by eight")
    selected = _indices(samples, splits)
    runtime_device = _device(device)
    normalized = normalize_windows(samples.windows)
    rng = np.random.default_rng(seed)
    population = _random_population(rng, population_size)
    first_q_count = {str(q): population_size // len(INITIAL_Q_GRID)
                     for q in INITIAL_Q_GRID}
    best_active = best_profitable = None
    best_active_fitness = best_profitable_fitness = -math.inf
    complete_path_trials = active_trials = profitable_trials = 0
    history = []
    train_indices = selected["train"]
    for generation in range(1, generations + 1):
        train_scores = _scores(normalized[train_indices], population,
                               device=runtime_device, chunk_rows=chunk_rows)
        fitness = np.empty(population_size, dtype=np.float64)
        generation_active = 0
        for member in range(population_size):
            threshold, provenance = derive_train_threshold(
                train_scores[:, member], float(population[member, -1]))
            scores = np.full(len(samples.windows), np.nan, dtype=np.float32)
            scores[train_indices] = train_scores[:, member]
            result = simulate_portfolio(
                samples, train_indices, scores, threshold=threshold,
                cost_bps=cost_bps, allocation=allocation,
                max_positions=max_positions, initial_equity=initial_equity)
            fitness[member] = _fitness(result)
            complete_path_trials += not result["incomplete_data"]
            if _active(result):
                active_trials += 1
                generation_active += 1
                if fitness[member] > best_active_fitness:
                    best_active_fitness = float(fitness[member])
                    best_active = (population[member].copy(), threshold, provenance)
            if qualifies_sparse_strategy(result):
                profitable_trials += 1
                if fitness[member] > best_profitable_fitness:
                    best_profitable_fitness = float(fitness[member])
                    best_profitable = (population[member].copy(), threshold, provenance)
        rank = np.argsort(-fitness, kind="stable")
        history.append({"generation": generation,
                        "best_train_fitness": float(fitness[rank[0]]),
                        "median_train_fitness": float(np.median(fitness)),
                        "activity_qualified_this_generation": generation_active})
        if generation != generations:
            population = _breed(rng, population[rank], generation)

    def evaluate_frozen(candidate) -> dict | None:
        if candidate is None:
            return None
        genome, threshold, provenance = candidate
        scores = score_sparse_genome(samples.windows, genome,
                                     device=runtime_device, chunk_rows=chunk_rows)
        results = {}
        scenarios = {}
        coverage = {}
        for name in ("train", "validation", "test"):
            rows = selected[name]
            results[name] = _summary(simulate_portfolio(
                samples, rows, scores, threshold=threshold,
                cost_bps=cost_bps, allocation=allocation,
                max_positions=max_positions, initial_equity=initial_equity))
            raw = simulate_missing_scenarios(
                samples, rows, scores, threshold=threshold,
                cost_bps=cost_bps, allocation=allocation,
                max_positions=max_positions, initial_equity=initial_equity)
            scenarios[name] = {
                "unresolved_selected": raw["unresolved_selected"],
                "no_fill": _summary(raw["no_fill"]),
                "full_budget_loss": _summary(raw["full_budget_loss"]),
            }
            coverage[name] = float(np.mean(scores[rows] > threshold))
        return {
            "research_only": True, "deployment_allowed": False,
            "train_only_selection": True,
            "genome_sha256": hashlib.sha256(genome.tobytes()).hexdigest(),
            "genome": genome.tolist(),
            "threshold_provenance": provenance,
            "frozen_numeric_score_threshold": threshold,
            "candidate_coverage_by_split": coverage,
            "train_fitness": _fitness(simulate_portfolio(
                samples, selected["train"], scores, threshold=threshold,
                cost_bps=cost_bps, allocation=allocation,
                max_positions=max_positions, initial_equity=initial_equity)),
            "results": results,
            "missing_target_scenarios": scenarios,
        }

    best_active_report = evaluate_frozen(best_active)
    selected_report = evaluate_frozen(best_profitable)
    no_trade_scores = np.full(len(samples.windows), -1.0, dtype=np.float32)
    always_scores = np.full(len(samples.windows), 1.0, dtype=np.float32)
    momentum_scores = (np.asarray(samples.windows[:, -1, 3], dtype=np.float64)
                       / np.asarray(samples.windows[:, -6, 3], dtype=np.float64) - 1).astype(np.float32)
    baselines = {}
    for baseline, scores in {
        "no_trade": no_trade_scores,
        "always_buy_top10_liquidity_rank": always_scores,
        "positive_5day_momentum_top10": momentum_scores,
    }.items():
        baselines[baseline] = {
            name: _summary(simulate_portfolio(
                samples, selected[name], scores, threshold=0.0,
                cost_bps=cost_bps, allocation=allocation,
                max_positions=max_positions, initial_equity=initial_equity))
            for name in ("train", "validation", "test")
        }
    sensitivity = None
    if best_profitable is not None:
        genome, threshold, _ = best_profitable
        scores = score_sparse_genome(samples.windows, genome,
                                     device=runtime_device, chunk_rows=chunk_rows)
        sensitivity = {}
        for cost in tuple(dict.fromkeys((*cost_grid_bps, cost_bps))):
            if not math.isfinite(cost) or not 0 <= cost < 10_000:
                raise ValueError("Cost grid requires finite nonnegative bps")
            sensitivity[str(cost)] = _summary(simulate_portfolio(
                samples, selected["test"], scores, threshold=threshold,
                cost_bps=float(cost), allocation=allocation,
                max_positions=max_positions, initial_equity=initial_equity))
    report = {
        "experiment": "mark1-4-v2-sparse-random-neural-evolution",
        "research_only": True, "deployment_allowed": False,
        "decision": "30 completed bars through t; rank and choose before t+1 open",
        "genome": {"architecture": "150-to-12 tanh hidden-to-linear score, plus natural log target train coverage",
                   "genes": GENOME_SIZE, "initial_q_grid": list(INITIAL_Q_GRID),
                   "train_quantile_rule": "numeric threshold from training scores only; strict greater-than",
                   "hidden_activation": "tanh", "final_activation": "linear"},
        "search": {"seed": seed, "device": runtime_device,
                   "population_size": population_size, "generations": generations,
                   "fitness_trials": population_size * generations,
                   "complete_path_trials": complete_path_trials,
                   "activity_qualified_trials": active_trials,
                   "profitable_activity_qualified_trials": profitable_trials,
                   "initial_q_grid_counts": first_q_count,
                   "fitness_definition": "train exact integer-share: 252*mean daily net - 0.5*sqrt(252)*downside RMS - 0.25*max drawdown; unknown selected outcome disqualified",
                   "activity_gate": "at least 20 executed train sessions and 50 executed trades",
                   "cash_gate": "positive exact train compound return AND positive preregistered fitness",
                   "validation_or_test_used_in_selection": False,
                   "history": history},
        "no_trade_train_compound_benchmark": 0.0,
        "no_profitable_strategy": best_profitable is None,
        "best_active_exploratory": best_active_report,
        "selected_strategy": selected_report,
        "results": selected_report["results"] if selected_report else None,
        "missing_target_scenarios": (selected_report["missing_target_scenarios"]
                                     if selected_report else None),
        "baselines": baselines,
        "frozen_test_cost_sensitivity_bps": sensitivity,
        "splits": {name: {"candidates": len(rows),
                          "first_target": min(samples.target_dates[rows].tolist()),
                          "last_target": max(samples.target_dates[rows].tolist())}
                   for name, rows in selected.items()},
        "source": samples.source,
        "limitations": [
            "Historical catalog may introduce survivorship bias despite calibration-frozen liquid selection.",
            "Unobserved selected next-day outcomes invalidate exact portfolio returns; scenario estimates are not observations.",
            "OPEN/CLOSE prices do not prove auction fills; costs approximate spread, impact and fees.",
            "This is adaptive historical research, not a pristine live profitability estimate.",
            "A positive train strategy may fail on validation or test; neither holdout can rescue or select it.",
            "No GUI, order entry or deployable signal is connected.",
        ],
    }
    return report, best_profitable[0].copy() if best_profitable is not None else None
