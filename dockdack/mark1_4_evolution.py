"""Research-only Mark1.4 neuroevolution of arbitrary 30-bar buy scorers.

An individual is a small nonlinear neural network, not a hand-written trading
rule. Genetic selection sees training dates only. Completed bars through t are
the sole model input; t+1 OPEN/CLOSE are outcomes for an explicitly hypothetical
same-session round trip. Nothing in this module calls a broker or writes data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np
import torch


LOOKBACK = 30
CHANNELS = 5
HIDDEN = 12
INPUTS = LOOKBACK * CHANNELS
GENOME_SIZE = INPUTS * HIDDEN + HIDDEN + HIDDEN + 1 + 1


@dataclass(frozen=True)
class EvolutionSamples:
    """Causal windows plus *separate* next-session outcomes.

    ``target_ordinals`` are zero-based positions in the market session calendar.
    The caller must create chronological splits with an embargo before evolving.
    """

    windows: np.ndarray
    target_dates: np.ndarray
    target_ordinals: np.ndarray
    symbol_ids: np.ndarray
    entry_open: np.ndarray
    exit_close: np.ndarray
    source: dict

    def __post_init__(self) -> None:
        shape = np.shape(self.windows)
        if len(shape) != 3 or shape[1:] != (LOOKBACK, CHANNELS):
            raise ValueError("windows must have shape [N,30,5]")
        count = shape[0]
        if count == 0 or any(np.shape(getattr(self, name)) != (count,) for name in (
            "target_dates", "target_ordinals", "symbol_ids", "entry_open", "exit_close"
        )):
            raise ValueError("All sample arrays must have the same nonzero length")
        if not isinstance(self.source, dict):
            raise ValueError("source must be a provenance dictionary")


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """Return stable [N,30,5] inputs using only each completed 30-bar window.

    OHLC become clipped log ratios to the preceding completed close; the first
    bar uses its own open as reference. Volume is log1p(volume) centered on the
    median of the *same completed window*. No cross-sample/future statistics.
    """
    raw = np.asarray(windows, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (LOOKBACK, CHANNELS):
        raise ValueError("Expected [samples,30,5] completed OHLCV windows")
    if not np.isfinite(raw).all() or np.any(raw[:, :, :4] <= 0) or np.any(raw[:, :, 4] < 0):
        raise ValueError("Window prices must be positive and volume nonnegative")
    reference = np.empty((len(raw), LOOKBACK), dtype=np.float64)
    reference[:, 0] = raw[:, 0, 0]
    reference[:, 1:] = raw[:, :-1, 3]
    normalized = np.empty(raw.shape, dtype=np.float32)
    normalized[:, :, :4] = np.clip(np.log(raw[:, :, :4] / reference[:, :, None]), -1.5, 1.5)
    log_volume = np.log1p(raw[:, :, 4])
    normalized[:, :, 4] = np.clip(
        log_volume - np.median(log_volume, axis=1)[:, None], -5, 5
    )
    return normalized


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return requested


def _population_scores(normalized: np.ndarray, genomes: np.ndarray, *,
                       device: str = "auto", chunk_rows: int = 2048) -> np.ndarray:
    """Batch all individuals on CPU/CUDA; bound temporary device memory."""
    population = np.asarray(genomes, dtype=np.float32)
    if population.ndim != 2 or population.shape[1] != GENOME_SIZE:
        raise ValueError(f"genomes must have shape [population,{GENOME_SIZE}]")
    if not np.isfinite(population).all() or chunk_rows < 1:
        raise ValueError("Genomes must be finite and chunk_rows positive")
    flattened = np.asarray(normalized, dtype=np.float32).reshape(-1, INPUTS)
    target = _device(device)
    weights = torch.as_tensor(population, device=target)
    stop1 = INPUTS * HIDDEN
    stop2 = stop1 + HIDDEN
    stop3 = stop2 + HIDDEN
    first = weights[:, :stop1].reshape(-1, INPUTS, HIDDEN)
    bias1 = weights[:, stop1:stop2]
    second = weights[:, stop2:stop3]
    bias2 = weights[:, stop3]
    output = np.empty((len(flattened), len(population)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(flattened), chunk_rows):
            x = torch.as_tensor(flattened[start:start + chunk_rows], device=target)
            hidden = torch.tanh(torch.einsum("ni,pih->nph", x, first) + bias1)
            scores = torch.tanh((hidden * second).sum(dim=2) + bias2)
            output[start:start + len(x)] = scores.cpu().numpy()
    return output


def score_genome(windows: np.ndarray, genome: np.ndarray, *, device: str = "auto",
                 chunk_rows: int = 2048) -> np.ndarray:
    """Score every causal window; buy only scores above ``genome[-1]``."""
    individual = np.asarray(genome, dtype=np.float32)
    if individual.shape != (GENOME_SIZE,):
        raise ValueError(f"genome must contain {GENOME_SIZE} finite genes")
    return _population_scores(normalize_windows(windows), individual[None, :],
                              device=device, chunk_rows=chunk_rows)[:, 0]


def genome_to_payload(genome: np.ndarray) -> dict[str, np.ndarray]:
    """Named, NumPy-saveable frozen champion with nondeployment metadata."""
    individual = np.asarray(genome, dtype=np.float32)
    if individual.shape != (GENOME_SIZE,) or not np.isfinite(individual).all():
        raise ValueError(f"genome must contain {GENOME_SIZE} finite genes")
    stop1 = INPUTS * HIDDEN
    stop2 = stop1 + HIDDEN
    stop3 = stop2 + HIDDEN
    return {
        "first_weights": individual[:stop1].reshape(INPUTS, HIDDEN).copy(),
        "first_bias": individual[stop1:stop2].copy(),
        "second_weights": individual[stop2:stop3].copy(),
        "second_bias": np.asarray(individual[stop3], dtype=np.float32),
        "score_threshold": np.asarray(individual[-1], dtype=np.float32),
        "genome": individual.copy(),
        "lookback": np.asarray(LOOKBACK, dtype=np.int32),
        "channels": np.asarray(CHANNELS, dtype=np.int32),
        "research_only": np.asarray(True),
        "deployment_allowed": np.asarray(False),
    }


def _valid_indices(samples: EvolutionSamples, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(samples.windows)):
        raise ValueError("indices must be a nonempty one-dimensional sample subset")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("indices must not repeat sample rows")
    return indices


def simulate_portfolio(samples: EvolutionSamples, indices: np.ndarray,
                       scores: np.ndarray, *, threshold: float = 0.0,
                       cost_bps: float = 20.0, allocation: float = 0.1,
                       max_positions: int = 10,
                       initial_equity: float = 10_000_000.0) -> dict:
    """Cash-only integer-share t+1 OPEN-to-CLOSE *proxy* portfolio.

    Each selected name receives at most 10% of start-of-session cash equity.
    Any selected outcome without valid OPEN/CLOSE makes that day's return and
    all subsequent exact equity unknown; it is never treated as a flat trade.
    Known-day fractional returns are retained *only* for fitness diagnostics.
    """
    rows = _valid_indices(samples, indices)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(samples.windows),) or not np.isfinite(scores[rows]).all():
        raise ValueError("scores must be one finite value per candidate")
    if not math.isfinite(threshold) or not math.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("threshold and roundtrip cost must be finite and cost nonnegative")
    if (not 0 < allocation <= 1 or type(max_positions) is not int or max_positions < 1
            or max_positions * allocation > 1 + 1e-12):
        raise ValueError("Allocation and position cap must not exceed total equity")
    if not math.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial_equity must be positive and finite")
    dates = np.asarray(samples.target_dates)
    ordinals = np.asarray(samples.target_ordinals)
    symbols = np.asarray(samples.symbol_ids)
    opened = np.asarray(samples.entry_open, dtype=np.float64)
    closed = np.asarray(samples.exit_close, dtype=np.float64)
    if np.any(ordinals[rows] < 0):
        raise ValueError("target_ordinals must be nonnegative")
    if np.any(symbols[rows] < 0):
        raise ValueError("symbol_ids must be nonnegative")
    # Use integer session ordinal, not lexicographic symbol/date ordering.
    ordered_rows = rows[np.argsort(ordinals[rows], kind="stable")]
    populated_sessions, session_starts, session_counts = np.unique(
        ordinals[ordered_rows], return_index=True, return_counts=True)
    # Ordinals come from a market-session calendar, so interior gaps are real
    # cash-only sessions rather than disappearing from annualized denominators.
    sessions = np.arange(int(populated_sessions[0]), int(populated_sessions[-1]) + 1)
    populated_ranges = {int(day): (int(start), int(count)) for day, start, count in
                        zip(populated_sessions, session_starts, session_counts)}
    half = cost_bps / 20_000
    equity = float(initial_equity)
    peak = equity
    drawdown = 0.0
    exact = True
    daily_dates: list[str | None] = []
    daily_ordinals: list[int] = []
    daily_returns: list[float | None] = []
    daily_signals: list[int] = []
    daily_selected_symbol_ids: list[list[int]] = []
    daily_drawdowns: list[float | None] = []
    proxy_returns: list[float | None] = []
    signals = observed_signals = active_days = unresolved_days = exact_trades = executed_days = 0
    for ordinal in sessions:
        daily_ordinals.append(int(ordinal))
        if int(ordinal) not in populated_ranges:
            daily_dates.append(None)
            daily_returns.append(0.0 if exact else None)
            daily_signals.append(0)
            daily_selected_symbol_ids.append([])
            daily_drawdowns.append(equity / peak - 1 if exact else None)
            proxy_returns.append(0.0)
            continue
        start, count = populated_ranges[int(ordinal)]
        day_rows = ordered_rows[start:start + count]
        if not np.all(dates[day_rows] == dates[day_rows[0]]):
            raise ValueError("An ordinal must identify exactly one session date")
        daily_dates.append(str(dates[day_rows[0]]))
        eligible = day_rows[scores[day_rows] > threshold]
        # Stable tie-break across devices, input row ordering, and population.
        rank = np.lexsort((symbols[eligible], -scores[eligible]))
        selected = eligible[rank[:max_positions]]
        daily_signals.append(len(selected))
        daily_selected_symbol_ids.append([int(symbol) for symbol in symbols[selected]])
        if len(np.unique(symbols[selected])) != len(selected):
            raise ValueError("Duplicate symbol within one selected session")
        signals += len(selected)
        active_days += bool(len(selected))
        valid = (np.isfinite(opened[selected]) & np.isfinite(closed[selected])
                 & (opened[selected] > 0) & (closed[selected] > 0))
        observed_signals += int(valid.sum())
        if not valid.all():
            unresolved_days += 1
            exact = False
            daily_returns.append(None)
            daily_drawdowns.append(None)
            proxy_returns.append(None)
            continue
        net_per_share = closed[selected] * (1 - half) - opened[selected] * (1 + half)
        gross_per_share = opened[selected] * (1 + half)
        fractional_net = closed[selected] * (1 - half) / gross_per_share - 1
        proxy_returns.append(float(allocation * fractional_net.sum()))
        if not exact:
            daily_returns.append(None)
            daily_drawdowns.append(None)
            continue
        start_equity = equity
        # Buying power is cash-only and the 10% budget includes entry costs.
        quantities = np.floor(allocation * start_equity / gross_per_share).astype(np.int64)
        exact_trades += int(np.count_nonzero(quantities))
        executed_days += bool(np.any(quantities))
        debit = float(np.dot(quantities, gross_per_share))
        if debit > start_equity + 1e-7:
            raise AssertionError("Position sizing exceeded cash equity")
        pnl = float(np.dot(quantities, net_per_share))
        equity += pnl
        if equity < -1e-7:
            raise AssertionError("Cash-only strategy produced negative equity")
        daily_returns.append(equity / start_equity - 1)
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1)
        daily_drawdowns.append(equity / peak - 1)
    complete_daily = np.asarray(daily_returns, dtype=np.float64) if exact else None
    return {
        "sessions": len(sessions), "signals": signals,
        "calendar_sessions_missing_candidate": len(sessions) - len(populated_sessions),
        "observed_signals": observed_signals,
        "unresolved_signals": signals - observed_signals,
        "executed_trades": exact_trades if exact else None,
        "executed_trades_on_exact_prefix": exact_trades,
        "executed_sessions": executed_days if exact else None,
        "active_sessions": active_days, "unresolved_sessions": unresolved_days,
        "daily_dates": daily_dates, "daily_ordinals": daily_ordinals,
        "daily_returns": daily_returns, "daily_signals": daily_signals,
        "daily_selected_symbol_ids": daily_selected_symbol_ids,
        "daily_drawdowns": daily_drawdowns,
        "observed_day_proxy_returns": proxy_returns,
        "final_equity": equity if exact else None,
        "compound_net_return": equity / initial_equity - 1 if exact else None,
        "max_drawdown": drawdown if exact else None,
        "annualized_mean_daily_return": (float(252 * complete_daily.mean())
                                         if exact else None),
        "signal_active_fraction": active_days / len(sessions),
        "executed_active_fraction": executed_days / len(sessions) if exact else None,
        "incomplete_data": not exact,
        "allocation": allocation, "max_positions": max_positions,
        "initial_equity": initial_equity, "cost_bps": cost_bps,
        "sizing": "integer shares, floor(10% of start-of-day equity / entry cost), no leverage",
        "entry_exit": "hypothetical next-open purchase and same-day close sale; fills unverified",
        "session_denominator": "all calendar ordinals from first to last candidate in this split; interior empty sessions are cash days",
    }


def simulate_missing_scenarios(samples: EvolutionSamples, indices: np.ndarray,
                               scores: np.ndarray, *, threshold: float = 0.0,
                               cost_bps: float = 20.0, allocation: float = 0.1,
                               max_positions: int = 10,
                               initial_equity: float = 10_000_000.0) -> dict:
    """Reporting-only sensitivity for selected next-day outcomes that are absent.

    Neither scenario is an observed fill or a trainable fitness. ``no_fill``
    assumes each unresolved selected slot was never filled and leaves its cash
    untouched. ``full_budget_loss`` assumes its *entire* 10%-of-start-day
    allocation was forfeited. Observed selected positions still use integer
    shares and declared transaction costs. No scenario refills vacant slots.
    """
    rows = _valid_indices(samples, indices)
    scores = np.asarray(scores, dtype=np.float64)
    # Reuse exact simulator's input guards; its incomplete result is not used
    # as a fitness or substituted for either scenario.
    exact = simulate_portfolio(samples, rows, scores, threshold=threshold,
                               cost_bps=cost_bps, allocation=allocation,
                               max_positions=max_positions,
                               initial_equity=initial_equity)
    ordinals = np.asarray(samples.target_ordinals)
    symbols = np.asarray(samples.symbol_ids)
    opened = np.asarray(samples.entry_open, dtype=np.float64)
    closed = np.asarray(samples.exit_close, dtype=np.float64)
    ordered_rows = rows[np.argsort(ordinals[rows], kind="stable")]
    populated, starts, counts = np.unique(ordinals[ordered_rows],
                                           return_index=True, return_counts=True)
    bounds = {int(day): (int(start), int(count)) for day, start, count in
              zip(populated, starts, counts)}
    sessions = np.arange(int(populated[0]), int(populated[-1]) + 1)
    states = {
        name: {"equity": float(initial_equity), "peak": float(initial_equity),
               "max_drawdown": 0.0, "daily_returns": [],
               "executed_trades": 0, "executed_sessions": 0,
               "unknown_budget_forfeited_total": 0.0}
        for name in ("no_fill", "full_budget_loss")
    }
    half = cost_bps / 20_000
    for day in sessions:
        if int(day) not in bounds:
            for state in states.values():
                state["daily_returns"].append(0.0 if state["equity"] > 0 else None)
            continue
        start, count = bounds[int(day)]
        day_rows = ordered_rows[start:start + count]
        eligible = day_rows[scores[day_rows] > threshold]
        ranked = np.lexsort((symbols[eligible], -scores[eligible]))
        selected = eligible[ranked[:max_positions]]
        valid = (np.isfinite(opened[selected]) & np.isfinite(closed[selected])
                 & (opened[selected] > 0) & (closed[selected] > 0))
        known = selected[valid]
        unknown_count = int((~valid).sum())
        entry_cost = opened[known] * (1 + half)
        net_per_share = closed[known] * (1 - half) - entry_cost
        for name, state in states.items():
            start_equity = state["equity"]
            if start_equity == 0:
                quantities = np.zeros(len(known), dtype=np.int64)
            else:
                quantities = np.floor(allocation * start_equity / entry_cost).astype(np.int64)
            known_pnl = float(np.dot(quantities, net_per_share))
            forfeited = (unknown_count * allocation * start_equity
                         if name == "full_budget_loss" else 0.0)
            state["unknown_budget_forfeited_total"] += forfeited
            state["executed_trades"] += int(np.count_nonzero(quantities))
            state["executed_sessions"] += bool(np.any(quantities))
            following = start_equity + known_pnl - forfeited
            if following < -1e-7:
                raise AssertionError("Scenario exceeded cash-only equity")
            following = max(0.0, following)
            state["equity"] = following
            state["daily_returns"].append(
                following / start_equity - 1 if start_equity > 0 else None)
            state["peak"] = max(state["peak"], following)
            state["max_drawdown"] = min(
                state["max_drawdown"], following / state["peak"] - 1)
    scenarios = {}
    for name, state in states.items():
        daily = (np.asarray(state["daily_returns"], dtype=np.float64)
                 if all(value is not None for value in state["daily_returns"])
                 else None)
        scenarios[name] = {
            "research_only": True, "not_exact": True,
            "assumption": (
                "Missing selected positions never filled; their selected slots are not backfilled"
                if name == "no_fill" else
                "Entire 10%-of-scenario-start-day-equity budget lost for every missing selected position"
            ),
            "sessions": exact["sessions"], "signals": exact["signals"],
            "observed_signals": exact["observed_signals"],
            "unresolved_signals": exact["unresolved_signals"],
            "calendar_sessions_missing_candidate": exact["calendar_sessions_missing_candidate"],
            "active_sessions": exact["active_sessions"],
            "executed_trades": state["executed_trades"],
            "executed_sessions": state["executed_sessions"],
            "unknown_budget_forfeited_total": state["unknown_budget_forfeited_total"],
            "daily_returns": state["daily_returns"],
            "final_equity": state["equity"],
            "compound_net_return": state["equity"] / initial_equity - 1,
            "max_drawdown": state["max_drawdown"],
            "annualized_mean_daily_return": (float(252 * daily.mean())
                                             if daily is not None else None),
            "zero_equity_exhausted": state["equity"] == 0,
            "allocation": allocation, "max_positions": max_positions,
            "initial_equity": initial_equity, "cost_bps": cost_bps,
        }
    return {"unresolved_selected": exact["unresolved_signals"], **scenarios}


def _training_fitness(result: Mapping) -> float:
    """Optimize exact integer-share PnL; never reward unknown or zero fills."""
    if result["incomplete_data"]:
        # No approximate return substitutes for an unknown selected execution.
        return -1_000_000.0 - result["unresolved_signals"] / max(1, result["signals"])
    if result["executed_trades"] == 0:
        return -0.02
    realized = np.asarray(result["daily_returns"], dtype=np.float64)
    annualized_mean = 252 * float(realized.mean())
    downside = float(np.sqrt(np.mean(np.square(np.minimum(realized, 0)))))
    activity_fraction = result["executed_sessions"] / result["sessions"]
    # A 2%-of-days executed-activity floor is a soft penalty.
    inactivity = max(0.0, 0.02 - activity_fraction)
    return (annualized_mean - 0.5 * math.sqrt(252) * downside
            - 0.25 * abs(result["max_drawdown"]) - 0.05 * inactivity)


def _random_population(rng: np.random.Generator, size: int) -> np.ndarray:
    genomes = np.empty((size, GENOME_SIZE), dtype=np.float32)
    stop1 = INPUTS * HIDDEN
    stop2 = stop1 + HIDDEN
    stop3 = stop2 + HIDDEN
    genomes[:, :stop1] = rng.normal(0, 0.5, (size, stop1)).astype(np.float32)
    genomes[:, stop1:stop2] = rng.normal(0, 0.15, (size, HIDDEN)).astype(np.float32)
    genomes[:, stop2:stop3] = rng.normal(0, 0.45, (size, HIDDEN)).astype(np.float32)
    genomes[:, stop3] = rng.normal(0, 0.15, size).astype(np.float32)
    genomes[:, -1] = rng.uniform(-0.4, 0.4, size).astype(np.float32)
    return genomes


def _breed(rng: np.random.Generator, ranked: np.ndarray, size: int) -> np.ndarray:
    elite_count = max(2, size // 8)
    elites = ranked[:elite_count]
    children = [individual.copy() for individual in elites]
    while len(children) < size:
        left = elites[int(rng.integers(elite_count))]
        right = elites[int(rng.integers(elite_count))]
        mask = rng.random(GENOME_SIZE) < 0.5
        child = np.where(mask, left, right).astype(np.float32)
        mutated = rng.random(GENOME_SIZE) < 0.04
        child[mutated] += rng.normal(0, 0.12, int(mutated.sum())).astype(np.float32)
        child[-1] = np.clip(child[-1], -0.95, 0.95)
        children.append(child)
    return np.stack(children)


def _summary(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in {
        "daily_dates", "daily_returns", "observed_day_proxy_returns"
    }}


def evolve(samples: EvolutionSamples, splits: Mapping[str, np.ndarray], *,
           seed: int = 42, population_size: int = 32, generations: int = 20,
           device: str = "auto", chunk_rows: int = 2048, cost_bps: float = 20.0,
           allocation: float = 0.1, max_positions: int = 10,
           initial_equity: float = 10_000_000.0,
           cost_grid_bps: tuple[float, ...] = (0, 10, 20, 40, 80)) -> tuple[dict, np.ndarray]:
    """Evolve train-only random nonlinear rules; freeze before validation/test.

    All reported performance is a historical open/close proxy, not a verified
    execution result or a deployable trading signal. Validation and test are
    evaluated once after champion selection and never enter reproduction.
    """
    if type(seed) is not int or seed < 0 or not 4 <= population_size <= 256:
        raise ValueError("seed and population_size must be valid")
    if type(generations) is not int or not 1 <= generations <= 500:
        raise ValueError("generations must be in [1,500]")
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("splits must contain train, validation, and test")
    selected = {name: _valid_indices(samples, splits[name]) for name in splits}
    if len(np.unique(np.concatenate(tuple(selected.values())))) != sum(map(len, selected.values())):
        raise ValueError("Chronological split indices overlap")
    train_days = np.asarray(samples.target_ordinals)[selected["train"]]
    validation_days = np.asarray(samples.target_ordinals)[selected["validation"]]
    test_days = np.asarray(samples.target_ordinals)[selected["test"]]
    if not (train_days.max() < validation_days.min() < validation_days.max() < test_days.min()):
        raise ValueError("Expected strictly ordered train, validation, and test sessions")
    # The split creator, not this routine, defines the market calendar; enforce
    # a minimum 30-session separation as an additional guard.
    if validation_days.min() - train_days.max() <= LOOKBACK or test_days.min() - validation_days.max() <= LOOKBACK:
        raise ValueError("Chronological split embargo must exceed 30 sessions")
    runtime_device = _device(device)
    normalized = normalize_windows(samples.windows)
    rng = np.random.default_rng(seed)
    population = _random_population(rng, population_size)
    best_fitness = -math.inf
    champion: np.ndarray | None = None
    active_best_fitness = -math.inf
    active_champion: np.ndarray | None = None
    active_eligible_trials = 0
    fitness_history: list[dict] = []
    # Scores for an entire generation are batched on the selected device.
    for generation in range(1, generations + 1):
        training_scores = _population_scores(normalized[selected["train"]], population,
                                             device=runtime_device, chunk_rows=chunk_rows)
        fitness = np.empty(population_size, dtype=np.float64)
        for member in range(population_size):
            full_scores = np.full(len(samples.windows), np.nan, dtype=np.float32)
            full_scores[selected["train"]] = training_scores[:, member]
            simulated = simulate_portfolio(
                samples, selected["train"], full_scores,
                threshold=float(population[member, -1]), cost_bps=cost_bps,
                allocation=allocation, max_positions=max_positions,
                initial_equity=initial_equity)
            fitness[member] = _training_fitness(simulated)
            if (not simulated["incomplete_data"]
                    and simulated["executed_active_fraction"] >= 0.02
                    and simulated["executed_trades"] > 0):
                active_eligible_trials += 1
                if fitness[member] > active_best_fitness:
                    active_best_fitness = float(fitness[member])
                    active_champion = population[member].copy()
        rank = np.argsort(-fitness, kind="stable")
        leading = int(rank[0])
        # Strict > freezes earliest generation on ties; no validation peeking.
        if fitness[leading] > best_fitness:
            best_fitness = float(fitness[leading])
            champion = population[leading].copy()
        fitness_history.append({"generation": generation,
                                "best_train_fitness": float(fitness[leading]),
                                "median_train_fitness": float(np.median(fitness))})
        if generation != generations:
            population = _breed(rng, population[rank], population_size)
    assert champion is not None
    frozen_scores = score_genome(samples.windows, champion, device=runtime_device,
                                 chunk_rows=chunk_rows)
    results = {
        name: _summary(simulate_portfolio(
            samples, selected[name], frozen_scores,
            threshold=float(champion[-1]), cost_bps=cost_bps,
            allocation=allocation, max_positions=max_positions,
            initial_equity=initial_equity))
        for name in ("train", "validation", "test")
    }
    # Frozen, untuned controls use precisely the same candidates, capital,
    # costs and splits. Momentum is a comparator, never part of the genome.
    no_trade_scores = np.full(len(samples.windows), -1.0, dtype=np.float32)
    always_scores = np.full(len(samples.windows), 1.0, dtype=np.float32)
    previous_close = np.asarray(samples.windows[:, -6, 3], dtype=np.float64)
    recent_close = np.asarray(samples.windows[:, -1, 3], dtype=np.float64)
    momentum_scores = (recent_close / previous_close - 1).astype(np.float32)
    baseline_score_map = {
        "no_trade": no_trade_scores,
        "always_buy_top10_liquidity_rank": always_scores,
        "positive_5day_momentum_top10": momentum_scores,
    }
    baselines = {}
    for baseline_name, baseline_scores in baseline_score_map.items():
        baselines[baseline_name] = {
            name: _summary(simulate_portfolio(
                samples, selected[name], baseline_scores, threshold=0.0,
                cost_bps=cost_bps, allocation=allocation,
                max_positions=max_positions, initial_equity=initial_equity))
            for name in ("train", "validation", "test")
        }
    def scenario_report(scores: np.ndarray, threshold: float) -> dict:
        by_split = {}
        for name in ("train", "validation", "test"):
            raw = simulate_missing_scenarios(
                samples, selected[name], scores, threshold=threshold,
                cost_bps=cost_bps, allocation=allocation,
                max_positions=max_positions, initial_equity=initial_equity)
            by_split[name] = {
                "unresolved_selected": raw["unresolved_selected"],
                "no_fill": _summary(raw["no_fill"]),
                "full_budget_loss": _summary(raw["full_budget_loss"]),
            }
        return by_split

    missing_scenarios = {
        "research_only": True, "not_exact": True,
        "not_used_for_genetic_selection": True,
        "champion": scenario_report(frozen_scores, float(champion[-1])),
        "baselines": {name: scenario_report(scores, 0.0)
                      for name, scores in baseline_score_map.items()},
    }
    active_contender = None
    if active_champion is not None:
        active_scores = score_genome(samples.windows, active_champion,
                                     device=runtime_device, chunk_rows=chunk_rows)
        active_results = {
            name: _summary(simulate_portfolio(
                samples, selected[name], active_scores,
                threshold=float(active_champion[-1]), cost_bps=cost_bps,
                allocation=allocation, max_positions=max_positions,
                initial_equity=initial_equity))
            for name in ("train", "validation", "test")
        }
        active_contender = {
            "exploratory": True, "deployment_allowed": False,
            "selection": "best train fitness among complete-path candidates executing in >=2% of train sessions",
            "train_fitness": active_best_fitness,
            "genome_sha256": hashlib.sha256(active_champion.tobytes()).hexdigest(),
            "genome": active_champion.tolist(),
            "score_threshold": float(active_champion[-1]),
            "results": active_results,
            "missing_target_scenarios": scenario_report(
                active_scores, float(active_champion[-1])),
        }
    sensitivity = {}
    for cost in tuple(dict.fromkeys((*cost_grid_bps, cost_bps))):
        if not math.isfinite(cost) or not 0 <= cost < 10_000:
            raise ValueError("Cost sensitivity grid must contain finite nonnegative bps")
        sensitivity[str(cost)] = _summary(simulate_portfolio(
            samples, selected["test"], frozen_scores,
            threshold=float(champion[-1]), cost_bps=float(cost),
            allocation=allocation, max_positions=max_positions,
            initial_equity=initial_equity))
    report = {
        "experiment": "mark1-4-random-neural-genetic-daily-open-close-v1",
        "research_only": True, "deployment_allowed": False,
        "decision": "after 30 completed daily bars through t, before t+1 open",
        "genome": {"architecture": "random 150-to-12 tanh-to-1 tanh scorer plus threshold",
                   "genes": GENOME_SIZE, "input": "30x5 causal log-normalized completed OHLCV",
                   "random_initialization": True, "selection": "train fitness only",
                   "recombination": "uniform crossover among top 1/8 plus Gaussian mutation"},
        "search": {"seed": seed, "device": runtime_device,
                   "population_size": population_size, "generations": generations,
                   "fitness_trials": population_size * generations,
                   "active_eligible_trials": active_eligible_trials,
                   "champion_train_fitness": best_fitness,
                   "fitness_definition": "complete-path integer-share daily PnL: 252*mean - 0.5*sqrt(252)*downside RMS - 0.25*max drawdown - 0.05*executed-activity deficit below 2%; unknown selected outcome disqualified",
                   "validation_or_test_used_in_selection": False,
                   "history": fitness_history},
        "policy": {"score_threshold": float(champion[-1]),
                   "rank": "descending score, ascending symbol ID tie-break",
                   "position_fraction_of_start_day_equity": allocation,
                   "max_same_day_positions": max_positions,
                   "entry": "t+1 opening auction proxy",
                   "exit": "t+1 closing auction proxy",
                   "roundtrip_cost_bps": cost_bps,
                   "initial_equity": initial_equity},
        "champion_sha256": hashlib.sha256(champion.tobytes()).hexdigest(),
        "champion_training_path_complete": not results["train"]["incomplete_data"],
        "splits": {name: {"candidates": len(indices),
                          "first_target": min(samples.target_dates[indices].tolist()),
                          "last_target": max(samples.target_dates[indices].tolist())}
                   for name, indices in selected.items()},
        "results": results, "baselines": baselines,
        "active_contender": active_contender,
        "missing_target_scenarios": missing_scenarios,
        "frozen_test_cost_sensitivity_bps": sensitivity,
        "source": samples.source,
        "limitations": [
            "Historical catalog may introduce survivorship bias; source metadata must disclose selection.",
            "Only selected outcomes with valid next-day prices can establish a complete equity path.",
            "Missing selected outcomes invalidate final equity and compound return; they are not zero-PnL fills.",
            "Missing-target no-fill and full-budget-loss scenarios are reporting-only assumptions, not observed returns or exact upper/lower bounds.",
            "OPEN/CLOSE observations do not prove auction fills or available shares; spread and impact are approximated by declared bps.",
            "Daily bars cannot resolve intraday barrier order; this experiment does not claim stop-loss/take-profit fills.",
            "Many adaptive fitness trials overfit; frozen historical test is not a pristine live experiment.",
            "CUDA floating-point reductions can break bitwise reproducibility at near-tied scores; CPU seeded runs are deterministic.",
            "Currencies and markets are not mixed; each run requires its own initial equity and cost assumptions.",
            "Leading/trailing calendar sessions without any candidate cannot be inferred from split indices and are not included in session counts.",
        ],
    }
    return report, champion
