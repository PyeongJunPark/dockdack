"""Train-quantile / 2021-only policy choice for Mark1.4 follow-up scorers.

The model weights are fitted elsewhere on 2018--2020 target outcomes. This
module freezes a numeric threshold from those *training scores*, then chooses
among fixed seeds and coverages using only exact 2021 calibration portfolios.
It never reads outcomes outside the supplied calibration indices and never
places an order. The best exact, sufficiently active candidate is retained
even when its calibration objective is below the zero-return cash benchmark;
that shortfall is reported rather than silently changing the preregistered
selection rule.
"""

from __future__ import annotations

from datetime import date
import hashlib
import math
from typing import Mapping

import numpy as np

from dockdack.mark1_4_evolution import EvolutionSamples, simulate_portfolio


DEFAULT_TARGET_COVERAGES = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)


def _period_rows(samples: EvolutionSamples, indices: np.ndarray, *,
                 earliest: str, latest: str, name: str) -> np.ndarray:
    rows = np.asarray(indices)
    if (rows.ndim != 1 or len(rows) == 0
            or not np.issubdtype(rows.dtype, np.integer)
            or np.any(rows < 0) or np.any(rows >= len(samples.windows))
            or len(np.unique(rows)) != len(rows)):
        raise ValueError(f"{name}_indices must be unique in-range integers")
    selected = rows.astype(np.int64, copy=False)
    period_dates = [str(day) for day in np.asarray(samples.target_dates)[selected]]
    try:
        parsed = [date.fromisoformat(day) for day in period_dates]
    except ValueError as exc:
        raise ValueError(f"{name} target dates must be ISO dates") from exc
    start = date.fromisoformat(earliest)
    end = date.fromisoformat(latest)
    if any(day < start or day > end for day in parsed):
        raise ValueError(f"{name} target dates must be in {earliest}..{latest}")
    return selected


def select_calibrated_policy(
    samples: EvolutionSamples,
    scores_by_seed: Mapping[int, np.ndarray],
    train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    *,
    target_coverages: tuple[float, ...] = DEFAULT_TARGET_COVERAGES,
    cost_bps: float = 20.0,
    allocation: float = 0.1,
    max_positions: int = 10,
    initial_equity: float = 10_000_000.0,
    min_executed_trades: int = 20,
) -> dict:
    """Select at most one seed/coverage without reading 2022+ outcomes.

    ``scores_by_seed`` contains already fitted all-row scores for one model
    variant. Values at 2022+ rows are *never accessed*. For each seed and q,
    a numeric threshold is the 1-q quantile of 2018--2020 scores. The same
    threshold is evaluated once on 2021, with the declared cash-only,
    integer-share, next-open-to-close proxy. Only complete paths with at least
    ``min_executed_trades`` can compete, and the scalar objective is
    ``compound_net_return - 0.5 * abs(max_drawdown)``. Cash has objective 0
    as a comparator, not an eligibility gate. Strictly greater objective
    and stable seed/q order resolve ties.
    """
    train = _period_rows(samples, train_indices, earliest="2018-01-01",
                         latest="2020-12-31", name="train")
    calibration = _period_rows(samples, calibration_indices,
                               earliest="2021-01-01", latest="2021-12-31",
                               name="calibration")
    if len(np.intersect1d(train, calibration)):
        raise ValueError("Training and calibration rows overlap")
    ordinals = np.asarray(samples.target_ordinals)
    if int(ordinals[train].max()) >= int(ordinals[calibration].min()):
        raise ValueError("Training sessions must precede calibration sessions")
    if not scores_by_seed:
        raise ValueError("At least one fitted seed is required")
    if (not isinstance(target_coverages, tuple) or not target_coverages
            or any(not math.isfinite(float(q)) or not 0 < float(q) < 1
                   for q in target_coverages)
            or len(set(target_coverages)) != len(target_coverages)):
        raise ValueError("target_coverages must contain unique finite fractions in (0,1)")
    if type(min_executed_trades) is not int or min_executed_trades < 1:
        raise ValueError("min_executed_trades must be a positive integer")
    if (not math.isfinite(cost_bps) or not 0 <= cost_bps < 10_000
            or not math.isfinite(initial_equity) or initial_equity <= 0):
        raise ValueError("Invalid cost or starting equity")
    if (not 0 < allocation <= 1 or type(max_positions) is not int
            or max_positions < 1 or allocation * max_positions > 1 + 1e-12):
        raise ValueError("Allocation and position cap must fit cash equity")

    records: list[dict] = []
    selected: dict | None = None
    eligible_data_activity = 0
    for seed in sorted(scores_by_seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed keys must be nonnegative integers")
        scores = np.asarray(scores_by_seed[seed], dtype=np.float32)
        if scores.shape != (len(samples.windows),) or not np.isfinite(scores[train]).all() or not np.isfinite(scores[calibration]).all():
            raise ValueError("Each seed needs finite scores for train and calibration candidates")
        train_scores = np.asarray(scores[train], dtype=np.float64)
        score_hash = hashlib.sha256(np.ascontiguousarray(scores[train]).tobytes()).hexdigest()
        for q in target_coverages:
            numeric_threshold = float(np.quantile(train_scores, 1 - q, method="linear"))
            outcome = simulate_portfolio(
                samples, calibration, scores, threshold=numeric_threshold,
                cost_bps=cost_bps, allocation=allocation,
                max_positions=max_positions, initial_equity=initial_equity,
            )
            complete = not outcome["incomplete_data"]
            fills = outcome["executed_trades"] if complete else None
            active = complete and fills >= min_executed_trades
            eligible_data_activity += bool(active)
            objective = (float(outcome["compound_net_return"]
                               - 0.5 * abs(outcome["max_drawdown"]))
                         if active else None)
            if not complete:
                status = "incomplete_selected_outcome"
            elif fills < min_executed_trades:
                status = "insufficient_integer_share_fills"
            elif objective <= 0:
                status = "eligible_below_or_equal_cash"
            else:
                status = "eligible_above_cash"
            record = {
                "seed": seed,
                "target_train_candidate_coverage": float(q),
                "achieved_train_candidate_coverage": float(np.mean(train_scores > numeric_threshold)),
                "train_numeric_score_threshold": numeric_threshold,
                "train_score_sha256": score_hash,
                "calibration_objective": objective,
                "status": status,
                "calibration": {key: value for key, value in outcome.items()
                                if key not in {"daily_dates", "daily_returns",
                                               "observed_day_proxy_returns"}},
            }
            records.append(record)
            if objective is not None and (
                    selected is None or objective > selected["calibration_objective"]):
                selected = record
    return {
        "experiment": "mark1-4-followup-train-quantile-2021-selection",
        "research_only": True,
        "deployment_allowed": False,
        "decision": "selected" if selected is not None else "cash",
        "selected": selected,
        "no_eligible_strategy": selected is None,
        "cash_objective": 0.0,
        "objective": "2021 exact compound net return - 0.5 * abs(max drawdown)",
        "eligibility": (
            "complete 2021 portfolio path, at least "
            f"{min_executed_trades} integer-share fills; cash zero is reported, not a gate"
        ),
        "calibration_candidates_meeting_data_and_activity": eligible_data_activity,
        "evaluated_seed_coverage_pairs": len(records),
        "candidates": records,
        "train_target_range": [str(min(samples.target_dates[train])),
                               str(max(samples.target_dates[train]))],
        "calibration_target_range": [str(min(samples.target_dates[calibration])),
                                     str(max(samples.target_dates[calibration]))],
        "threshold_source": "2018-2020 training candidate scores only; fixed numeric threshold",
        "selection_source": "2021 calibration exact integer-share portfolio only",
        "later_outcomes_or_scores_used": False,
        "policy": {"cost_bps": float(cost_bps), "allocation": float(allocation),
                   "max_positions": max_positions,
                   "initial_equity": float(initial_equity)},
    }
