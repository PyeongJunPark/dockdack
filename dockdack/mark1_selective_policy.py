"""Research-only rare-signal policy selection on a caller-owned holdout.

This module never fits probabilities or touches models, databases, or orders.
The caller must keep this policy-selection period separate from probability
calibration, model fitting, and final evaluation. Selection intervals are not
adjusted for trying the grid and are not independent evidence of profitability.
Returns are equal-weight daily-bar proxies, not executable portfolio returns.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from dockdack.mark1_deep_validation import _ordered_dates
from dockdack.mark1_metrics import _probabilities, binary_metrics


THRESHOLDS = (.50, .55, .60, .65, .70, .75, .80, .85, .90, .95)
STOP_PROBABILITY_CAPS = (1.0, .25)
BLOCK_LENGTH = 10
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42
CONFIDENCE = .95
MIN_SIGNALS = 50
MIN_SIGNAL_DAYS = 20
MIN_SYMBOLS = 10
QUALIFICATION_COST_BPS = 20.0
MIN_PRECISION = .65
MIN_PRECISION_LOWER = .579


def _finite_real(value) -> bool:
    return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(value))


def _integer(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _symbols(symbol_ids, count: int) -> np.ndarray:
    values = np.asarray(symbol_ids)
    if values.ndim != 1 or len(values) != count:
        raise ValueError("symbol_ids must be a one-dimensional array matching labels")
    if not count:
        return np.empty(0, dtype=np.int64)
    if values.dtype.kind in "iu":
        return values
    if values.dtype.kind in "US":
        keys = values.astype(str)
        if any(not value.strip() for value in keys):
            raise ValueError("symbol_ids must not contain blank identifiers")
        return keys
    if values.dtype.kind == "O":
        if all(_integer(value) for value in values):
            # Object integer identifiers need not fit in int64.
            return np.asarray([str(value) for value in values], dtype=str)
        if all(isinstance(value, (str, np.str_)) and str(value).strip()
               for value in values):
            return values.astype(str)
    raise ValueError("symbol_ids must contain only nonmissing string or integer identifiers")


def _selected_mask(selected, count: int, probabilities: np.ndarray) -> np.ndarray:
    mask = np.asarray(selected)
    if mask.ndim != 1 or len(mask) != count or (count and mask.dtype.kind != "b"):
        raise ValueError("selected must be a one-dimensional boolean mask matching labels")
    mask = mask.astype(bool, copy=False)
    if np.any(mask & (probabilities <= .5)):
        raise ValueError("selected signals must satisfy the strict probability > 0.5 rule")
    return mask


def _stop_probabilities(values, count: int) -> np.ndarray | None:
    if values is None:
        return None
    result = _probabilities(values)
    if len(result) != count:
        raise ValueError("stop_probabilities must match probabilities")
    return result


def _policy(policy: dict) -> tuple[float, float]:
    if not isinstance(policy, dict):
        raise ValueError("policy must be a dictionary")
    threshold, cap = policy.get("threshold"), policy.get("stop_probability_cap")
    if not _finite_real(threshold) or not .5 <= threshold <= 1:
        raise ValueError("policy threshold must be finite and in [0.5, 1]")
    if not _finite_real(cap) or not 0 <= cap <= 1:
        raise ValueError("stop_probability_cap must be finite and in [0, 1]")
    return float(threshold), float(cap)


def apply_policy(probabilities, policy: dict, stop_probabilities=None) -> np.ndarray:
    """Return a new strict-confidence, optional inclusive-stop-cap mask.

    ``stop_probabilities`` represents stop_only + both_touch. Its construction
    is the caller's responsibility; it is never fitted or recalibrated here.
    A cap of 1 is unrestricted and does not require stop probabilities.
    """
    values = _probabilities(probabilities)
    threshold, cap = _policy(policy)
    stops = _stop_probabilities(stop_probabilities, len(values))
    if cap < 1 and stops is None:
        raise ValueError("a restricted stop cap requires stop_probabilities")
    result = (values > .5) & (values > threshold)
    if cap < 1:
        result &= stops <= cap
    return result


def _block_bootstrap(labels, returns, groups, selected, count_days,
                     signal_days, symbol_count, cost_bps) -> dict:
    signal_count = int(selected.sum())
    result = {
        "method": "moving_block_percentile", "block_length": BLOCK_LENGTH,
        "confidence": CONFIDENCE, "n_boot": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED, "count_days": int(count_days),
        "signal_count": signal_count, "signal_days": signal_days,
        "symbol_count": symbol_count, "cost_bps": float(cost_bps),
        "minimum_signals": MIN_SIGNALS, "minimum_signal_days": MIN_SIGNAL_DAYS,
        "minimum_symbols": MIN_SYMBOLS, "valid_replicates": 0,
        "skipped_zero_signal_replicates": 0,
        "precision_lower": None, "precision_upper": None,
        "net_mean_lower": None, "net_mean_upper": None, "reason": None,
        "adjusted_for_policy_search": False,
    }
    if (signal_count < MIN_SIGNALS or signal_days < MIN_SIGNAL_DAYS
            or symbol_count < MIN_SYMBOLS):
        result["reason"] = "insufficient_signals_signal_days_or_symbols_for_interval"
        return result
    counts = np.bincount(groups[selected], minlength=count_days)
    wins = np.bincount(groups[selected], weights=labels[selected], minlength=count_days)
    selected_returns = returns[selected]
    scale = max(1.0, float(np.max(np.abs(selected_returns))))
    return_sums = np.bincount(groups[selected], weights=selected_returns / scale,
                              minlength=count_days)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    block_count = (count_days + BLOCK_LENGTH - 1) // BLOCK_LENGTH
    offsets = np.arange(BLOCK_LENGTH)
    precisions, net_means = [], []
    for _ in range(BOOTSTRAP_REPLICATES):
        starts = generator.integers(0, count_days - BLOCK_LENGTH + 1, size=block_count)
        sampled = (starts[:, None] + offsets).reshape(-1)[:count_days]
        denominator = float(counts[sampled].sum())
        if denominator == 0:
            result["skipped_zero_signal_replicates"] += 1
            continue
        precision = float(wins[sampled].sum() / denominator)
        scaled_mean = float(return_sums[sampled].sum() / denominator)
        net_mean = float(np.clip(scaled_mean, -1, 1) * scale - cost_bps / 10000)
        if not math.isfinite(precision) or not math.isfinite(net_mean):
            raise ValueError("nonfinite bootstrap estimate")
        precisions.append(precision)
        net_means.append(net_mean)
    result["valid_replicates"] = len(precisions)
    if not precisions:
        result["reason"] = "no_valid_bootstrap_replicates"
        return result
    precision_interval = np.quantile(precisions, [.025, .975])
    net_interval = np.quantile(net_means, [.025, .975])
    if not (np.isfinite(precision_interval).all() and np.isfinite(net_interval).all()):
        raise ValueError("nonfinite bootstrap interval")
    result.update(precision_lower=float(precision_interval[0]),
                  precision_upper=float(precision_interval[1]),
                  net_mean_lower=float(net_interval[0]), net_mean_upper=float(net_interval[1]))
    return result


def _evaluate(labels, probabilities, returns, groups, symbols, selected,
              count_days, overall, cost_bps) -> dict:
    count = len(labels)
    signal_count = int(selected.sum())
    signal_days = int(np.unique(groups[selected]).size)
    symbol_count = int(np.unique(symbols[selected]).size)
    gross_mean = None
    if signal_count:
        selected_returns = returns[selected]
        scale = max(1.0, float(np.max(np.abs(selected_returns))))
        gross_mean = float((selected_returns / scale).mean() * scale)
    result = {
        "overall": overall, "count": count, "signal_count": signal_count,
        "signal_days": signal_days, "symbol_count": symbol_count,
        "count_days": int(count_days), "coverage": signal_count / count if count else None,
        "precision": float(labels[selected].mean()) if signal_count else None,
        "gross_mean_return": gross_mean,
        "net_mean_return": gross_mean - cost_bps / 10000 if gross_mean is not None else None,
        "cost_bps": float(cost_bps),
    }
    result["block_bootstrap"] = _block_bootstrap(
        labels, returns, groups, selected, count_days, signal_days, symbol_count, cost_bps)
    return result


def _validated(labels, probabilities, gross_returns, dates, symbol_ids, cost_bps):
    if gross_returns is None:
        raise ValueError("gross_returns are required")
    # This validates all numeric vectors and preserves the original probability
    # errors, ranking, and strict >.5 signal metrics under the 'overall' key.
    overall = binary_metrics(labels, probabilities, gross_returns=gross_returns, cost_bps=cost_bps)
    count = overall["count"]
    unique, groups = _ordered_dates(dates, count)
    symbols = _symbols(symbol_ids, count)
    return (np.asarray(labels, dtype=np.float64), np.asarray(probabilities, dtype=np.float64),
            np.asarray(gross_returns, dtype=np.float64), groups, symbols, len(unique), overall)


def evaluate_signals(labels, probabilities, gross_returns, dates, symbol_ids, selected,
                     cost_bps: float = 20) -> dict:
    """Evaluate an explicit mask without changing any supplied probabilities.

    CI: 2,000 seed-42 non-circular moving-block replicates of 10 consecutive
    supplied sessions, with the last block truncated. Every represented date,
    including no-signal dates, participates; each date's symbols stay together.
    The caller must supply all evaluation rows, not just selected trades.
    Dates absent from the input cannot be reconstructed. Intervals are withheld
    below 50 events, 20 signal dates, or 10 selected symbols. They capture some
    temporal dependence, not policy-search or data-selection uncertainty.
    """
    labels, probabilities, returns, groups, symbols, count_days, overall = _validated(
        labels, probabilities, gross_returns, dates, symbol_ids, cost_bps)
    mask = _selected_mask(selected, len(labels), probabilities)
    return _evaluate(labels, probabilities, returns, groups, symbols, mask,
                     count_days, overall, float(cost_bps))


def _eligible(metrics: dict) -> bool:
    return (metrics["signal_count"] >= MIN_SIGNALS
            and metrics["signal_days"] >= MIN_SIGNAL_DAYS
            and metrics["symbol_count"] >= MIN_SYMBOLS
            and metrics["block_bootstrap"]["reason"] is None
            and _finite_real(metrics["block_bootstrap"]["precision_lower"])
            and _finite_real(metrics["block_bootstrap"]["net_mean_lower"]))


def qualification(metrics: dict) -> dict:
    """Fail-closed research gate, never permission to trade or deploy."""
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary")
    reasons = []
    for key, minimum in (("signal_count", MIN_SIGNALS), ("signal_days", MIN_SIGNAL_DAYS),
                         ("symbol_count", MIN_SYMBOLS)):
        if not _integer(metrics.get(key)) or metrics[key] < minimum:
            reasons.append(f"requires_at_least_{minimum}_{key}")
    if not _finite_real(metrics.get("precision")) or not MIN_PRECISION <= metrics["precision"] <= 1:
        reasons.append("precision_must_be_at_least_0_65")
    block = metrics.get("block_bootstrap")
    if not isinstance(block, dict):
        return {"qualified": False, "reasons": reasons + ["missing_block_bootstrap"]}
    if (not _finite_real(metrics.get("cost_bps"))
            or metrics["cost_bps"] != QUALIFICATION_COST_BPS
            or not _finite_real(block.get("cost_bps"))
            or block["cost_bps"] != QUALIFICATION_COST_BPS):
        reasons.append("requires_20_bps_round_trip_cost")
    expected = {"method": "moving_block_percentile", "block_length": BLOCK_LENGTH,
                "n_boot": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                "confidence": CONFIDENCE, "minimum_signals": MIN_SIGNALS,
                "minimum_signal_days": MIN_SIGNAL_DAYS, "minimum_symbols": MIN_SYMBOLS}
    if (any(block.get(key) != value for key, value in expected.items())
            or any(block.get(key) != metrics.get(key)
                   for key in ("signal_count", "signal_days", "symbol_count", "count_days"))
            or block.get("reason", "missing") is not None):
        reasons.append("invalid_or_suppressed_block_interval")
    valid = block.get("valid_replicates")
    skipped = block.get("skipped_zero_signal_replicates")
    count, days = metrics.get("count"), metrics.get("count_days")
    signals, signal_days, symbols = (metrics.get(key) for key in
                                    ("signal_count", "signal_days", "symbol_count"))
    if (not _integer(valid) or not _integer(skipped) or valid < 1 or skipped < 0
            or valid + skipped != BOOTSTRAP_REPLICATES
            or not all(_integer(value) for value in (count, days, signals, signal_days, symbols))
            or not 0 <= signal_days <= days <= count
            or not 0 <= max(signal_days, symbols) <= signals <= count):
        reasons.append("invalid_bootstrap_replicate_or_population_counts")
    lower, upper = block.get("precision_lower"), block.get("precision_upper")
    if (not _finite_real(lower) or not _finite_real(upper)
            or not MIN_PRECISION_LOWER < lower <= upper <= 1):
        reasons.append("precision_lower_must_exceed_0_579")
    lower, upper = block.get("net_mean_lower"), block.get("net_mean_upper")
    if not _finite_real(lower) or not _finite_real(upper) or not 0 < lower <= upper:
        reasons.append("net_mean_lower_must_be_positive_after_cost")
    return {"qualified": not reasons, "reasons": reasons}


def select_policy(labels, probabilities, gross_returns, dates, symbol_ids,
                  stop_probabilities=None) -> dict:
    """Select from a declared grid on a separate chronological policy holdout.

    Among sufficiently supported candidates, maximize precision CI lower bound,
    then net-return lower bound, then event count, then prefer no stop cap and a
    lower threshold. Qualification is reported separately, never imposed by
    silently altering the grid. With no eligible candidate, the >.5 unrestricted
    policy remains diagnostic-only and is explicitly unqualified.
    """
    labels, probabilities, returns, groups, symbols, count_days, overall = _validated(
        labels, probabilities, gross_returns, dates, symbol_ids, QUALIFICATION_COST_BPS)
    stops = _stop_probabilities(stop_probabilities, len(labels))
    grid = []
    caps = STOP_PROBABILITY_CAPS if stops is not None else (1.0,)
    for cap in caps:
        for threshold in THRESHOLDS:
            policy = {"threshold": threshold, "stop_probability_cap": cap}
            selected = apply_policy(probabilities, policy, stops)
            metrics = _evaluate(labels, probabilities, returns, groups, symbols, selected,
                                count_days, overall, QUALIFICATION_COST_BPS)
            grid.append({"policy": policy, "metrics": metrics, "eligible": _eligible(metrics)})
    eligible = [item for item in grid if item["eligible"]]
    if eligible:
        chosen = max(eligible, key=lambda item: (
            item["metrics"]["block_bootstrap"]["precision_lower"],
            item["metrics"]["block_bootstrap"]["net_mean_lower"],
            item["metrics"]["signal_count"], item["policy"]["stop_probability_cap"],
            -item["policy"]["threshold"]))
        reason = "max_precision_lower_then_net_lower_count_and_simplicity"
    else:
        chosen, reason = grid[0], "no_eligible_candidate_diagnostic_only"
    gate = qualification(chosen["metrics"])
    return {"grid": grid, "chosen_policy": chosen["policy"], "chosen_metrics": chosen["metrics"],
            "calibration_qualified": bool(eligible and gate["qualified"]), "qualification": gate,
            "selection_reason": reason, "research_only": True, "deployment_allowed": False,
            "selection_intervals_adjusted_for_grid_search": False}
