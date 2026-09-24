"""Research-only signal evaluation with a fixed moving-block bootstrap gate.

Returns describe equally weighted *per-signal daily OHLC proxies*, not a
portfolio or achievable fills. Passing the research gate does not authorize
live orders. All supplied session dates participate, including no-signal days;
the caller must supply every represented evaluation session, not just trades.
Missing sessions cannot be reconstructed without an exchange calendar.
"""

from __future__ import annotations

from datetime import date, datetime
import math
from numbers import Integral, Real

import numpy as np

from dockdack.mark1_metrics import binary_metrics


BLOCK_LENGTH = 10
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 42
CONFIDENCE = 0.95
QUALIFICATION_COST_BPS = 20.0


def _ordered_dates(dates, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return chronological session keys and group indices, rejecting NaT."""
    values = np.asarray(dates)
    if values.ndim != 1 or len(values) != count:
        raise ValueError("dates must be a one-dimensional array matching labels")
    if not count:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if values.dtype.kind in "iu":
        keys = values
    elif values.dtype.kind == "M":
        if np.isnat(values).any():
            raise ValueError("dates must not contain NaT")
        keys = values.astype("datetime64[D]").astype(np.int64)
    elif values.dtype.kind in "USO":
        keys = []
        integer_dates = values.dtype.kind == "O" and all(
            isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            for value in values)
        if integer_dates:
            keys = np.asarray(values.tolist(), dtype=np.int64)
        else:
            for value in values:
                if isinstance(value, datetime):
                    parsed = value.date()
                elif isinstance(value, date):
                    parsed = value
                elif isinstance(value, (str, np.str_)):
                    try:
                        parsed = date.fromisoformat(str(value))
                    except (ValueError, TypeError) as exc:
                        raise ValueError("dates must contain valid ISO calendar dates") from exc
                else:
                    raise ValueError("dates must contain integer day IDs or calendar dates")
                keys.append(parsed.toordinal())
            keys = np.asarray(keys, dtype=np.int64)
    else:
        raise ValueError("dates must contain integer day IDs or calendar dates")
    unique, inverse = np.unique(keys, return_inverse=True)
    return unique, inverse


def _block_bootstrap(labels, probabilities, gross_returns, groups, count_days,
                     cost_bps: float) -> dict:
    selected = probabilities > 0.5
    signal_count = int(selected.sum())
    count_by_day = np.bincount(groups, weights=selected, minlength=count_days)
    signal_days = int(np.count_nonzero(count_by_day))
    result = {
        "method": "moving_block_percentile", "block_length": BLOCK_LENGTH,
        "confidence": CONFIDENCE, "n_boot": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED, "count_days": int(count_days),
        "signal_count": signal_count, "signal_days": signal_days,
        "cost_bps": float(cost_bps), "valid_replicates": 0,
        "skipped_zero_signal_replicates": 0,
        "precision_lower": None, "precision_upper": None,
        "net_mean_lower": None, "net_mean_upper": None, "reason": None,
    }
    if signal_count < 100 or signal_days < 30:
        result["reason"] = "insufficient_signals_or_signal_days_for_interval"
        return result

    wins_by_day = np.bincount(groups, weights=labels * selected,
                              minlength=count_days)
    # Scale before summing to avoid overflow for finite, very large returns.
    selected_returns = gross_returns[selected]
    scale = max(1.0, float(np.max(np.abs(selected_returns))))
    returns_by_day = np.bincount(groups[selected],
                                 weights=selected_returns / scale,
                                 minlength=count_days)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    block_count = (count_days + BLOCK_LENGTH - 1) // BLOCK_LENGTH
    offsets = np.arange(BLOCK_LENGTH)
    precisions, net_means = [], []
    for _ in range(BOOTSTRAP_REPLICATES):
        starts = generator.integers(0, count_days - BLOCK_LENGTH + 1,
                                    size=block_count)
        sampled = (starts[:, None] + offsets).reshape(-1)[:count_days]
        denominator = float(count_by_day[sampled].sum())
        if denominator == 0:
            result["skipped_zero_signal_replicates"] += 1
            continue
        precision = float(wins_by_day[sampled].sum() / denominator)
        scaled_mean = float(returns_by_day[sampled].sum() / denominator)
        net_mean = float(np.clip(scaled_mean, -1.0, 1.0) * scale
                         - cost_bps / 10000)
        if not math.isfinite(precision) or not math.isfinite(net_mean):
            raise ValueError("Non-finite bootstrap estimate")
        precisions.append(precision)
        net_means.append(net_mean)
    result["valid_replicates"] = len(precisions)
    if not precisions:
        result["reason"] = "no_valid_bootstrap_replicates"
        return result
    precision_interval = np.quantile(precisions, [.025, .975])
    net_interval = np.quantile(net_means, [.025, .975])
    if not (np.isfinite(precision_interval).all() and np.isfinite(net_interval).all()):
        raise ValueError("Non-finite bootstrap quantile")
    result.update(precision_lower=float(precision_interval[0]),
                  precision_upper=float(precision_interval[1]),
                  net_mean_lower=float(net_interval[0]),
                  net_mean_upper=float(net_interval[1]))
    return result


def evaluate(labels, probabilities, gross_returns, dates, cost_bps: float = 20) -> dict:
    """Evaluate already-calibrated probabilities and fixed strict >50% signals.

    The 95% interval uses 500 non-circular moving-block replicates (seed 42),
    each containing 10 consecutive supplied session-date clusters. Blocks are
    sampled with replacement and the last block is truncated to the original
    number of dates. This preserves within-day dependence and some serial
    dependence, not all dependence or uncertainty from repeated model search.
    Intervals are suppressed below 100 signals or 30 signal-producing dates.
    No calibration is fitted here and no model or threshold is selected here.
    """
    # Existing metrics validate labels/probabilities/returns/cost before any
    # aggregation; normalized integer dates preserve chronological sorting.
    result = binary_metrics(labels, probabilities, gross_returns=gross_returns,
                            cost_bps=cost_bps)
    if gross_returns is None:
        raise ValueError("gross_returns are required for research evaluation")
    unique, groups = _ordered_dates(dates, result["count"])
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    gross_returns = np.asarray(gross_returns, dtype=np.float64)
    result["block_bootstrap"] = _block_bootstrap(
        labels, probabilities, gross_returns, groups, len(unique), float(cost_bps))
    # Keep the legacy key explicitly undefined: the reported uncertainty is
    # the serial-dependence-aware block interval, not an independent-day one.
    result["precision_date_bootstrap"] = None
    return result


def _finite_real(value) -> bool:
    return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(value))


def qualification(metrics: dict) -> dict:
    """Fail-closed research gate; never permission to deploy or submit orders."""
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an evaluation dictionary")
    reasons = []
    signal_count = metrics.get("signal_count")
    if (not isinstance(signal_count, Integral) or isinstance(signal_count, bool)
            or signal_count < 200):
        reasons.append("requires_at_least_200_signals")
    block = metrics.get("block_bootstrap")
    if not isinstance(block, dict):
        return {"qualified": False, "reasons": reasons + ["missing_block_bootstrap"]}
    signal_days = block.get("signal_days")
    if (not isinstance(signal_days, Integral) or isinstance(signal_days, bool)
            or signal_days < 50):
        reasons.append("requires_at_least_50_signal_days")
    if (not _finite_real(metrics.get("cost_bps"))
            or metrics["cost_bps"] != QUALIFICATION_COST_BPS
            or not _finite_real(block.get("cost_bps"))
            or block["cost_bps"] != QUALIFICATION_COST_BPS):
        reasons.append("requires_20_bps_round_trip_cost")
    if (block.get("method") != "moving_block_percentile"
            or block.get("block_length") != BLOCK_LENGTH
            or block.get("n_boot") != BOOTSTRAP_REPLICATES
            or block.get("seed") != BOOTSTRAP_SEED
            or block.get("confidence") != CONFIDENCE
            or block.get("signal_count") != signal_count
            or block.get("reason", "missing") is not None):
        reasons.append("invalid_or_suppressed_block_interval")
    valid, skipped = block.get("valid_replicates"), block.get("skipped_zero_signal_replicates")
    count_days = block.get("count_days")
    if (not isinstance(valid, Integral) or isinstance(valid, (bool, np.bool_))
            or not isinstance(skipped, Integral) or isinstance(skipped, (bool, np.bool_))
            or valid < 1 or skipped < 0 or valid + skipped != BOOTSTRAP_REPLICATES
            or not isinstance(count_days, Integral) or isinstance(count_days, (bool, np.bool_))
            or not isinstance(signal_days, Integral) or not 0 <= signal_days <= count_days):
        reasons.append("invalid_bootstrap_replicate_or_date_counts")
    precision_lower, precision_upper = block.get("precision_lower"), block.get("precision_upper")
    net_lower, net_upper = block.get("net_mean_lower"), block.get("net_mean_upper")
    if (not _finite_real(precision_lower) or not _finite_real(precision_upper)
            or not 0.5 < precision_lower <= precision_upper <= 1):
        reasons.append("precision_lower_must_exceed_0_5")
    if (not _finite_real(net_lower) or not _finite_real(net_upper)
            or not 0 < net_lower <= net_upper):
        reasons.append("net_mean_lower_must_be_positive_after_cost")
    return {"qualified": not reasons, "reasons": reasons}
