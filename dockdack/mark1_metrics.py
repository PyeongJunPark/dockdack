"""Dependency-light binary evaluation and monotone probability calibration.

Signals use the strict rule ``probability > 0.5``. Reported returns are simple
per-signal hypothetical returns, not an executable portfolio backtest. A fixed
round-trip cost is subtracted from each selected gross return. Date-cluster
bootstrap resamples entire target dates, preserving same-day cross-symbol
dependence; it does not eliminate serial dependence between different dates.
"""

from __future__ import annotations

from datetime import date, datetime
import math

import numpy as np


_LOG_LOSS_EPSILON = 1e-15


def _numeric_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a one-dimensional numeric array")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _labels(values) -> np.ndarray:
    labels = _numeric_vector(values, "labels")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("labels must contain only 0 or 1")
    return labels


def _probabilities(values) -> np.ndarray:
    probabilities = _numeric_vector(values, "probabilities")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return probabilities


def _paired(labels, probabilities) -> tuple[np.ndarray, np.ndarray]:
    labels, probabilities = _labels(labels), _probabilities(probabilities)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have equal lengths")
    return labels, probabilities


def _date_groups(dates, count: int) -> tuple[np.ndarray, int]:
    values = np.asarray(dates)
    if values.ndim != 1 or len(values) != count:
        raise ValueError("dates must be a one-dimensional array matching labels")
    if not count:
        return np.empty(0, dtype=np.int64), 0
    if values.dtype.kind == "M":
        if np.isnat(values).any():
            raise ValueError("dates must not contain NaT")
        keys = values.astype("datetime64[D]").astype(str)
    elif values.dtype.kind in "iu":
        # Integer calendar-date IDs (e.g. YYYYMMDD) are accepted unchanged.
        keys = values
    elif values.dtype.kind in "US":
        keys = values.astype(str)
        if any(not item.strip() for item in keys):
            raise ValueError("dates must not contain empty date identifiers")
    elif values.dtype.kind == "O":
        keys = []
        for value in values:
            if isinstance(value, datetime):
                key = value.date().isoformat()
            elif isinstance(value, date):
                key = value.isoformat()
            elif isinstance(value, str) and value.strip():
                key = value
            elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                key = str(value)
            else:
                raise ValueError("dates must contain nonmissing date identifiers")
            keys.append(key)
        keys = np.asarray(keys, dtype=str)
    else:
        raise ValueError("dates must contain strings, dates, or integer date identifiers")
    unique, inverse = np.unique(keys, return_inverse=True)
    return inverse, len(unique)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _ranking_metrics(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float | None, float | None]:
    """Group equal scores before accumulating ROC area and average precision."""
    count, positives = len(labels), int(labels.sum())
    negatives = count - positives
    if not count or not positives:
        return None, None
    order = np.argsort(-probabilities, kind="stable")
    ranked_labels, ranked_scores = labels[order], probabilities[order]
    ends = np.r_[np.flatnonzero(ranked_scores[:-1] != ranked_scores[1:]), count - 1]
    cumulative_positive = np.cumsum(ranked_labels)[ends]
    cumulative_count = ends + 1
    group_positive = np.diff(np.r_[0.0, cumulative_positive])
    average_precision = float(np.sum(
        (group_positive / positives) * (cumulative_positive / cumulative_count)
    ))
    if not negatives:
        return None, average_precision
    group_negative = np.diff(np.r_[0.0, cumulative_count - cumulative_positive])
    negatives_below = negatives - np.cumsum(group_negative)
    auc = float(np.sum(group_positive * (negatives_below + 0.5 * group_negative))
                / (positives * negatives))
    return auc, average_precision


def reliability_bins(labels, probabilities, *, n_bins: int = 10) -> list[dict]:
    """Equal-width calibration bins; the final bin includes probability one."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    labels, probabilities = _paired(labels, probabilities)
    bucket = np.minimum((probabilities * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(bucket, minlength=n_bins)
    probability_sum = np.bincount(bucket, weights=probabilities, minlength=n_bins)
    label_sum = np.bincount(bucket, weights=labels, minlength=n_bins)
    return [
        {"lower": index / n_bins, "upper": (index + 1) / n_bins,
         "count": int(counts[index]),
         "mean_probability": float(probability_sum[index] / counts[index]) if counts[index] else None,
         "observed_rate": float(label_sum[index] / counts[index]) if counts[index] else None}
        for index in range(n_bins)
    ]


def day_bootstrap_precision(labels, probabilities, dates, n_boot: int = 400,
                            seed: int = 42) -> dict | None:
    """Percentile 95% interval resampling target-date clusters, not IID samples.

    All supplied dates participate, including dates with no signal. Replicates
    containing no signal cannot estimate precision and are explicitly counted
    as invalid, rather than assigning them a made-up zero precision.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    labels, probabilities = _paired(labels, probabilities)
    groups, count_days = _date_groups(dates, len(labels))
    selected = probabilities > 0.5
    if not selected.any():
        return None
    selected_per_day = np.bincount(groups, weights=selected, minlength=count_days)
    wins_per_day = np.bincount(groups, weights=selected * labels, minlength=count_days)
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        sampled = generator.integers(0, count_days, size=count_days)
        denominator = selected_per_day[sampled].sum()
        if denominator:
            estimates.append(float(wins_per_day[sampled].sum() / denominator))
    lower, upper = np.quantile(estimates, [0.025, 0.975]) if estimates else (None, None)
    return {
        "point": float(labels[selected].mean()),
        "lower": float(lower) if lower is not None else None,
        "upper": float(upper) if upper is not None else None,
        "confidence": 0.95, "n_boot": n_boot, "count_days": count_days,
        "signal_days": int((selected_per_day > 0).sum()),
        "valid_replicates": len(estimates),
    }


def binary_metrics(labels, probabilities, *, gross_returns=None, cost_bps: float = 20,
                   dates=None) -> dict:
    """JSON-safe evaluation; undefined statistics use ``None``, never NaN/Inf."""
    labels, probabilities = _paired(labels, probabilities)
    if (isinstance(cost_bps, bool) or not isinstance(cost_bps, (int, float))
            or not math.isfinite(cost_bps) or not 0 <= cost_bps <= 10000):
        raise ValueError("cost_bps must be finite and in [0, 10000]")
    returns = None
    if gross_returns is not None:
        returns = _numeric_vector(gross_returns, "gross_returns")
        if returns.shape != labels.shape or (returns < -1).any():
            raise ValueError("gross_returns must match labels and be at least -1")
    if dates is not None:
        _date_groups(dates, len(labels))
    count = len(labels)
    selected = probabilities > 0.5
    signal_count = int(selected.sum())
    precision = float(labels[selected].mean()) if signal_count else None
    gross_mean = None
    if returns is not None and signal_count:
        selected_returns = returns[selected]
        return_scale = max(1.0, float(np.abs(selected_returns).max()))
        gross_mean = float((selected_returns / return_scale).mean() * return_scale)
    bins = reliability_bins(labels, probabilities)
    auc, ap = _ranking_metrics(labels, probabilities)
    clipped = np.clip(probabilities, _LOG_LOSS_EPSILON, 1 - _LOG_LOSS_EPSILON)
    result = {
        "count": count,
        "base_rate": float(labels.mean()) if count else None,
        "brier": float(np.square(probabilities - labels).mean()) if count else None,
        "log_loss": float(-(labels * np.log(clipped)
                            + (1 - labels) * np.log1p(-clipped)).mean()) if count else None,
        "roc_auc": auc, "pr_auc": ap,
        "accuracy": float((selected == labels).mean()) if count else None,
        "signal_count": signal_count,
        "coverage": signal_count / count if count else None,
        "precision": precision, "winrate": precision,
        "gross_mean_return": gross_mean,
        "net_mean_return": gross_mean - cost_bps / 10000 if gross_mean is not None else None,
        "cost_bps": float(cost_bps),
        "ece10": sum(item["count"] / count * abs(item["mean_probability"] - item["observed_rate"])
                     for item in bins if item["count"]) if count else None,
        "reliability_bins": bins,
        "precision_date_bootstrap": day_bootstrap_precision(labels, probabilities, dates)
                                    if dates is not None else None,
    }
    return result


def calibrated_probability(logits, calibration: dict) -> np.ndarray:
    """Apply a positive-slope Platt transform with an overflow-safe sigmoid."""
    logits = _numeric_vector(logits, "logits")
    if not isinstance(calibration, dict) or calibration.get("method") != "platt_monotone":
        raise ValueError("calibration method must be platt_monotone")
    slope, bias = calibration.get("slope"), calibration.get("bias")
    for name, value in (("slope", slope), ("bias", bias)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f"calibration {name} must be finite")
    if slope <= 0:
        raise ValueError("calibration slope must be positive")
    # Overflow to +/-Inf is a correctly saturated probability, not an error.
    with np.errstate(over="ignore"):
        transformed = slope * logits + bias
    return _sigmoid(transformed)


def fit_calibration(logits, labels) -> dict:
    """Fit unweighted BCE on a separate calibration set using damped Newton steps.

    The positive-slope constraint preserves ranking. A tiny L2 stabilizer
    (reported in metadata) prevents unbounded estimates on separable data.
    Empty or single-class calibration sets are rejected; there is no silent
    identity fallback described as a fitted probability model.
    """
    logits, labels = _numeric_vector(logits, "logits"), _labels(labels)
    if len(logits) != len(labels) or not len(labels):
        raise ValueError("calibration logits and labels must be nonempty and equal length")
    if np.unique(labels).size != 2:
        raise ValueError("calibration requires both positive and negative labels")
    # A bounded, centered design keeps the 2x2 Newton solve well-conditioned.
    scale = max(1.0, float(np.abs(logits).max()))
    standardized = logits / scale
    center = float(standardized.mean())
    standardized = standardized - center
    design = np.column_stack((standardized, np.ones(len(labels))))
    base_rate = float(labels.mean())
    parameters = np.array([1.0, math.log(base_rate / (1 - base_rate))])
    regularization = 1e-6
    min_slope = 1e-8

    def objective(values):
        scores = design @ values
        return float((np.logaddexp(0.0, scores) - labels * scores).mean()
                     + 0.5 * regularization * np.dot(values, values))

    loss = objective(parameters)
    iterations = 0
    for iterations in range(1, 101):
        probabilities = _sigmoid(design @ parameters)
        residual = probabilities - labels
        gradient = design.T @ residual / len(labels) + regularization * parameters
        weights = probabilities * (1 - probabilities)
        hessian = (design.T * weights) @ design / len(labels)
        hessian += regularization * np.eye(2)
        direction = np.linalg.solve(hessian, gradient)
        if parameters[0] <= min_slope * 1.001 and gradient[0] > 0:
            # At the positive-slope boundary, optimize the intercept along the
            # feasible edge instead of repeatedly proposing a negative slope.
            direction = np.array([0.0, gradient[1] / hessian[1, 1]])
        step = 1.0
        candidate = parameters
        candidate_loss = loss
        for _ in range(40):
            proposed = parameters - step * direction
            proposed[0] = max(min_slope, proposed[0])
            proposed_loss = objective(proposed)
            if proposed_loss < loss:
                candidate, candidate_loss = proposed, proposed_loss
                break
            step *= 0.5
        change = float(np.max(np.abs(candidate - parameters)))
        improvement = loss - candidate_loss
        parameters, loss = candidate, candidate_loss
        if change < 1e-9 or improvement < 1e-12:
            break
    slope = float(parameters[0] / scale)
    bias = float(parameters[1] - parameters[0] * center)
    if not math.isfinite(slope) or slope <= 0 or not math.isfinite(bias):
        raise ValueError("calibration is numerically unidentifiable for these logits")
    calibration = {
        "method": "platt_monotone", "slope": slope, "bias": bias,
        "fit_samples": len(labels), "iterations": iterations,
        "regularization": regularization, "weighted": False,
    }
    fitted = calibrated_probability(logits, calibration)
    clipped = np.clip(fitted, _LOG_LOSS_EPSILON, 1 - _LOG_LOSS_EPSILON)
    calibration["fit_log_loss"] = float(-(labels * np.log(clipped)
                                        + (1 - labels) * np.log1p(-clipped)).mean())
    return calibration
