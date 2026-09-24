"""Causal mark1 research inputs and labels for +0.5% take / -0.4% stop.

This is a new target contract, not a reinterpretation of frozen mark1 models.
The 30 completed historical OHLCV bars and candidate entry are the only feature
inputs. Daily HIGH/LOW cannot recover intraday ordering: touching both barriers
is a failure and the return proxy assumes the stop was reached first.
"""
from __future__ import annotations

import numpy as np

from .mark1_selective_features import (
    FEATURE_NAMES as _BASE_FEATURE_NAMES,
    LOOKBACK,
    WINDOWS,
    features_from_history as _base_features,
)


TARGET = "daily_high_ge_entry_0_5pct_and_low_gt_entry_minus_0_4pct_conservative"
TAKE_PROFIT_PCT = 0.5
STOP_LOSS_PCT = 0.4
TAKE_MULTIPLIER = 1.005
STOP_MULTIPLIER = 0.996
BARRIER_TOLERANCE = 1e-12
CLASS_NAMES = ("take_only", "stop_only", "both", "neither")
_BARRIER_KINDS = ("take_only", "stop_only", "both_touch", "neither")
_BARRIER_COLUMNS = {
    (window, kind): _BASE_FEATURE_NAMES.index(f"w{window}_past_{kind}_rate")
    for window in WINDOWS for kind in _BARRIER_KINDS
}
_RENAMED_COLUMNS = {
    index: f"w{window}_past_{kind}_rate_tp005_sl004"
    for (window, kind), index in _BARRIER_COLUMNS.items()
}
# There are exactly sixteen target-dependent fields in the reused feature
# implementation. Fail rather than silently retaining a newly added old-target
# historical barrier field if that implementation's schema changes.
if {i for i, name in enumerate(_BASE_FEATURE_NAMES) if "_past_" in name} != set(_RENAMED_COLUMNS):
    raise RuntimeError("Unexpected target-dependent base feature schema")
FEATURE_NAMES = tuple(_RENAMED_COLUMNS.get(i, name)
                      for i, name in enumerate(_BASE_FEATURE_NAMES))


def _barrier_hits(high, low, entry):
    """Shared vectorized comparisons for historical features and target labels."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        take = high / entry >= TAKE_MULTIPLIER * (1 - BARRIER_TOLERANCE)
        stop = low / entry <= STOP_MULTIPLIER * (1 + BARRIER_TOLERANCE)
    return take, stop


def features_from_history(history, entry_prices, *, validate=True) -> np.ndarray:
    """Return float32 [N,184] features, with all past barriers at +.5%/-.4%.

    The validated target-independent features are reused byte-for-byte from the
    existing implementation. All sixteen historical barrier rates are then
    replaced, and their feature names explicitly identify the new target.
    Historical barriers are relative to each completed bar's OPEN, not the
    candidate entry. The candidate changes only the existing query features.
    Inputs and their writeability flags are never modified.
    """
    result = _base_features(history, entry_prices, validate=validate)
    if not len(result):
        return result
    prices = np.asarray(history, dtype=np.float64)[..., :4]
    take, stop = _barrier_hits(prices[..., 1], prices[..., 2], prices[..., 0])
    events = {
        "take_only": take & ~stop,
        "stop_only": stop & ~take,
        "both_touch": take & stop,
        "neither": ~take & ~stop,
    }
    for (window, kind), column in _BARRIER_COLUMNS.items():
        result[:, column] = events[kind][:, -window:].mean(axis=1)
    return result


def barrier_outcomes(high, low, close, entry) -> dict[str, np.ndarray]:
    """Whole-session +.5%/-.4% labels and a stop-first daily return proxy.

    Take is inclusive HIGH >= entry*1.005; stop is inclusive LOW <= entry*.996.
    Relative tolerance 1e-12 is used only for numerical exact-touch rounding,
    consistently with historical feature comparisons. Both-hit days return
    -.004, take-only days +.005, and neither-hit days CLOSE/entry-1. These are
    hypothetical gross returns: no fill, fee, spread, or gap is modeled here.
    Broadcast-compatible positive finite inputs are accepted, including scalars.
    """
    high, low, close, entry = np.broadcast_arrays(
        *[np.asarray(value, dtype=np.float64) for value in (high, low, close, entry)])
    if (not all(np.isfinite(value).all() for value in (high, low, close, entry))
            or any((value <= 0).any() for value in (high, low, close, entry))
            or np.any(high < low) or np.any(close > high) or np.any(close < low)):
        raise ValueError("Finite positive prices and valid target high/low/close are required")
    take, stop = _barrier_hits(high, low, entry)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gross = np.where(stop, -.004, np.where(take, .005, close / entry - 1))
    if not np.isfinite(gross).all():
        raise ValueError("Non-finite hypothetical return")
    return {"success": take & ~stop, "both_touch": take & stop,
            "take_hit": take, "stop_hit": stop, "gross_return": gross}


def class_targets(ohlc, factors=(1.,)) -> np.ndarray:
    """Return int64 [N,F]: take_only=0, stop_only=1, both=2, neither=3.

    Candidate entries are target OPEN times predeclared positive factors. This
    supports counterfactual price augmentation; target-day HLC are labels only,
    never model feature inputs. Both=2 remains a success failure/stop-first exit.
    """
    raw_prices, raw_factors = np.asarray(ohlc), np.asarray(factors)
    if raw_prices.ndim != 2 or raw_prices.shape[1] != 4 or raw_prices.dtype.kind not in "iuf":
        raise ValueError("ohlc must be a numeric [N,4] OPEN,HIGH,LOW,CLOSE array")
    if raw_factors.ndim != 1 or not len(raw_factors) or raw_factors.dtype.kind not in "iuf":
        raise ValueError("factors must be a nonempty one-dimensional numeric sequence")
    prices = raw_prices.astype(np.float64, copy=False)
    multipliers = raw_factors.astype(np.float64, copy=False)
    if not np.isfinite(multipliers).all() or np.any(multipliers <= 0):
        raise ValueError("factors must be finite and positive")
    if (not np.isfinite(prices).all() or np.any(prices <= 0)
            or np.any(prices[:, 0] > prices[:, 1]) or np.any(prices[:, 0] < prices[:, 2])):
        raise ValueError("OHLC must be finite positive prices with OPEN inside HIGH/LOW")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        entries = prices[:, :1] * multipliers[None, :]
    outcomes = barrier_outcomes(prices[:, 1:2], prices[:, 2:3], prices[:, 3:4], entries)
    take, stop = outcomes["take_hit"], outcomes["stop_hit"]
    result = np.full(entries.shape, 3, dtype=np.int64)
    result[take & ~stop] = 0
    result[stop & ~take] = 1
    result[take & stop] = 2
    return result


def event_data(dataset, indices):
    """Build new-target OPEN-entry events; never reuse a cached old label."""
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= len(dataset.target_ohlc))):
        raise ValueError("Valid one-dimensional sample indices required")
    ohlc = dataset.target_ohlc[indices]
    classes = class_targets(ohlc)[:, 0]
    outcomes = barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], ohlc[:, 0])
    return {"labels": outcomes["success"], "gross": outcomes["gross_return"],
            "classes": classes, "dates": dataset.target_dates[indices],
            "symbols": dataset.symbol_ids[indices]}
