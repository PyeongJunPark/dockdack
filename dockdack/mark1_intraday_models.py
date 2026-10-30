"""Causal daily-history feature experts for separately trained MK1.13–MK1.17.

These are price-aware *daily proxy* models, not minute-bar models. A queried
intraday price is only an input at inference; the stored training event uses
the next session's observed OPEN. The target-day high/low/close never enter
the features.
"""
from __future__ import annotations

import numpy as np
from torch import nn


VARIANTS = {
    "mark1.13": ("entry_gap", "momentum_1", "momentum_5", "momentum_10",
                 "momentum_20", "volatility_5", "volatility_20", "volume_slope"),
    "mark1.14": ("entry_gap", "momentum_1", "momentum_3", "momentum_5",
                 "rsi_10", "drawdown_10", "drawdown_30", "close_location_5"),
    "mark1.15": ("entry_gap", "high_distance_10", "high_distance_30",
                 "low_distance_10", "range_5", "range_20", "volume_surprise",
                 "close_location_5"),
    "mark1.16": ("entry_gap", "volatility_5", "volatility_10", "volatility_20",
                 "range_5", "range_20", "gap_volatility", "momentum_5"),
    "mark1.17": ("entry_gap", "volume_surprise", "volume_slope",
                 "pressure_5", "pressure_20", "close_location_5",
                 "momentum_10", "volatility_10"),
}

ARCHITECTURES = {
    "mark1.13": (24, 8),
    "mark1.14": (16, 8),
    "mark1.15": (24, 12),
    "mark1.16": (16, 16),
    "mark1.17": (32, 8),
}


def feature_matrix(history, entries, variant: str) -> np.ndarray:
    """Return eight finite features per query without reading target-day bars."""
    if variant not in VARIANTS:
        raise ValueError("Unknown MK1 intraday-proxy variant")
    try:
        bars = np.asarray(history, dtype=np.float64)
        entry = np.asarray(entries, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Numeric completed 30-bar history and entry prices required") from exc
    if bars.ndim == 2:
        bars = bars[None, :, :]
    if entry.ndim == 0:
        entry = np.full(len(bars), float(entry), dtype=np.float64)
    if (bars.ndim != 3 or bars.shape[1:] != (30, 5) or not len(bars)
            or entry.shape != (len(bars),) or not np.isfinite(bars).all()
            or not np.isfinite(entry).all() or np.any(entry <= 0)
            or np.any(bars[..., :4] <= 0) or np.any(bars[..., 4] < 0)):
        raise ValueError("Finite positive 30x5 OHLCV and aligned prices required")
    prices = bars[..., :4]
    if (np.any(bars[..., 1] < np.max(prices, axis=2))
            or np.any(bars[..., 2] > np.min(prices, axis=2))):
        raise ValueError("Completed OHLCV contains inconsistent high or low")
    op, hi, lo, cl, volume = (bars[..., index] for index in range(5))
    log_close = np.log(cl)
    ret = np.diff(log_close, axis=1)
    day_range = np.log(hi) - np.log(lo)
    location = (cl - lo) / np.maximum(hi - lo, np.finfo(float).tiny) - .5
    log_volume = np.log1p(volume)
    volume_center = log_volume - log_volume.mean(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = {
            "entry_gap": np.log(entry) - log_close[:, -1],
            "momentum_1": ret[:, -1],
            "momentum_3": log_close[:, -1] - log_close[:, -4],
            "momentum_5": log_close[:, -1] - log_close[:, -6],
            "momentum_10": log_close[:, -1] - log_close[:, -11],
            "momentum_20": log_close[:, -1] - log_close[:, -21],
            "volatility_5": np.sqrt(np.mean(ret[:, -5:] ** 2, axis=1)),
            "volatility_10": np.sqrt(np.mean(ret[:, -10:] ** 2, axis=1)),
            "volatility_20": np.sqrt(np.mean(ret[:, -20:] ** 2, axis=1)),
            "rsi_10": (np.sum(np.maximum(ret[:, -10:], 0), axis=1)
                       / np.maximum(np.sum(np.abs(ret[:, -10:]), axis=1), 1e-12) - .5),
            "drawdown_10": log_close[:, -1] - np.max(log_close[:, -10:], axis=1),
            "drawdown_30": log_close[:, -1] - np.max(log_close, axis=1),
            "high_distance_10": np.log(entry) - np.log(np.max(hi[:, -10:], axis=1)),
            "high_distance_30": np.log(entry) - np.log(np.max(hi, axis=1)),
            "low_distance_10": np.log(entry) - np.log(np.min(lo[:, -10:], axis=1)),
            "range_5": np.mean(day_range[:, -5:], axis=1),
            "range_20": np.mean(day_range[:, -20:], axis=1),
            "volume_surprise": volume_center[:, -1],
            "volume_slope": np.mean(log_volume[:, -5:], axis=1)
                            - np.mean(log_volume[:, -20:-5], axis=1),
            "gap_volatility": np.sqrt(np.mean(
                (np.log(op[:, -10:]) - log_close[:, -11:-1]) ** 2, axis=1)),
            "pressure_5": np.mean(location[:, -5:] * volume_center[:, -5:], axis=1),
            "pressure_20": np.mean(location[:, -20:] * volume_center[:, -20:], axis=1),
            "close_location_5": np.mean(location[:, -5:], axis=1),
        }
    result = np.column_stack([values[name] for name in VARIANTS[variant]])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite MK1 intraday-proxy features")
    result = np.clip(result, -4., 4.).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("MK1 intraday-proxy features exceed float32")
    return result


def build_model(variant: str) -> nn.Module:
    if variant not in VARIANTS:
        raise ValueError("Unknown MK1 intraday-proxy variant")
    first, second = ARCHITECTURES[variant]
    return nn.Sequential(nn.Linear(8, first), nn.GELU(),
                         nn.Linear(first, second), nn.GELU(), nn.Linear(second, 1))
