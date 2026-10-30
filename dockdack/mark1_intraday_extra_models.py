"""Five additional causal daily-proxy feature hypotheses, MK1.18–MK1.22.

These use only completed daily bars plus a price query. They cannot establish
the unseen post-query intraday path or minute-bar first-touch outcome.
"""
from __future__ import annotations

import numpy as np
from torch import nn


VARIANTS = {
    "mark1.18": ("entry_gap", "return_1", "return_2", "rsi_5", "reversal_3",
                 "close_location_1", "range_1", "volume_surprise_5"),
    "mark1.19": ("entry_gap", "gap_1", "body_1", "upper_wick_1", "lower_wick_1",
                 "body_5", "close_location_5", "range_5"),
    "mark1.20": ("entry_gap", "momentum_3", "momentum_5", "momentum_10",
                 "acceleration_5", "acceleration_10", "volume_slope_10", "volatility_10"),
    "mark1.21": ("entry_gap", "volatility_5", "volatility_10", "volume_surprise_5",
                 "vol_volume_5", "pressure_5", "range_5", "momentum_5"),
    "mark1.22": ("entry_gap", "range_5", "range_20", "range_ratio", "volatility_5",
                 "volatility_20", "channel_width_20", "high_distance_20"),
}
ARCHITECTURES = {
    "mark1.18": "silu_12_8",
    "mark1.19": "tanh_20_10",
    "mark1.20": "gelu_32_16",
    "mark1.21": "gated_24_8",
    "mark1.22": "relu_32_16_8",
}
EFFECTIVE_LOOKBACK = {"mark1.18": 6, "mark1.19": 5, "mark1.20": 21,
                      "mark1.21": 11, "mark1.22": 21}


class GatedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(8, 24), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(8, 24), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(24, 8), nn.SiLU(), nn.Linear(8, 1))

    def forward(self, x):
        return self.out(self.value(x) * self.gate(x))


def build_model(variant: str) -> nn.Module:
    if variant == "mark1.18":
        return nn.Sequential(nn.Linear(8, 12), nn.SiLU(), nn.Linear(12, 8),
                             nn.SiLU(), nn.Linear(8, 1))
    if variant == "mark1.19":
        return nn.Sequential(nn.Linear(8, 20), nn.Tanh(), nn.Linear(20, 10),
                             nn.Tanh(), nn.Linear(10, 1))
    if variant == "mark1.20":
        return nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 16),
                             nn.GELU(), nn.Linear(16, 1))
    if variant == "mark1.21":
        return GatedExpert()
    if variant == "mark1.22":
        return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16),
                             nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
    raise ValueError("Unknown MK1 extra variant")


def feature_matrix(history, entries, variant: str) -> np.ndarray:
    if variant not in VARIANTS:
        raise ValueError("Unknown MK1 extra variant")
    try:
        bars = np.asarray(history, dtype=np.float64)
        entry = np.asarray(entries, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Numeric completed daily history and query prices required") from exc
    if bars.ndim == 2:
        bars = bars[None]
    if entry.ndim == 0:
        entry = np.full(len(bars), float(entry), dtype=np.float64)
    if (bars.ndim != 3 or bars.shape[1:] != (30, 5) or not len(bars)
            or entry.shape != (len(bars),) or not np.isfinite(bars).all()
            or not np.isfinite(entry).all() or np.any(entry <= 0)
            or np.any(bars[..., :4] <= 0) or np.any(bars[..., 4] < 0)):
        raise ValueError("Finite positive 30x5 OHLCV and aligned prices required")
    price = bars[..., :4]
    if (np.any(bars[..., 1] < np.max(price, axis=2))
            or np.any(bars[..., 2] > np.min(price, axis=2))):
        raise ValueError("Completed OHLCV high/low inconsistent")
    op, hi, lo, cl, volume = (bars[..., column] for column in range(5))
    lc, log_volume = np.log(cl), np.log1p(volume)
    returns = np.diff(lc, axis=1)
    ranges = np.log(hi) - np.log(lo)
    location = (cl - lo) / np.maximum(hi - lo, np.finfo(float).tiny) - .5
    body = np.log(cl) - np.log(op)
    centered_volume_5 = log_volume[:, -5:] - log_volume[:, -5:].mean(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = {
            "entry_gap": np.log(entry) - lc[:, -1],
            "return_1": returns[:, -1],
            "return_2": lc[:, -1] - lc[:, -3],
            "rsi_5": (np.sum(np.maximum(returns[:, -5:], 0), axis=1) /
                      np.maximum(np.sum(np.abs(returns[:, -5:]), axis=1), 1e-12) - .5),
            "reversal_3": returns[:, -1] - np.mean(returns[:, -4:-1], axis=1),
            "close_location_1": location[:, -1],
            "close_location_5": np.mean(location[:, -5:], axis=1),
            "range_1": ranges[:, -1],
            "range_5": np.mean(ranges[:, -5:], axis=1),
            "range_20": np.mean(ranges[:, -20:], axis=1),
            "range_ratio": np.mean(ranges[:, -5:], axis=1) /
                           np.maximum(np.mean(ranges[:, -20:], axis=1), 1e-5) - 1.,
            "volume_surprise_5": log_volume[:, -1] -
                                 np.mean(log_volume[:, -5:], axis=1),
            "gap_1": np.log(op[:, -1]) - lc[:, -2],
            "body_1": body[:, -1],
            "upper_wick_1": np.log(hi[:, -1]) -
                            np.log(np.maximum(op[:, -1], cl[:, -1])),
            "lower_wick_1": np.log(np.minimum(op[:, -1], cl[:, -1])) -
                            np.log(lo[:, -1]),
            "body_5": np.mean(body[:, -5:], axis=1),
            "momentum_3": lc[:, -1] - lc[:, -4],
            "momentum_5": lc[:, -1] - lc[:, -6],
            "momentum_10": lc[:, -1] - lc[:, -11],
            "acceleration_5": np.mean(returns[:, -5:], axis=1) -
                              np.mean(returns[:, -10:-5], axis=1),
            "acceleration_10": np.mean(returns[:, -10:], axis=1) -
                               np.mean(returns[:, -20:-10], axis=1),
            "volume_slope_10": np.mean(log_volume[:, -5:], axis=1) -
                               np.mean(log_volume[:, -10:-5], axis=1),
            "volatility_5": np.sqrt(np.mean(returns[:, -5:] ** 2, axis=1)),
            "volatility_10": np.sqrt(np.mean(returns[:, -10:] ** 2, axis=1)),
            "volatility_20": np.sqrt(np.mean(returns[:, -20:] ** 2, axis=1)),
            "vol_volume_5": np.sqrt(np.mean(returns[:, -5:] ** 2, axis=1)) *
                            (log_volume[:, -1] - np.mean(log_volume[:, -5:], axis=1)),
            "pressure_5": np.mean(location[:, -5:] * centered_volume_5, axis=1),
            "channel_width_20": np.log(np.max(hi[:, -20:], axis=1)) -
                                np.log(np.min(lo[:, -20:], axis=1)),
            "high_distance_20": np.log(entry) -
                                np.log(np.max(hi[:, -20:], axis=1)),
        }
    result = np.column_stack([values[name] for name in VARIANTS[variant]])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite MK1 extra features")
    result = np.clip(result, -4., 4.).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("MK1 extra features exceed float32")
    return result
