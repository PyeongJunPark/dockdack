"""Research-only models for a future target-price / fixed-horizon experiment.

All feature functions accept *exactly* ``(N, L, 5)`` completed OHLCV bars,
ordered oldest to newest, plus one query/entry price per window. The caller
must stop each window at the decision time: no current unfinished OHLCV,
target-day high/low/close/volume, label, or future bar is an input. At training
the query is the next session's observed OPEN; at runtime it is the current
intraday quote, a distribution shift that needs separate validation. Shape
validation rejects an accidentally appended future bar.

The common network input is :func:`sequence_features`, with shape
``(N, L, 6)``. The sixth channel is zero except at the last completed bar,
where it is log(query price / last completed close). All networks output
``(N, 2)``: column 0 is a target-hit
**logit** and column 1 is an unconstrained expected **net return** in
fractional units (``0.03`` means +3% after costs). Neither column is a
trading signal or calibrated probability by itself.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


LOOKBACKS = (10, 15, 20, 30)
MODEL_FAMILIES = ("linear", "mlp", "cnn", "gru")
RAW_CHANNELS = 5
FEATURE_CHANNELS = 6


def _check_lookback(lookback: int) -> int:
    if isinstance(lookback, bool) or not isinstance(lookback, (int, np.integer)):
        raise ValueError("lookback must be one of 10, 15, 20, 30")
    if int(lookback) not in LOOKBACKS:
        raise ValueError("lookback must be one of 10, 15, 20, 30")
    return int(lookback)


def validate_completed_bars(bars: object, lookback: int) -> np.ndarray:
    """Validate N chronological windows and return finite float64 OHLCV.

    Only a window of the exact declared length is accepted. This function
    cannot infer whether its final row was actually complete; the dataset
    builder must enforce the decision timestamp before calling it.
    """
    length = _check_lookback(lookback)
    try:
        value = np.asarray(bars, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric completed OHLCV windows required") from exc
    if value.ndim != 3 or value.shape[1:] != (length, RAW_CHANNELS) or not len(value):
        raise ValueError(f"expected nonempty (N, {length}, 5) completed OHLCV")
    if not np.isfinite(value).all():
        raise ValueError("completed OHLCV must be finite")
    prices = value[..., :4]
    if np.any(prices <= 0) or np.any(value[..., 4] < 0):
        raise ValueError("OHLC must be positive and volume nonnegative")
    opening, high, low, close = (value[..., index] for index in range(4))
    if (np.any(high < low) or np.any(high < opening) or np.any(high < close)
            or np.any(low > opening) or np.any(low > close)):
        raise ValueError("completed OHLC has inconsistent high or low")
    return np.ascontiguousarray(value)


def sequence_features(bars: object, lookback: int,
                      query_prices: object) -> np.ndarray:
    """Return completed-bar features plus the observed query-price gap.

    The channels are log(open / previous close), log(high / open),
    log(low / open), log(close / open), and log1p(volume) minus its
    **preceding-window** mean. The first bar's missing previous close and
    volume baseline are defined to yield zero. Thus no whole-dataset
    statistics, absolute prices, future bars, or target values enter.
    Channel six holds log(query price / last completed close) only at the
    final sequence position. This scalar is known at the decision point and
    is the only information from the new session; it is **not** its OHLCV.
    Clipping limits pathological but formally finite price jumps; it does
    not repair invalid OHLCV. Training-only fitted scalers, if any, belong
    outside this deterministic transform.
    """
    value = validate_completed_bars(bars, lookback)
    try:
        query = np.asarray(query_prices, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric query prices required") from exc
    if query.ndim == 0:
        query = np.full(len(value), float(query), dtype=np.float64)
    if query.shape != (len(value),) or not np.isfinite(query).all() or np.any(query <= 0):
        raise ValueError("one finite positive query price per completed window required")
    opening, high, low, close, volume = (value[..., index] for index in range(5))
    previous_close = np.concatenate((opening[:, :1], close[:, :-1]), axis=1)
    log_volume = np.log1p(volume)
    prior_volume_total = np.concatenate((
        np.zeros((len(value), 1), dtype=np.float64),
        np.cumsum(log_volume[:, :-1], axis=1),
    ), axis=1)
    prior_count = np.maximum(np.arange(value.shape[1], dtype=np.float64), 1.0)
    relative_volume = log_volume - prior_volume_total / prior_count
    relative_volume[:, 0] = 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        history_features = np.stack((
            np.log(opening) - np.log(previous_close),
            np.log(high) - np.log(opening),
            np.log(low) - np.log(opening),
            np.log(close) - np.log(opening),
            relative_volume,
        ), axis=-1)
        query_gap = np.log(query) - np.log(close[:, -1])
    if not np.isfinite(history_features).all() or not np.isfinite(query_gap).all():
        raise ValueError("completed OHLCV produced nonfinite features")
    features = np.zeros((len(value), value.shape[1], FEATURE_CHANNELS), dtype=np.float64)
    features[..., :RAW_CHANNELS] = history_features
    features[:, -1, RAW_CHANNELS] = query_gap
    result = np.clip(features, -4.0, 4.0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("completed OHLCV features exceed float32")
    return np.ascontiguousarray(result)


def tabular_features(bars: object, lookback: int,
                     query_prices: object) -> np.ndarray:
    """Flatten the same causal sequence into a position-preserving table."""
    features = sequence_features(bars, lookback, query_prices)
    return np.ascontiguousarray(features.reshape(features.shape[0], -1))


class _CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, values: Tensor) -> Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class TargetHorizonModel(nn.Module):
    """Small, comparable network families with the same feature contract."""

    def __init__(self, family: str, lookback: int):
        super().__init__()
        self.lookback = _check_lookback(lookback)
        if family not in MODEL_FAMILIES:
            raise ValueError(f"family must be one of {MODEL_FAMILIES}")
        self.family = family
        if family == "linear":
            self.network = nn.Linear(self.lookback * FEATURE_CHANNELS, 2)
        elif family == "mlp":
            self.network = nn.Sequential(
                nn.Linear(self.lookback * FEATURE_CHANNELS, 64), nn.GELU(),
                nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2),
            )
        elif family == "cnn":
            self.network = nn.Sequential(
                _CausalConv1d(RAW_CHANNELS, 16), nn.GELU(),
                _CausalConv1d(16, 16), nn.GELU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
            self.head = nn.Linear(17, 2)
        else:
            self.network = nn.GRU(FEATURE_CHANNELS, 24, batch_first=True)
            self.head = nn.Linear(24, 2)

    def forward(self, features: Tensor) -> Tensor:
        if (not isinstance(features, Tensor) or features.ndim != 3
                or features.shape[1:] != (self.lookback, FEATURE_CHANNELS)
                or features.shape[0] == 0 or not features.is_floating_point()):
            raise ValueError(f"expected nonempty float tensor (N, {self.lookback}, 6)")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("model features must be finite")
        if self.family in ("linear", "mlp"):
            return self.network(features.flatten(start_dim=1))
        if self.family == "cnn":
            history = self.network(features[..., :RAW_CHANNELS].transpose(1, 2))
            return self.head(torch.cat((history, features[:, -1, RAW_CHANNELS:]), dim=1))
        _, hidden = self.network(features)
        return self.head(hidden[-1])


def build_model(family: str, lookback: int) -> TargetHorizonModel:
    """Build a fresh two-task model; no weights or broker are loaded."""
    return TargetHorizonModel(family, lookback)
