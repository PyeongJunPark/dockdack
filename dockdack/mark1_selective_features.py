"""Causal, dependency-light tabular features for selective mark_1 research.

Only 30 completed OHLCV bars and a candidate entry price are accepted. The
candidate is not a completed candle. Historical barrier rates describe already
finished sessions, never the target session. The price-times-volume feature is
explicitly a proxy, not observed turnover or free float. There are no learned
cross-sample statistics, symbol identifiers, dates, or target-day HLCV inputs.
"""
from __future__ import annotations

import numpy as np


LOOKBACK = 30
WINDOWS = (5, 10, 20, 30)
RETURN_LAGS = (1, 2, 3, 5, 10, 20, 29)
CANDLE_LAGS = (1, 2, 3, 5)
FEATURE_CLIP = 20.0
BARRIER_TOLERANCE = 1e-12
RETURN_SCALE_FLOOR = 0.003
RANGE_SCALE_FLOOR = 0.003

_WINDOW_FEATURES = (
    "return_mean", "return_std", "return_min", "return_max",
    "downside_return_rms", "upside_return_rms", "positive_return_fraction",
    "return_sum", "log_range_mean", "log_range_std", "log_true_range_mean",
    "body_fraction_mean", "close_range_position_mean", "upper_wick_fraction_mean",
    "lower_wick_fraction_mean", "up_excursion_q50", "up_excursion_q90",
    "down_excursion_q10", "down_excursion_q50", "past_take_only_rate",
    "past_stop_only_rate", "past_both_touch_rate", "past_neither_rate",
    "log1p_volume_mean_div20", "log1p_volume_std", "last_log_volume_minus_mean",
    "zero_volume_fraction", "log_price_volume_proxy_mean_div20",
    "last_close_over_mean_log_close_vol_scaled", "last_close_over_max_high_vol_scaled",
    "last_close_over_min_low_vol_scaled", "trend_efficiency",
)
_CANDLE_FEATURES = (
    "log_body", "log_range", "body_fraction", "close_range_position",
    "upper_wick_fraction", "lower_wick_fraction",
)
_QUERY_FEATURES = (
    "query_log_price_div10", "query_log_gap", "query_gap_vol_scaled",
    "query_gap_log_true_range_scaled", "query_log_over_last_open",
    "query_log_over_last_high", "query_log_over_last_low", "historical_return_std",
    "historical_log_true_range_mean", "historical_vol_compression_5_20",
    "historical_vol_compression_10_29", "historical_last_log1p_volume_div20",
    "historical_log_volume_mean5_minus_mean20",
)
FEATURE_NAMES = (
    tuple(f"w{window}_{name}" for window in WINDOWS for name in _WINDOW_FEATURES)
    + tuple(f"close_log_return_lag{lag}" for lag in RETURN_LAGS)
    + tuple(f"candle_lag{lag}_{name}" for lag in CANDLE_LAGS for name in _CANDLE_FEATURES)
    + _QUERY_FEATURES
    + tuple(f"query_w{window}_{name}_vol_scaled" for window in WINDOWS
            for name in ("over_mean_log_close", "over_max_high", "over_min_low"))
)


def _validate(history, entry_prices, validate):
    if not isinstance(validate, (bool, np.bool_)):
        raise ValueError("validate must be boolean")
    history = np.asarray(history)
    entries = np.asarray(entry_prices)
    if (history.ndim != 3 or history.shape[1:] != (LOOKBACK, 5)
            or history.dtype.kind not in "fiu"):
        raise ValueError("history must be a numeric [N,30,5] OHLCV array")
    if entries.ndim != 1 or len(entries) != len(history) or entries.dtype.kind not in "fiu":
        raise ValueError("entry_prices must be a numeric [N] array")
    # Copying is unnecessary: every subsequent operation is out of place.
    history = history.astype(np.float64, copy=False)
    entries = entries.astype(np.float64, copy=False)
    if not np.isfinite(history).all() or not np.isfinite(entries).all():
        raise ValueError("history and entry_prices must be finite")
    prices, volumes = history[..., :4], history[..., 4]
    if np.any(prices <= 0) or np.any(entries <= 0) or np.any(volumes < 0):
        raise ValueError("prices must be positive and volume nonnegative")
    if validate and (np.any(prices[..., 1] < prices.max(axis=2))
                     or np.any(prices[..., 2] > prices.min(axis=2))):
        raise ValueError("historical OPEN/CLOSE must lie within LOW/HIGH")
    return history, entries


def _fraction(numerator, denominator, *, neutral=0.0):
    result = np.full_like(numerator, neutral, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def features_from_history(history, entry_prices, *, validate=True) -> np.ndarray:
    """Return immutable-input-safe float32 ``[N,184]`` causal features.

    All computations use float64 intermediates, then fixed [-20,20] clipping.
    A w-session return window uses the last min(w,29) observed close changes;
    candle/range/barrier/volume windows use exactly w completed bars. The
    log-true-range is max(log(H/L), abs(log(H/previous C)),
    abs(log(L/previous C))) and uses the 29 available previous closes. It is
    a dimensionless log-range measure, not a currency-denominated ATR.

    Flat candles have zero body/wicks and a neutral close position of 0.5.
    Zero volume is retained via log1p and an explicit fraction. Historical
    return and log-range denominators have fixed 0.003 floors. Population
    standard deviations are used. A historical exact +1% touch succeeds only
    if -0.9% was not touched; both touches are failure, tolerance 1e-12.

    Shape, finite, positive-price and nonnegative-volume checks always apply;
    validate=False skips only the OHLC ordering check for prevalidated banks.
    Empty batches are supported. No scaling statistics are fitted here.
    """
    history, entries = _validate(history, entry_prices, validate)
    if not len(history):
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    prices, volume = history[..., :4], history[..., 4]
    open_, high, low, close = (prices[..., index] for index in range(4))
    log_prices = np.log(prices)
    log_open, log_high, log_low, log_close = (log_prices[..., index] for index in range(4))
    log_volume = np.log1p(volume)
    returns = np.diff(log_close, axis=1)
    return_std = returns.std(axis=1)
    scale = np.maximum(return_std, RETURN_SCALE_FLOOR)
    log_range = log_high - log_low
    true_range = np.maximum.reduce((log_range[:, 1:],
                                   np.abs(log_high[:, 1:] - log_close[:, :-1]),
                                   np.abs(log_low[:, 1:] - log_close[:, :-1])))
    range_scale = np.maximum(true_range.mean(axis=1), RANGE_SCALE_FLOOR)
    spread = high - low
    body = _fraction(close - open_, spread)
    position = _fraction(close - low, spread, neutral=0.5)
    upper = _fraction(high - np.maximum(open_, close), spread)
    lower = _fraction(np.minimum(open_, close) - low, spread)
    up_excursion, down_excursion = log_high - log_open, log_low - log_open
    take = up_excursion >= np.log(1.01 * (1 - BARRIER_TOLERANCE))
    stop = down_excursion <= np.log(0.991 * (1 + BARRIER_TOLERANCE))
    columns = []
    for window in WINDOWS:
        ret = returns[:, -window:]
        ranges = log_range[:, -window:]
        volumes = log_volume[:, -window:]
        take_w, stop_w = take[:, -window:], stop[:, -window:]
        ret_sum = ret.sum(axis=1)
        absolute_sum = np.abs(ret).sum(axis=1)
        columns.extend((
            ret.mean(axis=1), ret.std(axis=1), ret.min(axis=1), ret.max(axis=1),
            np.sqrt(np.square(np.minimum(ret, 0)).mean(axis=1)),
            np.sqrt(np.square(np.maximum(ret, 0)).mean(axis=1)),
            (ret > 0).mean(axis=1), ret_sum, ranges.mean(axis=1), ranges.std(axis=1),
            true_range[:, -window:].mean(axis=1), body[:, -window:].mean(axis=1),
            position[:, -window:].mean(axis=1), upper[:, -window:].mean(axis=1),
            lower[:, -window:].mean(axis=1),
            np.quantile(up_excursion[:, -window:], 0.5, axis=1),
            np.quantile(up_excursion[:, -window:], 0.9, axis=1),
            np.quantile(down_excursion[:, -window:], 0.1, axis=1),
            np.quantile(down_excursion[:, -window:], 0.5, axis=1),
            (take_w & ~stop_w).mean(axis=1), (stop_w & ~take_w).mean(axis=1),
            (take_w & stop_w).mean(axis=1), (~take_w & ~stop_w).mean(axis=1),
            volumes.mean(axis=1) / 20, volumes.std(axis=1),
            log_volume[:, -1] - volumes.mean(axis=1),
            (volume[:, -window:] == 0).mean(axis=1),
            (log_close[:, -window:] + volumes).mean(axis=1) / 20,
            (log_close[:, -1] - log_close[:, -window:].mean(axis=1)) / scale,
            (log_close[:, -1] - log_high[:, -window:].max(axis=1)) / scale,
            (log_close[:, -1] - log_low[:, -window:].min(axis=1)) / scale,
            _fraction(ret_sum, absolute_sum),
        ))
    columns.extend(log_close[:, -1] - log_close[:, -1 - lag] for lag in RETURN_LAGS)
    for lag in CANDLE_LAGS:
        columns.extend((log_close[:, -lag] - log_open[:, -lag], log_range[:, -lag],
                        body[:, -lag], position[:, -lag], upper[:, -lag], lower[:, -lag]))
    query_log = np.log(entries)
    gap = query_log - log_close[:, -1]
    columns.extend((
        query_log / 10, gap, gap / scale, gap / range_scale,
        query_log - log_open[:, -1], query_log - log_high[:, -1], query_log - log_low[:, -1],
        return_std, true_range.mean(axis=1),
        returns[:, -5:].std(axis=1) / np.maximum(returns[:, -20:].std(axis=1), RETURN_SCALE_FLOOR),
        returns[:, -10:].std(axis=1) / scale, log_volume[:, -1] / 20,
        log_volume[:, -5:].mean(axis=1) - log_volume[:, -20:].mean(axis=1),
    ))
    for window in WINDOWS:
        columns.extend(((query_log - log_close[:, -window:].mean(axis=1)) / scale,
                        (query_log - log_high[:, -window:].max(axis=1)) / scale,
                        (query_log - log_low[:, -window:].min(axis=1)) / scale))
    result = np.column_stack(columns)
    if result.shape[1] != len(FEATURE_NAMES) or not np.isfinite(result).all():
        raise ValueError("Feature schema or numerical validity failure")
    return np.clip(result, -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)
