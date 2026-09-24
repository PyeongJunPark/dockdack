"""Causal walk-forward development folds and four-way daily barrier labels.

This module reads approved ``Mark1Dataset`` windows but never changes their
values or writeability flags. Fold construction uses dates and symbol IDs only,
not any target OHLC value. Calendar purging is in trading sessions, not days.
"""
from __future__ import annotations

import numpy as np

from .mark1_data import LOOKBACK, barrier_outcomes


FOLDS = {
    "walk_2022": {"train_start": "2012-01-01", "train_end": "2019-12-31",
                  "tune_year": 2020, "calibration_year": 2021, "selection_year": 2022},
    "walk_2024": {"train_start": "2014-01-01", "train_end": "2021-12-31",
                  "tune_year": 2022, "calibration_year": 2023, "selection_year": 2024},
}
SPLIT_NAMES = ("train", "tune", "calibration", "selection")
CLASS_NAMES = ("take_only", "stop_only", "both", "neither")


def _day(value: str) -> int:
    return int(np.datetime64(value, "D").astype(np.int64))


def _integer_vector(value, name):
    result = np.asarray(value)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if result.dtype.kind == "u" and result.size and result.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} is outside the supported integer range")
    return result.astype(np.int64, copy=False)


def _validate_sample_keys(dataset):
    dates = _integer_vector(dataset.target_dates, "target_dates")
    symbols = _integer_vector(dataset.symbol_ids, "symbol_ids")
    if len(dates) != len(symbols) or np.any(symbols < 0):
        raise ValueError("Sample dates and nonnegative symbol IDs must have matching lengths")
    if len(dates) > 1:
        order = np.lexsort((dates, symbols))
        previous, following = order[:-1], order[1:]
        if np.any((symbols[previous] == symbols[following]) & (dates[previous] == dates[following])):
            raise ValueError("Duplicate (symbol_id, target_date) approved sample")
    return dates, symbols


def _cap(indices, cap, seed, fold_number, split_number):
    if cap and len(indices) > cap:
        generator = np.random.default_rng(np.random.SeedSequence([seed, fold_number, split_number]))
        positions = np.sort(generator.choice(len(indices), cap, replace=False))
        indices = indices[positions]
    # These arrays own their own index storage; never freeze caller input arrays.
    indices.setflags(write=False)
    return indices


def make_splits(dataset, sessions: np.ndarray, fold_name: str, *,
                max_train: int = 750_000, max_tune: int = 100_000,
                seed: int = 42) -> dict[str, np.ndarray]:
    """Return approved dataset indices for a predeclared development fold.

    ``train_start`` and ``train_end`` bound training *target dates* inclusively;
    their causal historical context may predate train_start. For each later
    year, the first historical session (target calendar ordinal minus 30)
    must be strictly after the preceding December 31. Thus the first thirty
    target trading sessions of each later year are purged.

    The universe is fixed using all uncapped training-period samples before
    sampling. Train and tune caps are uniform without replacement and do not
    inspect labels; zero means uncapped. Calibration and selection are always
    uncapped. No fold admits 2025+ targets. Source values and flags are unchanged;
    returned integer index arrays are read-only and sorted by dataset position.
    """
    if not isinstance(fold_name, str) or fold_name not in FOLDS:
        raise ValueError(f"Unknown fold: {fold_name!r}")
    for value, name in ((max_train, "max_train"), (max_tune, "max_tune"), (seed, "seed")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    calendar = _integer_vector(sessions, "sessions")
    if not len(calendar) or np.any(calendar[1:] <= calendar[:-1]):
        raise ValueError("sessions must be a nonempty strictly increasing unique calendar")
    dates, symbols = _validate_sample_keys(dataset)
    ordinals = np.searchsorted(calendar, dates)
    if np.any(ordinals >= len(calendar)):
        raise ValueError("Approved target is absent from the supplied calendar")
    if np.any(calendar[ordinals] != dates):
        raise ValueError("Approved target is absent from the supplied calendar")
    if np.any(ordinals < LOOKBACK):
        raise ValueError("Calendar does not cover all 30 historical sessions before an approved target")
    historical_start_dates = calendar[ordinals - LOOKBACK]
    fold = FOLDS[fold_name]
    train_mask = (dates >= _day(fold["train_start"])) & (dates <= _day(fold["train_end"]))
    if not np.any(train_mask):
        raise ValueError(f"No approved training-period samples for {fold_name}")
    eligible_symbols = np.unique(symbols[train_mask])
    eligible_mask = np.isin(symbols, eligible_symbols)
    split_indices = {"train": np.flatnonzero(train_mask).astype(np.int64)}
    for name in SPLIT_NAMES[1:]:
        year = fold[name + "_year"]
        lower = _day(f"{year}-01-01")
        upper = _day(f"{year}-12-31")
        prior_end = _day(f"{year - 1}-12-31")
        mask = ((dates >= lower) & (dates <= upper) & eligible_mask
                & (historical_start_dates > prior_end))
        split_indices[name] = np.flatnonzero(mask).astype(np.int64)
    fold_number = tuple(FOLDS).index(fold_name)
    for split_number, name in enumerate(SPLIT_NAMES):
        cap = max_train if name == "train" else max_tune if name == "tune" else 0
        split_indices[name] = _cap(split_indices[name], int(cap), int(seed), fold_number, split_number)
    return split_indices


def class_targets(ohlc, factors=(1.,)) -> np.ndarray:
    """Return int64 [N,F] targets: take_only=0, stop_only=1, both=2, neither=3.

    Entry is the target OPEN times a predeclared positive factor. The existing
    conservative 1e-12 relative barrier tolerance is preserved exactly. Class
    ``both`` is separate for learning but counts as failure/stop-first for the
    trading rule. OHLC and factor inputs are not mutated. Target-day values are
    labels only and must never enter causal feature construction.
    """
    raw_prices = np.asarray(ohlc)
    raw_factors = np.asarray(factors)
    if raw_prices.ndim != 2 or raw_prices.shape[1] != 4 or raw_prices.dtype.kind not in "iuf":
        raise ValueError("ohlc must be a numeric [N,4] OPEN,HIGH,LOW,CLOSE array")
    if raw_factors.ndim != 1 or not len(raw_factors) or raw_factors.dtype.kind not in "iuf":
        raise ValueError("factors must be a nonempty one-dimensional numeric sequence")
    prices = raw_prices.astype(np.float64, copy=False)
    factors_array = raw_factors.astype(np.float64, copy=False)
    if not np.isfinite(factors_array).all() or np.any(factors_array <= 0):
        raise ValueError("factors must be finite and positive")
    if (not np.isfinite(prices).all() or np.any(prices <= 0)
            or np.any(prices[:, 0] > prices[:, 1]) or np.any(prices[:, 0] < prices[:, 2])):
        raise ValueError("OHLC must be finite positive prices with OPEN inside HIGH/LOW")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        entries = prices[:, :1] * factors_array[None, :]
    outcomes = barrier_outcomes(prices[:, 1:2], prices[:, 2:3], prices[:, 3:4], entries)
    take, stop = outcomes["take_hit"], outcomes["stop_hit"]
    result = np.full(entries.shape, 3, dtype=np.int64)
    result[take & ~stop] = 0
    result[stop & ~take] = 1
    result[take & stop] = 2
    return result


def split_manifest(dataset, splits: dict[str, np.ndarray]) -> dict:
    """Describe index membership without reading or summarizing outcome prices."""
    dates = _integer_vector(dataset.target_dates, "target_dates")
    symbols = _integer_vector(dataset.symbol_ids, "symbol_ids")
    if len(dates) != len(symbols):
        raise ValueError("Mismatched sample dates and symbols")
    result = {}
    for name, values in splits.items():
        indices = _integer_vector(values, f"{name} indices")
        if np.any(indices < 0) or np.any(indices >= len(dates)) or np.any(indices[1:] <= indices[:-1]):
            raise ValueError("Split indices must be unique, increasing and within the dataset")
        result[name] = {"count": len(indices), "symbols": int(len(np.unique(symbols[indices]))),
                        "first": str(np.datetime64(int(dates[indices].min()), "D")) if len(indices) else None,
                        "last": str(np.datetime64(int(dates[indices].max()), "D")) if len(indices) else None}
    return result
