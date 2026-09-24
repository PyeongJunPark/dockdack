"""Read-only, approved-window data and causal features for the mark_1 experiment.

The query token is a hypothetical entry price, not a completed target-day bar.
Daily OHLC cannot establish which intraday barrier was touched first. Labels
therefore use the conservative *whole-session* high/low event specified below.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

LOOKBACK = 30
TARGET = "daily_high_ge_entry_1pct_and_low_gt_entry_minus_0_9pct_conservative"
FEATURE_NAMES = (
    "log_open_or_entry_over_last_close", "log_high_over_last_close",
    "log_low_over_last_close", "log_close_over_last_close",
    "centered_log1p_volume", "close_log_change", "log_high_low_range",
    "historical_bar_mask", "entry_query_mask",
)
SPLIT_NAMES = ("train", "tune", "calibration", "selection", "test")
PERIOD_ENDS = ("2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31")
BARRIER_TOLERANCE = 1e-12


def features_from_history(history: torch.Tensor, entry_prices: torch.Tensor, *,
                          validate: bool = False) -> torch.Tensor:
    """Create [batch,31,9] causal features from ONLY 30 finished OHLCV bars.

The first return is zero; volume is centered within the historical window.
Only feature 0 and the query mask are nonzero in token 31. Log subtraction
avoids overflow in ratios for very small historical adjusted share prices.
"""
    if history.ndim != 3 or history.shape[1:] != (LOOKBACK, 5):
        raise ValueError("history must have shape [batch,30,5] in OHLCV order")
    if entry_prices.ndim != 1 or len(entry_prices) != len(history):
        raise ValueError("entry_prices must have shape [batch]")
    if not history.is_floating_point() or not entry_prices.is_floating_point():
        raise ValueError("history and entry prices must be floating point")
    if history.device != entry_prices.device:
        raise ValueError("history and entry prices must be on the same device")
    if validate:
        prices, volumes = history[..., :4], history[..., 4]
        if (not bool(torch.isfinite(history).all())
                or not bool(torch.isfinite(entry_prices).all())
                or bool((prices <= 0).any()) or bool((volumes < 0).any())
                or bool((entry_prices <= 0).any())
                or bool((prices[..., 1] < prices.amax(dim=-1)).any())
                or bool((prices[..., 2] > prices.amin(dim=-1)).any())):
            raise ValueError("Invalid historical OHLCV or entry prices")
    log_prices = history[..., :4].log()
    reference = log_prices[:, -1, 3]
    result = history.new_zeros((len(history), LOOKBACK + 1, len(FEATURE_NAMES)))
    result[:, :LOOKBACK, :4] = log_prices - reference[:, None, None]
    log_volume = history[..., 4].log1p()
    result[:, :LOOKBACK, 4] = log_volume - log_volume.mean(dim=1, keepdim=True)
    result[:, 1:LOOKBACK, 5] = log_prices[:, 1:, 3] - log_prices[:, :-1, 3]
    result[:, :LOOKBACK, 6] = log_prices[..., 1] - log_prices[..., 2]
    result[:, :LOOKBACK, 7] = 1
    result[:, LOOKBACK, 0] = entry_prices.log() - reference
    result[:, LOOKBACK, 8] = 1
    if validate and not bool(torch.isfinite(result).all()):
        raise ValueError("Non-finite mark_1 features")
    return result


def barrier_outcomes(high, low, close, entry) -> dict[str, np.ndarray]:
    """Whole-session labels and a conservative, hypothetical daily return.

    Relative tolerance 1e-12 includes a numerically rounded exact +1% touch
    and treats a numerically rounded exact -0.9% touch as failure. Both-hit
    days get -0.9% in gross_return (an assumption, NOT an observed sequence).
    Neither-hit days use close/entry-1 as a declared same-day timeout. There
    are no fees, gaps, execution costs, or guaranteed fills in this proxy.
    """
    high, low, close, entry = np.broadcast_arrays(
        *[np.asarray(value, dtype=np.float64) for value in (high, low, close, entry)])
    if (not all(np.isfinite(value).all() for value in (high, low, close, entry))
            or any((value <= 0).any() for value in (high, low, close, entry))
            or np.any(high < low) or np.any(close > high) or np.any(close < low)):
        raise ValueError("Finite positive prices and valid target high/low/close are required")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_high = high / entry
        relative_low = low / entry
        take_hit = relative_high >= 1.01 * (1 - BARRIER_TOLERANCE)
        stop_hit = relative_low <= .991 * (1 + BARRIER_TOLERANCE)
        gross_return = np.where(stop_hit, -.009, np.where(take_hit, .01, close / entry - 1))
    if not np.isfinite(gross_return).all():
        raise ValueError("Non-finite hypothetical return")
    return {"success": take_hit & ~stop_hit, "both_touch": take_hit & stop_hit,
            "take_hit": take_hit, "stop_hit": stop_hit, "gross_return": gross_return}


@dataclass
class Mark1Dataset:
    bars: np.ndarray
    starts: np.ndarray
    target_dates: np.ndarray
    symbol_ids: np.ndarray
    target_ohlc: np.ndarray
    splits: dict[str, np.ndarray]
    manifest: dict


def _validated_windows(rows, samples, sessions, market, as_of):
    """Equivalent clean-v1 span validation without using its obsolete label."""
    from examples.train_lstm30 import clean_rows
    dates, bars, _, dropped = clean_rows([row[:6] for row in rows], allow_float32_rounding=True)
    if any(dropped.values()) or len(dates) != len(rows):
        raise ValueError("Invalid cleaned bars; dropping/reconnecting rows is forbidden")
    positions, ordinals, segments = {}, [], []
    expected_currency = "KRW" if market == "domestic" else "USD"
    previous_day = None
    for index, row in enumerate(rows):
        day, volume, segment, currency = row[0], row[5], row[6], row[7]
        if (day not in sessions or day >= as_of or (previous_day is not None and day <= previous_day)
                or type(volume) is not int or volume < 0 or type(segment) is not int
                or segment < 1 or currency != expected_currency):
            raise ValueError("Invalid cleaned bar date, segment, currency or volume")
        positions[day] = index
        ordinals.append(sessions[day])
        segments.append(segment)
        previous_day = day
    ordinals = np.asarray(ordinals, dtype=np.int64)
    segments = np.asarray(segments, dtype=np.int64)
    if len(rows) > 1 and np.any((np.diff(ordinals) == 1) != (np.diff(segments) == 0)):
        raise ValueError("Cleaned segment does not match calendar continuity")
    starts, target_indices, seen = [], [], set()
    for first, endpoint, target, sample_segment in samples:
        try:
            begin, end, outcome = positions[first], positions[endpoint], positions[target]
        except (KeyError, TypeError) as exc:
            raise ValueError("Approved sample references a missing date") from exc
        if (end != begin + LOOKBACK - 1 or outcome != begin + LOOKBACK
                or type(sample_segment) is not int or sample_segment < 1
                or outcome in seen or rows[outcome][5] <= 0
                or np.any(segments[begin:outcome + 1] != sample_segment)
                or ordinals[outcome] - ordinals[begin] != LOOKBACK):
            raise ValueError("Approved sample is not a unique consecutive single-segment 30+1 span")
        seen.add(outcome)
        starts.append(begin)
        target_indices.append(outcome)
    starts = np.asarray(starts, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    exact_prices = np.asarray([row[1:5] for row in rows], dtype=np.float64)
    return dates, bars, starts, target_indices, ordinals, exact_prices


def load_dataset(database: Path, market: str, *, start: str = "2010-01-01",
                 max_train_samples: int = 200_000, max_eval_samples: int = 60_000,
                 seed: int = 42, purge_sessions: int = 30) -> Mark1Dataset:
    """Load approved spans only, with train-availability universe and fixed splits.

    Caps sample uniformly without replacement, never by label. Zero means no
    cap. Each later split drops windows whose historical input reaches the
    previous split: the minimum purge is 30 sessions. Larger purges may be
    requested. All stored source bars for eligible symbols after start are
    retained, while starts refer exclusively to approved, unpurged windows.
    """
    # Training runs from the source checkout. Inference/features remain usable
    # from an installed dockdack package without shipping the examples folder.
    from examples.train_lstm30 import _clean_database_contract, date_number
    database = Path(database).resolve()
    if not database.is_file():
        raise ValueError(f"Cleaned database not found: {database}")
    if market not in {"domestic", "us"}:
        raise ValueError("market must be domestic or us")
    if date_number(start) > date_number(PERIOD_ENDS[0]):
        raise ValueError("start must include the fixed training period ending 2021")
    if (type(max_train_samples) is not int or max_train_samples < 0
            or type(max_eval_samples) is not int or max_eval_samples < 0
            or type(purge_sessions) is not int or purge_sessions < LOOKBACK
            or type(seed) is not int or seed < 0):
        raise ValueError("Nonnegative integer caps/seed and purge_sessions >= 30 are required")
    started = time.monotonic()
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    bar_parts, start_parts, date_parts, id_parts, price_parts = [], [], [], [], []
    split_parts = {name: [] for name in SPLIT_NAMES}
    uncapped_counts = {name: 0 for name in SPLIT_NAMES}
    purge_counts = {name: 0 for name in SPLIT_NAMES}
    symbols, excluded_symbols, bar_offset, sample_offset = [], [], 0, 0
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        contract = _clean_database_contract(connection, market)
        if contract is None:
            raise ValueError("mark_1 requires an audited clean-daily-v1 database")
        metadata, sessions = contract
        ordered_sessions = np.asarray(list(sessions), dtype="datetime64[D]").astype(np.int64)
        cut_dates = np.asarray([date_number(day) for day in PERIOD_ENDS], dtype=np.int64)
        cut_ordinals = np.searchsorted(ordered_sessions, cut_dates, side="right") - 1
        exchanges = ("KRX",) if market == "domestic" else ("NA", "ND", "NY")
        placeholders = ",".join("?" for _ in exchanges)
        candidates = connection.execute(
            "SELECT symbol,exchange,COUNT(*),SUM(CASE WHEN target_date<=? THEN 1 ELSE 0 END) "
            f"FROM training_samples WHERE input_start_date>=? AND exchange IN ({placeholders}) "
            "GROUP BY symbol,exchange ORDER BY exchange,symbol",
            (PERIOD_ENDS[0], start, *exchanges)).fetchall()
        for symbol, exchange, approved_count, train_count in candidates:
            if not train_count:
                excluded_symbols.append({"symbol": symbol, "exchange": exchange,
                                         "reason": "no_approved_training_sample"})
                continue
            rows = connection.execute(
                "SELECT trade_date,open,high,low,close,volume,segment_id,currency FROM daily_bars "
                "WHERE symbol=? AND exchange=? AND trade_date>=? ORDER BY trade_date",
                (symbol, exchange, start)).fetchall()
            samples = connection.execute(
                "SELECT input_start_date,input_end_date,target_date,segment_id FROM training_samples "
                "WHERE symbol=? AND exchange=? AND input_start_date>=? ORDER BY target_date",
                (symbol, exchange, start)).fetchall()
            dates, bars, local_starts, targets, ordinals, exact = _validated_windows(
                rows, samples, sessions, market, metadata["as_of_exclusive"])
            target_dates = dates[targets]
            period = np.searchsorted(cut_dates, target_dates, side="left")
            keep = np.ones(len(samples), dtype=bool)
            for split_number, name in enumerate(SPLIT_NAMES):
                mask = period == split_number
                if split_number:
                    previous_cut = cut_dates[split_number - 1]
                    previous_ordinal = cut_ordinals[split_number - 1]
                    purged = mask & ((dates[local_starts] <= previous_cut)
                                     | (ordinals[targets] - previous_ordinal <= purge_sessions))
                    keep[purged] = False
                    purge_counts[name] += int(purged.sum())
            kept_period = period[keep]
            count = int(keep.sum())
            for split_number, name in enumerate(SPLIT_NAMES):
                local_indices = np.flatnonzero(kept_period == split_number).astype(np.int64)
                uncapped_counts[name] += len(local_indices)
                split_parts[name].append(local_indices + sample_offset)
            symbol_id = len(symbols)
            symbols.append({"symbol_id": symbol_id, "symbol": symbol, "exchange": exchange,
                            "approved_training_samples": int(train_count),
                            "approved_samples_before_purge": int(approved_count),
                            "samples_after_purge": count, "bars": len(bars)})
            bar_parts.append(bars)
            start_parts.append(local_starts[keep] + bar_offset)
            date_parts.append(target_dates[keep].astype(np.int32))
            id_parts.append(np.full(count, symbol_id, dtype=np.int32))
            price_parts.append(exact[targets[keep]])
            bar_offset += len(bars)
            sample_offset += count
            if len(symbols) % 500 == 0:
                print(json.dumps({"phase": "mark1_data", "market": market,
                                  "symbols": len(symbols), "samples": sample_offset,
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    finally:
        connection.close()
    if not symbols:
        raise ValueError("No instruments have approved training-period samples")
    splits = {name: np.concatenate(parts) for name, parts in split_parts.items()}
    for split_number, name in enumerate(SPLIT_NAMES):
        cap = max_train_samples if name == "train" else max_eval_samples
        if cap and len(splits[name]) > cap:
            rng = np.random.default_rng(np.random.SeedSequence([seed, split_number]))
            positions = np.sort(rng.choice(len(splits[name]), cap, replace=False))
            splits[name] = splits[name][positions]
    target_dates = np.concatenate(date_parts)
    manifest = {
        "database": str(database), "market": market, "start": start,
        "target": TARGET, "feature_names": list(FEATURE_NAMES), "seed": seed,
        "lookback": LOOKBACK, "tokens": LOOKBACK + 1, "purge_sessions": purge_sessions,
        "period_ends": list(PERIOD_ENDS), "max_train_samples": max_train_samples,
        "max_eval_samples": max_eval_samples,
        "sampling": "Uniform per split, fixed seed; no outcome stratification; caps do not use future labels",
        "universe": "Only instruments with an approved training-period sample; clean-v1 classifications retained",
        "candidate_symbols": len(candidates), "selected_symbols": len(symbols), "symbols": symbols,
        "excluded_symbols": excluded_symbols, "packed_bars": bar_offset,
        "samples_after_purge": sample_offset, "uncapped_split_counts": uncapped_counts,
        "purged_split_counts": purge_counts,
        "selected_counts": {name: len(indices) for name, indices in splits.items()},
        "split_dates": {name: {"first": str(np.datetime64(int(target_dates[indices].min()), "D")),
                               "last": str(np.datetime64(int(target_dates[indices].max()), "D"))}
                        if len(indices) else {"first": None, "last": None}
                        for name, indices in splits.items()},
        "source_fingerprints": metadata.get("source_fingerprints"),
        "cleaned_metadata": metadata,
        "limitations": list(metadata.get("limitations", [])) + [
            "Daily bars do not reveal intraday barrier order or extrema after a hypothetical entry",
            "Both-touch is failure for a conservative whole-session event, not proven stop-first execution",
            "Synthetic entry prices are counterfactual queries, not new independent market observations",
            "The historical target_up column is ignored; mark_1 must compute its own entry-relative labels",
            "Gross returns are hypothetical stop-first and day-close-timeout proxies without costs or fills",
            "The current historical catalog and train-availability universe omit some later IPOs",
        ],
    }
    return Mark1Dataset(np.concatenate(bar_parts), np.concatenate(start_parts), target_dates,
                        np.concatenate(id_parts), np.concatenate(price_parts), splits, manifest)
