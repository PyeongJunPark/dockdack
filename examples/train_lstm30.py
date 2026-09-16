"""Train pooled-market 30-candle LSTMs from the existing read-only SQLite DB.

Run from the repository root with ``python -m examples.train_lstm30 --help``.
Legacy DB labels use the NEXT valid observed bar's close gain >= 1%; cleaned
DBs use only approved next-scheduled-session targets and never bridge gaps.
Neither is an intraday barrier label. Profit/stop exits are execution rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dockdack.ml30 import CandleLSTM, FEATURE_NAMES, TARGET


LOOKBACK = 30
SPLIT_NAMES = ("train", "validation", "test")


def date_number(value: str) -> int:
    """Reject noncanonical dates so lexicographic SQLite date comparisons work."""
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"Expected ISO YYYY-MM-DD date, got {value!r}")
    return int(np.datetime64(value, "D").astype(np.int64))


def date_string(value: int) -> str:
    return str(np.datetime64(int(value), "D"))


def clean_rows(rows: list[tuple], *, allow_float32_rounding: bool = False
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Keep valid, ordered bars; labels retain float64 closes before packing.

    Invalid and zero-price rows are removed. Therefore a target is the next
    VALID observed trading bar, which can cross a suspension or a data gap.
    No price/volume fill or interpolation is performed.
    Cleaned DBs opt into their assessor's finite-positive float32 cast rule,
    including representable subnormals; legacy numeric bounds stay unchanged.
    """
    dates, values, seen = [], [], set()
    counts = {"invalid_date": 0, "duplicate_date": 0, "invalid_ohlcv": 0}
    for row in rows:
        try:
            number = date_number(str(row[0]))
        except (ValueError, TypeError, OverflowError):
            counts["invalid_date"] += 1
            continue
        if number in seen:
            counts["duplicate_date"] += 1
            continue
        seen.add(number)
        try:
            numbers = tuple(float(value) for value in row[1:6])
        except (ValueError, TypeError, OverflowError):
            counts["invalid_ohlcv"] += 1
            continue
        if len(numbers) != 5:
            counts["invalid_ohlcv"] += 1
            continue
        op, hi, lo, cl, vol = numbers
        if (not all(math.isfinite(value) for value in numbers)
                or min(op, hi, lo, cl) <= 0 or vol < 0
                or hi < max(op, lo, cl) or lo > min(op, hi, cl)):
            counts["invalid_ohlcv"] += 1
            continue
        if allow_float32_rounding:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                packed = np.asarray(numbers, dtype=np.float32)
            representable = np.isfinite(packed).all() and (packed[:4] > 0).all()
        else:
            representable = (max(numbers) <= np.finfo(np.float32).max
                             and min(op, hi, lo, cl) >= np.finfo(np.float32).tiny)
        if not representable:
            counts["invalid_ohlcv"] += 1
            continue
        dates.append(number)
        values.append(numbers)
    if not dates:
        return (np.empty(0, np.int32), np.empty((0, 5), np.float32),
                np.empty(0, np.float64), counts)
    order = np.argsort(dates, kind="stable")
    doubles = np.asarray(values, dtype=np.float64)[order]
    return (np.asarray(dates, dtype=np.int32)[order], doubles.astype(np.float32),
            doubles[:, 3], counts)


def make_sample_indices(dates: np.ndarray, closes: np.ndarray,
                        train_end: str, validation_end: str,
                        offset: int = 0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """31 raw bars make one sample; no window can cross symbol boundaries."""
    if len(dates) != len(closes):
        raise ValueError("Dates and closes must have the same length")
    if len(dates) > 1 and np.any(np.diff(dates) <= 0):
        raise ValueError("Dates must be strictly increasing")
    train_cut, validation_cut = date_number(train_end), date_number(validation_end)
    if train_cut >= validation_cut:
        raise ValueError("train_end must be earlier than validation_end")
    labels = np.zeros(len(dates), dtype=np.float32)
    if len(closes) > 1:
        # Tiny double precision tolerance includes mathematically exact +1%.
        returns = (closes[1:] - closes[:-1]) / closes[:-1]
        labels[1:] = (returns >= 0.01 - 1e-12).astype(np.float32)
    targets = np.arange(LOOKBACK, len(dates), dtype=np.int64)
    target_dates = dates[targets]
    masks = (target_dates <= train_cut,
             (target_dates > train_cut) & (target_dates <= validation_cut),
             target_dates > validation_cut)
    return ({name: targets[mask] - LOOKBACK + offset
             for name, mask in zip(SPLIT_NAMES, masks)}, labels)


def symbol_rank(seed: int, symbol: str, exchange: str) -> str:
    return hashlib.sha256(f"{seed}:{exchange}:{symbol}".encode("utf-8")).hexdigest()


def split_description(starts: np.ndarray, labels: np.ndarray,
                      dates: np.ndarray) -> dict:
    if not len(starts):
        return {"samples": 0, "target_start": None, "target_end": None,
                "positive_samples": 0, "positive_rate": None}
    targets = starts + LOOKBACK
    return {"samples": len(starts), "target_start": date_string(dates[targets].min()),
            "target_end": date_string(dates[targets].max()),
            "positive_samples": int(labels[targets].sum()),
            "positive_rate": float(labels[targets].mean())}


@dataclass
class PackedDataset:
    bars: np.ndarray
    labels: np.ndarray
    dates: np.ndarray
    symbol_ids: np.ndarray
    splits: dict[str, np.ndarray]
    manifest: dict


def _clean_database_contract(connection, market):
    """Recognise approved-index databases without silently treating them as raw."""
    objects = dict(connection.execute("SELECT name,type FROM sqlite_master WHERE type IN ('table','view')"))
    tables = {name for name, kind in objects.items() if kind == "table"}
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata")) if "metadata" in objects else {}
    except sqlite3.DatabaseError as exc:
        raise ValueError("Malformed database metadata table") from exc
    bar_columns = {row[1] for row in connection.execute("PRAGMA table_info(daily_bars)")}
    cleaned = ("training_samples" in objects or "segment_id" in bar_columns
               or "requires_training_samples" in metadata or "build_status" in metadata
               or "clean-daily" in str(metadata.get("schema_version", "")))
    if not cleaned:
        return None
    try:
        decoded = {key: json.loads(value) for key, value in metadata.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Malformed cleaned database JSON metadata") from exc
    policy = decoded.get("policy")
    if (decoded.get("schema_version") != "clean-daily-v1"
            or decoded.get("requires_training_samples") is not True
            or decoded.get("build_status") != "complete" or decoded.get("market") != market
            or not isinstance(policy, dict) or type(policy.get("lookback")) is not int
            or policy["lookback"] != LOOKBACK):
        raise ValueError("Incomplete or incompatible cleaned database metadata")
    try:
        date_number(decoded.get("as_of_exclusive"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Cleaned database requires a canonical as_of_exclusive date") from exc
    required = {
        "metadata": {"key", "value"},
        "instruments": {"symbol", "exchange"},
        "daily_bars": {"symbol", "exchange", "trade_date", "open", "high", "low", "close", "volume", "segment_id", "currency"},
        "training_samples": {"symbol", "exchange", "input_start_date", "input_end_date", "target_date", "target_up", "segment_id"},
        "sessions": {"session_date", "ordinal"},
    }
    for table, columns in required.items():
        actual = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if table not in tables or not columns <= actual:
            raise ValueError(f"Malformed cleaned database: missing {table} columns")
    session_rows = connection.execute("SELECT session_date,ordinal FROM sessions ORDER BY ordinal").fetchall()
    sessions, previous = {}, None
    for expected, (day, ordinal) in enumerate(session_rows):
        try:
            current = date_number(day)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Malformed cleaned database session date") from exc
        if type(ordinal) is not int or ordinal != expected or (previous is not None and current <= previous):
            raise ValueError("Malformed cleaned database session ordinals")
        sessions[day], previous = ordinal, current
    if not sessions:
        raise ValueError("Cleaned database has no session calendar")
    if connection.execute("""SELECT 1 FROM training_samples s LEFT JOIN instruments i
            ON i.symbol=s.symbol AND i.exchange=s.exchange WHERE i.symbol IS NULL LIMIT 1""").fetchone():
        raise ValueError("Cleaned sample references an unknown instrument")
    return decoded, sessions


def _approved_sample_indices(rows, samples, sessions, market, train_end, validation_end, as_of):
    """Check 30+1 exact rows and calendar/segment continuity; never infer windows."""
    dates, bars, closes, dropped = clean_rows([row[:6] for row in rows], allow_float32_rounding=True)
    if any(dropped.values()) or len(rows) != len(dates):
        raise ValueError("Cleaned database contains invalid bars; refusing to drop and reconnect them")
    positions, ordinals, segments = {}, [], []
    currency = "KRW" if market == "domestic" else "USD"
    for index, row in enumerate(rows):
        day, volume, segment, stored_currency = row[0], row[5], row[6], row[7]
        if (day not in sessions or day >= as_of or type(segment) is not int or segment < 1
                or type(volume) is not int or volume < 0 or stored_currency != currency):
            raise ValueError("Cleaned bar calendar, segment, volume or currency is invalid")
        ordinal = sessions[day]
        if ordinals and ((ordinal == ordinals[-1] + 1) != (segment == segments[-1])):
            raise ValueError("Cleaned bar segment does not match session continuity")
        positions[day] = index
        ordinals.append(ordinal)
        segments.append(segment)
    labels = np.zeros(len(bars), dtype=np.float32)
    starts = {name: [] for name in SPLIT_NAMES}
    training_coverage = np.zeros(len(bars) + 1, dtype=np.int64)
    seen_targets = set()
    for first, endpoint, target, label, segment in samples:
        try:
            for day in (first, endpoint, target):
                date_number(day)
            begin, end, outcome = positions[first], positions[endpoint], positions[target]
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Cleaned sample references missing or invalid dates") from exc
        if (end != begin + LOOKBACK - 1 or outcome != begin + LOOKBACK
                or type(segment) is not int or segment < 1
                or any(value != segment for value in segments[begin:outcome + 1])
                or ordinals[outcome] - ordinals[begin] != LOOKBACK
                or rows[outcome][5] <= 0
                or type(label) is not int or label not in (0, 1) or outcome in seen_targets):
            raise ValueError("Cleaned sample is not a unique contiguous 30+1 window in one segment")
        expected = int(Decimal(str(float(closes[outcome])))
                       >= Decimal(str(float(closes[end]))) * Decimal("1.01"))
        if label != expected:
            raise ValueError("Cleaned sample target label does not match its endpoint and target closes")
        seen_targets.add(outcome)
        labels[outcome] = label
        split = "train" if target <= train_end else "validation" if target <= validation_end else "test"
        starts[split].append(begin)
        if split == "train":
            training_coverage[begin] += 1
            training_coverage[outcome + 1] -= 1
    return (dates, bars, labels, {name: np.asarray(values, dtype=np.int64) for name, values in starts.items()},
            int(np.count_nonzero(np.cumsum(training_coverage[:-1]))), dropped)


def load_dataset(database: Path, market: str, *, start: str = "2015-01-01",
                 train_end: str = "2022-12-31", validation_end: str = "2024-12-31",
                 max_symbols: int = 512, min_train_bars: int = 300,
                 seed: int = 42, max_train_samples: int = 0) -> PackedDataset:
    """Read without mutating DB; selection never uses validation/test outcomes."""
    if not database.is_file():
        raise ValueError(f"Database not found: {database}")
    if market not in {"domestic", "us"}:
        raise ValueError("market must be domestic or us")
    if not date_number(start) <= date_number(train_end) < date_number(validation_end):
        raise ValueError("Expected start <= train_end < validation_end")
    if max_symbols < 1 or min_train_bars < LOOKBACK + 1 or max_train_samples < 0:
        raise ValueError("Invalid max_symbols, min_train_bars or max_train_samples")
    exchanges = ("KRX",) if market == "domestic" else ("NA", "ND", "NY")
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    bars_parts, label_parts, date_parts, id_parts = [], [], [], []
    split_parts = {name: [] for name in SPLIT_NAMES}
    entries, offset, selected, last_progress = [], 0, 0, time.monotonic()
    cleaned_contract = None
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        cleaned_contract = _clean_database_contract(connection, market)
        placeholders = ",".join("?" for _ in exchanges)
        has_catalog = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'instruments'").fetchone()
        if cleaned_contract:
            universe_filter = "clean-daily-v1 approved training_samples; no inferred adjacency"
            candidates = connection.execute(
                "SELECT symbol,exchange,COUNT(*),SUM(CASE WHEN target_date <= ? THEN 1 ELSE 0 END) "
                f"FROM training_samples WHERE input_start_date >= ? AND exchange IN ({placeholders}) "
                "GROUP BY symbol,exchange", (train_end, start, *exchanges)).fetchall()
        elif has_catalog:
            universe_filter = ("catalog_market_code IN ('0','10') (KOSPI/KOSDAQ)"
                               if market == "domestic" else "is_etf = 0")
            condition = ("i.catalog_market_code IN ('0','10')" if market == "domestic"
                         else "i.is_etf = 0")
            # Indexed per-symbol joins exclude domestic ELWs/ETFs and US ETFs.
            # Future counts are ONLY coverage reporting, never eligibility.
            candidates = connection.execute(
                "SELECT i.symbol, i.exchange, COUNT(b.trade_date), "
                "SUM(CASE WHEN b.trade_date <= ? THEN 1 ELSE 0 END) "
                "FROM instruments i LEFT JOIN daily_bars b ON b.symbol = i.symbol "
                "AND b.exchange = i.exchange AND b.trade_date >= ? "
                f"WHERE i.exchange IN ({placeholders}) AND {condition} "
                "GROUP BY i.symbol, i.exchange HAVING COUNT(b.trade_date) > 0",
                (train_end, start, *exchanges)).fetchall()
        else:
            # Small custom/test databases may intentionally omit a catalog.
            universe_filter = "exchange only; instruments catalog unavailable"
            candidates = connection.execute(
                "SELECT symbol, exchange, COUNT(*), "
                "SUM(CASE WHEN trade_date <= ? THEN 1 ELSE 0 END) "
                f"FROM daily_bars WHERE trade_date >= ? AND exchange IN ({placeholders}) "
                "GROUP BY symbol, exchange", (train_end, start, *exchanges)).fetchall()
        candidates.sort(key=lambda row: symbol_rank(seed, str(row[0]), str(row[1])))
        for rank, (symbol, exchange, raw_count, train_raw_count) in enumerate(candidates):
            entry = {"symbol": symbol, "exchange": exchange, "hash_rank": rank,
                     "selected": False}
            if cleaned_contract:
                entry.update(approved_samples_in_date_range=raw_count, approved_training_samples=train_raw_count)
            else:
                entry.update(raw_bars_in_date_range=raw_count, raw_training_bars=train_raw_count)
            entries.append(entry)
            if train_raw_count < (1 if cleaned_contract else min_train_bars):
                entry["exclusion_reason"] = "no_approved_training_samples" if cleaned_contract else "insufficient_training_history"
                continue
            if selected >= max_symbols:
                entry["exclusion_reason"] = "deterministic_symbol_cap"
                continue
            columns = "trade_date,open,high,low,close,volume" + (",segment_id,currency" if cleaned_contract else "")
            rows = connection.execute(f"SELECT {columns} FROM daily_bars "
                "WHERE symbol=? AND exchange=? AND trade_date>=? ORDER BY trade_date",
                (symbol, exchange, start)).fetchall()
            if cleaned_contract:
                samples = connection.execute("SELECT input_start_date,input_end_date,target_date,target_up,segment_id "
                    "FROM training_samples WHERE symbol=? AND exchange=? AND input_start_date>=? ORDER BY target_date",
                    (symbol, exchange, start)).fetchall()
                dates, bars, labels, local_splits, valid_training_bars, dropped = _approved_sample_indices(
                    rows, samples, cleaned_contract[1], market, train_end, validation_end,
                    cleaned_contract[0]["as_of_exclusive"])
                entry["raw_bars_in_date_range"] = len(rows)
                entry["raw_training_bars"] = int((dates <= date_number(train_end)).sum())
                entry["approved_training_bars"] = valid_training_bars
            else:
                dates, bars, closes, dropped = clean_rows(rows)
                valid_training_bars = int((dates <= date_number(train_end)).sum())
            entry.update({"valid_bars": len(bars), "dropped_rows": dropped})
            entry["valid_training_bars"] = valid_training_bars
            if valid_training_bars < min_train_bars:
                entry["exclusion_reason"] = "insufficient_valid_training_history"
                continue
            if not cleaned_contract:
                local_splits, labels = make_sample_indices(dates, closes, train_end, validation_end)
            entry.update({"selected": True, "symbol_id": selected,
                          "first_valid_date": date_string(dates[0]),
                          "last_valid_date": date_string(dates[-1]),
                          "splits": {name: split_description(indices, labels, dates)
                                     for name, indices in local_splits.items()}})
            bars_parts.append(bars)
            label_parts.append(labels)
            date_parts.append(dates)
            id_parts.append(np.full(len(bars), selected, dtype=np.int32))
            for name in SPLIT_NAMES:
                split_parts[name].append(local_splits[name] + offset)
            offset += len(bars)
            selected += 1
            if selected % 50 == 0 or time.monotonic() - last_progress > 20:
                print(f"Loaded {selected}/{max_symbols} symbols, {offset:,} valid raw bars", flush=True)
                last_progress = time.monotonic()
    finally:
        connection.close()
    if not selected:
        raise ValueError("No symbols have enough valid training-period history")
    packed_bars = np.concatenate(bars_parts)
    packed_labels = np.concatenate(label_parts)
    packed_dates = np.concatenate(date_parts)
    packed_ids = np.concatenate(id_parts)
    splits = {name: np.concatenate(parts) for name, parts in split_parts.items()}
    uncapped_train_count = len(splits["train"])
    if max_train_samples and uncapped_train_count > max_train_samples:
        rng = np.random.default_rng(seed)
        selected_train = np.sort(rng.choice(uncapped_train_count, max_train_samples, replace=False))
        splits["train"] = splits["train"][selected_train]
    for name, indices in splits.items():
        if not len(indices):
            raise ValueError(f"No {name} samples: adjust date boundaries or symbol coverage")
    train_counts = np.bincount(packed_ids[splits["train"] + LOOKBACK], minlength=selected)
    for entry in entries:
        if entry["selected"]:
            entry["training_samples_used"] = int(train_counts[entry["symbol_id"]])
    manifest = {
        "database": str(database.resolve()), "market": market, "start": start,
        "train_end": train_end, "validation_end": validation_end, "seed": seed,
        "selection": "SHA256(seed:exchange:symbol) rank; training availability only",
        "universe_filter": universe_filter,
        "min_train_bars": min_train_bars, "max_symbols": max_symbols,
        "max_train_samples": max_train_samples, "candidate_symbols": len(entries),
        "selected_symbols": selected, "valid_raw_bars": len(packed_bars),
        "lookback_raw_bars": LOOKBACK, "uncapped_training_samples": uncapped_train_count,
        "splits": {name: split_description(indices, packed_labels, packed_dates)
                   for name, indices in splits.items()},
        "symbols": entries,
        "limitations": ["Available downloaded universe can have survivorship bias",
                        ("Cleaned classification and liquidity policy applied; historical point-in-time universe is not verified"
                         if cleaned_contract else "Basic catalog stock filter, not verified common equity; preferred shares, REITs or SPACs may remain"),
                        ("Only explicit approved 30+1 sample windows are used; session/segment gaps are never bridged"
                         if cleaned_contract else "Invalid bars are dropped; next valid target may cross a gap"),
                        "Validation/test inputs can use earlier observed bars, never future bars",
                        "Current DB tail may lag the current market date",
                        "No fees, slippage or realized strategy profitability in these labels"],
    }
    if cleaned_contract:
        manifest["cleaned_dataset"] = {"schema_version": cleaned_contract[0]["schema_version"],
                                       "policy": cleaned_contract[0]["policy"],
                                       "as_of_exclusive": cleaned_contract[0].get("as_of_exclusive"),
                                       "requires_training_samples": True}
    return PackedDataset(packed_bars, packed_labels, packed_dates, packed_ids, splits, manifest)


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray,
                   training_positive_rate: float) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not len(labels) or len(probabilities) != len(labels):
        raise ValueError("Nonempty matching probabilities and labels are required")
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    prior = float(np.clip(training_positive_rate, 1e-7, 1 - 1e-7))
    thresholds = {}
    for threshold in (0.5, 0.55, 0.6):
        predicted = probabilities >= threshold
        tp = int((predicted & (labels == 1)).sum())
        pp, positives = int(predicted.sum()), int(labels.sum())
        thresholds[str(threshold)] = {
            "accuracy": float((predicted == labels).mean()),
            "precision": tp / pp if pp else None,
            "recall": tp / positives if positives else None,
            "predicted_buy_samples": pp, "predicted_buy_fraction": pp / len(labels),
        }
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    auc = None
    if positives and negatives:
        order = np.argsort(probabilities, kind="stable")
        sorted_p, sorted_y = probabilities[order], labels[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
        group_count = np.diff(np.r_[boundaries, len(labels)])
        group_positive = np.add.reduceat(sorted_y, boundaries)
        group_negative = group_count - group_positive
        earlier_negative = np.cumsum(group_negative) - group_negative
        auc = float(np.sum(group_positive * (earlier_negative + 0.5 * group_negative))
                    / (positives * negatives))
    return {"samples": len(labels), "positive_rate": float(labels.mean()),
            "bce": float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean()),
            "brier_score": float(((probabilities - labels) ** 2).mean()), "roc_auc": auc,
            "mean_probability": float(probabilities.mean()),
            "no_skill_training_prior": prior,
            "no_skill_bce": float(-(labels * math.log(prior) + (1 - labels) * math.log1p(-prior)).mean()),
            "no_skill_majority_accuracy": float((labels == int(prior >= 0.5)).mean()),
            "fixed_thresholds": thresholds}


class BatchSource:
    """Only compact raw bars live on GPU; overlapping windows form per batch."""
    def __init__(self, dataset: PackedDataset, device: torch.device, fraction: float):
        self.dataset, self.device = dataset, device
        self.gpu_resident = False
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            packed_bytes = dataset.bars.nbytes + dataset.labels.nbytes
            # Leave most of this process's allowance for model activations.
            self.gpu_resident = packed_bytes < min(free * 0.25, total * fraction * 0.30)
        storage = device if self.gpu_resident else torch.device("cpu")
        self.bars = torch.from_numpy(dataset.bars).to(storage)
        self.labels = torch.from_numpy(dataset.labels).to(storage)
        self.offsets = torch.arange(LOOKBACK, device=storage)

    def get(self, starts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.from_numpy(np.ascontiguousarray(starts)).to(self.bars.device)
        inputs = self.bars[indices[:, None] + self.offsets[None, :]]
        labels = self.labels[indices + LOOKBACK]
        return inputs.to(self.device), labels.to(self.device)


def evaluate(model: nn.Module, source: BatchSource, starts: np.ndarray,
             batch_size: int, training_positive_rate: float) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    probability_parts, label_parts = [], []
    with torch.inference_mode():
        for first in range(0, len(starts), batch_size):
            inputs, labels = source.get(starts[first:first + batch_size])
            logits = model(inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite evaluation output; no checkpoint will be approved")
            probability_parts.append(torch.sigmoid(logits).cpu().numpy())
            label_parts.append(labels.cpu().numpy())
    probabilities, labels = np.concatenate(probability_parts), np.concatenate(label_parts)
    return binary_metrics(probabilities, labels, training_positive_rate), probabilities, labels


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def save_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=("domestic", "us"), required=True)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--validation-end", default="2024-12-31")
    parser.add_argument("--max-symbols", type=int, default=512)
    parser.add_argument("--min-train-bars", type=int, default=300)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="0 means all training windows; positive value caps reproducibly")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cudnn", choices=("auto", "on", "off"), default="auto",
                        help="auto disables cuDNN on Windows CUDA to avoid the LSTM dropout shutdown crash")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume last_model.pt in the SAME output directory")
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("Device must be auto, cuda or cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable; CPU fallback is disabled")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        # CUDA allocator controls require a concrete device index in PyTorch.
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def configure_cudnn(requested: str, device: torch.device) -> tuple[bool, str]:
    """Choose execution backend only; do not alter dropout or model weights."""
    if requested not in {"auto", "on", "off"}:
        raise ValueError("cuDNN mode must be auto, on or off")
    windows_cuda = platform.system() == "Windows" and device.type == "cuda"
    enabled = requested == "on" or (requested == "auto" and not windows_cuda)
    torch.backends.cudnn.enabled = enabled
    if requested == "auto" and windows_cuda:
        note = ("Windows CUDA automatic safety workaround: cuDNN disabled to avoid the "
                "observed LSTM dropout process-shutdown crash; CUDA and dropout remain enabled")
    elif requested == "off":
        note = "cuDNN explicitly disabled; model architecture and configured device are unchanged"
    elif requested == "on":
        note = "cuDNN explicitly enabled by opt-in; Windows LSTM dropout shutdown crash may recur"
    else:
        note = "cuDNN enabled by the automatic non-Windows-CUDA policy"
    return enabled, note


def train(args: argparse.Namespace) -> dict:
    if (min(args.epochs, args.batch_size, args.hidden_size, args.num_layers,
            args.patience, args.threads) < 1 or not 0 <= args.dropout < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not 0 < args.gpu_memory_fraction <= 1):
        raise ValueError("Invalid training hyperparameters")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    cudnn_enabled, backend_note = configure_cudnn(args.cudnn, device)
    if args.cudnn == "auto" and device.type == "cuda" and not cudnn_enabled:
        print(backend_note, flush=True)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    database = args.db or Path(f"data/kiwoom_daily/{args.market}_daily.sqlite3")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or (args.resume.parent if args.resume
                                else Path(f"outputs/ml30/{args.market}-{stamp}"))
    if args.resume:
        if output.resolve() != args.resume.resolve().parent:
            raise ValueError("Resume must use the checkpoint's existing output directory")
        if not args.resume.is_file() or not (output / "best_model.pt").is_file():
            raise ValueError("Resume requires existing last_model.pt and best_model.pt")
    else:
        output.mkdir(parents=True, exist_ok=False)
    print(f"Loading {args.market} data read-only from {database}; device={device}", flush=True)
    started = time.monotonic()
    dataset = load_dataset(database, args.market, start=args.start,
                           train_end=args.train_end, validation_end=args.validation_end,
                           max_symbols=args.max_symbols, min_train_bars=args.min_train_bars,
                           seed=args.seed, max_train_samples=args.max_train_samples)
    manifest_hash = hashlib.sha256(json.dumps(dataset.manifest, sort_keys=True).encode()).hexdigest()
    architecture = {"hidden_size": args.hidden_size, "num_layers": args.num_layers,
                    "dropout": args.dropout}
    metadata = {"schema_version": 1, "lookback": LOOKBACK, "architecture": architecture,
                "market": args.market, "feature_names": list(FEATURE_NAMES), "target": TARGET,
                "buy_threshold": 0.5, "target_return": 0.01,
                "target_horizon": "next valid observed trading bar close",
                "label_price_basis": "next close versus last completed input close, not fill price",
                "take_profit_rate": 0.01, "stop_loss_rate": 0.008,
                "exit_rules_are_learned": False, "manifest_sha256": manifest_hash,
                "training_start": args.start, "train_end": args.train_end,
                "validation_end": args.validation_end, "created_at": stamp,
                "seed": args.seed}
    model = CandleLSTM(**architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    source = BatchSource(dataset, device, args.gpu_memory_fraction)
    train_indices = dataset.splits["train"]
    prior = float(dataset.labels[train_indices + LOOKBACK].mean())
    history, best_loss, best_epoch, stale_epochs, first_epoch = [], math.inf, 0, 0, 1
    if args.resume:
        previous = torch.load(args.resume, map_location=device, weights_only=True)
        old = previous["metadata"]
        if old["manifest_sha256"] != manifest_hash or old["architecture"] != architecture:
            raise ValueError("Resume dataset/architecture differs from the original run")
        metadata = old
        model.load_state_dict(previous["model_state_dict"])
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        history, best_loss, best_epoch = previous["history"], previous["best_loss"], previous["best_epoch"]
        stale_epochs, first_epoch = previous["stale_epochs"], previous["epoch"] + 1
        torch.set_rng_state(previous["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in previous:
            torch.cuda.set_rng_state_all([value.cpu() for value in previous["cuda_rng_state_all"]])
    save_json(output / "manifest.json", dataset.manifest)
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update({"output_dir": str(output.resolve()), "device_used": str(device),
                       "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                       "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                       "platform": platform.system(), "cudnn_enabled": cudnn_enabled,
                       "backend_note": backend_note,
                       "parameters": sum(parameter.numel() for parameter in model.parameters()),
                       "gpu_resident_raw_data": source.gpu_resident,
                       "packed_data_bytes": dataset.bars.nbytes + dataset.labels.nbytes,
                       "manifest_sha256": manifest_hash})
    save_json(output / "run_config.json", run_config)
    print(json.dumps({"symbols": dataset.manifest["selected_symbols"],
                      "raw_bars": len(dataset.bars), "parameters": run_config["parameters"],
                      "samples": {name: len(indices) for name, indices in dataset.splits.items()},
                      "device": run_config["device_name"], "gpu_resident": source.gpu_resident}), flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        epoch_started, last_progress = time.monotonic(), time.monotonic()
        # Seed per epoch also reproduces the order when resuming.
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_indices))
        cumulative_loss = torch.zeros((), device=device)
        count = 0
        for first in range(0, len(order), args.batch_size):
            starts = train_indices[order[first:first + args.batch_size]]
            inputs, labels = source.get(starts)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; stopping without approving this model")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            cumulative_loss += loss.detach() * len(starts)
            count += len(starts)
            if time.monotonic() - last_progress >= 20:
                print(f"Epoch {epoch}/{args.epochs}: {count:,}/{len(order):,} samples, "
                      f"BCE={cumulative_loss.item() / count:.5f}", flush=True)
                last_progress = time.monotonic()
        validation, _, _ = evaluate(model, source, dataset.splits["validation"], args.batch_size, prior)
        row = {"epoch": epoch, "train_bce": cumulative_loss.item() / count,
               "validation": validation, "seconds": time.monotonic() - epoch_started}
        history.append(row)
        if validation["bce"] < best_loss:
            best_loss, best_epoch, stale_epochs = validation["bce"], epoch, 0
            save_checkpoint(output / "best_model.pt", {"model_state_dict": model.state_dict(),
                            "metadata": metadata, "epoch": epoch, "validation": validation})
        else:
            stale_epochs += 1
        save_json(output / "history.json", history)
        state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "metadata": metadata, "epoch": epoch, "best_loss": best_loss,
                 "best_epoch": best_epoch, "stale_epochs": stale_epochs, "history": history,
                 "torch_rng_state": torch.get_rng_state()}
        if device.type == "cuda":
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        save_checkpoint(output / "last_model.pt", state)
        print(f"Epoch {epoch}: train BCE={row['train_bce']:.5f}, val BCE={validation['bce']:.5f}, "
              f"val AUC={validation['roc_auc']}, best={best_epoch}, seconds={row['seconds']:.1f}", flush=True)
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} non-improving validation epochs", flush=True)
            break
    selected = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(selected["model_state_dict"])
    # This is the FIRST test evaluation, after model selection is frozen.
    test_metrics, test_probabilities, test_labels = evaluate(
        model, source, dataset.splits["test"], args.batch_size, prior)
    targets = dataset.splits["test"] + LOOKBACK
    np.savez_compressed(output / "test_predictions.npz", probability=test_probabilities,
                        label=test_labels, target_date=dataset.dates[targets],
                        symbol_id=dataset.symbol_ids[targets], packed_window_start=dataset.splits["test"])
    metrics = {"market": args.market, "best_epoch": int(selected["epoch"]),
               "completed_epochs": len(history), "test": test_metrics,
               "best_validation": selected["validation"],
               "elapsed_seconds": time.monotonic() - started, "run": run_config,
               "data": {key: value for key, value in dataset.manifest.items() if key != "symbols"},
               "evaluation_note": "Classification only; not a profit backtest or validated live trading strategy"}
    if device.type == "cuda":
        metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        metrics["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    save_json(output / "metrics.json", metrics)
    print(json.dumps({"output_dir": str(output.resolve()), "best_epoch": metrics["best_epoch"],
                      "test": test_metrics}, ensure_ascii=False), flush=True)
    return metrics


def main(argv: list[str] | None = None) -> None:
    try:
        train(parse_args(argv))
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"Training failed: {exc}") from exc


if __name__ == "__main__":
    main()
