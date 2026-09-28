"""Read-only, gap-preserving forward paths for Mark1 daily-bar horizon research.

Inputs are the *already approved* Mark1Dataset sample indices.  The 30-bar
history remains in that frozen cache; this module reads only the corresponding
future clean daily bars, in one symbol-sized batch at a time.  The resulting
bank can be reused for many (lookback, target, horizon) combinations without
changing the source database, frozen cache, or operating account.

Entry is the next session's observed OPEN.  A take-profit touch is a daily-HIGH
proxy for a hypothetical limit fill, not evidence that an order was filled.
There is no stop-loss.  Absent, zero-volume, discontinuous, or cross-segment
paths are ineligible; they are never forward-filled or labelled as losses.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
import sqlite3
from typing import Mapping

import numpy as np

from .research_artifacts import ArtifactResolver


_SPLITS = ("train", "tune", "calibration", "selection", "test")
_MARKETS = {"domestic": ({"KRX"}, "KRW"),
            "us": ({"NA", "ND", "NY"}, "USD")}
_TOLERANCE = 1e-12


def _day_number(value: str) -> int:
    if type(value) is not str:
        raise ValueError("Date must be a canonical YYYY-MM-DD string")
    try:
        day = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Invalid date") from exc
    if day.isoformat() != value:
        raise ValueError("Date must be a canonical YYYY-MM-DD string")
    return (day - date(1970, 1, 1)).days


def _day_text(value: int) -> str:
    return str(np.datetime64(int(value), "D"))


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


@dataclass(frozen=True)
class HorizonOutcomes:
    """Arrays aligned with ``HorizonBank.sample_indices``; invalid rows are NaN/-1."""

    eligible: np.ndarray
    hit: np.ndarray
    exit_date: np.ndarray
    exit_session: np.ndarray
    exit_price: np.ndarray
    gross_return: np.ndarray
    net_return: np.ndarray


@dataclass(frozen=True)
class HorizonBank:
    """Compact reusable future paths; no full-market bar bank is retained."""

    market: str
    sample_indices: np.ndarray
    symbol_ids: np.ndarray
    split_names: np.ndarray
    entry_dates: np.ndarray
    entry_ordinals: np.ndarray
    entry_prices: np.ndarray
    future_dates: np.ndarray
    future_high: np.ndarray
    future_close: np.ndarray
    valid_prefix: np.ndarray
    split_start_ordinals: np.ndarray
    split_end_ordinals: np.ndarray
    source_sha256: str

    @property
    def max_horizon(self) -> int:
        return self.future_high.shape[1]

    def histories(self, dataset, *, lookback: int, indices: np.ndarray) -> np.ndarray:
        """Materialize only a requested local batch of L completed bars.

        ``indices`` are positions within this bank, not source dataset IDs.
        The frozen 30-bar bank is never copied or normalized in-place.
        """
        _positive_int(lookback, "lookback")
        if lookback > 30:
            raise ValueError("This frozen cache supports at most 30 completed bars")
        picked = np.asarray(indices)
        if (picked.ndim != 1 or picked.dtype.kind not in "iu"
                or (picked.size and (np.any(picked < 0) or np.any(picked >= len(self.sample_indices))))):
            raise ValueError("indices must be valid one-dimensional local bank positions")
        starts = np.asarray(dataset.starts)[self.sample_indices[picked]]
        offsets = np.arange(30 - lookback, 30, dtype=np.int64)
        history = np.asarray(dataset.bars)[starts[:, None] + offsets[None, :]]
        if history.shape != (len(picked), lookback, 5):
            raise ValueError("Frozen bar bank has an incompatible shape")
        return history

    def outcomes(self, *, target_pct: float = 3.0, horizon: int = 10,
                 cost_bps: float = 0.0, slippage_bps: float = 0.0,
                 split_name: str | None = None, lookback: int = 20) -> HorizonOutcomes:
        """First +p% HIGH touch or H-th session CLOSE, with separate friction.

        The entry session is day 1; ``horizon=10`` times out at its day-10
        close.  ``cost_bps`` is round-trip commission/tax friction and
        ``slippage_bps`` is an additional *per-side* adverse adjustment.  These
        costs do not change the historical touch label.  A daily high does not
        guarantee a real limit-order fill.
        """
        target_pct = _finite_nonnegative(target_pct, "target_pct")
        if target_pct <= 0 or target_pct >= 1000:
            raise ValueError("target_pct must be in (0,1000)")
        _positive_int(horizon, "horizon")
        _positive_int(lookback, "lookback")
        if horizon > self.max_horizon or lookback > 30:
            raise ValueError("Horizon or history exceeds the loaded bank")
        cost_bps = _finite_nonnegative(cost_bps, "cost_bps")
        slippage_bps = _finite_nonnegative(slippage_bps, "slippage_bps")
        side_friction = (cost_bps / 2 + slippage_bps) / 10_000
        if side_friction >= 1:
            raise ValueError("Combined one-side friction must be below 100%")
        if split_name is not None and split_name not in _SPLITS:
            raise ValueError("Unknown split name")

        eligible = (self.valid_prefix[:, horizon - 1]
                    & (self.entry_ordinals + horizon - 1 <= self.split_end_ordinals)
                    & (self.entry_ordinals - lookback >= self.split_start_ordinals))
        if split_name is not None:
            eligible &= self.split_names == split_name
        target = self.entry_prices * (1 + target_pct / 100)
        touched = self.future_high[:, :horizon] >= target[:, None] * (1 - _TOLERANCE)
        first_touch = np.argmax(touched, axis=1)
        hit = touched.any(axis=1) & eligible
        exit_session = np.where(hit, first_touch + 1, horizon).astype(np.int16)
        exit_session[~eligible] = -1
        row = np.arange(len(self.sample_indices))
        exit_date = self.future_dates[row, np.maximum(exit_session - 1, 0)].copy()
        exit_date[~eligible] = -1
        exit_price = np.where(hit, target, self.future_close[:, horizon - 1])
        gross = exit_price / self.entry_prices - 1
        net = exit_price * (1 - side_friction) / (self.entry_prices * (1 + side_friction)) - 1
        for array in (exit_price, gross, net):
            array[~eligible] = np.nan
        return HorizonOutcomes(eligible, hit, exit_date, exit_session,
                               exit_price, gross, net)


def _split_indices(dataset, sample_indices) -> tuple[np.ndarray, np.ndarray]:
    length = len(dataset.starts)
    codes = np.full(length, -1, dtype=np.int8)
    for code, name in enumerate(_SPLITS):
        values = np.asarray(dataset.splits.get(name, np.empty(0, dtype=np.int64)))
        if (values.ndim != 1 or values.dtype.kind not in "iu"
                or (values.size and (np.any(values < 0) or np.any(values >= length)))):
            raise ValueError(f"Invalid {name} split indices")
        if np.any(codes[values] != -1):
            raise ValueError("One approved sample belongs to multiple splits")
        codes[values] = code
    if sample_indices is None:
        selected = np.flatnonzero(codes >= 0).astype(np.int64)
    else:
        selected = np.asarray(sample_indices)
        if (selected.ndim != 1 or selected.dtype.kind not in "iu"
                or (selected.size and (np.any(selected < 0) or np.any(selected >= length)))):
            raise ValueError("sample_indices must be in-range one-dimensional integers")
        selected = selected.astype(np.int64, copy=False)
        if len(np.unique(selected)) != len(selected) or np.any(codes[selected] < 0):
            raise ValueError("sample_indices must be unique approved split members")
    return selected, codes[selected]


def _split_boundaries(dataset, sessions: np.ndarray, selected_codes: np.ndarray,
                      *, split_end_dates: Mapping[str, str] | None,
                      split_start_dates: Mapping[str, str] | None,
                      as_of_exclusive: str) -> tuple[np.ndarray, np.ndarray]:
    if split_end_dates is None:
        period_ends = dataset.manifest.get("period_ends")
        if not isinstance(period_ends, list) or len(period_ends) != 4:
            raise ValueError("Frozen dataset split end dates are missing")
        split_end_dates = dict(zip(_SPLITS[:4], period_ends))
        split_end_dates["test"] = _day_text(sessions[-1])
    if set(split_end_dates) != set(_SPLITS):
        raise ValueError("All five split end dates are required")
    ends = np.asarray([np.searchsorted(sessions, _day_number(split_end_dates[name]), side="right") - 1
                       for name in _SPLITS], dtype=np.int64)
    if np.any(ends < 0) or np.any(ends[1:] <= ends[:-1]):
        raise ValueError("Split ends must be ordered within the session calendar")
    if split_end_dates["test"] >= as_of_exclusive:
        raise ValueError("Test boundary reaches an incomplete session")
    if split_start_dates is None:
        starts = np.r_[0, ends[:-1] + 1]
    else:
        if set(split_start_dates) != set(_SPLITS):
            raise ValueError("All five split start dates are required")
        starts = np.asarray([np.searchsorted(sessions, _day_number(split_start_dates[name]))
                             for name in _SPLITS], dtype=np.int64)
        if np.any(starts > ends) or np.any(starts[1:] <= ends[:-1]):
            raise ValueError("Split starts must follow the previous split end")
    # Train can use pre-2010 context, as in the already-approved frozen cache.
    starts[0] = 0
    return starts[selected_codes], ends[selected_codes]


def load_horizon_bank(dataset, source: Mapping[str, object], workspace: Path,
                      market: str, *, sample_indices=None, max_horizon: int = 20,
                      split_end_dates: Mapping[str, str] | None = None,
                      split_start_dates: Mapping[str, str] | None = None) -> HorizonBank:
    """Attach no more than ``max_horizon`` future sessions to approved samples.

    ``dataset, source, _ = mark1_2_data.load_dataset(market, workspace)`` is the
    production input.  Source paths are resolved and SHA-verified against its
    original frozen contract.  SQLite is opened immutable/query-only; the
    operating database and broker are never used.
    """
    if market not in _MARKETS or dataset.manifest.get("market") != market:
        raise ValueError("Dataset market differs from requested market")
    _positive_int(max_horizon, "max_horizon")
    if max_horizon > 60:
        raise ValueError("max_horizon exceeds the bounded research contract (60)")
    selected, codes = _split_indices(dataset, sample_indices)
    if not len(selected):
        raise ValueError("At least one approved sample is required")
    resolver = ArtifactResolver(Path(workspace))
    artifact = resolver.database(source, market)
    expected_exchanges, currency = _MARKETS[market]
    with closing(sqlite3.connect(artifact.physical_path.as_uri() + "?mode=ro&immutable=1",
                                 uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        metadata = {key: json.loads(value) for key, value in
                    connection.execute("SELECT key,value FROM metadata")}
        as_of = metadata.get("as_of_exclusive")
        if metadata.get("market") != market or type(as_of) is not str:
            raise ValueError("Cleaned source market or as-of contract is invalid")
        sessions = np.asarray([_day_number(row[0]) for row in
                               connection.execute("SELECT session_date FROM sessions ORDER BY ordinal")],
                              dtype=np.int32)
        if not len(sessions) or np.any(sessions[1:] <= sessions[:-1]):
            raise ValueError("Session calendar must be strictly ordered")
        session_ordinals = {_day_text(day): ordinal for ordinal, day in enumerate(sessions)}
        starts, ends = _split_boundaries(dataset, sessions, codes,
                                         split_end_dates=split_end_dates,
                                         split_start_dates=split_start_dates,
                                         as_of_exclusive=as_of)
        entry_dates = np.asarray(dataset.target_dates)[selected].astype(np.int32, copy=True)
        entry_ordinals = np.searchsorted(sessions, entry_dates).astype(np.int32)
        if (np.any(entry_ordinals >= len(sessions))
                or np.any(sessions[entry_ordinals] != entry_dates)
                or np.any(entry_ordinals < 30)):
            raise ValueError("Approved entry date is absent from the completed calendar")
        raw_entry = np.asarray(dataset.target_ohlc)[selected]
        if (raw_entry.shape != (len(selected), 4) or not np.isfinite(raw_entry).all()
                or np.any(raw_entry <= 0)):
            raise ValueError("Approved target OPEN/OHLC are invalid")
        prices = raw_entry[:, 0].astype(np.float64, copy=True)
        symbols = np.asarray(dataset.symbol_ids)[selected].astype(np.int32, copy=True)
        inventory = {}
        for row in dataset.manifest.get("symbols", ()):
            identifier = row.get("symbol_id")
            if (type(identifier) is not int or identifier in inventory
                    or row.get("exchange") not in expected_exchanges
                    or not isinstance(row.get("symbol"), str)):
                raise ValueError("Invalid frozen symbol inventory")
            inventory[identifier] = (row["symbol"], row["exchange"])
        if not set(np.unique(symbols)).issubset(inventory):
            raise ValueError("Approved sample has no frozen symbol identity")

        n = len(selected)
        future_dates = np.full((n, max_horizon), -1, dtype=np.int32)
        future_high = np.full((n, max_horizon), np.nan, dtype=np.float64)
        future_close = np.full((n, max_horizon), np.nan, dtype=np.float64)
        valid_prefix = np.zeros((n, max_horizon), dtype=bool)
        positions_by_symbol: dict[int, list[int]] = {}
        for position, symbol_id in enumerate(symbols):
            positions_by_symbol.setdefault(int(symbol_id), []).append(position)
        for symbol_id, positions in positions_by_symbol.items():
            symbol, exchange = inventory[symbol_id]
            first = int(min(entry_ordinals[positions]))
            last = min(len(sessions) - 1, int(max(entry_ordinals[positions])) + max_horizon - 1)
            rows = connection.execute(
                "SELECT trade_date,open,high,low,close,volume,segment_id,currency "
                "FROM daily_bars WHERE symbol=? AND exchange=? AND trade_date>=? "
                "AND trade_date<=? ORDER BY trade_date",
                (symbol, exchange, _day_text(sessions[first]), _day_text(sessions[last])))
            observed = {}
            for day, opened, high, low, close, volume, segment, denomination in rows:
                ordinal = session_ordinals.get(day)
                if ordinal is None:
                    raise ValueError("Stored price is outside the session calendar")
                try:
                    o, h, l, c = (float(value) for value in (opened, high, low, close))
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValueError("Malformed cleaned OHLC") from exc
                if (not all(math.isfinite(value) and value > 0 for value in (o, h, l, c))
                        or h < max(o, c, l) or l > min(o, c, h)
                        or type(volume) is not int or volume < 0
                        or type(segment) is not int or segment < 1
                        or denomination != currency or ordinal in observed):
                    raise ValueError("Malformed cleaned OHLCV, segment, or currency")
                observed[ordinal] = (o, h, l, c, volume, segment)
            for position in positions:
                ordinal = int(entry_ordinals[position])
                first_bar = observed.get(ordinal)
                if first_bar is None or not np.array_equal(
                        np.asarray(first_bar[:4]), raw_entry[position]):
                    raise ValueError("Frozen approved entry differs from the clean database")
                segment = first_bar[5]
                intact = True
                for offset in range(max_horizon):
                    current = ordinal + offset
                    if current >= len(sessions):
                        break
                    bar = observed.get(current)
                    if bar is None or bar[4] <= 0 or bar[5] != segment:
                        intact = False
                    future_dates[position, offset] = sessions[current]
                    if bar is not None:
                        future_high[position, offset] = bar[1]
                        future_close[position, offset] = bar[3]
                    valid_prefix[position, offset] = intact
    resolver.recheck(artifact, database=True)
    return HorizonBank(
        market=market, sample_indices=selected, symbol_ids=symbols,
        split_names=np.asarray(_SPLITS, dtype="U11")[codes],
        entry_dates=entry_dates, entry_ordinals=entry_ordinals,
        entry_prices=prices, future_dates=future_dates,
        future_high=future_high, future_close=future_close,
        valid_prefix=valid_prefix, split_start_ordinals=starts,
        split_end_ordinals=ends, source_sha256=artifact.sha256)
