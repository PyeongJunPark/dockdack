"""Pure, point-in-time daily-bar quality checks and eligible sample indexing.

Raw rows are never mutated. Hard-invalid rows remain in the assessment rather
than being removed and accidentally joining two separated trading sessions.
Liquidity thresholds are evaluated only through an input endpoint. Targets
must be observed with positive volume, but have no turnover/return-size cutoff.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, InvalidOperation
import math
import struct
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True, kw_only=True)
class QualityPolicy:
    min_median_turnover: float
    lookback: int = 30
    liquidity_window: int = 60
    median_window: int = 20
    min_median_volume: float = 10000
    min_active_fraction: float = 0.95
    max_zero_run: int = 2

    def __post_init__(self):
        for name in ("lookback", "liquidity_window", "median_window"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.median_window > self.liquidity_window:
            raise ValueError("median_window must not exceed liquidity_window")
        if type(self.max_zero_run) is not int or not 0 <= self.max_zero_run <= self.liquidity_window:
            raise ValueError("max_zero_run must be an integer within the liquidity window")
        for name in ("min_median_volume", "min_median_turnover", "min_active_fraction"):
            value = _number(getattr(self, name))
            if value is None or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.min_active_fraction > 1:
            raise ValueError("min_active_fraction must not exceed one")


@dataclass(frozen=True)
class BarAssessment:
    source_index: int
    date: str | None
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: int | None
    currency: str | None
    trade_value: Any
    liquidity_turnover: float | None
    hard_errors: tuple[str, ...]
    flags: tuple[str, ...]
    input_eligible: bool = False
    median_volume: float | None = None
    median_turnover: float | None = None
    active_fraction: float | None = None
    max_zero_run: int | None = None

    @property
    def valid(self) -> bool:
        return not self.hard_errors


@dataclass(frozen=True)
class SampleAssessment:
    input_start_index: int
    input_end_index: int
    target_index: int
    input_start_date: str
    input_end_date: str
    target_date: str
    target_up: bool


@dataclass(frozen=True)
class SeriesAssessment:
    bars: tuple[BarAssessment, ...]
    samples: tuple[SampleAssessment, ...]


def _get(row, key):
    # sqlite3.Row is mapping-like but does not implement dict.get().
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _price(value) -> float | None:
    result = _number(value)
    if result is None or result <= 0:
        return None
    try:
        float32 = struct.unpack("f", struct.pack("f", result))[0]
    except (OverflowError, struct.error):
        return None
    return result if math.isfinite(float32) and float32 > 0 else None


def _volume(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 and _number(value) is not None else None
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value >= 0 and value.is_integer() else None
    try:
        parsed = Decimal(str(value))
        if not parsed.is_finite() or parsed < 0 or parsed != parsed.to_integral_value():
            return None
        return int(parsed) if _number(parsed) is not None else None
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return None


def _iso_date(value) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def _median(ordered: list) -> float:
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    # Half first so two otherwise finite large values cannot overflow a sum.
    return float(ordered[middle - 1]) / 2 + float(ordered[middle]) / 2


def assess_series(rows: Iterable[Mapping[str, Any]], *, market, as_of: date,
                  session_dates: Sequence[str], policy: QualityPolicy) -> SeriesAssessment:
    """Assess one symbol, retaining original row indexes and chronological samples.

    ``session_dates`` must be a strictly ascending complete exchange calendar
    covering the intended history. All dates >= ``as_of`` are conservatively
    incomplete. ``liquidity_turnover`` must already be normalized to KRW/USD by
    the caller; raw ``trade_value`` is metadata, never a liquidity input.

    A sample needs ``lookback`` consecutive hard-valid input bars, a liquid
    input endpoint with a full consecutive liquidity history, and a hard-valid
    positive-volume target on the next scheduled session. Earlier zero-volume
    input bars are retained; the configured active fraction/run filters apply
    to the endpoint's trailing history. The label is a >= 1% next-close rise.
    """
    selected_market = getattr(market, "value", market)
    if selected_market not in {"domestic", "us"}:
        raise ValueError("market must be domestic or us")
    if type(as_of) is not date:
        raise ValueError("as_of must be a date, not a datetime or string")
    if not isinstance(policy, QualityPolicy):
        raise TypeError("policy must be a QualityPolicy")
    calendar = tuple(session_dates)
    if not calendar or any(_iso_date(day) is None for day in calendar):
        raise ValueError("session_dates must contain exact ISO session dates")
    if any(a >= b for a, b in zip(calendar, calendar[1:])):
        raise ValueError("session_dates must be strictly ascending and unique")
    session_positions = {day: index for index, day in enumerate(calendar)}
    expected_currency = "KRW" if selected_market == "domestic" else "USD"
    bars: list[BarAssessment] = []
    for index, row in enumerate(rows):
        errors, flags = [], []
        parsed_date = _iso_date(_get(row, "date"))
        day = parsed_date.isoformat() if parsed_date is not None else None
        if day is None:
            errors.append("INVALID_DATE")
        else:
            if parsed_date >= as_of:
                errors.append("INCOMPLETE_SESSION")
            if day < calendar[0] or day > calendar[-1]:
                errors.append("OUTSIDE_CALENDAR_COVERAGE")
            elif day not in session_positions:
                errors.append("NON_SESSION_DATE")
        prices = {name: _price(_get(row, name)) for name in ("open", "high", "low", "close")}
        for name, value in prices.items():
            if value is None:
                errors.append("INVALID_" + name.upper())
        if all(value is not None for value in prices.values()):
            if not (prices["low"] <= min(prices["open"], prices["close"])
                    <= max(prices["open"], prices["close"]) <= prices["high"]):
                errors.append("INVALID_OHLC_BOUNDS")
            elif prices["high"] / prices["low"] >= 2:
                flags.append("HIGH_LOW_RATIO_GE_2")
        volume = _volume(_get(row, "volume"))
        if volume is None:
            errors.append("INVALID_VOLUME")
        elif volume == 0:
            flags.append("ZERO_VOLUME")
        currency = _get(row, "currency")
        if currency != expected_currency:
            errors.append("CURRENCY_MISMATCH")
        turnover = _number(_get(row, "liquidity_turnover"))
        if turnover is None or turnover < 0:
            turnover = None
            flags.append("LIQUIDITY_TURNOVER_UNAVAILABLE")
        bars.append(BarAssessment(index, day, **prices, volume=volume, currency=currency,
                                  trade_value=_get(row, "trade_value"), liquidity_turnover=turnover,
                                  hard_errors=tuple(errors), flags=tuple(flags)))
    counts = Counter(bar.date for bar in bars if bar.date is not None)
    for index, bar in enumerate(bars):
        if bar.date is not None and counts[bar.date] > 1:
            bars[index] = replace(bar, hard_errors=bar.hard_errors + ("DUPLICATE_DATE",))
    valid_by_session = {session_positions[bar.date]: bar.source_index for bar in bars if bar.valid}

    history, recent, zero_runs = deque(), deque(), deque()
    volumes, turnovers = [], []
    active_count = missing_turnovers = consecutive = 0
    previous_session = None
    previous_close = None
    samples = []
    for session, index in sorted(valid_by_session.items()):
        bar = bars[index]
        if previous_session is None or session != previous_session + 1:
            history.clear()
            recent.clear()
            zero_runs.clear()
            volumes.clear()
            turnovers.clear()
            active_count = missing_turnovers = consecutive = 0
            previous_close = None
        flags = list(bar.flags)
        if previous_close is not None and abs(bar.close / previous_close - 1) >= 0.5:
            flags.append("ABS_RETURN_GE_50PCT")
        consecutive += 1
        history.append(bar.volume)
        active_count += bar.volume > 0
        if len(history) > policy.liquidity_window:
            active_count -= history.popleft() > 0
        if bar.volume == 0:
            if zero_runs and zero_runs[-1][1] == session - 1:
                zero_runs[-1][1] = session
            else:
                zero_runs.append([session, session])
        window_start = session - policy.liquidity_window + 1
        while zero_runs and zero_runs[0][1] < window_start:
            zero_runs.popleft()
        if zero_runs and zero_runs[0][0] < window_start:
            zero_runs[0][0] = window_start
        zero_run = max((end - start + 1 for start, end in zero_runs), default=0)
        recent.append((bar.volume, bar.liquidity_turnover))
        insort(volumes, bar.volume)
        if bar.liquidity_turnover is None:
            missing_turnovers += 1
        else:
            insort(turnovers, bar.liquidity_turnover)
        if len(recent) > policy.median_window:
            old_volume, old_turnover = recent.popleft()
            volumes.pop(bisect_left(volumes, old_volume))
            if old_turnover is None:
                missing_turnovers -= 1
            else:
                turnovers.pop(bisect_left(turnovers, old_turnover))
        median_volume = _median(volumes) if len(recent) == policy.median_window else None
        median_turnover = (_median(turnovers) if len(recent) == policy.median_window
                           and not missing_turnovers else None)
        active_fraction = active_count / len(history)
        if len(history) < policy.liquidity_window:
            flags.append("INSUFFICIENT_LIQUIDITY_HISTORY")
        if median_volume is not None and median_volume < policy.min_median_volume:
            flags.append("LOW_MEDIAN_VOLUME")
        if median_turnover is None:
            flags.append("INCOMPLETE_TURNOVER_WINDOW")
        elif median_turnover < policy.min_median_turnover:
            flags.append("LOW_MEDIAN_TURNOVER")
        if active_fraction < policy.min_active_fraction:
            flags.append("LOW_ACTIVE_FRACTION")
        if zero_run > policy.max_zero_run:
            flags.append("EXCESS_ZERO_RUN")
        eligible = (len(history) == policy.liquidity_window and bar.volume > 0
                    and median_volume is not None and median_volume >= policy.min_median_volume
                    and median_turnover is not None and median_turnover >= policy.min_median_turnover
                    and active_fraction >= policy.min_active_fraction and zero_run <= policy.max_zero_run)
        bars[index] = replace(bar, flags=tuple(flags), input_eligible=eligible,
                              median_volume=median_volume, median_turnover=median_turnover,
                              active_fraction=active_fraction, max_zero_run=zero_run)
        if eligible and consecutive >= policy.lookback and session + 1 in valid_by_session:
            target = bars[valid_by_session[session + 1]]
            if target.volume > 0:
                first = bars[valid_by_session[session - policy.lookback + 1]]
                target_up = Decimal(str(target.close)) >= Decimal(str(bar.close)) * Decimal("1.01")
                samples.append(SampleAssessment(first.source_index, index, target.source_index,
                                                first.date, bar.date, target.date, target_up))
        previous_session, previous_close = session, bar.close
    return SeriesAssessment(tuple(bars), tuple(samples))
