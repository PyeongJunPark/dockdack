"""Fail-closed pre-open input collection for the Mark1.4 daily-bar scorer.

This module does not fetch quotes or submit orders. The caller owns the broker
worker and passes its existing history cache; a successful result is only a
complete, same-session batch of model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Mapping

from dockdack.history import market_time
from dockdack.market_schedule import session_on
from dockdack.models import Market


@dataclass(frozen=True)
class PreopenResult:
    candidates: tuple[dict, ...] = ()
    reason: str | None = None
    session_open: datetime | None = None
    ranking_fetched_at: datetime | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None and len(self.candidates) == 100


class _Unavailable(ValueError):
    pass


def _aware(value, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise _Unavailable(f"{label}: timezone-aware timestamp required")
    return value


def _expected_dates(market: Market, trading_day: date) -> tuple[date, ...]:
    days = []
    cursor = trading_day - timedelta(days=1)
    for _ in range(180):
        if session_on(market, cursor) is not None:
            days.append(cursor)
            if len(days) == 30:
                return tuple(reversed(days))
        cursor -= timedelta(days=1)
    raise _Unavailable("completed_sessions_unavailable: fewer than 30 exchange sessions")


def _number(value, *, positive: bool) -> float:
    if isinstance(value, bool):
        raise _Unavailable("history_invalid_ohlcv: boolean value")
    try:
        decimal = Decimal(str(value))
        result = float(decimal)
    except (InvalidOperation, ValueError, OverflowError, TypeError) as exc:
        raise _Unavailable("history_invalid_ohlcv: nonnumeric value") from exc
    if not decimal.is_finite() or not isfinite(result) or (result <= 0 if positive else result < 0):
        raise _Unavailable("history_invalid_ohlcv: nonfinite or out-of-range value")
    return result


def _history_row(history, item, market: Market, trading_day: date, expected: tuple[date, ...]) -> dict:
    inst = item.instrument
    if (getattr(history, "market", None) != market
            or getattr(history, "symbol", None) != inst.symbol
            or getattr(history, "exchange", None) != inst.exchange
            or getattr(history, "currency", None) != inst.currency):
        raise _Unavailable(f"history_identity_mismatch:{item.id}")
    try:
        bars = tuple(history.bars)
        dates = tuple(bar.day for bar in bars)
    except (AttributeError, TypeError) as exc:
        raise _Unavailable(f"history_missing_bars:{item.id}") from exc
    if (not dates or any(type(day) is not date for day in dates)
            or any(left >= right for left, right in zip(dates, dates[1:]))
            or any(day > trading_day for day in dates)):
        raise _Unavailable(f"history_dates_invalid_or_future:{item.id}")
    completed = tuple(bar for bar in bars if bar.day < trading_day)[-30:]
    if tuple(bar.day for bar in completed) != expected:
        raise _Unavailable(f"history_missing_consecutive_sessions:{item.id}")
    values = []
    for bar in completed:
        opening, high, low, close = (_number(getattr(bar, key), positive=True)
                                     for key in ("open", "high", "low", "close"))
        volume = _number(bar.volume, positive=False)
        if high < max(opening, low, close) or low > min(opening, high, close):
            raise _Unavailable(f"history_invalid_ohlcv:{item.id}")
        values.append((opening, high, low, close, volume))
    return {
        "watch_id": item.id,
        "symbol": inst.symbol,
        "exchange": inst.exchange,
        "bars": tuple(values),
        "dates": tuple(day.isoformat() for day in expected),
        "last_completed_date": expected[-1].isoformat(),
    }


def collect_preopen_candidates(store, engine, market: Market | str, *, clock=None, cancelled=None) -> PreopenResult:
    """Return all current volume TOP100 windows or a reason and no candidates.

    Collection may run only from the exchange's pre-open ranking slot until the
    regular open. The ranking and every 30-bar window must belong to that same
    trading session. A slow fetch, cancellation, or single invalid name voids
    the entire batch; the caller must not reuse an earlier result for the day.
    Composite visible watchlists use their separate volume snapshot, including
    model candidates outside the active 100-name buy watchlist.
    """
    opened = fetched_at = None
    try:
        market = Market(market)
        clock = clock or engine.clock
        cancelled = cancelled or engine._stop.is_set

        def guard() -> datetime:
            if cancelled():
                raise _Unavailable("cancelled: pre-open collection stopped")
            now = _aware(clock(), "clock")
            if opened is not None and not opened - timedelta(minutes=10) <= now < opened:
                raise _Unavailable("preopen_deadline_passed: outside the current pre-open slot")
            return now

        now = guard()
        trading_day = market_time(market, now).date()
        session = session_on(market, trading_day)
        if session is None:
            raise _Unavailable("market_closed: no exchange session today")
        opened = session.opened
        guard()
        slot = opened - timedelta(minutes=10)
        expected = _expected_dates(market, trading_day)

        visible = tuple(row for row in store.rankings()
                        if isinstance(row, Mapping) and row.get("market") == market.value)
        composite = (len(visible) == 100
                     and {row.get("ranking_scheme") for row in visible} == {"composite"})
        if composite:
            rows = tuple({**row, "ranking_basis": "volume"}
                         for row in store.model_volume_rankings(market))
            if {row.get("fetched_at") for row in rows} != {row.get("fetched_at") for row in visible}:
                raise _Unavailable("ranking_timestamp_mismatch: visible and model snapshots differ")
        else:
            rows = visible
        now = guard()
        if len(rows) != 100:
            raise _Unavailable(f"ranking_count_invalid: expected 100, got {len(rows)}")
        if (any(type(row.get("rank")) is not int for row in rows)
                or {row["rank"] for row in rows} != set(range(1, 101))):
            raise _Unavailable("ranking_rank_invalid: ranks must be exactly 1..100")
        rows = tuple(sorted(rows, key=lambda row: row["rank"]))
        if any(row.get("ranking_basis") != "volume" for row in rows):
            raise _Unavailable("ranking_basis_invalid: volume ranking required")
        volumes = []
        for row in rows:
            try:
                volume = Decimal(str(row.get("volume")))
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise _Unavailable("ranking_volume_invalid: traded share volume required") from exc
            if not volume.is_finite() or volume < 0 or volume != volume.to_integral_value():
                raise _Unavailable("ranking_volume_invalid: nonnegative integer shares required")
            volumes.append(volume)
        if any(left < right for left, right in zip(volumes, volumes[1:])):
            raise _Unavailable("ranking_order_invalid: rank must follow descending volume")
        stamps = {row.get("fetched_at") for row in rows}
        if len(stamps) != 1:
            raise _Unavailable("ranking_timestamp_mismatch: one pre-open refresh required")
        try:
            fetched_at = _aware(datetime.fromisoformat(str(next(iter(stamps))).replace("Z", "+00:00")),
                                "ranking_fetched_at")
        except ValueError as exc:
            raise _Unavailable("ranking_timestamp_invalid: timezone-aware ISO timestamp required") from exc
        if not slot <= fetched_at <= now or fetched_at >= opened:
            raise _Unavailable("ranking_not_current_preopen_slot: refresh must finish in this session's slot")
        watch_ids = tuple(row.get("watch_id") for row in rows)
        if (any(not isinstance(key, str) or not key for key in watch_ids)
                or len(set(watch_ids)) != 100):
            raise _Unavailable("ranking_candidates_invalid: 100 distinct watch IDs required")
        if composite:
            from dockdack.gui_service import Instrument
            from dockdack.watchlist import WatchItem
            try:
                items = {row["watch_id"]: WatchItem(
                    Instrument(market, row["symbol"], row["exchange"]), row["name"], 31)
                    for row in rows}
            except (KeyError, ValueError, TypeError) as exc:
                raise _Unavailable("ranking_candidates_invalid: model share-volume identity") from exc
            if any(items[key].id != key or items[key].instrument.currency != row["currency"]
                   for key, row in zip(watch_ids, rows)):
                raise _Unavailable("ranking_candidates_invalid: model share-volume identity")
        else:
            items = {item.id: item for item in store.items()}
            if any(key not in items or items[key].instrument.market != market for key in watch_ids):
                raise _Unavailable("ranking_watchlist_mismatch: ranked names must be active in this market")

        candidates = []
        for key in watch_ids:
            guard()
            item = items[key]
            try:
                history = engine.histories.get(item, 31)
            except Exception as exc:
                raise _Unavailable(f"history_fetch_failed:{key}:{type(exc).__name__}: {str(exc)[:160]}") from exc
            guard()
            candidates.append(_history_row(history, item, market, trading_day, expected))
        guard()
        return PreopenResult(tuple(candidates), None, opened, fetched_at)
    except _Unavailable as exc:
        return PreopenResult((), str(exc), opened, fetched_at)
    except Exception as exc:
        return PreopenResult((), f"preopen_input_unavailable:{type(exc).__name__}: {str(exc)[:160]}", opened, fetched_at)
