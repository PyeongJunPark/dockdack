"""Read-only, gap-preserving cleaned daily prices for mark_1 backtests.

Signals still require the approved training-sample index. Holdings, however,
must be valued and tested for exits on *all* retained cleaned bars, including
dates that are not themselves approved signal targets. Missing sessions are
reported, never forward-filled or joined across without the caller knowing.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path

EPOCH = date(1970, 1, 1)


def _day(value: int | str) -> tuple[int, str]:
    if type(value) is int:
        try:
            parsed = EPOCH + timedelta(days=value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Date outside supported calendar") from exc
        return value, parsed.isoformat()
    if type(value) is not str:
        raise ValueError("Day must be an integer epoch day or canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Day must be an integer epoch day or canonical ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError("Date must have YYYY-MM-DD form")
    return (parsed - EPOCH).days, value


def load_price_panel(database: Path, market: str, symbols: list[dict],
                     start_day: int | str, end_day: int | str):
    """Return ``(prices, sessions, metadata)`` from an audited clean-v1 DB.

    ``symbols`` is the original cache manifest's symbol mapping; its IDs are
    preserved. ``prices[(symbol_id, epoch_day)]`` is an unrounded float64-style
    Python-float OHLC tuple. ``sessions`` includes every scheduled day in the
    requested interval, even when no retained price exists on that day.

    Zero-volume bars are retained for valuation but explicitly listed in
    ``metadata['zero_volume_keys']`` so execution engines can prohibit fills.
    Missing prices are absent keys; they are not fabricated or bridged. The
    source must remain unchanged; caller can additionally verify its file hash.
    """
    from examples.train_lstm30 import _clean_database_contract

    database = Path(database).resolve()
    if not database.is_file():
        raise ValueError(f"Cleaned database not found: {database}")
    if market not in {"domestic", "us"}:
        raise ValueError("market must be domestic or us")
    start_number, start_text = _day(start_day)
    end_number, end_text = _day(end_day)
    if start_number > end_number:
        raise ValueError("start_day must not exceed end_day")
    exchanges = {"KRX"} if market == "domestic" else {"NA", "ND", "NY"}
    expected_currency = "KRW" if market == "domestic" else "USD"
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("A nonempty cache-manifest symbol list is required")
    seen_ids, seen_symbols, validated_symbols = set(), set(), []
    for item in symbols:
        if not isinstance(item, dict):
            raise ValueError("Malformed cache-manifest symbol")
        symbol_id, symbol, exchange = (item.get(key) for key in ("symbol_id", "symbol", "exchange"))
        if (type(symbol_id) is not int or symbol_id < 0 or symbol_id in seen_ids
                or type(symbol) is not str or not symbol or exchange not in exchanges
                or (symbol, exchange) in seen_symbols):
            raise ValueError("Invalid or duplicate cache-manifest symbol mapping")
        seen_ids.add(symbol_id)
        seen_symbols.add((symbol, exchange))
        validated_symbols.append((symbol_id, symbol, exchange))

    prices, symbol_coverage, zero_volume_keys = {}, [], []
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        contract = _clean_database_contract(connection, market)
        if contract is None:
            raise ValueError("Backtesting requires an audited clean-daily-v1 database")
        source_metadata, calendar = contract
        if end_text >= source_metadata["as_of_exclusive"]:
            raise ValueError("Requested end day reaches the incomplete as-of session")
        if start_text < next(iter(calendar)) or end_text > next(reversed(calendar)):
            raise ValueError("Requested range lies outside the stored session calendar")
        sessions = [_day(day)[0] for day in calendar if start_text <= day <= end_text]
        if not sessions:
            raise ValueError("Requested range has no scheduled trading sessions")
        session_set = set(sessions)
        for symbol_id, symbol, exchange in validated_symbols:
            if connection.execute("SELECT 1 FROM instruments WHERE symbol=? AND exchange=?",
                                  (symbol, exchange)).fetchone() is None:
                raise ValueError(f"Unknown cached instrument: {exchange}:{symbol}")
            rows = connection.execute(
                "SELECT trade_date,open,high,low,close,volume,currency FROM daily_bars "
                "WHERE symbol=? AND exchange=? AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
                (symbol, exchange, start_text, end_text))
            observed_days = []
            for day, open_, high, low, close, volume, currency in rows:
                day_number, _ = _day(day)
                try:
                    ohlc = tuple(float(value) for value in (open_, high, low, close))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"Invalid OHLC for {exchange}:{symbol} on {day}") from exc
                if (day_number not in session_set or currency != expected_currency
                        or type(volume) is not int or volume < 0
                        or not all(math.isfinite(value) and value > 0 for value in ohlc)
                        or ohlc[1] < max(ohlc) or ohlc[2] > min(ohlc)
                        or (observed_days and day_number <= observed_days[-1])):
                    raise ValueError(f"Invalid cleaned price/calendar/currency for {exchange}:{symbol} on {day}")
                key = (symbol_id, day_number)
                prices[key] = ohlc
                observed_days.append(day_number)
                if volume == 0:
                    zero_volume_keys.append([symbol_id, day_number])
            observed_set = set(observed_days)
            internal_missing = (sum(day not in observed_set for day in sessions
                                    if observed_days[0] <= day <= observed_days[-1])
                                if observed_days else 0)
            symbol_coverage.append({
                "symbol_id": symbol_id, "symbol": symbol, "exchange": exchange,
                "bars": len(observed_days), "missing_sessions": len(sessions) - len(observed_days),
                "internal_missing_sessions": internal_missing,
                "first": _day(observed_days[0])[1] if observed_days else None,
                "last": _day(observed_days[-1])[1] if observed_days else None,
            })
    except sqlite3.DatabaseError as exc:
        raise ValueError("Malformed cleaned price database") from exc
    finally:
        connection.close()
    metadata = {
        "database": str(database), "market": market, "start": start_text, "end": end_text,
        "scheduled_sessions": len(sessions), "symbols": len(validated_symbols),
        "stored_price_rows": len(prices),
        "missing_symbol_sessions": len(sessions) * len(validated_symbols) - len(prices),
        "internal_missing_symbol_sessions": sum(item["internal_missing_sessions"] for item in symbol_coverage),
        "zero_volume_keys": zero_volume_keys, "symbol_coverage": symbol_coverage,
        "source_metadata": source_metadata,
        "coverage_rule": "All retained clean bars for the training-available universe; not limited to approved signal targets",
        "missing_price_rule": "Absent keys remain missing; the loader does not fill or bridge them",
        "currency": expected_currency,
    }
    return prices, sessions, metadata
