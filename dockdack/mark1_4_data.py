"""Read-only, point-in-time *candidate* data for Mark1.4 research.

Selection is frozen at the last scheduled session on/before calibration_end.
The current Kiwoom catalog is not a historical security master, so even this
selection retains catalog survivorship bias. No broker, GUI or order code is
imported here, and the raw database is opened with SQLite query-only mode.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from contextlib import closing
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from dockdack.clean_daily_dataset import load_sessions, normalized_turnover, open_source
from dockdack.dataset_identity import classify_instrument
from dockdack.dataset_quality import QualityPolicy, assess_series
from dockdack.mark1_4_evolution import EvolutionSamples


LOOKBACK = 30


def _rows(db, symbol: str, exchange: str, first: str, last: str, market: str):
    rows = [dict(row) for row in db.execute(
        "SELECT trade_date AS date,open,high,low,close,volume,trade_value,currency "
        "FROM daily_bars WHERE symbol=? AND exchange=? AND trade_date>=? "
        "AND trade_date<=? ORDER BY trade_date",
        (symbol, exchange, first, last),
    )]
    for row in rows:
        row["liquidity_turnover"] = normalized_turnover(row["trade_value"], market)
    return rows


def _validate_calendar(sessions: tuple[str, ...], start: str, test_end: str,
                       calibration_end: str, train_end: str,
                       required_history: int):
    if not sessions or sessions != tuple(sorted(set(sessions))):
        raise ValueError("Session calendar must be nonempty, ordered and unique")
    if sessions[0] < start or sessions[-1] > test_end:
        raise ValueError("Session calendar falls outside requested dates")
    cutoff_index = bisect_right(sessions, calibration_end) - 1
    if cutoff_index < required_history - 1:
        raise ValueError(f"Calibration needs at least {required_history} scheduled sessions")
    if cutoff_index >= len(sessions) - 1 or sessions[-1] <= train_end:
        raise ValueError("Calendar must extend past calibration and train-end dates")
    return cutoff_index


def load_mark14_candidates(database: Path, market: str, *, start: str,
                           calibration_end: str, train_end: str, test_end: str,
                           max_symbols: int = 100, session_dates=None,
                           policy: QualityPolicy | None = None) -> EvolutionSamples:
    """Return original 30×OHLCV windows and *later* open/close outcomes.

    Only the calibration-period bars choose the fixed 100-symbol universe.
    Each later candidate uses t-only quality and 30 completed bars; a missing
    or zero-volume target t+1 is retained with NaN outcome, never silently
    removed from the signal denominator. Target ordinals are 0-based offsets
    into ``session_dates`` for chronological 30-session embargo splits.
    """
    if market not in {"domestic", "us"}:
        raise ValueError("Market must be domestic or us")
    if type(max_symbols) is not int or max_symbols < 1:
        raise ValueError("max_symbols must be a positive integer")
    first, cutoff, train_last, last = (
        date.fromisoformat(value) for value in
        (start, calibration_end, train_end, test_end)
    )
    if not first < cutoff < train_last < last:
        raise ValueError("Expected start < calibration_end < train_end < test_end")
    if policy is None:
        policy = QualityPolicy(min_median_turnover=(
            1_000_000_000 if market == "domestic" else 1_000_000))
    if policy.lookback != LOOKBACK:
        raise ValueError("Mark1.4 requires exactly 30 completed bars")
    if session_dates is None:
        session_dates, _ = load_sessions(market, first, last + timedelta(days=1))
    sessions = tuple(session_dates)
    cutoff_index = _validate_calendar(sessions, start, test_end,
                                      calibration_end, train_end,
                                      max(LOOKBACK, policy.liquidity_window))
    selection_day = sessions[cutoff_index]
    selection_sessions = sessions[:cutoff_index + 1]
    # Assess a bounded recent slice for selection. Sixty consecutive sessions
    # are sufficient for a fully mature t-only liquidity endpoint; including
    # 30 more makes data gaps explicit rather than linking across them.
    selection_first = sessions[max(0, cutoff_index - max(policy.liquidity_window, LOOKBACK) - 30)]
    session_index = {day: index for index, day in enumerate(sessions)}
    windows, dates, ordinals, symbols, opens, closes = [], [], [], [], [], []
    ranked, catalog_rows = [], 0
    missing_targets = 0
    with closing(open_source(database)) as db:
        db.execute("BEGIN")  # Stable read snapshot even if a collector is active.
        metadata = db.execute("SELECT value FROM metadata WHERE key='market'").fetchone()
        if metadata is None or metadata[0] != market:
            raise ValueError("Raw database market metadata does not match requested market")
        catalog = [dict(row) for row in db.execute("SELECT * FROM instruments")]
        catalog_rows = len(catalog)
        for item in catalog:
            status, _ = classify_instrument(item, market)
            if status != "eligible":
                continue
            symbol, exchange = item["symbol"], item["exchange"]
            rows = _rows(db, symbol, exchange, selection_first, selection_day, market)
            if not rows:
                continue
            assessment = assess_series(rows, market=market,
                                       as_of=cutoff + timedelta(days=1),
                                       session_dates=selection_sessions, policy=policy)
            endpoint = next((bar for bar in reversed(assessment.bars)
                             if bar.date == selection_day), None)
            if endpoint is None or not endpoint.input_eligible:
                continue
            ranked.append((endpoint.median_turnover, symbol, exchange,
                           endpoint.median_volume))
        # Descending liquidity, deterministic code tie-break. No later bar or
        # target price participates in universe selection.
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        chosen = ranked[:max_symbols]
        if len(chosen) < max_symbols:
            raise ValueError(f"Only {len(chosen)} calibration-eligible symbols; "
                             f"requested {max_symbols}")
        selected = []
        for _, symbol, exchange, _ in chosen:
            symbol_id = len(selected)
            rows = _rows(db, symbol, exchange, start, test_end, market)
            assessment = assess_series(rows, market=market,
                                       as_of=last + timedelta(days=1),
                                       session_dates=sessions, policy=policy)
            by_session = {session_index[bar.date]: bar for bar in assessment.bars
                          if bar.valid}
            before = len(windows)
            for bar in assessment.bars:
                if not bar.input_eligible:
                    continue
                t = session_index[bar.date]
                if t < cutoff_index or t + 1 >= len(sessions):
                    continue
                target_day = sessions[t + 1]
                if target_day > test_end:
                    continue
                history = [by_session.get(index)
                           for index in range(t - LOOKBACK + 1, t + 1)]
                if any(past is None for past in history):
                    continue
                target = by_session.get(t + 1)
                observed = target is not None and target.volume > 0
                windows.append(np.asarray(
                    [(past.open, past.high, past.low, past.close, past.volume)
                     for past in history], dtype=np.float32))
                dates.append(target_day)
                ordinals.append(t + 1)
                symbols.append(symbol_id)
                opens.append(target.open if observed else np.nan)
                closes.append(target.close if observed else np.nan)
                missing_targets += not observed
            selected.append({"symbol_id": symbol_id, "symbol": symbol,
                             "exchange": exchange,
                             "calibration_median_turnover": float(chosen[symbol_id][0]),
                             "calibration_median_volume": float(chosen[symbol_id][3]),
                             "preopen_candidates": len(windows) - before})
    if not windows:
        raise ValueError("No eligible post-calibration 30-bar candidates")
    if len(set(zip(symbols, dates))) != len(dates):
        raise ValueError("Duplicate symbol and target session in candidate data")
    values = np.stack(windows)
    if values.shape[1:] != (LOOKBACK, 5) or not np.isfinite(values).all():
        raise ValueError("Invalid causal 30-bar windows")
    return EvolutionSamples(
        windows=values,
        target_dates=np.asarray(dates, dtype="U10"),
        target_ordinals=np.asarray(ordinals, dtype=np.int32),
        symbol_ids=np.asarray(symbols, dtype=np.int32),
        entry_open=np.asarray(opens, dtype=np.float64),
        exit_close=np.asarray(closes, dtype=np.float64),
        source={"database": str(Path(database).resolve()), "market": market,
                "catalog_rows": catalog_rows,
                "calibration_start": start,
                "calibration_cutoff_requested": calibration_end,
                "calibration_cutoff_session": selection_day,
                "selection": "fixed liquid top N by t-only 20-session median turnover "
                             "on last scheduled calibration session",
                "calibration_eligible_symbols": len(ranked),
                "selected_symbols": selected,
                "eligible_preopen_candidates": len(dates),
                "unobserved_or_untradable_target": missing_targets,
                "target_ordinal_convention": "zero_based_session_index",
                "quality_policy": asdict(policy),
                "read_only": True,
                "research_only": True,
                "catalog_point_in_time": False},
    )


def mark14_chronological_splits(samples: EvolutionSamples, sessions, *,
                                train_start: str, train_end: str,
                                validation_end: str, test_end: str,
                                embargo_sessions: int = 30):
    """Return date-separated train, validation and test candidate indices.

    Samples whose target dates precede ``train_start`` are calibration-only.
    The embargo is measured on the shared *market session* calendar, not the
    number of samples; candidate gaps do not shorten it.
    """
    if type(embargo_sessions) is not int or embargo_sessions < LOOKBACK:
        raise ValueError("Embargo must span at least the 30-bar lookback")
    if not train_start <= train_end < validation_end < test_end:
        raise ValueError("Expected train_start <= train_end < validation_end < test_end")
    calendar = tuple(sessions)
    if not calendar or calendar != tuple(sorted(set(calendar))):
        raise ValueError("Session calendar must be ordered and unique")
    if len(samples.target_ordinals) != len(samples.target_dates):
        raise ValueError("Target ordinal/date arrays do not match")
    if np.any(samples.target_ordinals < 0) or np.any(samples.target_ordinals >= len(calendar)):
        raise ValueError("Target ordinal is outside the calendar")
    for ordinal, day in zip(samples.target_ordinals, samples.target_dates):
        if calendar[int(ordinal)] != day:
            raise ValueError("Target ordinal does not match target date")
    cut_train = bisect_right(calendar, train_end) - 1
    cut_validation = bisect_right(calendar, validation_end) - 1
    cut_test = bisect_right(calendar, test_end) - 1
    start_train = bisect_left(calendar, train_start)
    if (start_train > cut_train or cut_train < 0 or cut_validation >= len(calendar)
            or cut_test >= len(calendar)):
        raise ValueError("Split boundaries fall outside the calendar")
    targets = samples.target_ordinals
    splits = {
        "train": np.flatnonzero((targets >= start_train) & (targets <= cut_train)),
        "validation": np.flatnonzero((targets > cut_train + embargo_sessions)
                                     & (targets <= cut_validation)),
        "test": np.flatnonzero((targets > cut_validation + embargo_sessions)
                               & (targets <= cut_test)),
    }
    if any(not len(indices) for indices in splits.values()):
        raise ValueError("Empty chronological split after 30-session embargo")
    return splits
