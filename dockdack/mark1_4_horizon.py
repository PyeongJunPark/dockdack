"""Research-only fixed-horizon comparison for frozen Mark1.4 daily signals.

The signal sees only the 30 completed bars through session t.  Its existing
``target_ordinals`` identify the *next* session, t+1, whose OPEN is the
hypothetical entry.  Holding H sessions exits at the CLOSE at target ordinal
``+(H - 1)``.  All outcomes are loaded separately from the original SQLite DB;
this module neither changes the source DB nor connects to orders or the GUI.
"""

from __future__ import annotations

from contextlib import closing
from datetime import date, timedelta
import math
from pathlib import Path

import numpy as np

from dockdack.clean_daily_dataset import normalized_turnover, open_source
from dockdack.dataset_quality import QualityPolicy, assess_series
from dockdack.mark1_4_evolution import EvolutionSamples, simulate_portfolio


def _calendar_and_symbols(samples: EvolutionSamples, sessions) -> tuple[tuple[str, ...], dict[int, tuple[str, str]]]:
    calendar = tuple(sessions)
    if not calendar or calendar != tuple(sorted(set(calendar))):
        raise ValueError("sessions must be a nonempty, ordered, unique market calendar")
    try:
        if any(date.fromisoformat(day).isoformat() != day for day in calendar):
            raise ValueError("sessions must contain ISO dates")
    except (TypeError, ValueError) as exc:
        raise ValueError("sessions must contain ISO dates") from exc
    ordinals = np.asarray(samples.target_ordinals)
    if np.any(ordinals < 0) or np.any(ordinals >= len(calendar)):
        raise ValueError("target ordinal is outside the supplied calendar")
    if any(calendar[int(ordinal)] != day for ordinal, day in
           zip(ordinals, samples.target_dates)):
        raise ValueError("target ordinal and date do not match the calendar")
    source_symbols = samples.source.get("selected_symbols")
    if not isinstance(source_symbols, list):
        raise ValueError("source.selected_symbols is required for raw DB lookup")
    symbols: dict[int, tuple[str, str]] = {}
    for item in source_symbols:
        if not isinstance(item, dict):
            raise ValueError("selected symbol entries must be dictionaries")
        symbol_id = item.get("symbol_id")
        symbol, exchange = item.get("symbol"), item.get("exchange")
        if (type(symbol_id) is not int or symbol_id < 0 or symbol_id in symbols
                or not isinstance(symbol, str) or not symbol
                or not isinstance(exchange, str) or not exchange):
            raise ValueError("invalid or duplicate selected symbol identity")
        symbols[symbol_id] = (symbol, exchange)
    if any(int(symbol_id) not in symbols for symbol_id in samples.symbol_ids):
        raise ValueError("sample symbol has no selected-symbol identity")
    if len(set(symbols.values())) != len(symbols):
        raise ValueError("selected symbols must be unique by symbol and exchange")
    return calendar, symbols


def load_horizon_closes(database: Path, samples: EvolutionSamples, sessions,
                        horizons=(3, 5)) -> dict[int, np.ndarray]:
    """Read aligned Hth-session closes from the raw DB, leaving gaps as NaN.

    A missing calendar session, hard-invalid/zero-volume bar or trailing exit
    outside ``sessions`` is *not* forward-filled.  The same hard validity rules
    as Mark1.4 candidate loading are used for exit bars.  The source opens in
    SQLite read-only/query-only mode inside one stable read transaction.
    """
    requested = tuple(horizons)
    if not requested or any(type(value) is not int or value < 1 for value in requested):
        raise ValueError("horizons must contain positive integer session counts")
    if len(set(requested)) != len(requested):
        raise ValueError("horizons must be unique")
    calendar, symbols = _calendar_and_symbols(samples, sessions)
    market = samples.source.get("market")
    if market not in {"domestic", "us"}:
        raise ValueError("source.market must be domestic or us")
    selected = {horizon: np.full(len(samples.windows), np.nan, dtype=np.float64)
                for horizon in requested}
    if 1 in selected:
        selected[1][:] = np.asarray(samples.exit_close, dtype=np.float64)
    if all(horizon == 1 for horizon in requested):
        return selected
    target_ordinals = np.asarray(samples.target_ordinals, dtype=np.int64)
    last_exit = min(len(calendar) - 1,
                    int(target_ordinals.max()) + max(requested) - 1)
    first_date = calendar[int(target_ordinals.min())]
    last_date = calendar[last_exit]
    policy_data = samples.source.get("quality_policy")
    if isinstance(policy_data, dict):
        policy = QualityPolicy(**policy_data)
    else:
        policy = QualityPolicy(min_median_turnover=0)
    raw_path = Path(database).resolve(strict=True)
    declared = samples.source.get("database")
    if declared is not None and Path(declared).resolve(strict=True) != raw_path:
        raise ValueError("database does not match the candidate source")
    by_symbol = {symbol_id: np.flatnonzero(np.asarray(samples.symbol_ids) == symbol_id)
                 for symbol_id in symbols}
    with closing(open_source(raw_path)) as db:
        db.execute("BEGIN")
        metadata = db.execute("SELECT value FROM metadata WHERE key='market'").fetchone()
        if metadata is None or metadata[0] != market:
            raise ValueError("raw database market does not match the candidate source")
        for symbol_id, (symbol, exchange) in symbols.items():
            sample_rows = by_symbol[symbol_id]
            if not len(sample_rows):
                continue
            raw = [dict(row) for row in db.execute(
                "SELECT trade_date AS date,open,high,low,close,volume,trade_value,currency "
                "FROM daily_bars WHERE symbol=? AND exchange=? AND trade_date>=? "
                "AND trade_date<=? ORDER BY trade_date",
                (symbol, exchange, first_date, last_date),
            )]
            for bar in raw:
                bar["liquidity_turnover"] = normalized_turnover(bar["trade_value"], market)
            assessed = assess_series(raw, market=market,
                                     as_of=date.fromisoformat(last_date) + timedelta(days=1),
                                     session_dates=calendar, policy=policy)
            valid_closes = {bar.date: float(bar.close) for bar in assessed.bars
                            if bar.valid and bar.volume is not None and bar.volume > 0}
            for horizon in requested:
                if horizon == 1:
                    continue
                exit_ordinals = target_ordinals[sample_rows] + horizon - 1
                in_calendar = exit_ordinals < len(calendar)
                aligned = selected[horizon]
                for row, ordinal in zip(sample_rows[in_calendar], exit_ordinals[in_calendar]):
                    aligned[row] = valid_closes.get(calendar[int(ordinal)], np.nan)
    return selected


def _eligible_entry_rows(samples: EvolutionSamples, indices: np.ndarray, *, horizon: int,
                         entry_stride: int | None, split_start_ordinal: int | None,
                         split_end_ordinal: int | None):
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if entry_stride is None:
        entry_stride = horizon
    if type(entry_stride) is not int or entry_stride < horizon:
        raise ValueError("entry_stride must be an integer >= horizon to prevent overlap")
    rows = np.asarray(indices, dtype=np.int64)
    if (rows.ndim != 1 or not len(rows) or np.any(rows < 0)
            or np.any(rows >= len(samples.windows)) or len(np.unique(rows)) != len(rows)):
        raise ValueError("indices must be a nonempty, unique sample subset")
    ordinals = np.asarray(samples.target_ordinals, dtype=np.int64)
    first = int(ordinals[rows].min()) if split_start_ordinal is None else split_start_ordinal
    last = None if split_end_ordinal is None else split_end_ordinal
    if type(first) is not int or first < 0 or (last is not None and
                                               (type(last) is not int or last < first)):
        raise ValueError("invalid split boundaries")
    if np.any(ordinals[rows] < first) or (last is not None and np.any(ordinals[rows] > last)):
        raise ValueError("indices fall outside declared split boundaries")
    candidate_days = np.unique(ordinals[rows])
    stride_days = candidate_days[(candidate_days - first) % entry_stride == 0]
    trailing = np.empty(0, dtype=np.int64)
    if last is not None:
        trailing = stride_days[stride_days + horizon - 1 > last]
        stride_days = stride_days[stride_days + horizon - 1 <= last]
    if not len(stride_days):
        raise ValueError("no eligible entry sessions under cadence and split boundary")
    mask = np.isin(ordinals[rows], stride_days)
    return rows[mask], stride_days, trailing, first, last, entry_stride


def simulate_fixed_horizon(samples: EvolutionSamples, indices: np.ndarray,
                           scores: np.ndarray, exit_close: np.ndarray, *, horizon: int,
                           threshold: float = 0.0, cost_bps: float = 20.0,
                           allocation: float = 0.1, max_positions: int = 10,
                           initial_equity: float = 10_000_000.0,
                           entry_stride: int | None = None,
                           split_start_ordinal: int | None = None,
                           split_end_ordinal: int | None = None) -> dict:
    """Cash-only nonoverlapping H-session hypothetical roundtrip portfolio.

    The entry signal and ranking are frozen before the target session.  At an
    eligible session's OPEN, each chosen name receives at most ``allocation``
    of start equity using integer shares.  All positions exit at that day's
    ``+(horizon-1)`` session CLOSE.  No intermediate reinvestment, leverage or
    stop/limit behavior is inferred from daily OHLC.  Drawdown is measured at
    realized exit equity points only, not at interim mark-to-market prices.
    """
    rows, days, trailing, first, last, stride = _eligible_entry_rows(
        samples, indices, horizon=horizon, entry_stride=entry_stride,
        split_start_ordinal=split_start_ordinal, split_end_ordinal=split_end_ordinal)
    score_values = np.asarray(scores, dtype=np.float64)
    exits = np.asarray(exit_close, dtype=np.float64)
    if (score_values.shape != (len(samples.windows),)
            or exits.shape != (len(samples.windows),)
            or not np.isfinite(score_values[rows]).all()):
        raise ValueError("scores and exit_close must align with all candidates")
    if not math.isfinite(threshold) or not math.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("threshold and roundtrip cost must be finite and cost nonnegative")
    if (not math.isfinite(allocation) or not 0 < allocation <= 1
            or type(max_positions) is not int or max_positions < 1
            or max_positions * allocation > 1 + 1e-12):
        raise ValueError("allocation and position cap must not exceed total equity")
    if not math.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial_equity must be finite and positive")
    ordinals = np.asarray(samples.target_ordinals, dtype=np.int64)
    symbols = np.asarray(samples.symbol_ids, dtype=np.int64)
    entry_opens = np.asarray(samples.entry_open, dtype=np.float64)
    half = cost_bps / 20_000
    equity = float(initial_equity)
    peak = equity
    drawdown_at_exits = 0.0
    exact = True
    signals = observed_signals = executed_trades = active_entries = unresolved_entries = 0
    block_returns: list[float | None] = []
    entry_signal_counts: list[int] = []
    entry_selected_symbol_ids: list[list[int]] = []
    drawdowns_at_exit_points: list[float | None] = []
    entry_ordinals: list[int] = []
    exit_ordinals: list[int] = []
    entry_dates: list[str] = []
    for day in days:
        day_rows = rows[ordinals[rows] == day]
        if not np.all(samples.target_dates[day_rows] == samples.target_dates[day_rows[0]]):
            raise ValueError("one target ordinal must identify one date")
        entry_ordinals.append(int(day))
        exit_ordinals.append(int(day + horizon - 1))
        entry_dates.append(str(samples.target_dates[day_rows[0]]))
        eligible = day_rows[score_values[day_rows] > threshold]
        ranked = eligible[np.lexsort((symbols[eligible], -score_values[eligible]))]
        chosen = ranked[:max_positions]
        entry_signal_counts.append(len(chosen))
        entry_selected_symbol_ids.append([int(symbol) for symbol in symbols[chosen]])
        if len(np.unique(symbols[chosen])) != len(chosen):
            raise ValueError("duplicate selected symbol on one entry session")
        signals += len(chosen)
        active_entries += bool(len(chosen))
        valid = (np.isfinite(entry_opens[chosen]) & np.isfinite(exits[chosen])
                 & (entry_opens[chosen] > 0) & (exits[chosen] > 0))
        observed_signals += int(valid.sum())
        if not valid.all():
            unresolved_entries += 1
            exact = False
            block_returns.append(None)
            drawdowns_at_exit_points.append(None)
            continue
        if not exact:
            block_returns.append(None)
            drawdowns_at_exit_points.append(None)
            continue
        gross_buy = entry_opens[chosen] * (1 + half)
        net_sale = exits[chosen] * (1 - half)
        quantities = np.floor(allocation * equity / gross_buy).astype(np.int64)
        executed_trades += int(np.count_nonzero(quantities))
        debit = float(np.dot(quantities, gross_buy))
        if debit > equity + 1e-7:
            raise AssertionError("position sizing exceeded cash equity")
        starting_equity = equity
        equity += float(np.dot(quantities, net_sale - gross_buy))
        if equity < -1e-7:
            raise AssertionError("cash-only strategy produced negative equity")
        block_returns.append(equity / starting_equity - 1)
        peak = max(peak, equity)
        drawdown_at_exits = min(drawdown_at_exits, equity / peak - 1)
        drawdowns_at_exit_points.append(equity / peak - 1)
    return {
        "horizon_sessions": horizon,
        "entry_stride_sessions": stride,
        "entry_sessions": len(days),
        "ineligible_trailing_entry_sessions": len(trailing),
        "split_start_ordinal": first,
        "split_end_ordinal": last,
        "entry_ordinals": entry_ordinals,
        "exit_ordinals": exit_ordinals,
        "entry_dates": entry_dates,
        "entry_signal_counts": entry_signal_counts,
        "entry_selected_symbol_ids": entry_selected_symbol_ids,
        "signals": signals,
        "observed_signals": observed_signals,
        "unresolved_signals": signals - observed_signals,
        "active_entry_sessions": active_entries,
        "unresolved_entry_sessions": unresolved_entries,
        "executed_trades": executed_trades if exact else None,
        "executed_trades_on_exact_prefix": executed_trades,
        "realized_block_returns": block_returns,
        "drawdowns_at_exit_points": drawdowns_at_exit_points,
        "final_equity": equity if exact else None,
        "compound_net_return": equity / initial_equity - 1 if exact else None,
        "max_drawdown_at_exits_only": drawdown_at_exits if exact else None,
        "incomplete_data": not exact,
        "initial_equity": initial_equity,
        "cost_bps": cost_bps,
        "allocation": allocation,
        "max_positions": max_positions,
        "entry_exit": "hypothetical next-session open and Hth-session close; fills unverified",
        "sizing": "integer shares, floor(allocation * start equity / entry cost); cash only",
        "interim_mark_to_market": False,
    }


def compare_fixed_horizon(samples: EvolutionSamples, indices: np.ndarray,
                          scores: np.ndarray, exit_close: np.ndarray, *, horizon: int,
                          threshold: float = 0.0, cost_bps: float = 20.0,
                          allocation: float = 0.1, max_positions: int = 10,
                          initial_equity: float = 10_000_000.0,
                          split_start_ordinal: int | None = None,
                          split_end_ordinal: int | None = None) -> dict:
    """Compare H-session hold versus one-day exit at exactly the same entries.

    Only full-horizon entries within the declared split are admitted to both
    arms.  A non-observed selected exit remains unresolved rather than being
    silently dropped or replaced with a later bar.  The one-day arm is cash
    during the remaining H-1 sessions; score threshold and rank are identical.
    """
    subset = np.asarray(indices, dtype=np.int64)
    if subset.ndim != 1 or not len(subset):
        raise ValueError("indices must be a nonempty sample subset")
    ordinals = np.asarray(samples.target_ordinals, dtype=np.int64)
    boundary = int(ordinals[subset].max()) if split_end_ordinal is None else split_end_ordinal
    eligible, days, trailing, start, _, _ = _eligible_entry_rows(
        samples, subset, horizon=horizon, entry_stride=horizon,
        split_start_ordinal=split_start_ordinal, split_end_ordinal=boundary)
    one_day = simulate_portfolio(
        samples, eligible, scores, threshold=threshold, cost_bps=cost_bps,
        allocation=allocation, max_positions=max_positions, initial_equity=initial_equity)
    held = simulate_fixed_horizon(
        samples, eligible, scores, exit_close, horizon=horizon,
        threshold=threshold, cost_bps=cost_bps, allocation=allocation,
        max_positions=max_positions, initial_equity=initial_equity,
        entry_stride=horizon, split_start_ordinal=start,
        split_end_ordinal=boundary)
    if held["entry_ordinals"] != [int(day) for day in days] or one_day["signals"] != held["signals"]:
        raise AssertionError("comparison arms did not use identical entry signals")
    return {
        "horizon_sessions": horizon,
        "entry_stride_sessions": horizon,
        "entry_ordinals": [int(day) for day in days],
        "ineligible_trailing_entry_sessions": len(trailing),
        "split_start_ordinal": start,
        "split_end_ordinal": boundary,
        "one_day": one_day,
        "multi_day": held,
        "same_entry_signals": True,
        "research_only": True,
    }
