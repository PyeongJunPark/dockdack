"""Isolated configurable-barrier accounting for mark_1 +0.5%/-0.4% research.

This is a separate copy of the proven daily portfolio engine: old source,
models and backtests are not modified. Explicit barrier arguments permit a
matched old-target comparison without changing any module globals.

This is an idealized daily-bar simulation, not evidence of executable fills.
Both barriers in one bar mean STOP FIRST. No same-day exit proceeds or slots
are available to other OPEN entries. Fee assumptions are explicit. Missing
held-symbol prices lock capital and use stale marks until an observed OPEN
permits liquidation. All affected trades and portfolio metrics are uncertain.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Integral, Real

TAKE_PROFIT = .005
STOP_LOSS = .004
TOLERANCE = 1e-12


def _integer(value, name: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _number(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _ohlc(value) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError("Price rows must contain OPEN,HIGH,LOW,CLOSE")
    o, h, l, c = (_number(x, "OHLC", positive=True) for x in value)
    if h < max(o, l, c) or l > min(o, h, c):
        raise ValueError("Inconsistent OHLC price range")
    return o, h, l, c


@dataclass
class _Position:
    symbol_id: int
    entry_date: int
    entry_index: int
    entry_price: float
    quantity: int
    entry_fee: float
    probability: float
    liquidity_shares: float
    last_close: float
    last_observed_date: int
    path_uncertain: bool = False


def simulate_portfolio(candidates, prices, sessions, *, initial_cash,
                       exit_mode="carry", max_positions=20,
                       position_fraction=.05, cost_bps=20,
                       volume_fraction=.001, take_profit=TAKE_PROFIT,
                       stop_loss=STOP_LOSS) -> dict:
    """Simulate a single-currency, long-only, unlevered daily-bar portfolio.

    ``candidates``: iterable of dictionaries containing integer ``symbol_id``,
    epoch-day ``date``, ``probability``, and backward-looking ``liquidity_shares``.
    ``prices``: mapping (symbol_id, epoch-day) -> (open, high, low, close).
    ``sessions``: strictly increasing exchange-session epoch days. All bars are
    validated, but price rows outside the session range are allowed.

    Entries at actual OPEN are ranked by decreasing probability then symbol ID;
    exactly .5 does not buy. Each target is position_fraction of start-of-day
    OPEN-marked equity, capped by cash, available slots, and integer
    floor(liquidity_shares * volume_fraction). Position notional excludes fees.
    ``take_profit`` and ``stop_loss`` are positive fractions, defaulting to
    +0.005/-0.004; stop_loss must be below one. They are per-call values and do
    not mutate the old engine or any concurrent comparison.
    ``cost_bps`` is a roundtrip *rate*, half applied to each side's actual
    notional (therefore currency fees vary with exit price). No spread, market
    impact, price limits, auction delay, or partial fills are modeled.

    Prior holdings gap beyond a barrier exit at OPEN, before new entries. A
    symbol that exits cannot reenter that day. Other intraday exits are resolved
    only after all OPEN purchases. In carry mode neither-hit holdings persist;
    eod mode liquidates them at CLOSE. Missing held-symbol bars keep capital
    locked and mark to last known CLOSE; resumption forces OPEN liquidation,
    because intervening barrier touches are unknown. Terminal still-missing
    holdings receive explicitly uncertain last-price accounting liquidation.
    The last supplied session liquidates observed holdings at CLOSE.

    Returns JSON-native summary, daily equity (with initial anchor), trades, and
    rejection counts plus up to 10,000 eligible-signal rejection examples.
    """
    take_profit = _number(take_profit, "take_profit", positive=True)
    stop_loss = _number(stop_loss, "stop_loss", positive=True)
    if stop_loss >= 1:
        raise ValueError("stop_loss must be below one")
    cash = _number(initial_cash, "initial_cash", positive=True)
    initial_cash = cash
    max_positions = _integer(max_positions, "max_positions", nonnegative=True)
    if max_positions == 0:
        raise ValueError("max_positions must be positive")
    position_fraction = _number(position_fraction, "position_fraction", positive=True)
    cost_bps = _number(cost_bps, "cost_bps")
    volume_fraction = _number(volume_fraction, "volume_fraction", positive=True)
    if position_fraction > 1 or volume_fraction > 1 or not 0 <= cost_bps < 20_000:
        raise ValueError("Fractions must be at most one; cost_bps in [0,20000)")
    if exit_mode not in ("carry", "eod"):
        raise ValueError("exit_mode must be carry or eod")
    sessions = [_integer(day, "session date") for day in sessions]
    if not sessions or any(b <= a for a, b in zip(sessions, sessions[1:])):
        raise ValueError("sessions must be nonempty and strictly increasing")
    session_set = set(sessions)
    for key, value in prices.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Price keys must be (symbol_id,date)")
        _integer(key[0], "price symbol_id", nonnegative=True)
        _integer(key[1], "price date")
        _ohlc(value)

    by_day = defaultdict(list)
    seen = set()
    rejected = Counter()
    rejection_examples = []
    candidate_count = 0
    eligible_count = 0

    def reject(candidate, reason):
        rejected[reason] += 1
        if reason != "BELOW_THRESHOLD" and len(rejection_examples) < 10_000:
            rejection_examples.append({"date": candidate["date"],
                                       "symbol_id": candidate["symbol_id"],
                                       "probability": candidate["probability"],
                                       "reason": reason})

    for row in candidates:
        symbol = _integer(row["symbol_id"], "candidate symbol_id", nonnegative=True)
        date = _integer(row["date"], "candidate date")
        probability = _number(row["probability"], "probability")
        liquidity = _number(row["liquidity_shares"], "liquidity_shares")
        if not 0 <= probability <= 1 or liquidity < 0 or date not in session_set:
            raise ValueError("Invalid candidate probability, liquidity, or session date")
        key = (symbol, date)
        if key in seen:
            raise ValueError("Duplicate candidate symbol/date")
        seen.add(key)
        candidate_count += 1
        normalized = {"symbol_id": symbol, "date": date,
                      "probability": probability, "liquidity_shares": liquidity}
        if probability <= .5:
            reject(normalized, "BELOW_THRESHOLD")
            continue
        eligible_count += 1
        by_day[date].append(normalized)

    fee_rate = cost_bps / 20_000
    positions: dict[int, _Position] = {}
    trades = []
    equity = [{"date": sessions[0] - 1, "equity": cash, "cash": cash,
               "market_value": 0., "exposure": 0., "position_count": 0,
               "stale_position_count": 0, "fully_observed_to_date": True,
               "daily_return": 0., "is_initial": True}]
    entry_fees = exit_fees = 0.
    intraday_exposures = []
    exit_counts = Counter()
    gap_valuation_days = 0
    gap_calendar_days = set()

    def sell(symbol, date, index, price, reason, *, both_touch=False):
        nonlocal cash, exit_fees
        position = positions.pop(symbol)
        notional = position.quantity * price
        exit_fee = notional * fee_rate
        entry_notional = position.quantity * position.entry_price
        fees = position.entry_fee + exit_fee
        gross_pnl = notional - entry_notional
        net_pnl = gross_pnl - fees
        cash += notional - exit_fee
        exit_fees += exit_fee
        exit_counts[reason] += 1
        trades.append({"symbol_id": symbol, "entry_date": position.entry_date,
                       "exit_date": date, "entry_price": position.entry_price,
                       "exit_price": price, "quantity": position.quantity,
                       "probability": position.probability,
                       "liquidity_shares": position.liquidity_shares,
                       "holding_days": index - position.entry_index + 1,
                       "gross_pnl": gross_pnl, "net_pnl": net_pnl,
                       "gross_return": price / position.entry_price - 1,
                       "net_return": net_pnl / (entry_notional + position.entry_fee),
                       "entry_fee": position.entry_fee, "exit_fee": exit_fee,
                       "fees": fees, "exit_reason": reason,
                       "both_touch": bool(both_touch),
                       "path_uncertain": position.path_uncertain,
                       "exit_price_date": position.last_observed_date if reason == "END_STALE_MARK" else date})

    for index, date in enumerate(sessions):
        exited_today = set()
        # Mark all opening holdings before any gap exit or new allocation.
        start_equity = cash + sum(p.quantity * (float(prices[(s, date)][0]) if (s, date) in prices else p.last_close)
                                  for s, p in positions.items())
        for symbol, position in list(positions.items()):
            if (symbol, date) not in prices:
                position.path_uncertain = True
                gap_valuation_days += 1
                gap_calendar_days.add(date)
                continue
            opened = float(prices[(symbol, date)][0])
            if position.path_uncertain:
                sell(symbol, date, index, opened, "DATA_GAP_RESUMPTION")
                exited_today.add(symbol)
                continue
            take = position.entry_price * (1 + take_profit)
            stop = position.entry_price * (1 - stop_loss)
            if opened <= stop * (1 + TOLERANCE):
                sell(symbol, date, index, opened, "GAP_STOP")
                exited_today.add(symbol)
            elif opened >= take * (1 - TOLERANCE):
                sell(symbol, date, index, opened, "GAP_TAKE")
                exited_today.add(symbol)

        target_notional = start_equity * position_fraction
        for row in sorted(by_day.get(date, ()), key=lambda r: (-r["probability"], r["symbol_id"])):
            symbol = row["symbol_id"]
            if symbol in exited_today:
                reject(row, "ALREADY_EXITED_TODAY")
                continue
            if symbol in positions:
                reject(row, "ALREADY_HELD")
                continue
            if len(positions) >= max_positions:
                reject(row, "POSITION_CAP")
                continue
            bar = prices.get((symbol, date))
            if bar is None:
                reject(row, "MISSING_PRICE")
                continue
            opened = float(bar[0])
            liquidity_limit = math.floor(row["liquidity_shares"] * volume_fraction)
            if liquidity_limit == 0:
                reject(row, "LIQUIDITY_CAP")
                continue
            target_limit = math.floor(target_notional / opened)
            if target_limit == 0:
                reject(row, "POSITION_TOO_SMALL")
                continue
            cash_limit = math.floor(cash / (opened * (1 + fee_rate)))
            quantity = min(target_limit, liquidity_limit, cash_limit)
            if quantity <= 0:
                reject(row, "INSUFFICIENT_CASH")
                continue
            notional = quantity * opened
            fee = notional * fee_rate
            cash -= notional + fee
            # Floating-point subtraction may leave a tiny negative residue only.
            if cash < -max(1., initial_cash) * 1e-12:
                raise AssertionError("Backtest cash became negative")
            cash = max(cash, 0.)
            entry_fees += fee
            positions[symbol] = _Position(symbol, date, index, opened, quantity, fee,
                                           row["probability"], row["liquidity_shares"], opened, date)

        opening_value = sum(p.quantity * (float(prices[(s, date)][0]) if (s, date) in prices else p.last_close)
                            for s, p in positions.items())
        opening_equity = cash + opening_value
        intraday_exposures.append(opening_value / opening_equity if opening_equity else 0.)
        for symbol, position in list(positions.items()):
            if (symbol, date) not in prices:
                if index == len(sessions) - 1:
                    sell(symbol, date, index, position.last_close, "END_STALE_MARK")
                continue
            _, high, low, close = (float(x) for x in prices[(symbol, date)])
            position.last_close = close
            position.last_observed_date = date
            take = position.entry_price * (1 + take_profit)
            stop = position.entry_price * (1 - stop_loss)
            take_hit = high >= take * (1 - TOLERANCE)
            stop_hit = low <= stop * (1 + TOLERANCE)
            if stop_hit:
                sell(symbol, date, index, stop, "STOP", both_touch=take_hit)
            elif take_hit:
                sell(symbol, date, index, take, "TAKE")
            elif exit_mode == "eod":
                sell(symbol, date, index, close, "EOD_CLOSE")
            elif index == len(sessions) - 1:
                sell(symbol, date, index, close, "END_OF_DATA")

        market_value = sum(p.quantity * p.last_close for p in positions.values())
        total = cash + market_value
        previous = equity[-1]["equity"]
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Portfolio accounting overflow or nonpositive equity")
        equity.append({"date": date, "equity": total, "cash": cash,
                       "market_value": market_value,
                       "exposure": market_value / total,
                       "position_count": len(positions),
                       "stale_position_count": sum((s, date) not in prices for s in positions),
                       "fully_observed_to_date": gap_valuation_days == 0,
                       "daily_return": total / previous - 1,
                       "is_initial": False})

    returns = [row["daily_return"] for row in equity[1:]]
    mean_return = math.fsum(returns) / len(returns)
    variance = (math.fsum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
                if len(returns) > 1 else 0.)
    sharpe = mean_return / math.sqrt(variance) * math.sqrt(252) if variance > 0 else None
    peak, max_drawdown = initial_cash, 0.
    for row in equity:
        peak = max(peak, row["equity"])
        max_drawdown = min(max_drawdown, row["equity"] / peak - 1)
    wins = [trade["net_pnl"] for trade in trades if trade["net_pnl"] > 0]
    losses = [trade["net_pnl"] for trade in trades if trade["net_pnl"] < 0]
    total_return = equity[-1]["equity"] / initial_cash - 1
    annualized_exponent = math.log(equity[-1]["equity"] / initial_cash) * 252 / len(sessions)
    annualized = math.expm1(annualized_exponent) if annualized_exponent < 709 else None
    summary = {"initial_cash": initial_cash, "final_equity": equity[-1]["equity"],
               "total_return": total_return, "annualized_return": annualized,
               "max_drawdown": max_drawdown, "sharpe": sharpe,
               "trade_count": len(trades), "win_rate": len(wins) / len(trades) if trades else None,
               "profit_factor": math.fsum(wins) / abs(math.fsum(losses)) if losses else None,
               "avg_net_return": math.fsum(t["net_return"] for t in trades) / len(trades) if trades else None,
               "gross_pnl": math.fsum(t["gross_pnl"] for t in trades),
               "net_pnl": math.fsum(t["net_pnl"] for t in trades),
               "fees": entry_fees + exit_fees, "entry_fees": entry_fees,
               "exit_fees": exit_fees, "mean_exposure": math.fsum(intraday_exposures) / len(sessions),
               "mean_close_exposure": math.fsum(r["exposure"] for r in equity[1:]) / len(sessions),
               "max_open_exposure": max(intraday_exposures),
               "forced_gap_count": exit_counts["DATA_GAP_RESUMPTION"],
               "stale_terminal_count": exit_counts["END_STALE_MARK"],
               "gap_valuation_days": gap_valuation_days,
               "gap_calendar_days": len(gap_calendar_days),
               "uncertain_trades": sum(t["path_uncertain"] for t in trades),
               "fully_observed": gap_valuation_days == 0,
               "end_of_data_count": exit_counts["END_OF_DATA"],
               "closing_counts": dict(exit_counts), "sessions": len(sessions),
               "start_date": sessions[0], "end_date": sessions[-1],
               "candidate_count": candidate_count, "eligible_signal_count": eligible_count,
               "rejected_count": sum(rejected.values()),
               "both_touch_stop_count": sum(t["both_touch"] for t in trades),
               "exit_mode": exit_mode, "cost_bps": cost_bps,
               "take_profit": take_profit, "stop_loss": stop_loss,
               "max_positions": max_positions, "position_fraction": position_fraction,
               "volume_fraction": volume_fraction,
               "assumptions": ["Actual OPEN entries; signal > 0.5; both touches STOP FIRST",
                               "Opening gaps execute at OPEN; intraday exits never finance OPEN entries",
                               "One entry per symbol/day; integer shares; long only; no leverage",
                               "Liquidity cap uses supplied past-only share volume",
                               "Roundtrip bps split equally across actual entry/exit notionals",
                               "No bid/ask spread, market impact, partial fills, or price-limit restrictions",
                               "Missing held prices lock capital and use stale marks; resume at next observed OPEN",
                               "Missing-price paths and terminal stale-price liquidation are uncertain accounting, not observed execution",
                               "Last session forced CLOSE exit is an evaluation convention",
                               "Annualization and Sharpe use 252 sessions/year, zero risk-free rate"]}
    reconciliation = initial_cash + summary["net_pnl"] - equity[-1]["equity"]
    if abs(reconciliation) > max(1., initial_cash, equity[-1]["equity"]) * 1e-9:
        raise AssertionError("Portfolio cash and trade P&L do not reconcile")
    return {"summary": summary, "equity": equity, "trades": trades,
            "rejections": {"counts": dict(rejected), "examples": rejection_examples,
                           "examples_truncated": max(0, sum(v for k, v in rejected.items()
                                                          if k != "BELOW_THRESHOLD") - len(rejection_examples))}}


