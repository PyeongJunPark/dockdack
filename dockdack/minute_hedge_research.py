"""Isolated minute-bar research for long stock / explicit market hedge.

This module cannot place an order.  A synthetic short is only a paper index
cash-flow.  Alternatively, real inverse-ETF minute bars can price a *paper*
long-ETF hedge; that is not equivalent to a constant short-index position.
Bars must be completed, genuine minute OHLCV supplied by the caller.  Broker
time labels are retained without assuming start/end semantics.  Because a
label may denote the *end* of a candle and the adapter finalizes it only
after another full interval, paper fills are delayed to the third later
candle's open.  No performance is asserted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from math import fsum, isfinite, log, sqrt
from typing import Sequence


EXECUTION_DELAY_BARS = 3


def _finite(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


@dataclass(frozen=True, slots=True)
class MinuteBar:
    time: datetime  # Broker bar label; not necessarily the execution timestamp.
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Actual minute turnover in the stock's quote currency, if available.
    # The VWAP variant abstains when it is missing; OHLCV cannot recover it.
    turnover: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.time, datetime) or self.time.tzinfo is None or self.time.utcoffset() is None:
            raise ValueError("minute bar time must be timezone-aware")
        if self.time.second or self.time.microsecond:
            raise ValueError("minute bar time must be aligned to a whole minute")
        for name in ("open", "high", "low", "close"):
            _finite(getattr(self, name), name, positive=True)
        volume = _finite(self.volume, "volume")
        if volume < 0 or not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError("invalid minute OHLCV")
        if self.turnover is not None:
            turnover = _finite(self.turnover, "turnover")
            if turnover < 0 or (volume == 0) != (turnover == 0):
                raise ValueError("minute turnover and volume disagree")


@dataclass(frozen=True, slots=True)
class MinuteSession:
    """One contiguous exchange session for one stock and one market index."""

    stock_id: str
    benchmark_index_id: str
    stock_currency: str
    interval_minutes: int
    stock: tuple[MinuteBar, ...]
    benchmark_index: tuple[MinuteBar, ...]
    source: str
    sector_id: str | None = None
    sector: tuple[MinuteBar, ...] | None = None
    inverse_etf_id: str | None = None
    inverse_etf_currency: str | None = None
    inverse_etf_index_multiple: float | None = None
    inverse_etf: tuple[MinuteBar, ...] | None = None
    previous_close_day: date | None = None
    previous_stock_close: float | None = None
    previous_index_close: float | None = None

    def __post_init__(self) -> None:
        if (not self.stock_id.strip() or not self.benchmark_index_id.strip()
                or not self.stock_currency.strip() or not self.source.strip()):
            raise ValueError("stock, currency, benchmark index, and minute data source are required")
        if self.stock_id == self.benchmark_index_id or not self.stock or len(self.stock) != len(self.benchmark_index):
            raise ValueError("stock and market-index minute series must be distinct and aligned")
        if type(self.interval_minutes) is not int or self.interval_minutes not in {1, 3, 5, 10, 15, 30, 45, 60}:
            raise ValueError("supported minute-bar interval required")
        if (self.sector is None) != (self.sector_id is None):
            raise ValueError("sector identity and minute series must be supplied together")
        if self.sector is not None and (not self.sector_id.strip() or len(self.sector) != len(self.stock)):
            raise ValueError("sector minute series must be aligned")
        if (self.inverse_etf is None) != (self.inverse_etf_id is None) or (
                self.inverse_etf is None) != (self.inverse_etf_currency is None) or (
                self.inverse_etf is None) != (self.inverse_etf_index_multiple is None):
            raise ValueError("inverse ETF identity, currency, index multiple, and minute series must be supplied together")
        if self.inverse_etf is not None and (
                not self.inverse_etf_id.strip() or self.inverse_etf_id in {self.stock_id, self.benchmark_index_id}
                or self.inverse_etf_currency != self.stock_currency or len(self.inverse_etf) != len(self.stock)):
            raise ValueError("inverse ETF must share stock currency and have distinct aligned minute bars")
        if self.inverse_etf_index_multiple is not None and (
                _finite(self.inverse_etf_index_multiple, "inverse_etf_index_multiple") != -1):
            raise ValueError("this paper hedge supports only an explicitly verified -1x inverse ETF")
        first = self.stock[0].time
        for index, bar in enumerate(self.stock):
            expected = first + timedelta(minutes=index * self.interval_minutes)
            if bar.time != expected or bar.time.date() != first.date():
                raise ValueError("minute series must be contiguous within one local session")
            if self.benchmark_index[index].time != expected or (
                    self.sector is not None and self.sector[index].time != expected) or (
                    self.inverse_etf is not None and self.inverse_etf[index].time != expected):
                raise ValueError("stock, market index, sector, and inverse ETF minute times must match")
        for name in ("previous_stock_close", "previous_index_close"):
            if getattr(self, name) is not None:
                _finite(getattr(self, name), name, positive=True)
        if self.previous_close_day is not None and (
                not isinstance(self.previous_close_day, date) or self.previous_close_day >= first.date()):
            raise ValueError("previous close day must precede the minute session")

    @property
    def day(self) -> date:
        return self.stock[0].time.date()


class Variant(str, Enum):
    RESIDUAL_Z = "residual_z"
    SECTOR_RELATIVE = "sector_relative"
    TURNOVER_VWAP = "turnover_vwap"
    ATR_DROP = "atr_drop"
    GAP_RELATIVE = "gap_relative"
    VOLUME_SHOCK_REVERSAL = "volume_shock_reversal"
    RANGE_RECOVERY = "range_recovery"


class HedgeLeg(str, Enum):
    SYNTHETIC_INDEX_SHORT = "synthetic_index_short"
    LONG_INVERSE_ETF = "long_inverse_etf"


@dataclass(frozen=True, slots=True)
class HedgePolicy:
    variant: Variant
    lookback: int
    entry_threshold: float
    take_profit: float
    stop_loss: float
    max_hold_minutes: int
    min_prior_volume: float
    min_prior_turnover: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.variant, Variant) or not isinstance(self.lookback, int) or self.lookback < 4:
            raise ValueError("known variant and at least four prior minutes required")
        if not isinstance(self.max_hold_minutes, int) or self.max_hold_minutes < 1:
            raise ValueError("max_hold_minutes must be a positive integer")
        for name in ("entry_threshold", "take_profit", "stop_loss"):
            _finite(getattr(self, name), name, positive=True)
        for name in ("min_prior_volume", "min_prior_turnover"):
            if _finite(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True, slots=True)
class PaperCosts:
    commission_bps_per_side: float
    slippage_bps_per_side: float
    index_short_borrow_bps_annual: float
    trading_minutes_per_year: int

    def __post_init__(self) -> None:
        for name in ("commission_bps_per_side", "slippage_bps_per_side", "index_short_borrow_bps_annual"):
            if _finite(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must not be negative")
        if not isinstance(self.trading_minutes_per_year, int) or self.trading_minutes_per_year <= 0:
            raise ValueError("trading_minutes_per_year must be positive")


@dataclass(frozen=True, slots=True)
class PaperTrade:
    variant: Variant
    hedge_leg: HedgeLeg
    entry_time: datetime
    exit_time: datetime
    stock_entry: float
    stock_exit: float
    index_entry: float
    index_exit: float
    hedge_entry: float
    hedge_exit: float
    long_stock_pnl: float
    synthetic_index_short_pnl: float
    inverse_etf_pnl: float
    # ETF interval return minus -1x of the *specified* benchmark. This is NOT
    # fund tracking error if the ETF's actual target index differs (e.g., a
    # KOSPI200 futures inverse ETF compared with spot KOSPI200).
    inverse_etf_vs_benchmark_return_gap: float | None
    round_trip_cost: float
    index_short_borrow_cost: float
    net_pnl: float
    exit_reason: str


@dataclass(frozen=True, slots=True)
class SessionResult:
    day: date
    stock_id: str
    benchmark_index_id: str
    hedge_leg: HedgeLeg
    trades: tuple[PaperTrade, ...]
    net_pnl: float
    # No position remains unmarked at the edge of a session.


def _prior_liquid(session: MinuteSession, policy: HedgePolicy, t: int) -> bool:
    prior = session.stock[t - policy.lookback:t]  # Excludes the signal minute.
    if fsum(bar.volume for bar in prior) < policy.min_prior_volume:
        return False
    if policy.min_prior_turnover:
        if any(bar.turnover is None for bar in prior):
            return False
        if fsum(bar.turnover for bar in prior if bar.turnover is not None) < policy.min_prior_turnover:
            return False
    return True


def _entry_score(session: MinuteSession, policy: HedgePolicy, t: int) -> float | None:
    stock = session.stock
    market = session.benchmark_index
    start = t - policy.lookback
    if policy.variant is Variant.RESIDUAL_Z:
        stock_returns = [log(stock[k].close / stock[k - 1].close) for k in range(start + 1, t)]
        market_returns = [log(market[k].close / market[k - 1].close) for k in range(start + 1, t)]
        mean_m = fsum(market_returns) / len(market_returns)
        variance_m = fsum((x - mean_m) ** 2 for x in market_returns)
        if variance_m <= 1e-16:
            return None
        mean_s = fsum(stock_returns) / len(stock_returns)
        beta = fsum((s - mean_s) * (m - mean_m) for s, m in zip(stock_returns, market_returns)) / variance_m
        # Absurd betas usually indicate near-zero benchmark variance.
        if not isfinite(beta) or abs(beta) > 5:
            return None
        spreads = [log(stock[k].close / stock[start].close)
                   - beta * log(market[k].close / market[start].close) for k in range(start, t)]
        mean = fsum(spreads) / len(spreads)
        sd = sqrt(fsum((x - mean) ** 2 for x in spreads) / len(spreads))
        if sd <= 1e-10:
            return None
        current = log(stock[t].close / stock[start].close) - beta * log(market[t].close / market[start].close)
        return (current - mean) / sd
    if policy.variant is Variant.SECTOR_RELATIVE:
        sector = session.sector
        if sector is None:
            return None
        return log(stock[t].close / stock[start].close) - log(sector[t].close / sector[start].close)
    if policy.variant is Variant.TURNOVER_VWAP:
        prior = stock[start:t]
        if any(bar.turnover is None for bar in prior):
            return None
        volume = fsum(bar.volume for bar in prior)
        if volume <= 0:
            return None
        vwap = fsum(bar.turnover for bar in prior if bar.turnover is not None) / volume
        if vwap <= 0 or not isfinite(vwap):
            return None
        return stock[t].close / vwap - 1
    if policy.variant is Variant.ATR_DROP:
        true_ranges = [max(stock[k].high - stock[k].low,
                           abs(stock[k].high - stock[k - 1].close),
                           abs(stock[k].low - stock[k - 1].close)) for k in range(start + 1, t)]
        atr = fsum(true_ranges) / len(true_ranges)
        if atr <= 1e-10:
            return None
        return (stock[t].close - stock[start].close) / atr
    if policy.variant is Variant.GAP_RELATIVE:
        previous_stock = session.previous_stock_close
        previous_index = session.previous_index_close
        if previous_stock is None or previous_index is None:
            return None
        opening_gap = log(stock[0].open / previous_stock) - log(market[0].open / previous_index)
        if opening_gap > -policy.entry_threshold / 2:
            return None
        return log(stock[t].close / previous_stock) - log(market[t].close / previous_index)
    if policy.variant is Variant.VOLUME_SHOCK_REVERSAL:
        prior_mean_volume = fsum(bar.volume for bar in stock[start:t]) / policy.lookback
        if prior_mean_volume <= 0 or stock[t].volume < 2 * prior_mean_volume:
            return None
        return log(stock[t].close / stock[start].close) - log(market[t].close / market[start].close)
    if policy.variant is Variant.RANGE_RECOVERY:
        prior_low = min(bar.low for bar in stock[start:t])
        # Both the low and the rebound are from a *completed* signal candle.
        # It is filled only after the conservative three-bar delay.
        if stock[t].low >= prior_low or stock[t].close / stock[t].low - 1 < 0.005:
            return None
        return log(stock[t].close / stock[start].close) - log(market[t].close / market[start].close)
    raise AssertionError("unknown policy variant")


def simulate_session(session: MinuteSession, policy: HedgePolicy, costs: PaperCosts, *,
                     notional_per_leg: float, hedge_leg: HedgeLeg) -> SessionResult:
    """Paper-only delayed-open simulation; no session-edge liquidation fiction."""
    notional = _finite(notional_per_leg, "notional_per_leg", positive=True)
    if policy.variant is Variant.SECTOR_RELATIVE and session.sector is None:
        raise ValueError("sector-relative policy requires aligned sector minutes")
    if policy.variant is Variant.GAP_RELATIVE and (
            session.previous_close_day is None or session.previous_stock_close is None
            or session.previous_index_close is None):
        raise ValueError("gap-relative policy requires dated previous completed closes")
    if not isinstance(hedge_leg, HedgeLeg):
        raise ValueError("hedge_leg must be explicitly selected")
    if policy.max_hold_minutes < session.interval_minutes or policy.max_hold_minutes % session.interval_minutes:
        raise ValueError("max_hold_minutes must be a positive multiple of the session bar interval")
    max_hold_bars = policy.max_hold_minutes // session.interval_minutes
    if hedge_leg is HedgeLeg.LONG_INVERSE_ETF and session.inverse_etf is None:
        raise ValueError("inverse ETF hedge requires aligned ETF minute bars; synthetic index is not a substitute")
    trades: list[PaperTrade] = []
    pending_entry_due: int | None = None
    pending_exit_due: int | None = None
    pending_exit_reason: str | None = None
    entry_index: int | None = None
    stock_entry = index_entry = hedge_entry = 0.0
    for t, (stock_bar, index_bar) in enumerate(zip(session.stock, session.benchmark_index)):
        hedge_bar = session.inverse_etf[t] if hedge_leg is HedgeLeg.LONG_INVERSE_ETF else index_bar
        if pending_exit_due == t:
            assert entry_index is not None
            assert pending_exit_reason is not None
            long_pnl = notional * (stock_bar.open / stock_entry - 1)
            short_pnl = (notional * (1 - index_bar.open / index_entry)
                         if hedge_leg is HedgeLeg.SYNTHETIC_INDEX_SHORT else 0.0)
            etf_pnl = (notional * (hedge_bar.open / hedge_entry - 1)
                       if hedge_leg is HedgeLeg.LONG_INVERSE_ETF else 0.0)
            inverse_vs_spot_gap = (hedge_bar.open / hedge_entry - 1 + index_bar.open / index_entry - 1
                                   if hedge_leg is HedgeLeg.LONG_INVERSE_ETF else None)
            round_trip = 4 * notional * (costs.commission_bps_per_side + costs.slippage_bps_per_side) / 10_000
            elapsed = (stock_bar.time - session.stock[entry_index].time).total_seconds() / 60
            borrow = (notional * costs.index_short_borrow_bps_annual / 10_000 * elapsed
                      / costs.trading_minutes_per_year if hedge_leg is HedgeLeg.SYNTHETIC_INDEX_SHORT else 0.0)
            net = long_pnl + short_pnl + etf_pnl - round_trip - borrow
            trades.append(PaperTrade(policy.variant, hedge_leg, session.stock[entry_index].time, stock_bar.time,
                                     stock_entry, stock_bar.open, index_entry, index_bar.open,
                                     hedge_entry, hedge_bar.open, long_pnl, short_pnl, etf_pnl,
                                     inverse_vs_spot_gap, round_trip, borrow, net, pending_exit_reason))
            pending_exit_due = None
            pending_exit_reason = None
            entry_index = None
        elif pending_entry_due == t:
            entry_index = t
            stock_entry, index_entry = stock_bar.open, index_bar.open
            hedge_entry = hedge_bar.open
            pending_entry_due = None

        if entry_index is not None and pending_exit_due is None:
            stock_return = stock_bar.close / stock_entry - 1
            if stock_return >= policy.take_profit:
                pending_exit_reason = "take_profit"
            elif stock_return <= -policy.stop_loss:
                pending_exit_reason = "stop_loss"
            elif t - entry_index + 1 >= max_hold_bars:
                pending_exit_reason = "time_limit"
            if pending_exit_reason is not None:
                pending_exit_due = t + EXECUTION_DELAY_BARS
        elif (entry_index is None and pending_entry_due is None and t >= policy.lookback
              and t + max_hold_bars + 2 * EXECUTION_DELAY_BARS - 1 < len(session.stock)):
            if _prior_liquid(session, policy, t):
                score = _entry_score(session, policy, t)
                if score is not None and isfinite(score) and score <= -policy.entry_threshold:
                    pending_entry_due = t + EXECUTION_DELAY_BARS
    if entry_index is not None or pending_entry_due is not None or pending_exit_due is not None:
        raise AssertionError("session ended with an unclosed paper position")
    return SessionResult(session.day, session.stock_id, session.benchmark_index_id, hedge_leg,
                         tuple(trades), fsum(trade.net_pnl for trade in trades))


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    training_days: tuple[date, ...]
    test_days: tuple[date, ...]
    selected_policy: HedgePolicy | None
    training_net_pnl: float
    test_results: tuple[SessionResult, ...]


def walk_forward(sessions: Sequence[MinuteSession], candidates: Sequence[HedgePolicy], costs: PaperCosts,
                 *, notional_per_leg: float, min_train_sessions: int, test_sessions: int = 1,
                 min_train_trades: int = 2, hedge_leg: HedgeLeg) -> tuple[WalkForwardFold, ...]:
    """Expanding-window policy/threshold selection using *training only*.

    A fold abstains if no candidate has sufficient training trades and a
    positive after-cost training PnL.  Test sessions never select parameters.
    Sessions are one stock/index pair per date, strictly chronological.
    """
    if not candidates or min_train_sessions < 1 or test_sessions < 1 or min_train_trades < 1:
        raise ValueError("nonempty candidates and positive split sizes are required")
    _finite(notional_per_leg, "notional_per_leg", positive=True)
    ordered = tuple(sessions)
    if any(a.day >= b.day for a, b in zip(ordered, ordered[1:])):
        raise ValueError("sessions must have strictly increasing dates")
    if ordered and any((s.stock_id, s.stock_currency, s.interval_minutes,
                        s.benchmark_index_id, s.sector_id, s.inverse_etf_id) !=
                       (ordered[0].stock_id, ordered[0].stock_currency, ordered[0].interval_minutes,
                        ordered[0].benchmark_index_id, ordered[0].sector_id, ordered[0].inverse_etf_id)
                       for s in ordered):
        raise ValueError("walk-forward sessions must describe one consistent stock/index/sector pair")
    folds: list[WalkForwardFold] = []
    for start in range(min_train_sessions, len(ordered), test_sessions):
        training, testing = ordered[:start], ordered[start:start + test_sessions]
        ranked: list[tuple[float, int, HedgePolicy]] = []
        for ordinal, policy in enumerate(candidates):
            results = [simulate_session(session, policy, costs, notional_per_leg=notional_per_leg,
                                        hedge_leg=hedge_leg)
                       for session in training]
            count = sum(len(result.trades) for result in results)
            if count >= min_train_trades:
                ranked.append((fsum(result.net_pnl for result in results), -ordinal, policy))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        best = ranked[0] if ranked and ranked[0][0] > 0 else None
        policy = best[2] if best else None
        result = tuple(simulate_session(session, policy, costs, notional_per_leg=notional_per_leg,
                                        hedge_leg=hedge_leg)
                       for session in testing) if policy else ()
        folds.append(WalkForwardFold(tuple(s.day for s in training), tuple(s.day for s in testing),
                                     policy, best[0] if best else 0.0, result))
    return tuple(folds)
