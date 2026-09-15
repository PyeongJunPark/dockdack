"""Read-only, rate-limited holdings snapshots for the monitoring dashboard.

The GUI's existing single worker owns refreshes.  This module never starts a
thread, arms automation, submits orders, or persists account responses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable

from dockdack.gui_service import Instrument, TradingService
from dockdack.models import AccountSnapshot, Market, Position


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PortfolioMarketState:
    market: Market
    snapshot: AccountSnapshot | None = None
    fetched_at: datetime | None = None
    last_attempt: datetime | None = None
    error: str = ""

    @property
    def positions(self) -> tuple[Position, ...]:
        return () if self.snapshot is None else tuple(
            position for position in self.snapshot.positions if position.quantity > 0
        )

    def status(self, now: datetime | None = None) -> str:
        if self.error:
            return "error"
        if self.snapshot is None or self.fetched_at is None:
            return "unknown"
        if ((now or utc_now()) - self.fetched_at).total_seconds() > 120:
            return "stale"
        return "ok" if self.positions else "empty"


class PortfolioCache:
    """One full account query per market, at most once every 60 seconds.

    ``force`` requests the earliest safe refresh even when a longer refresh
    interval is configured.  It never bypasses the 60-second API limit.
    Failed attempts are throttled too and retain the last successful snapshot.
    """

    def __init__(self, service: TradingService, *, clock: Callable[[], datetime] = utc_now,
                 refresh_seconds: float = 60):
        self.service = service
        self.clock = clock
        self.refresh_seconds = max(60, refresh_seconds)
        self._states = {market: PortfolioMarketState(market) for market in Market}

    def snapshot(self) -> dict[Market, PortfolioMarketState]:
        return dict(self._states)

    @staticmethod
    def _validate(account: AccountSnapshot, market: Market) -> None:
        currency = "KRW" if market is Market.DOMESTIC else "USD"
        if not isinstance(account, AccountSnapshot) or account.market is not market or account.currency != currency:
            raise ValueError("잔고 응답의 시장/통화를 확인할 수 없습니다.")
        for position in account.positions:
            if position.market is not market or position.currency != currency or not position.symbol:
                raise ValueError("보유종목의 시장/통화/코드를 확인할 수 없습니다.")
            for name in ("quantity", "sellable_quantity", "average_price", "current_price", "evaluation_amount",
                         "profit_loss", "profit_rate"):
                value = getattr(position, name)
                if not isinstance(value, Decimal) or not value.is_finite():
                    raise ValueError("보유종목의 수량/가격을 확인할 수 없습니다.")
                if name not in {"profit_loss", "profit_rate"} and value < 0:
                    raise ValueError("보유종목의 수량/가격이 올바르지 않습니다.")
        for name in ("cash", "available_to_order", "total_purchase", "total_evaluation", "total_profit_loss", "profit_rate"):
            value = getattr(account, name)
            if value is not None and (not isinstance(value, Decimal) or not value.is_finite()):
                raise ValueError("계좌 합계 금액을 확인할 수 없습니다.")

    def refresh_due(self, force: bool = False, stopped: Callable[[], bool] | None = None
                    ) -> dict[Market, PortfolioMarketState]:
        for market in Market:
            if stopped is not None and stopped():
                break
            state = self._states[market]
            now = self.clock()
            minimum = 60 if force else self.refresh_seconds
            if state.last_attempt is not None and (now - state.last_attempt).total_seconds() < minimum:
                continue
            state = replace(state, last_attempt=now)
            self._states[market] = state
            instrument = Instrument(market, "005930" if market is Market.DOMESTIC else "AAPL",
                                    "KRX" if market is Market.DOMESTIC else "ND")
            try:
                # safety_account validates that all balance pages are present;
                # representative symbols select the market, not one holding.
                account = self.service.safety_account(instrument)
                self._validate(account, market)
            except Exception as exc:
                self._states[market] = replace(state, error=str(exc)[:500] or type(exc).__name__)
            else:
                self._states[market] = PortfolioMarketState(
                    market, account, self.clock(), now,
                )
        return self.snapshot()
