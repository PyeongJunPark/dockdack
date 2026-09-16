"""Calendar-only market badges; these describe regular sessions, not order access."""

from datetime import datetime, timedelta
from typing import Literal, TypedDict

from dockdack.history import market_time
from dockdack.market_schedule import session_on
from dockdack.models import Market


class MarketStatus(TypedDict):
    market: Market
    state: Literal["open", "preopen", "preparing", "closed", "holiday", "unknown"]
    is_open: bool | None
    text: str
    detail: str
    checked_at: datetime


def _range(opened: datetime, closed: datetime) -> str:
    """Include both dates when a US session crosses midnight in Korea."""
    end = closed.strftime("%H:%M" if opened.date() == closed.date() else "%m/%d %H:%M")
    return f"{opened:%m/%d %H:%M}~{end} {opened.tzname()}"


def market_status(market: Market, now: datetime) -> MarketStatus:
    """Return an honest, locally calculated badge without contacting a broker.

    ``preparing`` means the ten-minute ranking-preparation window is eligible,
    not that a ranking refresh is currently running or an order can be placed.
    ``preopen`` includes all earlier times on an exchange-local session date;
    it does not mean that extended-hours trading is available. Unknown calendar
    dates/errors remain unknown rather than falling back to guessed hours.
    """
    name = "한국" if market is Market.DOMESTIC else "미국"
    result: MarketStatus = {
        "market": market,
        "state": "unknown",
        "is_open": None,
        "text": f"{name} · 장 시간 확인 필요",
        "detail": "정규장 기준 · 장 시간을 확인할 수 없습니다.",
        "checked_at": now,
    }
    try:
        local_now = market_time(market, now)
        session = session_on(market, local_now.date())
        if session is None:
            result.update(
                state="holiday", is_open=False, text=f"{name} · 휴장",
                detail=f"정규장 기준 · 현지 {local_now:%Y-%m-%d} 휴장일(주말·거래소 휴일). "
                       "장전·시간외 거래 가능 여부를 뜻하지 않습니다.",
            )
            return result
        opened = market_time(market, session.opened)
        closed = market_time(market, session.closed)
        if closed <= opened or opened.date() != local_now.date():
            raise ValueError("거래소 캘린더의 개장·폐장 시간이 올바르지 않습니다.")
        if opened <= local_now < closed:
            state, label = "open", "장중"
        elif opened - timedelta(minutes=10) <= local_now < opened:
            state, label = "preparing", "개장 준비 시간"
        elif local_now < opened:
            state, label = "preopen", "장전"
        else:
            state, label = "closed", "장 마감"
        if market is Market.US:
            hours = (f"뉴욕 {_range(opened, closed)} · "
                     f"한국 {_range(market_time(Market.DOMESTIC, opened), market_time(Market.DOMESTIC, closed))}")
        else:
            hours = f"한국 {_range(opened, closed)}"
        detail = f"정규장 기준 · {hours}. 장전·시간외 거래 가능 여부를 뜻하지 않습니다."
        if state == "preparing":
            detail += " 관심종목 초기 선정 가능 시간이며, 주문 개시를 뜻하지 않습니다."
        result.update(state=state, is_open=state == "open", text=f"{name} · {label}", detail=detail)
    except Exception as exc:
        # This is display-only: one calendar failure must not claim a market is
        # closed/open, or suppress the independent status of the other market.
        result["detail"] = f"정규장 확인 불가 · {exc}"
    return result


def market_statuses(now: datetime) -> dict[Market, MarketStatus]:
    """The two markets use their own local date and fail independently."""
    return {market: market_status(market, now) for market in Market}
