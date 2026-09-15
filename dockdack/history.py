"""Daily OHLCV data and exchange-local time, independent of Qt."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

from dockdack.exceptions import BrokerAPIError
from dockdack.http import KiwoomHTTPClient
from dockdack.models import Market


def market_time(market: Market, now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("시간에는 시간대 정보가 필요합니다.")
    return now.astimezone(ZoneInfo("Asia/Seoul" if market is Market.DOMESTIC else "America/New_York"))


def regular_session(market: Market, now: datetime) -> bool:
    """Holiday, DST, delayed-open and early-close aware; calendar errors fail closed."""
    from dockdack.market_schedule import is_open
    return is_open(market, now)


def day_count(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 1000:
        raise ValueError("조회 기간은 1~1000 거래일의 정수여야 합니다.")
    return value


@dataclass(frozen=True)
class DailyBar:
    day: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True)
class DailyHistory:
    market: Market
    symbol: str
    exchange: str
    currency: str
    requested_days: int
    bars: tuple[DailyBar, ...]  # Oldest first; today's bar may still be forming.

    @property
    def complete(self) -> bool:
        return len(self.bars) >= self.requested_days


def _number(value, *, volume=False) -> Decimal:
    try:
        result = Decimal(str(value).replace(",", "").strip())
    except InvalidOperation as exc:
        raise BrokerAPIError("일봉 응답에 잘못된 가격/거래량이 있습니다.") from exc
    if not volume:
        result = abs(result)  # Kiwoom prefixes price fields with a change sign.
    if not result.is_finite() or (result < 0 if volume else result <= 0):
        raise BrokerAPIError("일봉 가격은 양수, 거래량은 0 이상의 유한한 숫자여야 합니다.")
    return result


def fetch_daily_history(http: KiwoomHTTPClient, market: Market, symbol: str, exchange: str,
                        days: int, *, as_of: date | None = None) -> DailyHistory:
    day_count(days)
    end = as_of or market_time(market).date()
    if type(end) is not date:
        raise ValueError("기준일은 datetime.date여야 합니다.")
    if market is Market.DOMESTIC:
        api_id, path, key = "ka10081", "/api/dostk/chart", "stk_dt_pole_chart_qry"
        suffix = {"KRX": "", "NXT": "_NX", "SOR": "_AL"}[exchange]
        body = {"stk_cd": symbol + suffix, "base_dt": end.strftime("%Y%m%d"), "upd_stkpc_tp": "1"}
    else:
        api_id, path, key = "usa06012", "/api/us/chart", "result_list"
        body = {"stex_tp": exchange, "stk_cd": symbol,
                # Despite the field's name, the demo API returns bars backwards from this date.
                "strt_dt": end.strftime("%Y%m%d"),
                "upd_stkpc_tp": "1", "exrt_appl_tp": "0"}
    bars: dict[date, DailyBar] = {}
    for page in http.iter_pages(api_id=api_id, path=path, body=body, max_pages=20):
        returned = page.body.get("stk_cd")
        if returned and returned not in {symbol, body["stk_cd"]}:
            raise BrokerAPIError("요청 종목과 일봉 응답 종목이 다릅니다.")
        rows = page.body.get(key)
        if not isinstance(rows, list):
            raise BrokerAPIError("일봉 응답의 데이터 목록이 없거나 형식이 잘못되었습니다.")
        for row in rows:
            if not isinstance(row, dict):
                raise BrokerAPIError("일봉 데이터는 객체 목록이어야 합니다.")
            try:
                day = datetime.strptime(str(row["dt"]), "%Y%m%d").date()
            except (KeyError, ValueError) as exc:
                raise BrokerAPIError("일봉 응답의 거래일을 확인할 수 없습니다.") from exc
            if day > end:
                continue
            bar = DailyBar(day, *(_number(row.get(key)) for key in
                                  ("open_pric", "high_pric", "low_pric", "cur_prc")),
                           _number(row.get("trde_qty" if market is Market.DOMESTIC else "acc_trde_qty"), volume=True))
            if bar.high < bar.low:
                raise BrokerAPIError("일봉의 고가가 저가보다 낮습니다.")
            if day in bars and bars[day] != bar:
                raise BrokerAPIError("연속조회 중 동일 거래일의 일봉 값이 달라졌습니다. 다시 조회하세요.")
            bars[day] = bar
        # Both APIs return the newest bars first. Overlapping boundary dates are deduplicated.
        if len(bars) >= days:
            break
    ordered = tuple(bars[day] for day in sorted(bars)[-days:])
    if not ordered:
        raise BrokerAPIError("조회 가능한 일봉이 없습니다.")
    return DailyHistory(market, symbol, exchange, "KRW" if market is Market.DOMESTIC else "USD", days, ordered)
