"""Kiwoom's market-specific daily turnover rankings, normalized to KRW / USD."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from dockdack.exceptions import BrokerAPIError
from dockdack.http import KiwoomHTTPClient
from dockdack.models import Market
from dockdack.symbols import normalize_symbol
from dockdack.equity_policy import common_equities


@dataclass(frozen=True)
class RankedStock:
    market: Market
    symbol: str
    exchange: str
    name: str
    rank: int
    turnover: Decimal
    currency: str


def top_turnover(http: KiwoomHTTPClient, market: Market, limit: int = 100) -> tuple[RankedStock, ...]:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("거래대금 순위는 1~100개를 요청할 수 있습니다.")
    domestic = market is Market.DOMESTIC
    api_id, path, key = ("ka10032", "/api/dostk/rkinfo", "trde_prica_upper") if domestic else (
        "usa20540", "/api/us/rkinfo", "result_list")
    body = {"mrkt_tp": "000", "mang_stk_incls": "0", "stex_tp": "1"} if domestic else {
        "stex_tp": "0", "inds_cd": "000", "stk_tp": "1", "trde_qty_tp": "0",
        "stk_cnd": "0", "pric_cnd": "0", "trde_prica_cnd": "0"}
    results = {}
    for page in http.iter_pages(api_id=api_id, path=path, body=body, max_pages=20):
        rows = page.body.get(key)
        if not isinstance(rows, list):
            raise BrokerAPIError("거래대금 순위 응답 목록을 확인할 수 없습니다.")
        page_results = []
        for row in rows:
            if not isinstance(row, dict):
                raise BrokerAPIError("잘못된 순위 응답입니다.")
            symbol = normalize_symbol(str(row.get("stk_cd", "")))
            exchange = "KRX" if domestic else str(row.get("stex_tp", "")).strip().upper()
            if symbol and not domestic and exchange == "NP":
                # The all-market US ranking includes NP listings (observed for
                # TCEHY), outside our ND/NY/NA execution and equity-classification
                # scope. Exclude them without remapping their exchange or
                # counting them toward the required common-share total.
                continue
            if not symbol or exchange not in ({"KRX"} if domestic else {"ND", "NY", "NA"}):
                raise BrokerAPIError("순위 종목 또는 거래소를 확인할 수 없습니다.")
            try:
                amount = Decimal(str(row.get("trde_prica")).replace(",", ""))
                rank = int(str(row.get("now_rank" if domestic else "rank")))
            except (ValueError, InvalidOperation) as exc:
                raise BrokerAPIError("순위/거래대금을 숫자로 해석할 수 없습니다.") from exc
            if not amount.is_finite() or amount < 0 or rank <= 0:
                raise BrokerAPIError("잘못된 순위/거래대금 값입니다.")
            # Domestic response: million KRW; US response: thousand USD.
            page_results.append(RankedStock(market, symbol, exchange,
                str(row.get("stk_nm") or row.get("stk_enm") or symbol), rank,
                amount * (1_000_000 if domestic else 1_000), "KRW" if domestic else "USD"))
        eligible = common_equities(http, market, ((r.symbol, r.exchange) for r in page_results))
        for result in page_results:
            if (result.symbol, result.exchange) in eligible:
                results.setdefault((result.symbol, result.exchange), result)
        if len(results) >= limit:
            break
    if len(results) < limit:
        raise BrokerAPIError(f"분류가 확인된 일반기업 보통주가 {len(results)}개뿐이어서 상위 {limit}개를 확정하지 않았습니다.")
    # Trust the API's ranked candidate pages, then order by normalized turnover within the market.
    selected = sorted(results.values(), key=lambda item: (-item.turnover, item.rank, item.symbol))[:limit]
    return tuple(RankedStock(r.market, r.symbol, r.exchange, r.name, i, r.turnover, r.currency)
                 for i, r in enumerate(selected, 1))
