"""Kiwoom market rankings for the visible watchlist and model volume input."""

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
    volume: int | None = None
    ranking_basis: str = "turnover"


def _category_rows(http: KiwoomHTTPClient, market: Market, *, api_id: str,
                   key: str, body: dict[str, str], metric: str, basis: str,
                   limit: int, descending: bool = True) -> tuple[RankedStock, ...]:
    """Read a complete broker top-N and retain only classified common shares."""
    domestic = market is Market.DOMESTIC
    path = "/api/dostk/rkinfo" if domestic else "/api/us/rkinfo"
    candidates: dict[tuple[str, str], tuple[Decimal, RankedStock]] = {}
    ordinal = 0
    for page in http.iter_pages(api_id=api_id, path=path, body=body, max_pages=20):
        rows = page.body.get(key)
        if not isinstance(rows, list):
            raise BrokerAPIError(f"{basis} 순위 응답 목록을 확인할 수 없습니다.")
        parsed = []
        for row in rows:
            ordinal += 1
            if not isinstance(row, dict):
                raise BrokerAPIError(f"{basis} 순위 행이 잘못되었습니다.")
            symbol = normalize_symbol(str(row.get("stk_cd", "")))
            exchange = "KRX" if domestic else str(row.get("stex_tp", "")).strip().upper()
            if symbol and not domestic and exchange == "NP":
                continue
            if not symbol or exchange not in ({"KRX"} if domestic else {"ND", "NY", "NA"}):
                raise BrokerAPIError(f"{basis} 순위 종목 또는 거래소를 확인할 수 없습니다.")
            try:
                value = Decimal(str(row.get(metric)).replace(",", ""))
                rank = ordinal if domestic else int(str(row.get("rank")))
            except (ValueError, InvalidOperation, TypeError) as exc:
                raise BrokerAPIError(f"{basis} 순위 값을 해석할 수 없습니다.") from exc
            if not value.is_finite() or rank <= 0 or (basis == "market_cap" and value < 0):
                raise BrokerAPIError(f"{basis} 순위 값이 올바르지 않습니다.")
            if basis == "gainers" and value <= 0 or basis == "decliners" and value >= 0:
                continue
            stock = RankedStock(market, symbol, exchange,
                                str(row.get("stk_nm") or row.get("stk_enm") or symbol),
                                rank, Decimal(0), "KRW" if domestic else "USD",
                                ranking_basis=basis)
            parsed.append((value, stock))
        eligible = common_equities(http, market, ((stock.symbol, stock.exchange) for _, stock in parsed))
        for value, stock in parsed:
            if (stock.symbol, stock.exchange) in eligible:
                candidates.setdefault((stock.symbol, stock.exchange), (value, stock))
        if len(candidates) >= limit:
            break
    ordered = sorted(candidates.values(), key=lambda pair: (
        -pair[0] if descending else pair[0], pair[1].rank, pair[1].symbol))[:limit]
    if len(ordered) != limit:
        raise BrokerAPIError(f"분류가 확인된 {basis} 순위가 {len(ordered)}개뿐이어서 상위 {limit}개를 확정하지 않았습니다.")
    return tuple(RankedStock(stock.market, stock.symbol, stock.exchange, stock.name, index,
                             stock.turnover, stock.currency, stock.volume, basis)
                 for index, (_, stock) in enumerate(ordered, 1))


def top_change(http: KiwoomHTTPClient, market: Market, *, gainers: bool,
               limit: int = 20) -> tuple[RankedStock, ...]:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("등락률 순위는 1~100개를 요청할 수 있습니다.")
    domestic = market is Market.DOMESTIC
    body = ({"mrkt_tp": "000", "sort_tp": "1" if gainers else "3",
             "trde_qty_cnd": "0000", "stk_cnd": "3", "crd_cnd": "0",
             "updown_incls": "1", "pric_cnd": "0", "trde_prica_cnd": "0", "stex_tp": "1"}
            if domestic else
            {"stex_tp": "0", "inds_cd": "000", "inds_cls_tp": "0",
             "sort_tp": "1" if gainers else "4", "stk_tp": "1", "stk_cnd": "0",
             "pric_cnd": "0", "trde_prica_cnd": "0", "trde_qty_tp": "0"})
    return _category_rows(http, market, api_id="ka10027" if domestic else "usa20910",
                          key="pred_pre_flu_rt_upper" if domestic else "result_list",
                          body=body, metric="flu_rt", basis="gainers" if gainers else "decliners",
                          limit=limit, descending=gainers)


def top_market_cap(http: KiwoomHTTPClient, market: Market,
                   limit: int = 20) -> tuple[RankedStock, ...]:
    """Use verified US usa20550; domestic has no verified rank here."""
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("시가총액 순위는 1~100개를 요청할 수 있습니다.")
    if market is Market.DOMESTIC:
        return ()
    return _category_rows(http, market, api_id="usa20550", key="result_list",
                          body={"stex_tp": "0", "inds_cd": "000", "stk_tp": "1",
                                "trde_qty_tp": "0", "stk_cnd": "0", "pric_cnd": "0",
                                "trde_prica_cnd": "0"}, metric="mac", basis="market_cap",
                          limit=limit)


def top_watchlist(http: KiwoomHTTPClient, market: Market,
                  limit: int = 100) -> tuple[RankedStock, ...]:
    """Combine 20 per supported category, then fill duplicates by turnover."""
    if type(limit) is not int or limit != 100:
        raise ValueError("복합 관심종목은 시장별 정확히 100개만 확정합니다.")
    turnover = top_turnover(http, market, 100)
    sources = (("turnover", turnover[:20]),
               ("volume", top_volume(http, market, 20)),
               ("gainers", top_change(http, market, gainers=True)),
               ("decliners", top_change(http, market, gainers=False)),
               ("market_cap", top_market_cap(http, market)))
    selected: dict[tuple[str, str], RankedStock] = {}
    for basis, rows in sources:
        for row in rows:
            selected.setdefault((row.symbol, row.exchange),
                                RankedStock(row.market, row.symbol, row.exchange, row.name,
                                            0, row.turnover, row.currency, row.volume, basis))
    for row in turnover:
        if len(selected) >= 100:
            break
        selected.setdefault((row.symbol, row.exchange), row)
    if len(selected) != 100:
        raise BrokerAPIError(f"복합 관심종목이 {len(selected)}개뿐이어서 100개를 확정하지 않았습니다.")
    return tuple(RankedStock(row.market, row.symbol, row.exchange, row.name, index,
                             row.turnover, row.currency, row.volume, row.ranking_basis)
                 for index, row in enumerate(selected.values(), 1))


def top_volume(http: KiwoomHTTPClient, market: Market, limit: int = 100) -> tuple[RankedStock, ...]:
    """Rank verified common shares by actual traded shares, never by amount.

    Official contracts: ka10030 sort_tp=1 / trde_qty (no rank field),
    usa20530 qry_tp=0 / acc_trde_qty. Volume is in individual shares in both
    markets. In the pre-open preparation window this is the broker's latest
    available daily snapshot; an empty/short snapshot must not invent names.
    """
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("거래량 순위는 1~100개를 요청할 수 있습니다.")
    domestic = market is Market.DOMESTIC
    api_id, path, key = ("ka10030", "/api/dostk/rkinfo", "tdy_trde_qty_upper") if domestic else (
        "usa20530", "/api/us/rkinfo", "result_list")
    body = {"mrkt_tp": "000", "sort_tp": "1", "mang_stk_incls": "4", "crd_tp": "0",
            "trde_qty_tp": "0", "pric_tp": "0", "trde_prica_tp": "0",
            "mrkt_open_tp": "0", "stex_tp": "1"} if domestic else {
        "stex_tp": "0", "inds_cd": "000", "stk_tp": "1", "trde_qty_tp": "0",
        "qry_tp": "0", "stk_cnd": "0", "pric_cnd": "0", "trde_prica_cnd": "0"}
    results, ordinal = {}, 0
    for page in http.iter_pages(api_id=api_id, path=path, body=body, max_pages=20):
        rows = page.body.get(key)
        if not isinstance(rows, list):
            raise BrokerAPIError("거래량 순위 응답 목록을 확인할 수 없습니다.")
        candidates = []
        for row in rows:
            ordinal += 1
            if not isinstance(row, dict):
                raise BrokerAPIError("잘못된 거래량 순위 응답입니다.")
            symbol = normalize_symbol(str(row.get("stk_cd", "")))
            exchange = "KRX" if domestic else str(row.get("stex_tp", "")).strip().upper()
            if symbol and not domestic and exchange == "NP":
                continue
            if not symbol or exchange not in ({"KRX"} if domestic else {"ND", "NY", "NA"}):
                raise BrokerAPIError("거래량 순위 종목 또는 거래소를 확인할 수 없습니다.")
            try:
                volume = Decimal(str(row.get("trde_qty" if domestic else "acc_trde_qty")).replace(",", ""))
                # ka10030 defines no rank: preserve page order for tie-breaking.
                rank = ordinal if domestic else int(str(row.get("rank")))
                amount = Decimal(str(row.get("trde_amt" if domestic else "trde_prica", "0")).replace(",", ""))
            except (ValueError, InvalidOperation) as exc:
                raise BrokerAPIError("순위/거래량/거래대금을 숫자로 해석할 수 없습니다.") from exc
            if (not volume.is_finite() or volume < 0 or volume != volume.to_integral_value()
                    or not amount.is_finite() or amount < 0 or rank <= 0):
                raise BrokerAPIError("잘못된 순위/거래량/거래대금 값입니다.")
            candidates.append(RankedStock(market, symbol, exchange,
                str(row.get("stk_nm") or row.get("stk_enm") or symbol), rank,
                amount * (1_000_000 if domestic else 1_000), "KRW" if domestic else "USD",
                int(volume), "volume"))
        eligible = common_equities(http, market, ((r.symbol, r.exchange) for r in candidates))
        for result in candidates:
            if (result.symbol, result.exchange) in eligible:
                results.setdefault((result.symbol, result.exchange), result)
        if len(results) >= limit:
            break
    if len(results) < limit:
        raise BrokerAPIError(f"분류가 확인된 일반기업 보통주가 {len(results)}개뿐이어서 거래량 상위 {limit}개를 확정하지 않았습니다.")
    selected = sorted(results.values(), key=lambda item: (-item.volume, item.rank, item.symbol))[:limit]
    return tuple(RankedStock(r.market, r.symbol, r.exchange, r.name, i, r.turnover,
                            r.currency, r.volume, "volume") for i, r in enumerate(selected, 1))


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
