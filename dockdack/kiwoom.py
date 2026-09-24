"""High-level Kiwoom broker functions for Korean and US stocks."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from dockdack.conditions import ConnectFactory, KiwoomConditionClient
from dockdack.config import KiwoomConfig
from dockdack.exceptions import LiveOrderConfirmationRequired, OrderOutcomeUnknown
from dockdack.http import HttpTransport, KiwoomHTTPClient
from dockdack.history import DailyHistory, fetch_daily_history
from dockdack.universe import RankedStock, top_turnover
from dockdack.symbols import normalize_symbol, normalize_us_exchange
from dockdack.order_prices import current_limit_price, validate_us_order_price
from dockdack.models import (
    AccountSnapshot,
    CancelResult,
    ConditionMatch,
    DailyBar,
    DomesticExchange,
    ExecutionHistoryRecord,
    Market,
    OpenOrder,
    OrderRequest,
    OrderExecution,
    OrderResult,
    OrderSide,
    Position,
    Quote,
    SavedCondition,
    StockInfo,
    TradingMode,
    USExchange,
)


LIVE_ORDER_CONFIRMATION = "LIVE_ORDER"


class KiwoomBroker:
    """A safe high-level façade over the Kiwoom REST and WebSocket APIs."""

    def __init__(
        self,
        config: KiwoomConfig,
        *,
        us_config: KiwoomConfig | None = None,
        transport: HttpTransport | None = None,
        websocket_connect_factory: ConnectFactory | None = None,
    ) -> None:
        if us_config is not None and us_config.mode is not config.mode:
            raise ValueError("국내주식과 미국주식 config의 거래 환경은 같아야 합니다.")
        self.config = config
        self.domestic_config = config
        self.us_config = us_config or config
        self._domestic_http = KiwoomHTTPClient(config, transport=transport)
        self._us_http = (
            self._domestic_http
            if self.us_config == self.domestic_config
            else KiwoomHTTPClient(self.us_config, transport=transport)
        )
        self._websocket_connect_factory = websocket_connect_factory

    @classmethod
    def from_env(cls, mode: TradingMode | str | None = None) -> "KiwoomBroker":
        return cls(
            KiwoomConfig.from_env(mode, market=Market.DOMESTIC),
            us_config=KiwoomConfig.from_env(mode, market=Market.US),
        )

    @property
    def mode(self) -> TradingMode:
        return self.domestic_config.mode

    def _config_for(self, market: Market) -> KiwoomConfig:
        return self.domestic_config if market is Market.DOMESTIC else self.us_config

    def _http_for(self, market: Market) -> KiwoomHTTPClient:
        return self._domestic_http if market is Market.DOMESTIC else self._us_http

    def quote_domestic(
        self,
        symbol: str,
        *,
        exchange: DomesticExchange | str = DomesticExchange.KRX,
    ) -> Quote:
        selected_exchange = _domestic_exchange(exchange)
        api_symbol = _domestic_api_symbol(symbol, selected_exchange)
        body = self._domestic_http.request(
            api_id="ka10001",
            path="/api/dostk/stkinfo",
            body={"stk_cd": api_symbol},
        ).body
        return Quote(
            market=Market.DOMESTIC,
            symbol=_clean_domestic_symbol(str(body.get("stk_cd") or symbol)),
            name=str(body.get("stk_nm", "")),
            exchange=selected_exchange.value,
            price=_required_decimal(body.get("cur_prc"), "현재가", absolute=True),
            currency="KRW",
            change=_decimal(body.get("pred_pre")),
            change_rate=_decimal(body.get("flu_rt")),
            volume=_decimal(body.get("trde_qty"), absolute=True),
            raw=body,
        )

    def quote_us(
        self,
        symbol: str,
        *,
        exchange: USExchange | str,
    ) -> Quote:
        selected_exchange = _us_exchange(exchange, allow_all=False)
        body = self._us_http.request(
            api_id="usa20100",
            path="/api/us/mrkcond",
            body={"stex_tp": selected_exchange.value, "stk_cd": _symbol(symbol)},
        ).body
        return Quote(
            market=Market.US,
            symbol=str(body.get("stk_cd") or _symbol(symbol)),
            name=str(body.get("stk_nm") or body.get("stk_enm") or ""),
            exchange=str(body.get("stex_tp") or selected_exchange.value),
            price=_required_decimal(body.get("cur_prc"), "현재가", absolute=True),
            currency=str(body.get("curr_unit") or "USD"),
            change=_decimal(body.get("pred_pre")),
            change_rate=_decimal(body.get("flu_rt")),
            volume=_decimal(body.get("acc_trde_qty"), absolute=True),
            raw=body,
        )

    def resolve_us_exchange(self, symbol: str) -> USExchange:
        """Resolve an exact ticker using Kiwoom's exchange lookup (usa10098)."""
        ticker = _symbol(symbol)
        body = self._us_http.request(
            api_id="usa10098",
            path="/api/us/stkinfo",
            body={"stk_cd": ticker},
        ).body
        exchanges = {
            _us_exchange(str(row.get("stex_tp", "")), allow_all=False)
            for row in _records(body.get("list", []), _US_STOCK_KEYS)
            if str(row.get("stk_cd", "")).strip() == ticker
        }
        if len(exchanges) != 1:
            raise ValueError(
                f"{ticker}의 거래소를 하나로 확인할 수 없습니다. "
                "티커를 확인하거나 --exchange NASDAQ/NYSE/AMEX를 지정하세요."
            )
        return exchanges.pop()

    def get_quote(
        self,
        market: Market | str,
        symbol: str,
        *,
        exchange: DomesticExchange | USExchange | str,
    ) -> Quote:
        selected_market = _market(market)
        if selected_market is Market.DOMESTIC:
            return self.quote_domestic(symbol, exchange=exchange)
        return self.quote_us(symbol, exchange=exchange)

    def top_turnover(self, market: Market | str, limit: int = 100) -> tuple[RankedStock, ...]:
        selected = _market(market)
        return top_turnover(self._http_for(selected), selected, limit)

    def top_volume(self, market: Market | str, limit: int = 100) -> tuple[RankedStock, ...]:
        from dockdack.universe import top_volume
        selected = _market(market)
        return top_volume(self._http_for(selected), selected, limit)

    def common_equities(self, market: Market, candidates):
        from dockdack.equity_policy import common_equities
        selected = _market(market)
        return common_equities(self._http_for(selected), selected, candidates)

    def daily_history(self, market: Market | str, symbol: str, *,
                      exchange: DomesticExchange | USExchange | str, days: int = 30,
                      as_of: date | None = None) -> DailyHistory:
        selected = _market(market)
        venue = _domestic_exchange(exchange) if selected is Market.DOMESTIC else _us_exchange(exchange, allow_all=False)
        ticker = _clean_domestic_symbol(_symbol(symbol)) if selected is Market.DOMESTIC else _symbol(symbol)
        return fetch_daily_history(self._http_for(selected), selected, ticker, venue.value, days, as_of=as_of)

    def list_domestic_stocks(
        self,
        *,
        market_codes: Sequence[str] = ("0", "10"),
        query: str = "",
        limit: int | None = None,
        max_pages: int = 10,
    ) -> tuple[StockInfo, ...]:
        _validate_limit(limit)
        stocks: list[StockInfo] = []
        normalized_query = query.casefold().strip()
        for market_code in market_codes:
            for page in self._domestic_http.iter_pages(
                api_id="ka10099",
                path="/api/dostk/stkinfo",
                body={"mrkt_tp": str(market_code)},
                max_pages=max_pages,
            ):
                for row in _records(page.body.get("list", []), _DOMESTIC_STOCK_KEYS):
                    stock = StockInfo(
                        market=Market.DOMESTIC,
                        symbol=_clean_domestic_symbol(str(row.get("code", ""))),
                        name=str(row.get("name", "")),
                        exchange=str(row.get("marketName") or row.get("marketCode") or ""),
                        previous_close=_decimal(row.get("lastPrice"), absolute=True),
                        is_etf=_looks_true(row.get("isEtf")) if "isEtf" in row else None,
                        raw=row,
                    )
                    if _stock_matches(stock, normalized_query):
                        stocks.append(stock)
                        if limit is not None and len(stocks) >= limit:
                            return tuple(stocks)
        return tuple(stocks)

    def list_us_stocks(
        self,
        *,
        exchange: USExchange | str = USExchange.ALL,
        query: str = "",
        limit: int | None = None,
        max_pages: int = 10,
    ) -> tuple[StockInfo, ...]:
        _validate_limit(limit)
        selected_exchange = _us_exchange(exchange, allow_all=True)
        normalized_query = query.casefold().strip()
        stocks: list[StockInfo] = []
        for page in self._us_http.iter_pages(
            api_id="usa10099",
            path="/api/us/stkinfo",
            body={"stex_tp": selected_exchange.value},
            max_pages=max_pages,
        ):
            for row in _records(page.body.get("list", []), _US_STOCK_KEYS):
                stock = StockInfo(
                    market=Market.US,
                    symbol=str(row.get("stk_cd", "")),
                    name=str(row.get("stk_nm") or row.get("stk_enm") or ""),
                    english_name=str(row.get("stk_enm") or "") or None,
                    exchange=str(row.get("stex_tp") or row.get("mkgb") or ""),
                    is_etf=_looks_true(row.get("isEtf")),
                    raw=row,
                )
                if _stock_matches(stock, normalized_query):
                    stocks.append(stock)
                    if limit is not None and len(stocks) >= limit:
                        return tuple(stocks)
        return tuple(stocks)

    def iter_daily_bars_domestic(
        self,
        symbol: str,
        *,
        exchange: DomesticExchange | str = DomesticExchange.KRX,
        base_date: date | str | None = None,
        adjusted: bool = True,
        max_pages: int = 1_000,
    ) -> Iterator[tuple[DailyBar, ...]]:
        """Yield every Kiwoom daily-chart page available for a Korean symbol."""
        selected_exchange = _domestic_exchange(exchange)
        api_symbol = _domestic_api_symbol(symbol, selected_exchange)
        body = {
            "stk_cd": api_symbol,
            "base_dt": _api_date(base_date),
            "upd_stkpc_tp": "1" if adjusted else "0",
        }
        for page in self._domestic_http.iter_pages(
            api_id="ka10081",
            path="/api/dostk/chart",
            body=body,
            max_pages=max_pages,
        ):
            yield tuple(
                _domestic_daily_bar(row, symbol, selected_exchange)
                for row in _records(page.body.get("stk_dt_pole_chart_qry", []))
                if row.get("dt")
            )

    def daily_bars_domestic(
        self,
        symbol: str,
        *,
        exchange: DomesticExchange | str = DomesticExchange.KRX,
        base_date: date | str | None = None,
        adjusted: bool = True,
        max_pages: int = 1_000,
    ) -> tuple[DailyBar, ...]:
        """Return Korean daily OHLCV bars, newest page first from Kiwoom."""
        return tuple(
            bar
            for page in self.iter_daily_bars_domestic(
                symbol,
                exchange=exchange,
                base_date=base_date,
                adjusted=adjusted,
                max_pages=max_pages,
            )
            for bar in page
        )

    def iter_daily_bars_us(
        self,
        symbol: str,
        *,
        exchange: USExchange | str,
        start_date: date | str | None = None,
        adjusted: bool = True,
        apply_exchange_rate: bool = False,
        max_pages: int = 1_000,
    ) -> Iterator[tuple[DailyBar, ...]]:
        """Yield every Kiwoom daily-chart page available for a US symbol."""
        selected_exchange = _us_exchange(exchange, allow_all=False)
        body = {
            "stex_tp": selected_exchange.value,
            "stk_cd": _symbol(symbol),
            "upd_stkpc_tp": "1" if adjusted else "0",
            "exrt_appl_tp": "1" if apply_exchange_rate else "0",
        }
        if start_date is not None:
            body["strt_dt"] = _api_date(start_date)
        for page in self._us_http.iter_pages(
            api_id="usa06012",
            path="/api/us/chart",
            body=body,
            max_pages=max_pages,
        ):
            yield tuple(
                _us_daily_bar(row, symbol, selected_exchange, apply_exchange_rate)
                for row in _records(page.body.get("result_list", []))
                if row.get("dt")
            )

    def daily_bars_us(
        self,
        symbol: str,
        *,
        exchange: USExchange | str,
        start_date: date | str | None = None,
        adjusted: bool = True,
        apply_exchange_rate: bool = False,
        max_pages: int = 1_000,
    ) -> tuple[DailyBar, ...]:
        """Return US daily OHLCV bars, newest page first from Kiwoom."""
        return tuple(
            bar
            for page in self.iter_daily_bars_us(
                symbol,
                exchange=exchange,
                start_date=start_date,
                adjusted=adjusted,
                apply_exchange_rate=apply_exchange_rate,
                max_pages=max_pages,
            )
            for bar in page
        )

    def search_stocks(
        self,
        query: str,
        *,
        market: Market | str,
        exchange: USExchange | str = USExchange.ALL,
        limit: int = 20,
        domestic_market_codes: Sequence[str] = ("0", "10"),
    ) -> tuple[StockInfo, ...]:
        if not query.strip():
            raise ValueError("검색어가 필요합니다.")
        selected_market = _market(market)
        if selected_market is Market.DOMESTIC:
            return self.list_domestic_stocks(
                market_codes=domestic_market_codes,
                query=query,
                limit=limit,
            )
        return self.list_us_stocks(exchange=exchange, query=query, limit=limit)

    def screen_catalog(
        self,
        market: Market | str,
        condition: Callable[[StockInfo], bool],
        *,
        exchange: USExchange | str = USExchange.ALL,
        domestic_market_codes: Sequence[str] = ("0", "10"),
        limit: int | None = None,
    ) -> tuple[StockInfo, ...]:
        """Filter the broker stock catalog with a caller-provided predicate."""
        selected_market = _market(market)
        universe = (
            self.list_domestic_stocks(market_codes=domestic_market_codes)
            if selected_market is Market.DOMESTIC
            else self.list_us_stocks(exchange=exchange)
        )
        matches: list[StockInfo] = []
        for stock in universe:
            if condition(stock):
                matches.append(stock)
                if limit is not None and len(matches) >= limit:
                    break
        return tuple(matches)

    def screen_quotes(
        self,
        symbols: Iterable[str],
        condition: Callable[[Quote], bool],
        *,
        market: Market | str,
        exchange: DomesticExchange | USExchange | str,
        max_symbols: int = 100,
    ) -> tuple[Quote, ...]:
        """Fetch and filter a bounded candidate list while respecting client throttling."""
        candidates = tuple(symbols)
        if len(candidates) > max_symbols:
            raise ValueError(f"후보 종목은 최대 {max_symbols}개까지 조회할 수 있습니다.")
        matches: list[Quote] = []
        for symbol in candidates:
            quote = self.get_quote(market, symbol, exchange=exchange)
            if condition(quote):
                matches.append(quote)
        return tuple(matches)

    def account_domestic(
        self,
        *,
        exchange: DomesticExchange | str = DomesticExchange.KRX,
        max_pages: int = 10,
    ) -> AccountSnapshot:
        selected_exchange = _domestic_exchange(exchange)
        if selected_exchange is DomesticExchange.SOR:
            raise ValueError("국내 잔고 조회 exchange는 KRX 또는 NXT여야 합니다.")
        balance_pages = tuple(
            self._domestic_http.iter_pages(
                api_id="kt00018",
                path="/api/dostk/acnt",
                body={"qry_tp": "1", "dmst_stex_tp": selected_exchange.value},
                max_pages=max_pages,
            )
        )
        deposit = self._domestic_http.request(
            api_id="kt00001",
            path="/api/dostk/acnt",
            body={"qry_tp": "3"},
        ).body
        summary = balance_pages[0].body if balance_pages else {}
        positions: list[Position] = []
        for page in balance_pages:
            for row in _records(
                page.body.get("acnt_evlt_remn_indv_tot", []),
                _DOMESTIC_POSITION_KEYS,
            ):
                positions.append(_domestic_position(row, selected_exchange.value))
        return AccountSnapshot(
            market=Market.DOMESTIC,
            currency="KRW",
            positions=tuple(positions),
            cash=_decimal(deposit.get("entr")),
            available_to_order=_decimal(deposit.get("ord_alow_amt")),
            total_purchase=_decimal(summary.get("tot_pur_amt")),
            total_evaluation=_decimal(summary.get("tot_evlt_amt")),
            total_profit_loss=_decimal(summary.get("tot_evlt_pl")),
            profit_rate=_decimal(summary.get("tot_prft_rt")),
            raw={
                "balance": [page.body for page in balance_pages],
                "deposit": deposit,
            },
            cash_d1=_settlement_decimal(deposit.get("d1_entra")),
            cash_d2=_settlement_decimal(deposit.get("d2_entra")),
            cash_receivable=_settlement_decimal(deposit.get("ch_uncla")),
            cash_settlement_source="kt00001:d2_entra" if _settlement_decimal(deposit.get("d2_entra")) is not None else "",
        )

    def account_us(
        self,
        *,
        exchange: USExchange | str = USExchange.ALL,
        symbol: str = "",
        max_pages: int = 10,
    ) -> AccountSnapshot:
        selected_exchange = _us_exchange(exchange, allow_all=True)
        exchange_value = "" if selected_exchange is USExchange.ALL else selected_exchange.value
        balance_pages = tuple(
            self._us_http.iter_pages(
                api_id="ust21070",
                path="/api/us/acnt",
                body={"stex_tp": exchange_value, "stk_cd": _symbol(symbol) if symbol else ""},
                max_pages=max_pages,
            )
        )
        deposit = self._us_http.request(
            api_id="ust21110",
            path="/api/us/acnt",
            body={},
        ).body
        summary = balance_pages[0].body if balance_pages else {}
        positions: list[Position] = []
        for page in balance_pages:
            for row in _records(page.body.get("result_list", []), _US_POSITION_KEYS):
                positions.append(_us_position(row))
        usd_deposit = _find_currency(
            _records(deposit.get("result_list", []), _US_DEPOSIT_KEYS),
            "USD",
        )
        return AccountSnapshot(
            market=Market.US,
            currency="USD",
            positions=tuple(positions),
            cash=_decimal(usd_deposit.get("fc_entra")) if usd_deposit else None,
            available_to_order=_decimal(usd_deposit.get("fc_ord_alowa")) if usd_deposit else None,
            total_purchase=_decimal(summary.get("tot_prch_amt")),
            total_evaluation=_decimal(summary.get("tot_evlt_amt")),
            total_profit_loss=_decimal(summary.get("tot_pl_amt")),
            profit_rate=_decimal(summary.get("tot_pl_rt")),
            raw={
                "balance": [page.body for page in balance_pages],
                "deposit": deposit,
                "krw_cash": deposit.get("krw_entra"),
            },
            # The top-level ch_uncla is KRW. Only the USD row belongs in this
            # market's cash summary; this endpoint supplies no D+1/D+2 cash.
            cash_receivable=_settlement_decimal(usd_deposit.get("fc_ch_uncla")) if usd_deposit else None,
        )

    def get_account(
        self,
        market: Market | str,
        *,
        exchange: DomesticExchange | USExchange | str,
    ) -> AccountSnapshot:
        selected_market = _market(market)
        if selected_market is Market.DOMESTIC:
            return self.account_domestic(exchange=exchange)
        return self.account_us(exchange=exchange)

    def list_open_orders(
        self,
        market: Market | str,
        *,
        exchange: DomesticExchange | USExchange | str,
        symbol: str = "",
        max_pages: int = 10,
        strict: bool = False,
    ) -> tuple[OpenOrder, ...]:
        """Return outstanding orders so API acceptance is not mistaken for a fill."""
        selected_market = _market(market)
        rows: list[dict[str, Any]] = []
        if selected_market is Market.DOMESTIC:
            selected_exchange = _domestic_exchange(exchange)
            stex_tp = {
                DomesticExchange.KRX: "1",
                DomesticExchange.NXT: "2",
                DomesticExchange.SOR: "0",
            }[selected_exchange]
            body = {
                "all_stk_tp": "1" if symbol else "0",
                "trde_tp": "0",
                "stex_tp": stex_tp,
                "stk_cd": _symbol(symbol) if symbol else "",
            }
            for page in self._domestic_http.iter_pages(
                api_id="ka10075",
                path="/api/dostk/acnt",
                body=body,
                max_pages=max_pages,
            ):
                if strict:
                    _validate_order_rows(page.body.get("oso"), "oso_qty")
                rows.extend(_records(page.body.get("oso", []), _DOMESTIC_OPEN_ORDER_KEYS))
            return tuple(_domestic_open_order(row) for row in rows)

        selected_exchange = _us_exchange(exchange, allow_all=True)
        body = {
            "ord_dt": "",
            "slby_tp": "0",
            "stex_tp": "" if selected_exchange is USExchange.ALL else selected_exchange.value,
            "stk_cd": _symbol(symbol) if symbol else "",
        }
        for page in self._us_http.iter_pages(
            api_id="ust21050",
            path="/api/us/acnt",
            body=body,
            max_pages=max_pages,
        ):
            if strict:
                _validate_order_rows(page.body.get("result_list"), "ord_remnq")
            rows.extend(_records(page.body.get("result_list", []), _US_OPEN_ORDER_KEYS))
        return tuple(_us_open_order(row) for row in rows)

    def list_order_executions(
        self,
        market: Market | str,
        *,
        symbol: str,
        exchange: DomesticExchange | USExchange | str,
        max_pages: int = 10,
        strict: bool = False,
    ) -> tuple[OrderExecution, ...]:
        """Read today's order/fill records, including actual fill quantity and price."""
        selected_market = _market(market)
        if selected_market is Market.DOMESTIC:
            selected_exchange = _domestic_exchange(exchange)
            api_id, path = "ka10076", "/api/dostk/acnt"
            body = {"stk_cd": _clean_domestic_symbol(_symbol(symbol)), "qry_tp": "1",
                    "sell_tp": "0", "ord_no": "",
                    "stex_tp": {DomesticExchange.KRX: "1", DomesticExchange.NXT: "2", DomesticExchange.SOR: "0"}[selected_exchange]}
        else:
            api_id, path = "ust21510", "/api/us/acnt"
            body = {"stk_cd": _symbol(symbol), "slby_tp": "0",
                    "stex_tp": _us_exchange(exchange, allow_all=False).value}
        orders: list[OrderExecution] = []
        for page in self._http_for(selected_market).iter_pages(
            api_id=api_id, path=path, body=body, max_pages=max_pages,
        ):
            fallback = None if strict else []
            rows = page.body.get("cntr", fallback) if selected_market is Market.DOMESTIC else (
                page.body.get("result_list", page.body.get("result_lsit", fallback))
            )
            if strict:
                remaining_key = "oso_qty" if selected_market is Market.DOMESTIC else "ord_remnq"
                # Missing quantity fields must never be interpreted as a completed fill.
                _validate_order_rows(rows, remaining_key)
                for row in rows:
                    for key in ("ord_qty", "cntr_qty"):
                        if _required_decimal(row.get(key), key) < 0:
                            raise ValueError("체결 수량이 음수입니다.")
            for row in _records(rows):
                orders.append(OrderExecution(
                    order_number=str(row.get("ord_no", "")),
                    symbol=_clean_domestic_symbol(str(row.get("stk_cd", ""))) if selected_market is Market.DOMESTIC else str(row.get("stk_cd", "")),
                    side=str(row.get("io_tp_nm") or row.get("slby_tp_nm") or row.get("slby_tp", "")),
                    status=str(row.get("ord_stt") or row.get("ord_stat", "")),
                    order_quantity=_decimal(row.get("ord_qty")) or Decimal(0),
                    filled_quantity=_decimal(row.get("cntr_qty")) or Decimal(0),
                    remaining_quantity=_decimal(row.get("oso_qty", row.get("ord_remnq"))) or Decimal(0),
                    order_price=_decimal(row.get("ord_pric", row.get("ord_uv")), absolute=True) or Decimal(0),
                    fill_price=_decimal(row.get("cntr_pric") if selected_market is Market.DOMESTIC
                                        else row.get("cntr_uv"), absolute=True) or Decimal(0),
                    order_time=str(row.get("ord_tm") or row.get("ord_time", "")),
                ))
        return tuple(orders)

    def list_execution_history(
        self,
        market: Market | str,
        day: date,
        *,
        max_pages: int = 10,
    ) -> tuple[ExecutionHistoryRecord, ...]:
        """Read complete account/day order history without per-symbol calls.

        Official schemas: Kiwoom-Securities/Kiwoom-REST-API, examples/
        국내주식/계좌/get_domestic_account_order_fill_detail.py (kt00007), and
        미국주식/계좌/get_overseas_orders_by_period.py (ust21180).
        US searches the local date and following Korean date in one request;
        each returned order date/time is correlated by FillRecovery, not
        assumed to use a particular undocumented timezone.
        """
        if type(day) is not date:
            raise ValueError("체결 조회일은 datetime.date 형식이어야 합니다.")
        if type(max_pages) is not int or not 1 <= max_pages <= 100:
            raise ValueError("체결 조회 페이지 상한은 1~100 사이 정수여야 합니다.")
        selected = _market(market)
        if selected is Market.DOMESTIC:
            api_id, path = "kt00007", "/api/dostk/acnt"
            body = {"ord_dt": day.strftime("%Y%m%d"), "qry_tp": "1",
                    "stk_bond_tp": "1", "sell_tp": "0", "stk_cd": "",
                    "fr_ord_no": "", "dmst_stex_tp": "%"}
        else:
            api_id, path = "ust21180", "/api/us/acnt"
            body = {"strt_dt": day.strftime("%Y%m%d"),
                    "end_dt": (day + timedelta(days=1)).strftime("%Y%m%d"),
                    "slby_tp": "0", "stex_tp": "", "stk_cd": "", "oppo_trde_tp": "%"}
        records: list[ExecutionHistoryRecord] = []
        by_order: dict[tuple[date | None, str], ExecutionHistoryRecord] = {}
        for page in self._http_for(selected).iter_pages(
            api_id=api_id, path=path, body=body, max_pages=max_pages,
        ):
            if selected is Market.DOMESTIC:
                rows = page.body.get("acnt_ord_cntr_prps_dtl")
            else:
                # The official JSON schema and response example differ here.
                rows = page.body.get("result_list", page.body.get("result_lsit"))
            if not isinstance(rows, list):
                raise ValueError("체결 내역 응답 목록이 없거나 잘못되었습니다.")
            for row in rows:
                record = _execution_history_record(row, selected, day, api_id)
                key = record.broker_order_date, record.order_number.lstrip("0")
                previous = by_order.get(key)
                if previous is not None:
                    # Never add snapshots as if they were distinct executions.
                    if previous != record:
                        raise ValueError("동일 주문번호의 체결 내역이 서로 달라 가격을 확정할 수 없습니다.")
                    continue
                by_order[key] = record
                records.append(record)
        return tuple(records)

    def build_order(
        self,
        *,
        market: Market | str,
        side: OrderSide | str,
        symbol: str,
        quantity: int,
        exchange: DomesticExchange | USExchange | str,
        price: Decimal | int | float | str | None = None,
        order_type: str | None = None,
        stop_price: Decimal | int | float | str | None = None,
    ) -> OrderRequest:
        selected_market = _market(market)
        selected_side = _side(side)
        _validate_order_quantity(quantity)
        if selected_market is Market.US:
            if price is not None:
                price = validate_us_order_price(_decimal(price), "미국 주문 가격")
            if stop_price is not None:
                stop_price = validate_us_order_price(_decimal(stop_price), "미국 스톱 가격")
        selected_price = _order_decimal(price, "주문 가격")
        selected_stop_price = _order_decimal(stop_price, "스톱 가격")

        if selected_market is Market.DOMESTIC:
            selected_exchange = _domestic_exchange(exchange).value
            api_order_type = _domestic_order_type(order_type, selected_price)
            if api_order_type in {"0", "5", "28"} and selected_price is None:
                raise ValueError("국내 지정가/조건부/스톱 주문에는 price가 필요합니다.")
        else:
            selected_exchange = _us_exchange(exchange, allow_all=False).value
            api_order_type = _us_order_type(order_type, selected_price)
            if api_order_type in {"00", "26", "27", "30", "34"} and selected_price is None:
                raise ValueError("미국 지정가 계열 주문에는 price가 필요합니다.")
            if api_order_type in {"34", "35"} and selected_stop_price is None:
                raise ValueError("STOP/STOP LIMIT 주문에는 stop_price가 필요합니다.")

        if api_order_type in {"3", "03"} and selected_price is not None:
            raise ValueError("시장가 주문에는 price를 지정할 수 없습니다.")

        return OrderRequest(
            market=selected_market,
            side=selected_side,
            symbol=_symbol(symbol),
            quantity=quantity,
            exchange=selected_exchange,
            order_type=api_order_type,
            price=selected_price,
            stop_price=selected_stop_price,
        )

    def build_order_at_current_price(
        self,
        *,
        market: Market | str,
        side: OrderSide | str,
        symbol: str,
        quantity: int,
        exchange: DomesticExchange | USExchange | str | None = None,
    ) -> OrderRequest:
        """Fetch a quote and build a side-conservative legal limit, without submitting."""
        selected_market = _market(market)
        selected_side = _side(side)
        _validate_order_quantity(quantity)
        selected_symbol = _symbol(symbol)
        if selected_market is Market.DOMESTIC:
            selected_symbol = _clean_domestic_symbol(selected_symbol)
            selected_exchange = _domestic_exchange(exchange if exchange is not None else DomesticExchange.KRX)
        else:
            selected_exchange = (
                self.resolve_us_exchange(selected_symbol)
                if exchange is None else _us_exchange(exchange, allow_all=False)
            )
        quote = self.get_quote(selected_market, selected_symbol, exchange=selected_exchange)
        if _symbol(quote.symbol) != selected_symbol:
            raise ValueError("조회한 현재가의 종목코드가 요청 종목과 다릅니다. 주문을 중단합니다.")
        return self.build_order(
            market=selected_market, side=selected_side, symbol=selected_symbol,
            quantity=quantity, exchange=selected_exchange,
            price=current_limit_price(selected_market, selected_side, quote.price), order_type="limit",
        )

    def buy_at_current_price(
        self,
        *,
        market: Market | str,
        symbol: str,
        quantity: int,
        exchange: DomesticExchange | USExchange | str | None = None,
        confirm_live_order: str | None = None,
    ) -> OrderResult:
        """Buy once at a legal limit no higher than the quote; a fill is not guaranteed."""
        return self._trade_at_current_price(
            market=market, side=OrderSide.BUY, symbol=symbol, quantity=quantity,
            exchange=exchange, confirm_live_order=confirm_live_order,
        )

    def sell_at_current_price(
        self,
        *,
        market: Market | str,
        symbol: str,
        quantity: int,
        exchange: DomesticExchange | USExchange | str | None = None,
        confirm_live_order: str | None = None,
    ) -> OrderResult:
        """Sell once at a legal limit no lower than the quote; a fill is not guaranteed."""
        return self._trade_at_current_price(
            market=market, side=OrderSide.SELL, symbol=symbol, quantity=quantity,
            exchange=exchange, confirm_live_order=confirm_live_order,
        )

    def _trade_at_current_price(
        self,
        *,
        market: Market | str,
        side: OrderSide,
        symbol: str,
        quantity: int,
        exchange: DomesticExchange | USExchange | str | None,
        confirm_live_order: str | None,
    ) -> OrderResult:
        self._check_live_order(_market(market), confirm_live_order)
        request = self.build_order_at_current_price(
            market=market, side=side, symbol=symbol, quantity=quantity, exchange=exchange,
        )
        return self.place_order(request, confirm_live_order=confirm_live_order)

    def place_order(
        self,
        request: OrderRequest,
        *,
        confirm_live_order: str | None = None,
    ) -> OrderResult:
        # Also validate directly constructed OrderRequest values before any HTTP call.
        request = self.build_order(
            market=request.market, side=request.side, symbol=request.symbol,
            quantity=request.quantity, exchange=request.exchange, price=request.price,
            order_type=request.order_type, stop_price=request.stop_price,
        )
        self._check_live_order(request.market, confirm_live_order)
        if request.market is Market.DOMESTIC:
            api_id = "kt10000" if request.side is OrderSide.BUY else "kt10001"
            body: dict[str, Any] = {
                "dmst_stex_tp": request.exchange,
                "stk_cd": request.symbol,
                "ord_qty": str(request.quantity),
                "trde_tp": request.order_type,
                "ord_uv": _api_decimal(request.price),
                "cond_uv": _api_decimal(request.stop_price),
            }
            path = "/api/dostk/ordr"
        else:
            api_id = "ust20000" if request.side is OrderSide.BUY else "ust20001"
            body = {
                "stex_tp": request.exchange,
                "stk_cd": request.symbol,
                "ord_qty": str(request.quantity),
                "trde_tp": request.order_type,
                "ord_uv": _api_decimal(request.price),
            }
            if request.side is OrderSide.SELL:
                body["stop_pric"] = _api_decimal(request.stop_price)
            path = "/api/us/ordr"
        response = self._http_for(request.market).request(
            api_id=api_id,
            path=path,
            body=body,
            retry_auth=False,
        ).body
        order_number = _confirmed_order_number(response, "주문")
        return OrderResult(
            accepted=True,
            mode=self.mode,
            request=request,
            order_number=order_number,
            message=str(response.get("return_msg", "")),
            raw=response,
        )

    def buy(
        self,
        *,
        market: Market | str,
        symbol: str,
        quantity: int,
        exchange: DomesticExchange | USExchange | str,
        price: Decimal | int | float | str | None = None,
        order_type: str | None = None,
        confirm_live_order: str | None = None,
    ) -> OrderResult:
        request = self.build_order(
            market=market,
            side=OrderSide.BUY,
            symbol=symbol,
            quantity=quantity,
            exchange=exchange,
            price=price,
            order_type=order_type,
        )
        return self.place_order(request, confirm_live_order=confirm_live_order)

    def sell(
        self,
        *,
        market: Market | str,
        symbol: str,
        quantity: int,
        exchange: DomesticExchange | USExchange | str,
        price: Decimal | int | float | str | None = None,
        order_type: str | None = None,
        stop_price: Decimal | int | float | str | None = None,
        confirm_live_order: str | None = None,
    ) -> OrderResult:
        request = self.build_order(
            market=market,
            side=OrderSide.SELL,
            symbol=symbol,
            quantity=quantity,
            exchange=exchange,
            price=price,
            order_type=order_type,
            stop_price=stop_price,
        )
        return self.place_order(request, confirm_live_order=confirm_live_order)

    def cancel_order(
        self,
        *,
        market: Market | str,
        original_order_number: str,
        symbol: str,
        exchange: DomesticExchange | USExchange | str,
        quantity: int = 0,
        confirm_live_order: str | None = None,
    ) -> CancelResult:
        """Cancel an open order. Domestic quantity 0 means cancel all remaining shares."""
        selected_market = _market(market)
        self._check_live_order(selected_market, confirm_live_order)
        if not isinstance(original_order_number, str) or not original_order_number.strip():
            raise ValueError("원주문번호가 필요합니다.")
        original_order_number = original_order_number.strip()
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
            raise ValueError("취소 수량은 0 이상의 정수여야 합니다.")
        if quantity > 999_999_999_999:
            raise ValueError("취소 수량은 12자리 이하여야 합니다.")

        if selected_market is Market.DOMESTIC:
            body = {
                "dmst_stex_tp": _domestic_exchange(exchange).value,
                "orig_ord_no": original_order_number,
                "stk_cd": _symbol(symbol),
                "cncl_qty": str(quantity),
            }
            api_id = "kt10003"
            path = "/api/dostk/ordr"
        else:
            body = {
                "orig_ord_no": original_order_number,
                "stex_tp": _us_exchange(exchange, allow_all=False).value,
                "stk_cd": _symbol(symbol),
            }
            api_id = "ust20003"
            path = "/api/us/ordr"
        response = self._http_for(selected_market).request(
            api_id=api_id,
            path=path,
            body=body,
            retry_auth=False,
        ).body
        order_number = _confirmed_order_number(response, "취소 주문")
        cancelled_quantity = _decimal(response.get("cncl_qty") or response.get("cncl_ord_qty"))
        return CancelResult(
            accepted=True,
            mode=self.mode,
            market=selected_market,
            original_order_number=original_order_number,
            cancel_order_number=order_number,
            cancelled_quantity=cancelled_quantity,
            message=str(response.get("return_msg", "")),
            raw=response,
        )

    async def list_saved_conditions(
        self,
        market: Market | str,
    ) -> tuple[SavedCondition, ...]:
        selected_market = _market(market)
        return await self._condition_client(selected_market).list_conditions(selected_market)

    async def run_saved_condition(
        self,
        market: Market | str,
        sequence: str,
        *,
        domestic_exchange: str = "K",
        max_pages: int = 10,
    ) -> tuple[ConditionMatch, ...]:
        selected_market = _market(market)
        return await self._condition_client(selected_market).run_condition(
            selected_market,
            sequence,
            domestic_exchange=domestic_exchange,
            max_pages=max_pages,
        )

    def _condition_client(self, market: Market) -> KiwoomConditionClient:
        config = self._config_for(market)
        http = self._http_for(market)
        return KiwoomConditionClient(
            config,
            http.get_access_token,
            connect_factory=self._websocket_connect_factory,
        )

    def _check_live_order(self, market: Market, confirmation: str | None) -> None:
        config = self._config_for(market)
        if config.mode is TradingMode.DEMO:
            return
        if not config.allow_live_orders:
            raise LiveOrderConfirmationRequired(
                "실전 주문이 비활성화되어 있습니다. allow_live_orders=True를 명시하세요."
            )
        if confirmation != LIVE_ORDER_CONFIRMATION:
            raise LiveOrderConfirmationRequired(
                f'실전 주문에는 confirm_live_order="{LIVE_ORDER_CONFIRMATION}"가 필요합니다.'
            )


def _market(value: Market | str) -> Market:
    if isinstance(value, Market):
        return value
    try:
        return Market(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError("market은 'domestic' 또는 'us'여야 합니다.") from exc


def _side(value: OrderSide | str) -> OrderSide:
    if isinstance(value, OrderSide):
        return value
    try:
        return OrderSide(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError("side는 'buy' 또는 'sell'이어야 합니다.") from exc


def _domestic_exchange(value: DomesticExchange | str) -> DomesticExchange:
    if isinstance(value, DomesticExchange):
        return value
    try:
        return DomesticExchange(str(value).strip().upper())
    except ValueError as exc:
        raise ValueError("국내 거래소는 KRX, NXT 또는 SOR이어야 합니다.") from exc


def _us_exchange(value: USExchange | str, *, allow_all: bool) -> USExchange:
    if isinstance(value, USExchange):
        result = value
    else:
        normalized = str(value).strip().upper()
        aliases = {"AMEX": "NA", "NASDAQ": "ND", "NYSE": "NY", "ALL": "%"}
        try:
            result = USExchange(aliases.get(normalized, normalized))
        except ValueError as exc:
            raise ValueError("미국 거래소는 AMEX, NASDAQ 또는 NYSE여야 합니다.") from exc
    if result is USExchange.ALL and not allow_all:
        raise ValueError("이 기능에서는 미국 거래소를 하나 선택해야 합니다.")
    return result


def _symbol(value: str) -> str:
    result = normalize_symbol(str(value))
    if not result:
        raise ValueError("종목코드가 필요합니다.")
    return result


def _domestic_api_symbol(symbol: str, exchange: DomesticExchange) -> str:
    cleaned = _clean_domestic_symbol(_symbol(symbol))
    if exchange is DomesticExchange.NXT:
        return f"{cleaned}_NX"
    if exchange is DomesticExchange.SOR:
        return f"{cleaned}_AL"
    return cleaned


def _clean_domestic_symbol(symbol: str) -> str:
    result = symbol.strip().upper()
    if result.startswith("A") and result[1:].isdigit():
        result = result[1:]
    for suffix in ("_NX", "_AL"):
        if result.endswith(suffix):
            result = result[: -len(suffix)]
    return result


def _validate_order_rows(rows, remaining_key: str):
    if not isinstance(rows, list):
        raise ValueError("미체결 응답 목록이 없거나 잘못되었습니다. 자동 주문을 차단합니다.")
    for row in rows:
        if not isinstance(row, dict) or not row.get("stk_cd") or not row.get("ord_no"):
            raise ValueError("미체결 응답의 종목·주문번호를 확인할 수 없습니다.")
        quantity = _required_decimal(row.get(remaining_key), "미체결 잔량")
        if quantity < 0:
            raise ValueError("미체결 잔량이 음수입니다.")


def _execution_history_record(row: Any, market: Market, day: date,
                              api_id: str) -> ExecutionHistoryRecord:
    if not isinstance(row, dict):
        raise ValueError("체결 내역 행이 올바른 객체가 아닙니다.")

    def text_field(key: str, *, required: bool = False) -> str:
        value = row.get(key, "")
        if not isinstance(value, str) or (required and not value.strip()):
            raise ValueError(f"체결 내역의 {key} 값을 확인할 수 없습니다.")
        return value.strip()

    def quantity(key: str) -> Decimal:
        value = _required_decimal(row.get(key), key)
        if value < 0 or value != value.to_integral_value():
            raise ValueError(f"체결 내역의 {key} 수량이 올바르지 않습니다.")
        return value

    def price(key: str) -> Decimal | None:
        raw = row.get(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return None
        value = _required_decimal(raw, key)
        if value < 0:
            raise ValueError(f"체결 내역의 {key} 가격이 음수입니다.")
        return value if value > 0 else None

    order_number = text_field("ord_no", required=True)
    if not order_number.isascii() or not order_number.isdigit() or int(order_number) == 0:
        raise ValueError("체결 내역의 주문번호가 올바르지 않습니다.")
    symbol = text_field("stk_cd", required=True)
    exchange = ""
    broker_order_date = None
    if market is Market.DOMESTIC:
        symbol = _clean_domestic_symbol(symbol)
        # The account endpoint also documents J:ELW / Q:ETN prefixes. Keep
        # those prefixes so unrelated account instruments cannot alias a stock.
        digits = symbol[1:] if symbol.startswith(("J", "Q")) else symbol
        if len(digits) != 6 or not digits.isascii() or not digits.isdigit():
            raise ValueError("체결 내역의 국내 종목번호가 올바르지 않습니다.")
        exchange = text_field("dmst_stex_tp", required=True).upper()
        if exchange not in {"KRX", "NXT", "SOR"}:
            raise ValueError("체결 내역의 국내 거래소를 확인할 수 없습니다.")
        side_text = text_field("io_tp_nm", required=True)
        order_time, fill_time = text_field("ord_tm"), ""
        status = text_field("acpt_tp")
        original_order = text_field("ori_ord")
        currency = "KRW"
    else:
        symbol = _symbol(symbol)
        currency = text_field("crnc_code", required=True).upper()
        if currency != "USD":
            raise ValueError("미국 체결 내역의 통화가 USD가 아닙니다.")
        # stex_nm is a country display name (e.g. 미국), not ND/NY/NA.
        side_text = text_field("slby_tp_nm", required=True)
        order_time, fill_time = text_field("ord_time"), text_field("cntr_time")
        status = text_field("ord_stat_nm")
        original_order = ""
        if api_id == "ust21180":
            raw_day = text_field("ord_dt", required=True)
            if len(raw_day) != 8 or not raw_day.isascii() or not raw_day.isdigit():
                raise ValueError("미국 체결 내역의 반환 주문일이 올바르지 않습니다.")
            broker_order_date = datetime.strptime(raw_day, "%Y%m%d").date()
            if not day <= broker_order_date <= day + timedelta(days=1):
                raise ValueError("미국 체결 내역의 반환 주문일이 조회 범위를 벗어났습니다.")
    buy, sell = "매수" in side_text, "매도" in side_text
    if buy == sell:
        raise ValueError("체결 내역의 매수·매도 방향을 확인할 수 없습니다.")
    order_qty, filled, remaining = (quantity(key) for key in ("ord_qty", "cntr_qty", "ord_remnq"))
    if order_qty <= 0 or filled > order_qty or remaining > order_qty or filled + remaining > order_qty:
        raise ValueError("체결 내역의 주문·체결·잔량 수량이 일치하지 않습니다.")
    reported = price("cntr_uv")
    amount = price("cntr_amt") if api_id == "ust21180" else None
    fill_price = None
    if filled == 0:
        if amount is not None:
            raise ValueError("미체결 주문에 양수 체결금액이 반환되었습니다.")
        basis = "not_filled"
    elif amount is not None:
        # The account's cumulative executed amount is authoritative. cntr_uv
        # may be the last execution price, so never multiply it by all shares.
        basis, fill_price = "broker_average", amount / filled
    elif reported is None:
        basis = "missing"
    elif order_qty == filled == 1:
        basis, fill_price = "single_share", reported
    else:
        basis = "unverified_multi_share"
    return ExecutionHistoryRecord(
        market=market, order_date=day, order_number=order_number, symbol=symbol,
        exchange=exchange, side=OrderSide.BUY if buy else OrderSide.SELL,
        order_quantity=order_qty, filled_quantity=filled, remaining_quantity=remaining,
        order_price=price("ord_uv"), fill_price=fill_price, reported_fill_price=reported,
        price_basis=basis, order_time=order_time, fill_time=fill_time, status=status,
        currency=currency, source_api=api_id, original_order_number=original_order,
        broker_order_date=broker_order_date, fill_amount=amount,
    )


def _confirmed_order_number(response: Mapping[str, Any], operation: str) -> str:
    # Do not infer acceptance from HTTP 200 or coerce null/container values to strings.
    code = response.get("return_code")
    success = (type(code) is int and code == 0) or (
        isinstance(code, str) and code.strip() == "0"
    )
    number = response.get("ord_no")
    if not success or not isinstance(number, str) or not number.strip():
        raise OrderOutcomeUnknown(
            f"{operation} 응답의 성공 코드 또는 주문번호를 확인할 수 없습니다. "
            "접수 여부가 불명확하므로 재전송하지 말고 주문·체결 내역을 먼저 확인하세요."
        )
    return number.strip()


def _decimal(value: Any, *, absolute: bool = False) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    return abs(number) if absolute else number


def _settlement_decimal(value: Any) -> Decimal | None:
    """Keep signed cash amounts; missing or malformed forecasts stay unknown."""
    number = _decimal(value)
    return number if number is not None and number.is_finite() else None


def _required_decimal(value: Any, label: str, *, absolute: bool = False) -> Decimal:
    result = _decimal(value, absolute=absolute)
    if result is None or not result.is_finite():
        raise ValueError(f"키움 응답에서 {label} 값을 숫자로 해석할 수 없습니다: {value!r}")
    return result


def _validate_order_quantity(quantity: int) -> None:
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("주문 수량은 1 이상의 정수여야 합니다.")
    if quantity > 999_999_999_999:
        raise ValueError("주문 수량은 12자리 이하여야 합니다.")


def _order_decimal(value: Any, label: str) -> Decimal | None:
    if value is None:
        return None
    number = _decimal(value)
    if number is None or not number.is_finite() or number <= 0:
        raise ValueError(f"{label}은 0보다 큰 유한한 숫자여야 합니다.")
    if number.adjusted() > 11 or number.as_tuple().exponent < -12:
        raise ValueError(f"{label}이 API 입력 범위를 벗어났습니다.")
    if len(format(number, "f")) > 12:
        raise ValueError(f"{label}은 소수점을 포함해 12자리 이하여야 합니다.")
    return number


def _api_decimal(value: Decimal | None) -> str:
    if value is None:
        return ""
    return format(value, "f")


def _records(value: Any, keys: Sequence[str] = ()) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for row in value:
        if isinstance(row, Mapping):
            result.append(dict(row))
        elif keys and isinstance(row, (list, tuple)):
            result.append(dict(zip(keys, row)))
    return result


def _looks_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "y", "yes"}


def _stock_matches(stock: StockInfo, query: str) -> bool:
    if not query:
        return True
    values = (stock.symbol, stock.name, stock.english_name or "")
    return any(query in value.casefold() for value in values)


def _validate_limit(limit: int | None) -> None:
    if limit is not None and limit <= 0:
        raise ValueError("limit은 1 이상이어야 합니다.")


def _api_date(value: date | str | None) -> str:
    if value is None:
        return date.today().strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise ValueError("날짜는 YYYYMMDD 형식이어야 합니다.") from exc
    return text


def _trade_date(value: Any) -> date:
    try:
        return datetime.strptime(str(value).strip(), "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"키움 일봉 응답의 날짜 형식이 잘못되었습니다: {value!r}") from exc


def _domestic_daily_bar(
    row: dict[str, Any],
    fallback_symbol: str,
    exchange: DomesticExchange,
) -> DailyBar:
    return DailyBar(
        market=Market.DOMESTIC,
        symbol=_clean_domestic_symbol(fallback_symbol),
        exchange=exchange.value,
        trade_date=_trade_date(row.get("dt")),
        open=_decimal(row.get("open_pric"), absolute=True),
        high=_decimal(row.get("high_pric"), absolute=True),
        low=_decimal(row.get("low_pric"), absolute=True),
        close=_decimal(row.get("cur_prc"), absolute=True),
        volume=_decimal(row.get("trde_qty"), absolute=True),
        currency="KRW",
        trade_value=_decimal(row.get("trde_prica"), absolute=True),
        change=_decimal(row.get("pred_pre")),
        change_rate=_decimal(row.get("trde_tern_rt")),
        raw=row,
    )


def _us_daily_bar(
    row: dict[str, Any],
    fallback_symbol: str,
    exchange: USExchange,
    apply_exchange_rate: bool,
) -> DailyBar:
    return DailyBar(
        market=Market.US,
        symbol=_symbol(fallback_symbol),
        exchange=exchange.value,
        trade_date=_trade_date(row.get("dt")),
        open=_decimal(row.get("open_pric"), absolute=True),
        high=_decimal(row.get("high_pric"), absolute=True),
        low=_decimal(row.get("low_pric"), absolute=True),
        close=_decimal(row.get("cur_prc"), absolute=True),
        volume=_decimal(row.get("acc_trde_qty"), absolute=True),
        currency="KRW" if apply_exchange_rate else "USD",
        trade_value=_decimal(row.get("acc_trde_prica"), absolute=True),
        change=_decimal(row.get("pred_pre")),
        change_rate=_decimal(row.get("flu_rt")),
        adjustment_type=str(row.get("upd_stkpc_tp") or "") or None,
        adjustment_rate=_decimal(row.get("upd_rt")),
        raw=row,
    )


def _domestic_order_type(value: str | None, price: Decimal | None) -> str:
    if value is None:
        return "0" if price is not None else "3"
    aliases = {"LIMIT": "0", "MARKET": "3"}
    return aliases.get(value.strip().upper(), value.strip())


def _us_order_type(value: str | None, price: Decimal | None) -> str:
    if value is None:
        return "00" if price is not None else "03"
    aliases = {"LIMIT": "00", "MARKET": "03"}
    return aliases.get(value.strip().upper(), value.strip())


def _domestic_position(row: dict[str, Any], exchange: str) -> Position:
    return Position(
        market=Market.DOMESTIC,
        symbol=_clean_domestic_symbol(str(row.get("stk_cd", ""))),
        name=str(row.get("stk_nm", "")),
        exchange=exchange,
        currency="KRW",
        quantity=_decimal(row.get("rmnd_qty"), absolute=True) or Decimal(0),
        sellable_quantity=_decimal(row.get("trde_able_qty"), absolute=True) or Decimal(0),
        average_price=_decimal(row.get("pur_pric"), absolute=True) or Decimal(0),
        current_price=_decimal(row.get("cur_prc"), absolute=True) or Decimal(0),
        evaluation_amount=_decimal(row.get("evlt_amt")) or Decimal(0),
        profit_loss=_decimal(row.get("evltv_prft")) or Decimal(0),
        profit_rate=_decimal(row.get("prft_rt")) or Decimal(0),
        raw=row,
    )


def _us_position_exchange(row: dict[str, Any]) -> str:
    """Resolve explicit venue labels without guessing from a country or ticker.

    ust21070 can return a Korean exchange name instead of an orderable code.
    Keep unrecognized/missing or conflicting venues non-tradable; the original
    fields remain available on Position.raw for diagnosis.
    """
    venues = {
        normalize_us_exchange(row.get(key))
        for key in ("stex_code", "stex_tp", "stex_nm")
    } & {"ND", "NY", "NA"}
    return venues.pop() if len(venues) == 1 else ""


def _us_position(row: dict[str, Any]) -> Position:
    return Position(
        market=Market.US,
        symbol=normalize_symbol(str(row.get("stk_cd") or "")),
        name=str(row.get("frgn_stk_nm", "")),
        exchange=_us_position_exchange(row),
        currency=str(row.get("crnc_code") or "USD"),
        quantity=_decimal(row.get("poss_qty"), absolute=True) or Decimal(0),
        sellable_quantity=_decimal(row.get("sell_alowq"), absolute=True) or Decimal(0),
        average_price=_decimal(row.get("frgn_stk_book_uv"), absolute=True) or Decimal(0),
        current_price=_decimal(row.get("now_pric"), absolute=True) or Decimal(0),
        evaluation_amount=_decimal(row.get("evlt_amt")) or Decimal(0),
        profit_loss=_decimal(row.get("pl_amt")) or Decimal(0),
        profit_rate=_decimal(row.get("pl_rt")) or Decimal(0),
        raw=row,
    )


def _find_currency(rows: list[dict[str, Any]], currency: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("crnc_code", "")).upper() == currency.upper():
            return row
    return None


def _domestic_open_order(row: dict[str, Any]) -> OpenOrder:
    return OpenOrder(
        market=Market.DOMESTIC,
        order_number=str(row.get("ord_no", "")),
        symbol=_clean_domestic_symbol(str(row.get("stk_cd", ""))),
        name=str(row.get("stk_nm", "")),
        exchange=str(row.get("stex_tp_txt") or row.get("stex_tp") or ""),
        side=str(row.get("io_tp_nm", "")),
        status=str(row.get("ord_stt", "")),
        order_quantity=_decimal(row.get("ord_qty"), absolute=True) or Decimal(0),
        filled_quantity=_decimal(row.get("cntr_qty"), absolute=True) or Decimal(0),
        remaining_quantity=_decimal(row.get("oso_qty"), absolute=True) or Decimal(0),
        order_price=_decimal(row.get("ord_pric"), absolute=True) or Decimal(0),
        raw=row,
    )


def _us_open_order(row: dict[str, Any]) -> OpenOrder:
    return OpenOrder(
        market=Market.US,
        order_number=str(row.get("ord_no", "")),
        symbol=str(row.get("stk_cd", "")),
        name=str(row.get("frgn_stk_nm", "")),
        exchange=str(row.get("stex_nm", "")),
        side=str(row.get("slby_tp_nm") or row.get("frgn_trde_nm") or ""),
        status=str(row.get("ord_stat", "")),
        order_quantity=_decimal(row.get("ord_qty"), absolute=True) or Decimal(0),
        filled_quantity=_decimal(row.get("cntr_qty"), absolute=True) or Decimal(0),
        remaining_quantity=_decimal(row.get("ord_remnq"), absolute=True) or Decimal(0),
        order_price=_decimal(row.get("ord_uv"), absolute=True) or Decimal(0),
        raw=row,
    )


_DOMESTIC_STOCK_KEYS = (
    "code",
    "name",
    "listCount",
    "auditInfo",
    "regDay",
    "lastPrice",
    "state",
    "marketCode",
    "marketName",
    "upName",
    "upSizeName",
    "orderWarning",
    "companyClassName",
    "nxtEnable",
)

_US_STOCK_KEYS = ("stex_tp", "stk_cd", "stk_nm", "stk_enm", "mkgb", "upgb", "isEtf")

_DOMESTIC_POSITION_KEYS = (
    "stk_cd",
    "stk_nm",
    "evltv_prft",
    "prft_rt",
    "pur_pric",
    "pred_close_pric",
    "rmnd_qty",
    "trde_able_qty",
    "cur_prc",
    "pred_buyq",
    "pred_sellq",
    "tdy_buyq",
    "tdy_sellq",
    "pur_amt",
    "pur_cmsn",
    "evlt_amt",
    "sell_cmsn",
    "tax",
    "sum_cmsn",
    "poss_rt",
    "crd_tp",
    "crd_tp_nm",
    "crd_loan_dt",
)

_US_POSITION_KEYS = (
    "stex_nm",
    "crnc_code",
    "stk_cd",
    "frgn_stk_nm",
    "qty",
    "poss_qty",
    "sell_alowq",
    "pred_cntr_sellq",
    "pred_cntr_buyq",
    "tdy_cntr_sellq",
    "tdy_cntr_buyq",
    "frgn_stk_book_uv",
    "now_pric",
    "evlt_amt",
    "pl_amt",
    "pl_rt",
    "evlt_amt_krw",
    "pl_amt_krw",
    "natn_nm",
    "exch_rate",
    "frgn_stk_book_uv_krw",
    "now_pric_krw",
    "frgn_stk_book_amt",
    "frgn_stk_book_amt_krw",
)

_US_DEPOSIT_KEYS = (
    "crnc_code",
    "crnc_nm",
    "fc_entra",
    "fc_pymn_alowa",
    "futr_repl_profa",
    "fc_booka",
    "fc_ord_alowa",
    "futr_profa_booka",
    "fc_ch_uncla",
    "fc_etc_loana",
)

_DOMESTIC_OPEN_ORDER_KEYS = (
    "acnt_no",
    "ord_no",
    "mang_empno",
    "stk_cd",
    "tsk_tp",
    "ord_stt",
    "stk_nm",
    "ord_qty",
    "ord_pric",
    "oso_qty",
    "cntr_tot_amt",
    "orig_ord_no",
    "io_tp_nm",
    "trde_tp",
    "tm",
    "cntr_no",
    "cntr_pric",
    "cntr_qty",
    "cur_prc",
    "sel_bid",
    "buy_bid",
    "unit_cntr_pric",
    "unit_cntr_qty",
    "tdy_trde_cmsn",
    "tdy_trde_tax",
    "ind_invsr",
    "stex_tp",
    "stex_tp_txt",
    "sor_yn",
    "stop_pric",
)

_US_OPEN_ORDER_KEYS = (
    "ord_cntr_tp",
    "ord_no",
    "orig_ord_no",
    "stex_nm",
    "crnc_code",
    "stk_cd",
    "frgn_stk_nm",
    "frgn_trde_tp",
    "frgn_trde_nm",
    "slby_tp",
    "slby_tp_nm",
    "ord_qty",
    "ord_uv",
    "stop_pric",
    "cntr_qty",
    "cntr_uv",
    "mdfy_qty",
    "mdfy_uv",
    "cncl_qty",
    "ord_remnq",
    "ord_time",
    "ord_resp_time",
    "ord_stat",
    "rsrv_tp",
    "natn_nm",
)
