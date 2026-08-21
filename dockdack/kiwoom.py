"""High-level Kiwoom broker functions for Korean and US stocks."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Mapping, Sequence

from dockdack.conditions import ConnectFactory, KiwoomConditionClient
from dockdack.config import KiwoomConfig
from dockdack.exceptions import LiveOrderConfirmationRequired
from dockdack.http import HttpTransport, KiwoomHTTPClient
from dockdack.models import (
    AccountSnapshot,
    CancelResult,
    ConditionMatch,
    DomesticExchange,
    Market,
    OpenOrder,
    OrderRequest,
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
            rows.extend(_records(page.body.get("result_list", []), _US_OPEN_ORDER_KEYS))
        return tuple(_us_open_order(row) for row in rows)

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
        if quantity <= 0:
            raise ValueError("주문 수량은 1 이상이어야 합니다.")
        selected_price = _decimal(price)
        selected_stop_price = _decimal(stop_price)
        if selected_price is not None and selected_price <= 0:
            raise ValueError("주문 가격은 0보다 커야 합니다.")
        if selected_stop_price is not None and selected_stop_price <= 0:
            raise ValueError("스톱 가격은 0보다 커야 합니다.")

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

    def place_order(
        self,
        request: OrderRequest,
        *,
        confirm_live_order: str | None = None,
    ) -> OrderResult:
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
        ).body
        return OrderResult(
            accepted=True,
            mode=self.mode,
            request=request,
            order_number=str(response.get("ord_no", "")),
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
        original_order_number = str(original_order_number).strip()
        if not original_order_number:
            raise ValueError("원주문번호가 필요합니다.")
        if quantity < 0:
            raise ValueError("취소 수량은 0 이상이어야 합니다.")

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
        ).body
        cancelled_quantity = _decimal(response.get("cncl_qty") or response.get("cncl_ord_qty"))
        return CancelResult(
            accepted=True,
            mode=self.mode,
            market=selected_market,
            original_order_number=original_order_number,
            cancel_order_number=str(response.get("ord_no", "")),
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
    result = str(value).strip().upper()
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


def _decimal(value: Any, *, absolute: bool = False) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    return abs(number) if absolute else number


def _required_decimal(value: Any, label: str, *, absolute: bool = False) -> Decimal:
    result = _decimal(value, absolute=absolute)
    if result is None:
        raise ValueError(f"키움 응답에서 {label} 값을 숫자로 해석할 수 없습니다: {value!r}")
    return result


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


def _us_position(row: dict[str, Any]) -> Position:
    return Position(
        market=Market.US,
        symbol=str(row.get("stk_cd", "")),
        name=str(row.get("frgn_stk_nm", "")),
        exchange=str(row.get("stex_nm", "")),
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
    return rows[0] if rows else None


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
