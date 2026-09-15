"""Qt-independent operations bound to one explicitly selected trading environment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Callable
from collections import OrderedDict
from threading import Event

from dockdack import KiwoomBroker, KiwoomConfig, Market, OrderRequest, OrderResult, TradingMode
from dockdack.cli import identify_symbol
from dockdack.exceptions import ConfigurationError, OrderNotSent, OrderOutcomeUnknown
from dockdack.http import order_send_guard
from dockdack.kiwoom import LIVE_ORDER_CONFIRMATION


LIVE_RISK_ACKNOWLEDGEMENT = "REAL_TRADING_RISK_ACKNOWLEDGED"


@dataclass(frozen=True)
class Instrument:
    market: Market
    symbol: str
    exchange: str

    @property
    def currency(self) -> str:
        return "KRW" if self.market is Market.DOMESTIC else "USD"


def demo_broker(market: Market) -> KiwoomBroker:
    return KiwoomBroker(KiwoomConfig.from_env(TradingMode.DEMO, market=market))


class TradingService:
    """A single worker uses this service; brokers/tokens are reused per market.

    Switching environments requires a new service. Live permission is deliberately
    session-only and independent of the persisted SDK live-order configuration.
    """

    def __init__(self, broker_factory: Callable[[Market], KiwoomBroker] | None = None,
                 *, mode: TradingMode = TradingMode.DEMO):
        self._mode = TradingMode(mode)
        self.factory = broker_factory or (demo_broker if self.mode is TradingMode.DEMO else
                                         lambda market: KiwoomBroker(KiwoomConfig.from_env(self.mode, market=market)))
        self.brokers: dict[Market, KiwoomBroker] = {}
        self.exchanges: dict[str, str] = {}
        self._live_ack = Event()
        self._prepared = OrderedDict()
        self._frozen_configs = {}
        self._configuration_errors = {}
        self._storage_scope = "demo"
        if self.mode is TradingMode.REAL:
            # Freeze key selection without token/network requests. A changed key
            # requires a new service and therefore a different bound ledger.
            for market in Market:
                try:
                    if broker_factory is None:
                        config = KiwoomConfig.from_env(self.mode, market=market)
                    else:
                        broker = self.factory(market)
                        self._validate_broker(broker, market)
                        config = broker.domestic_config if market is Market.DOMESTIC else broker.us_config
                        self.brokers[market] = broker
                    self._frozen_configs[market] = config
                except ConfigurationError as exc:
                    self._configuration_errors[market] = exc
            # Lazy import avoids WatchStore -> gui_service -> WatchStore cycle.
            from dockdack.environment_store import scope_for_configs
            self._storage_scope = scope_for_configs(self._frozen_configs)
            if broker_factory is None:
                self.factory = self._configured_broker

    def _configured_broker(self, market):
        if market in self._configuration_errors:
            raise self._configuration_errors[market]
        return KiwoomBroker(self._frozen_configs[market])

    @property
    def mode(self) -> TradingMode:
        return self._mode

    @property
    def storage_scope(self) -> str:
        return self._storage_scope

    @property
    def live_risk_acknowledged(self) -> bool:
        return self._live_ack.is_set()

    def acknowledge_live_risk(self, confirmation: str):
        self.revoke_live_risk()
        if self.mode is not TradingMode.REAL or confirmation != LIVE_RISK_ACKNOWLEDGEMENT:
            raise ValueError("실전 환경의 실제 자금 손실 위험 확인이 필요합니다.")
        self._live_ack.set()

    def revoke_live_risk(self):
        self._live_ack.clear()
        self._prepared.clear()

    def _validate_broker(self, broker, market):
        expected_url = "https://api.kiwoom.com" if self.mode is TradingMode.REAL else "https://mockapi.kiwoom.com"
        selected = broker.domestic_config if market is Market.DOMESTIC else broker.us_config
        http_config = broker._http_for(market).config
        for config in (broker.config, selected, http_config):
            if broker.mode is not self.mode or config.mode is not self.mode or config.base_url != expected_url:
                label = "모의투자 전용" if self.mode is TradingMode.DEMO else "실전투자 전용"
                raise ValueError(f"현재 선택은 {label}입니다. API 환경·키·접속 주소가 일치해야 합니다.")
        if (selected.app_key, selected.secret_key) != (http_config.app_key, http_config.secret_key):
            raise ValueError("선택된 API 설정과 실제 전송 클라이언트의 키가 다릅니다.")
        if not selected.app_key.strip() or not selected.secret_key.strip():
            raise ValueError("선택한 환경의 App Key와 Secret Key가 필요합니다.")
        frozen = self._frozen_configs.get(market)
        if frozen is not None and (frozen.app_key, frozen.secret_key) != (selected.app_key, selected.secret_key):
            raise ValueError("API 키가 세션의 매매 기록 범위와 달라졌습니다. 환경을 다시 선택하세요.")

    def broker(self, market: Market) -> KiwoomBroker:
        market = Market(market)
        if market not in self.brokers:
            broker = self.factory(market)
            self._validate_broker(broker, market)
            self.brokers[market] = broker
        broker = self.brokers[market]
        self._validate_broker(broker, market)
        return broker

    def ensure_demo(self, instrument: Instrument):
        if self.mode is not TradingMode.DEMO:
            raise ValueError("이 기능은 모의투자 전용입니다.")
        self.ensure_environment(instrument)

    def ensure_environment(self, instrument: Instrument):
        self.broker(instrument.market)

    def ensure_order_permission(self, instrument: Instrument):
        broker = self.broker(instrument.market)
        if self.mode is TradingMode.REAL:
            if not self.live_risk_acknowledged:
                raise ValueError("이번 세션의 실전투자 위험 경고를 확인해야 주문할 수 있습니다.")
            selected = broker.domestic_config if instrument.market is Market.DOMESTIC else broker.us_config
            if not selected.allow_live_orders or not broker._http_for(instrument.market).config.allow_live_orders:
                raise ValueError("실전 주문 잠금: DOCKDACK_ALLOW_LIVE_ORDERS=true 설정이 필요합니다.")

    def resolve(self, symbol: str, exchange: str = "") -> Instrument:
        market, symbol = identify_symbol(symbol)
        if market is Market.DOMESTIC:
            if exchange not in {"", "KRX"}:
                raise ValueError("이 화면의 국내 거래소는 KRX입니다.")
            exchange = "KRX"
        elif exchange:
            if exchange not in {"ND", "NY", "NA"}:
                raise ValueError("미국 거래소는 NASDAQ, NYSE, AMEX 중 하나입니다.")
        else:
            if symbol not in self.exchanges:
                self.exchanges[symbol] = self.broker(market).resolve_us_exchange(symbol).value
            exchange = self.exchanges[symbol]
        return Instrument(market, symbol, exchange)

    def quote(self, instrument: Instrument):
        return self.broker(instrument.market).get_quote(
            instrument.market, instrument.symbol, exchange=instrument.exchange,
        )

    def history(self, instrument: Instrument, days: int):
        return self.broker(instrument.market).daily_history(
            instrument.market, instrument.symbol, exchange=instrument.exchange, days=days,
        )

    def top_turnover(self, market: Market, limit: int = 100):
        return self.broker(market).top_turnover(market, limit)

    def common_equities(self, market: Market, candidates):
        return self.broker(market).common_equities(market, candidates)

    def ensure_common_equity(self, instrument: Instrument):
        key = (instrument.symbol, instrument.exchange)
        if key not in self.common_equities(instrument.market, (key,)):
            raise ValueError("일반 기업 보통주로 분류되지 않은 종목은 자동주문하지 않습니다.")

    def prepare(self, instrument: Instrument, side: str, quantity: int,
                kind: str, price: Decimal | None = None) -> OrderRequest:
        broker = self.broker(instrument.market)
        if kind == "current":
            request = broker.build_order_at_current_price(
                market=instrument.market, side=side, symbol=instrument.symbol,
                quantity=quantity, exchange=instrument.exchange,
            )
            return self._remember_request(request, broker)
        if kind not in {"limit", "market"}:
            raise ValueError("지원하지 않는 주문 유형입니다.")
        if kind == "market" and instrument.market is Market.US and self.mode is TradingMode.DEMO:
            raise ValueError("미국 모의투자는 지정가만 가능합니다. 현재가 지정가를 선택하세요.")
        if kind == "limit" and instrument.market is Market.DOMESTIC and price is not None:
            if not price.is_finite() or price != price.to_integral_value():
                raise ValueError("국내 지정가는 정수 원 단위로 입력하세요.")
        request = broker.build_order(
            market=instrument.market, side=side, symbol=instrument.symbol,
            quantity=quantity, exchange=instrument.exchange, order_type=kind,
            price=price if kind == "limit" else None,
        )
        return self._remember_request(request, broker)

    def _remember_request(self, request, broker):
        # Identity, not equal values: an old preview cannot be reused in a newly
        # selected environment. Bound size also limits abandoned GUI previews.
        self._prepared[id(request)] = (request, broker, self.mode)
        while len(self._prepared) > 1024:
            self._prepared.popitem(last=False)
        return request

    def submit(self, request: OrderRequest) -> OrderResult:
        expected = self._prepared.pop(id(request), None)
        if expected is None or expected[0] is not request or expected[2] is not self.mode:
            raise OrderNotSent("현재 환경에서 새로 준비한 주문만 전송할 수 있습니다. 주문 미리보기를 다시 확인하세요.")
        broker = expected[1]

        def permission_before_send():
            try:
                self.ensure_order_permission(Instrument(request.market, request.symbol, request.exchange))
                if self.broker(request.market) is not broker or expected[2] is not self.mode:
                    raise ValueError("주문 준비 이후 거래 환경 또는 API 클라이언트가 변경되었습니다.")
            except Exception as exc:
                raise OrderNotSent(str(exc) or type(exc).__name__) from exc

        permission_before_send()
        with order_send_guard(permission_before_send):
            result = broker.place_order(request, **({"confirm_live_order": LIVE_ORDER_CONFIRMATION}
                                                   if self.mode is TradingMode.REAL else {}))
        if result.mode is not self.mode or result.request != request:
            raise OrderOutcomeUnknown("주문 응답의 거래 환경 또는 주문 내용이 요청과 다릅니다. 재전송하지 말고 주문 내역을 확인하세요.")
        return result

    def account(self, instrument: Instrument):
        broker = self.broker(instrument.market)
        return broker.account_domestic() if instrument.market is Market.DOMESTIC else broker.account_us()

    def orders(self, instrument: Instrument):
        return self.broker(instrument.market).list_open_orders(
            instrument.market, symbol=instrument.symbol, exchange=instrument.exchange,
        )

    def executions(self, instrument: Instrument):
        return self.broker(instrument.market).list_order_executions(
            instrument.market, symbol=instrument.symbol, exchange=instrument.exchange,
        )

    def execution_history(self, market: Market, day: date):
        """Dated read-only account history; one paginated request per market/day."""
        selected = Market(market)
        return self.broker(selected).list_execution_history(selected, day)

    def safety_orders(self, instrument: Instrument):
        return self.broker(instrument.market).list_open_orders(
            instrument.market, symbol=instrument.symbol, exchange=instrument.exchange, strict=True,
        )

    def safety_executions(self, instrument: Instrument):
        return self.broker(instrument.market).list_order_executions(
            instrument.market, symbol=instrument.symbol, exchange=instrument.exchange, strict=True,
        )

    def safety_account(self, instrument: Instrument):
        account = self.account(instrument)
        key, quantities = ("acnt_evlt_remn_indv_tot", ("rmnd_qty", "trde_able_qty")) if instrument.market is Market.DOMESTIC else (
            "result_list", ("poss_qty", "sell_alowq"))
        pages = account.raw.get("balance")
        if not isinstance(pages, list) or not pages:
            raise ValueError("잔고 응답을 검증할 수 없어 자동 주문을 차단합니다.")
        for page in pages:
            rows = page.get(key)
            if not isinstance(rows, list):
                raise ValueError("잔고 목록이 없거나 잘못되었습니다.")
            for row in rows:
                if not isinstance(row, dict) or not row.get("stk_cd"):
                    raise ValueError("잔고의 종목코드를 확인할 수 없습니다.")
                for field in quantities:
                    try:
                        quantity = Decimal(str(row.get(field)).replace(",", ""))
                    except ArithmeticError as exc:
                        raise ValueError("잔고 수량을 확인할 수 없습니다.") from exc
                    if not quantity.is_finite() or quantity < 0:
                        raise ValueError("잔고 수량이 올바르지 않습니다.")
        return account

    def protected_symbols(self, market: Market) -> set[str]:
        """Read the entire market's holdings/open orders before removing managed stocks."""
        inst = Instrument(market, "005930" if market is Market.DOMESTIC else "AAPL", "KRX" if market is Market.DOMESTIC else "ND")
        account = self.safety_account(inst)
        if account.market is not market or account.currency != inst.currency:
            raise ValueError("보호 종목 조회의 시장/통화가 다릅니다.")
        protected = set()
        for p in account.positions:
            if p.market is not market or p.currency != inst.currency or not p.quantity.is_finite() or p.quantity < 0:
                raise ValueError("보유 종목을 검증할 수 없어 목록을 교체하지 않습니다.")
            if p.quantity > 0:
                protected.add(p.symbol)
        orders = self.broker(market).list_open_orders(market, exchange="KRX" if market is Market.DOMESTIC else "%", strict=True)
        for order in orders:
            if not order.remaining_quantity.is_finite() or order.remaining_quantity < 0:
                raise ValueError("미체결 수량을 확인할 수 없습니다.")
            if order.remaining_quantity > 0:
                protected.add(order.symbol)
        return protected
