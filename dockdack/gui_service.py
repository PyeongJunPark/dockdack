"""Qt-independent operations used by the demo trading desktop."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from dockdack import KiwoomBroker, KiwoomConfig, Market, OrderRequest, OrderResult, TradingMode
from dockdack.cli import identify_symbol


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

    The first GUI release deliberately uses the demo environment only.
    """

    def __init__(self, broker_factory: Callable[[Market], KiwoomBroker] = demo_broker):
        self.factory = broker_factory
        self.brokers: dict[Market, KiwoomBroker] = {}
        self.exchanges: dict[str, str] = {}

    def broker(self, market: Market) -> KiwoomBroker:
        if market not in self.brokers:
            broker = self.factory(market)
            if broker.mode is not TradingMode.DEMO or broker.config.base_url != "https://mockapi.kiwoom.com":
                raise ValueError("이 화면은 모의투자 전용입니다. 모의투자 키를 확인하세요.")
            self.brokers[market] = broker
        return self.brokers[market]

    def resolve(self, symbol: str, exchange: str = "") -> Instrument:
        market, symbol = identify_symbol(symbol)
        if market is Market.DOMESTIC:
            if exchange not in {"", "KRX"}:
                raise ValueError("국내 모의투자 거래소는 KRX입니다.")
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

    def prepare(self, instrument: Instrument, side: str, quantity: int,
                kind: str, price: Decimal | None = None) -> OrderRequest:
        broker = self.broker(instrument.market)
        if kind == "current":
            return broker.build_order_at_current_price(
                market=instrument.market, side=side, symbol=instrument.symbol,
                quantity=quantity, exchange=instrument.exchange,
            )
        if kind not in {"limit", "market"}:
            raise ValueError("지원하지 않는 주문 유형입니다.")
        if kind == "market" and instrument.market is Market.US:
            raise ValueError("미국 모의투자는 지정가만 가능합니다. 현재가 지정가를 선택하세요.")
        if kind == "limit" and instrument.market is Market.DOMESTIC and price is not None:
            if not price.is_finite() or price != price.to_integral_value():
                raise ValueError("국내 지정가는 정수 원 단위로 입력하세요.")
        return broker.build_order(
            market=instrument.market, side=side, symbol=instrument.symbol,
            quantity=quantity, exchange=instrument.exchange, order_type=kind,
            price=price if kind == "limit" else None,
        )

    def submit(self, request: OrderRequest) -> OrderResult:
        return self.broker(request.market).place_order(request)

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
