"""Typed values returned by the broker client."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping


class TradingMode(str, Enum):
    DEMO = "demo"
    REAL = "real"


class Market(str, Enum):
    DOMESTIC = "domestic"
    US = "us"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class DomesticExchange(str, Enum):
    KRX = "KRX"
    NXT = "NXT"
    SOR = "SOR"


class USExchange(str, Enum):
    AMEX = "NA"
    NASDAQ = "ND"
    NYSE = "NY"
    ALL = "%"


@dataclass(frozen=True, slots=True)
class Quote:
    market: Market
    symbol: str
    name: str
    exchange: str
    price: Decimal
    currency: str
    change: Decimal | None = None
    change_rate: Decimal | None = None
    volume: Decimal | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class StockInfo:
    market: Market
    symbol: str
    name: str
    exchange: str
    english_name: str | None = None
    previous_close: Decimal | None = None
    is_etf: bool | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DailyBar:
    market: Market
    symbol: str
    exchange: str
    trade_date: date
    open: Decimal | None
    high: Decimal | None
    low: Decimal | None
    close: Decimal | None
    volume: Decimal | None
    currency: str
    trade_value: Decimal | None = None
    change: Decimal | None = None
    change_rate: Decimal | None = None
    adjustment_type: str | None = None
    adjustment_rate: Decimal | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Position:
    market: Market
    symbol: str
    name: str
    exchange: str
    currency: str
    quantity: Decimal
    sellable_quantity: Decimal
    average_price: Decimal
    current_price: Decimal
    evaluation_amount: Decimal
    profit_loss: Decimal
    profit_rate: Decimal
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    market: Market
    currency: str
    positions: tuple[Position, ...]
    cash: Decimal | None = None
    available_to_order: Decimal | None = None
    total_purchase: Decimal | None = None
    total_evaluation: Decimal | None = None
    total_profit_loss: Decimal | None = None
    profit_rate: Decimal | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class OrderRequest:
    market: Market
    side: OrderSide
    symbol: str
    quantity: int
    exchange: str
    order_type: str
    price: Decimal | None = None
    stop_price: Decimal | None = None

    @property
    def estimated_notional(self) -> Decimal | None:
        if self.price is None:
            return None
        return self.price * self.quantity


@dataclass(frozen=True, slots=True)
class OrderResult:
    accepted: bool
    mode: TradingMode
    request: OrderRequest
    order_number: str
    message: str
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class CancelResult:
    accepted: bool
    mode: TradingMode
    market: Market
    original_order_number: str
    cancel_order_number: str
    cancelled_quantity: Decimal | None
    message: str
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class OpenOrder:
    market: Market
    order_number: str
    symbol: str
    name: str
    exchange: str
    side: str
    status: str
    order_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    order_price: Decimal
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class OrderExecution:
    order_number: str
    symbol: str
    side: str
    status: str
    order_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    order_price: Decimal
    fill_price: Decimal
    order_time: str


@dataclass(frozen=True, slots=True)
class ExecutionHistoryRecord:
    """A dated broker order snapshot, not individual fill events.

    ``order_date`` is the explicit query date, not a date returned by the API;
    the US API does not document the query date's timezone. History retention
    and multi-share VWAP semantics are not assumed. ``fill_price`` is usable
    only for a verified one-share fill. An empty exchange means unknown venue.
    """

    market: Market
    order_date: date
    order_number: str
    symbol: str
    exchange: str
    side: OrderSide
    order_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    order_price: Decimal | None
    fill_price: Decimal | None
    reported_fill_price: Decimal | None
    price_basis: str
    order_time: str
    fill_time: str
    status: str
    currency: str
    source_api: str
    original_order_number: str = ""


@dataclass(frozen=True, slots=True)
class SavedCondition:
    sequence: str
    name: str
    market: Market


@dataclass(frozen=True, slots=True)
class ConditionMatch:
    market: Market
    symbol: str
    name: str
    exchange: str
    price: Decimal | None = None
    change_rate: Decimal | None = None
    volume: Decimal | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
