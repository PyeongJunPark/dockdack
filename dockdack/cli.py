"""Interactive terminal and one-shot commands for the Kiwoom broker."""

from __future__ import annotations

import argparse
import re
import sys
from decimal import Decimal, InvalidOperation
from typing import Callable, Sequence

from dockdack import (
    BrokerError, KiwoomBroker, KiwoomConfig, Market, TradingMode,
)
from dockdack.kiwoom import LIVE_ORDER_CONFIRMATION


class CommandParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def parser() -> argparse.ArgumentParser:
    root = CommandParser(prog="dockdack", description="키움 국내·미국주식 터미널 (기본: 모의투자)")
    root.add_argument("--mode", choices=("demo", "real"), default="demo")
    commands = root.add_subparsers(dest="command")
    commands.add_parser("shell", help="종목 입력 후 번호로 기능 선택 (기본 동작)")
    for name, help_text in (
        ("quote", "현재가"), ("buy", "매수"), ("sell", "매도"),
        ("orders", "종목별 미체결 주문"), ("balance", "해당 시장 잔고"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("symbol", help="한국 종목코드 또는 미국 티커")
        command.add_argument("--market", choices=("domestic", "us"))
        command.add_argument("--exchange", help="KRX/NXT/SOR 또는 NASDAQ/NYSE/AMEX")
        if name in ("buy", "sell"):
            command.add_argument("quantity", type=int, help="주문 수량 (정수)")
            command.add_argument("--type", choices=("limit", "market", "current"), required=True,
                                 help="limit: 지정가 / market: 시장가 / current: 조회한 현재가로 지정가")
            command.add_argument("--price", help="지정가 단가 (KRW/USD)")
            command.add_argument("--dry-run", action="store_true", help="주문 내용만 확인")
    return root


def identify_symbol(value: str, market: str | None = None) -> tuple[Market, str]:
    from dockdack.symbols import normalize_symbol
    symbol = normalize_symbol(value)
    if re.fullmatch(r"A[0-9]{6}", symbol):
        symbol = symbol[1:]
    domestic = re.fullmatch(r"[0-9][A-Z0-9]{5}", symbol) is not None
    selected = Market(market) if market else (Market.DOMESTIC if domestic else Market.US)
    if selected is Market.DOMESTIC:
        if not re.fullmatch(r"[A-Z0-9]{6}", symbol):
            raise ValueError("한국 종목코드는 6자리로 입력하세요. 예: 005930")
    elif not re.fullmatch(r"[A-Z][A-Za-z0-9.-]{0,11}", symbol):
        raise ValueError("미국 티커를 확인하세요. 예: AAPL (한국 코드는 005930)")
    return selected, symbol


def order_price(args: argparse.Namespace, market: Market) -> Decimal | None:
    if not 1 <= args.quantity <= 999_999_999_999:
        raise ValueError("주문 수량은 1 이상의 정수(최대 12자리)여야 합니다.")
    if args.type in {"market", "current"}:
        if args.price is not None:
            raise ValueError("시장가/현재가 주문에는 --price를 지정하지 마세요.")
        return None
    try:
        price = Decimal(args.price or "")
    except InvalidOperation as exc:
        raise ValueError("지정가 주문에는 숫자로 된 --price가 필요합니다.") from exc
    if not price.is_finite() or price <= 0:
        raise ValueError("가격은 0보다 큰 유한한 숫자여야 합니다.")
    if price.adjusted() > 11 or price.as_tuple().exponent < -12:
        raise ValueError("가격이 API 입력 범위를 벗어났습니다.")
    if len(format(price, "f")) > 12:
        raise ValueError("가격은 소수점을 포함해 12자리 이하여야 합니다.")
    if market is Market.DOMESTIC and price != price.to_integral_value():
        raise ValueError("한국주식 가격은 정수 원 단위로 입력하세요.")
    return price


def create_broker(mode: TradingMode, market: Market) -> KiwoomBroker:
    # Load only this market's credentials: a Korean-only setup must work too.
    return KiwoomBroker(KiwoomConfig.from_env(mode, market=market))


class Terminal:
    def __init__(
        self,
        mode: str,
        *,
        broker_factory: Callable[[TradingMode, Market], KiwoomBroker] = create_broker,
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
    ) -> None:
        self.mode = TradingMode(mode)
        self.broker_factory = broker_factory
        self.read = read
        self.write = write
        self.brokers: dict[Market, KiwoomBroker] = {}
        self.exchanges: dict[str, str] = {}

    @property
    def label(self) -> str:
        return "실전" if self.mode is TradingMode.REAL else "모의"

    def broker(self, market: Market) -> KiwoomBroker:
        if market not in self.brokers:
            broker = self.broker_factory(self.mode, market)
            if broker.mode is not self.mode:
                raise ValueError("요청한 거래 환경과 브로커 환경이 다릅니다.")
            self.brokers[market] = broker
        return self.brokers[market]

    def exchange(self, market: Market, symbol: str, value: str | None) -> str:
        if market is Market.DOMESTIC:
            exchange = (value or "KRX").upper()
            if exchange not in {"KRX", "NXT", "SOR"}:
                raise ValueError("한국 거래소는 KRX, NXT, SOR 중 하나입니다.")
            if self.mode is TradingMode.DEMO and exchange != "KRX":
                raise ValueError("국내 모의투자는 KRX만 지원합니다.")
            return exchange
        if value:
            aliases = {"NASDAQ": "ND", "NYSE": "NY", "AMEX": "NA"}
            exchange = aliases.get(value.upper(), value.upper())
            if exchange not in {"ND", "NY", "NA"}:
                raise ValueError("미국 거래소는 NASDAQ, NYSE, AMEX 중 하나입니다.")
            return exchange
        if symbol not in self.exchanges:
            self.exchanges[symbol] = self.broker(market).resolve_us_exchange(symbol).value
        return self.exchanges[symbol]

    def execute(self, args: argparse.Namespace) -> int:
        market, symbol = identify_symbol(args.symbol, args.market)
        is_order = args.command in {"buy", "sell"}
        price = order_price(args, market) if is_order else None
        exchange = self.exchange(market, symbol, args.exchange)
        broker = self.broker(market)
        if is_order:
            if args.type == "current":
                request = broker.build_order_at_current_price(
                    market=market, side=args.command, symbol=symbol,
                    quantity=args.quantity, exchange=exchange,
                )
                price = request.price
                self.write("현재가를 조회했습니다. 이 가격으로 지정가 주문을 준비합니다.")
            else:
                request = broker.build_order(
                    market=market, side=args.command, symbol=symbol, quantity=args.quantity,
                    exchange=exchange, price=price, order_type=args.type,
                )
            side = "매수" if args.command == "buy" else "매도"
            currency = "KRW" if market is Market.DOMESTIC else "USD"
            price_label = f"지정가 {price:f} {currency}" if price is not None else "시장가 (체결가격 미정)"
            self.write(f"[{self.label}] {symbol} / {exchange} / {side} / {args.quantity}주 / {price_label}")
            if request.estimated_notional is not None:
                self.write(f"주문금액: {request.estimated_notional:f} {currency} (수수료 제외)")
            if args.dry_run:
                self.write("미리보기 완료. 주문을 전송하지 않았습니다.")
                return 0
            if self.mode is TradingMode.REAL and not broker.config.allow_live_orders:
                raise ValueError("실전 주문은 .env의 DOCKDACK_ALLOW_LIVE_ORDERS=true 설정이 필요합니다.")
            confirmation = LIVE_ORDER_CONFIRMATION if self.mode is TradingMode.REAL else "y"
            if self.read(f"전송하려면 {confirmation} 입력 (Enter: 취소): ").strip() != confirmation:
                self.write("주문을 취소했습니다. 전송하지 않았습니다.")
                return 0
            try:
                result = broker.place_order(
                    request,
                    confirm_live_order=LIVE_ORDER_CONFIRMATION if self.mode is TradingMode.REAL else None,
                )
            except BrokerError:
                self.write("주문 응답을 확인하지 못했습니다. 재주문 전에 미체결/체결 내역을 확인하세요.")
                raise
            self.write(f"주문 접수: {result.order_number or '(주문번호 없음: 내역 확인 필요)'} | {result.message}")
            self.write("접수는 체결 완료가 아닙니다. 미체결 및 영웅문 체결내역을 확인하세요.")
        elif args.command == "quote":
            quote = broker.get_quote(market, symbol, exchange=exchange)
            self.write(f"[{self.label}] {quote.name} ({quote.symbol}) / {quote.exchange}")
            self.write(f"현재가: {quote.price:,.4f} {quote.currency}" if market is Market.US
                       else f"현재가: {quote.price:,.0f} {quote.currency}")
            self.write(f"전일대비: {quote.change if quote.change is not None else '-'} / "
                       f"등락률(%): {quote.change_rate if quote.change_rate is not None else '-'}")
        elif args.command == "orders":
            orders = broker.list_open_orders(market, exchange=exchange, symbol=symbol)
            self.write(f"[{self.label}] {symbol} 미체결 주문")
            for order in orders:
                self.write(f"{order.order_number} | {order.side} | {order.status} | "
                           f"주문 {order.order_quantity} / 체결 {order.filled_quantity} / 잔량 {order.remaining_quantity}")
            if not orders:
                self.write("미체결 주문이 없습니다. 체결 여부는 체결내역에서 확인하세요.")
        elif args.command == "balance":
            account = broker.get_account(market, exchange=exchange)
            self.write(f"[{self.label}] {market.value} 잔고 ({account.currency})")
            self.write(f"예수금: {account.cash} / 주문가능금액: {account.available_to_order}")
            for position in account.positions:
                self.write(f"{position.symbol} {position.name} | 보유 {position.quantity}주 | "
                           f"매도가능 {position.sellable_quantity}주 | 평가손익 {position.profit_loss}")
            if not account.positions:
                self.write("보유 종목이 없습니다.")
        return 0

    def shell(self) -> int:
        self.write(f"DockDack [{self.label}] — 한국: 005930 / 미국: AAPL / 종료: q")
        symbol = ""
        while True:
            try:
                if not symbol:
                    entered = self.read("종목코드/티커: ").strip()
                    if entered.lower() in {"q", "quit", "exit"}:
                        return 0
                    _, symbol = identify_symbol(entered)
                self.write(f"\n[{self.label}] {symbol} | 1 현재가 | 2 지정가 매수 | 3 지정가 매도 | "
                           "4 시장가 매수 | 5 시장가 매도 | 6 미체결 | 7 잔고 | 8 종목 변경 | "
                           "9 현재가 지정가 매수 | 10 현재가 지정가 매도 | q 종료")
                action = self.read("선택: ").strip().lower()
                if action in {"q", "quit", "exit"}:
                    return 0
                if action == "8":
                    symbol = ""
                    continue
                names = {"1": "quote", "2": "buy", "3": "sell", "4": "buy",
                         "5": "sell", "6": "orders", "7": "balance", "9": "buy", "10": "sell"}
                if action not in names:
                    raise ValueError("1~10 또는 q를 입력하세요.")
                argv = [names[action], symbol]
                if action in {"2", "3", "4", "5", "9", "10"}:
                    kind = "current" if action in {"9", "10"} else (
                        "limit" if action in {"2", "3"} else "market"
                    )
                    argv += [self.read("수량(주): ").strip(), "--type", kind]
                    if kind == "limit":
                        argv += ["--price", self.read("지정가 단가(KRW/USD): ").strip()]
                self.execute(parser().parse_args(argv))
            except (BrokerError, ValueError) as exc:
                self.write(f"오류: {exc}")
            except (EOFError, KeyboardInterrupt):
                self.write("\n종료합니다. 전송된 주문은 자동 취소되지 않습니다.")
                return 0


def main(argv: Sequence[str] | None = None) -> int:
    # Windows redirected output otherwise defaults to cp949; keep Korean readable.
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        args = parser().parse_args(argv)
        terminal = Terminal(args.mode)
        if args.command in (None, "shell"):
            return terminal.shell()
        return terminal.execute(args)
    except (BrokerError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\n중단했습니다. 이미 전송된 주문은 자동 취소되지 않습니다.", file=sys.stderr)
        return 130
