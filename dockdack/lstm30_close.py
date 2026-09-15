"""Explicitly authorised, DEMO-only, all-holdings pre-close liquidation.

The caller serialises tick() on the existing broker worker. This module never
arms an engine, starts a worker, cancels orders, or modifies ordinary strategy
limits. Only this close-only SELL path can use the entire verified sellable
holding without the ordinary per-order quantity/notional limits.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from threading import Lock
from uuid import NAMESPACE_URL, uuid5

from dockdack.exceptions import BrokerAPIError, OrderNotSent, OrderOutcomeUnknown
from dockdack.gui_service import Instrument
from dockdack.history import market_time
from dockdack.http import order_send_guard
from dockdack.market_schedule import session_on
from dockdack.models import AccountSnapshot, Market, OrderExecution, OrderResult, OrderSide, TradingMode
from dockdack.order_prices import current_limit_price
from dockdack.symbols import normalize_symbol, normalize_us_exchange
from dockdack.watchlist import TriggerKind, TriggerRule, WatchItem, positive, utc_now


CLOSE_CONFIRMATION = "DEMO_CLOSE_ALL_SELLABLE"
CLOSE_PREFIX = "close-"
_CANCELLED = {"취소", "취소완료", "취소확인", "취소확인완료", "cancelled", "canceled"}


def _quantity(value):
    try:
        number = Decimal(str(value).replace(",", ""))
    except (ArithmeticError, ValueError, TypeError) as exc:
        raise ValueError("보유/매도가능 수량을 확인할 수 없습니다.") from exc
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError("보유/매도가능 수량은 음수가 아닌 유한한 정수여야 합니다.")
    return number


def _symbol(value, market):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("잔고 종목코드를 확인할 수 없습니다.")
    symbol = normalize_symbol(value)
    if market is Market.DOMESTIC:
        if symbol.startswith("A") and symbol[1:].isdigit():
            symbol = symbol[1:]
        for suffix in ("_NX", "_AL"):
            if symbol.endswith(suffix):
                symbol = symbol[:-len(suffix)]
    return symbol


def _venue(value):
    return normalize_us_exchange(value)


def verified_holdings(account, market):
    """Validate complete raw quantities against normalised holdings, not prices.

    Unsupported venues remain identifiable holdings for an explicit unsold
    report. They are never guessed/remapped into a supported order venue.
    """
    currency = "KRW" if market is Market.DOMESTIC else "USD"
    if not isinstance(account, AccountSnapshot) or account.market is not market or account.currency != currency:
        raise ValueError("전체 잔고의 시장/통화가 일치하지 않습니다.")
    pages = account.raw.get("balance") if isinstance(account.raw, dict) else None
    if not isinstance(pages, list) or not pages:
        raise ValueError("전체 잔고의 원본 페이지가 없어 미보유로 간주하지 않습니다.")
    row_key, qty_key, sell_key = (("acnt_evlt_remn_indv_tot", "rmnd_qty", "trde_able_qty")
                                if market is Market.DOMESTIC else ("result_list", "poss_qty", "sell_alowq"))
    raw, normal, names = defaultdict(lambda: [Decimal(0), Decimal(0)]), defaultdict(lambda: [Decimal(0), Decimal(0)]), {}
    for page in pages:
        rows = page.get(row_key) if isinstance(page, dict) else None
        if not isinstance(rows, list):
            raise ValueError("전체 잔고의 원본 보유 목록이 없습니다.")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("잔고 보유 행 형식을 확인할 수 없습니다.")
            symbol = _symbol(row.get("stk_cd"), market)
            exchange = "KRX" if market is Market.DOMESTIC else _venue(row.get("stex_nm", ""))
            if market is Market.US and (row.get("crnc_code") or "USD") != currency:
                raise ValueError("원본 보유 종목의 통화가 다릅니다.")
            quantity, sellable = _quantity(row.get(qty_key)), _quantity(row.get(sell_key))
            if sellable > quantity:
                raise ValueError("매도가능 수량이 전체 보유 수량보다 많습니다.")
            raw[(symbol, exchange)][0] += quantity
            raw[(symbol, exchange)][1] += sellable
    for position in account.positions:
        if position.market is not market or position.currency != currency:
            raise ValueError("정규화된 보유 종목의 시장/통화가 다릅니다.")
        key = (_symbol(position.symbol, market), _venue(position.exchange))
        quantity, sellable = _quantity(position.quantity), _quantity(position.sellable_quantity)
        if sellable > quantity:
            raise ValueError("정규화된 매도가능 수량이 보유 수량보다 많습니다.")
        normal[key][0] += quantity
        normal[key][1] += sellable
        names[key] = position.name
    if dict(raw) != dict(normal):
        raise ValueError("원본 잔고와 정규화된 보유 수량이 일치하지 않습니다.")
    return tuple({"market": market.value, "symbol": symbol, "exchange": exchange, "currency": currency,
                  "name": names.get((symbol, exchange), ""), "quantity": quantity, "sellable_quantity": sellable}
                 for (symbol, exchange), (quantity, sellable) in normal.items() if quantity > 0)


class CloseLiquidator:
    def __init__(self, service, store, engine, *, enabled=False, minutes_before_close=5,
                 clock=utc_now, confirmation=None):
        if type(enabled) is not bool or type(minutes_before_close) is not int or not 1 <= minutes_before_close <= 30:
            raise ValueError("마감 청산 사용 여부와 1~30분의 시작 시간을 확인하세요.")
        if enabled and confirmation != CLOSE_CONFIRMATION:
            raise ValueError("전체 모의계좌 매도가능 수량 청산에 대한 명시적 확인이 필요합니다.")
        self.service, self.store, self.engine, self.clock = service, store, engine, clock
        self.enabled, self.minutes_before_close = enabled, minutes_before_close
        self._approved = enabled and confirmation == CLOSE_CONFIRMATION
        self._lock = Lock()
        self.errors, self.unsold = [], []
        self.last_checked_at = None
        self._check_mode()
        with self.store.connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS lstm30_close_intents (
                rule_id TEXT PRIMARY KEY REFERENCES rules(id), watch_id TEXT NOT NULL,
                market TEXT NOT NULL, session_day TEXT NOT NULL, quantity INTEGER NOT NULL,
                reference_price TEXT NOT NULL, limit_price TEXT NOT NULL,
                account_fetched_at TEXT NOT NULL, quote_fetched_at TEXT NOT NULL,
                authorisation TEXT NOT NULL, UNIQUE(market,session_day,watch_id))""")

    def _check_mode(self):
        if (getattr(self.service, "mode", None) is not TradingMode.DEMO or self.store.mode is not TradingMode.DEMO
                or self.engine.service is not self.service or self.engine.store is not self.store):
            self.engine.disarm()
            raise ValueError("마감 청산은 같은 모의 서비스·장부·GUI 주문 엔진에만 연결할 수 있습니다.")

    def _window(self, market, now):
        session = session_on(market, market_time(market, now).date())
        return bool(session and max(session.opened, session.closed - timedelta(minutes=self.minutes_before_close)) <= now < session.closed)

    def closing_markets(self, now=None):
        if not self.enabled or not self._approved:
            return frozenset()
        now = now or self.clock()
        result = set()
        for market in Market:
            try:
                if self._window(market, now):
                    result.add(market)
            except Exception:
                pass  # No sale when session hours cannot be established.
        return frozenset(result)

    def buy_blocked(self, market, now=None):
        if not self.enabled or not self._approved:
            return False
        try:
            market = Market(market)
            day = market_time(market, now or self.clock()).date()
            session = session_on(market, day)
            # A calendar lookup may take time. Sample the live clock only after
            # resolving bounds, and keep BUY blocked after the cutoff/close too.
            current = now if now is not None else self.clock()
            return (session is None or market_time(market, current).date() != day
                    or current >= max(session.opened, session.closed-timedelta(minutes=self.minutes_before_close)))
        except Exception:
            return True  # Calendar uncertainty must not admit a new BUY.

    def _running(self):
        external = getattr(self.engine, "external_stop", None)
        return (self.engine.orders_enabled and not self.engine._stop.is_set()
                and not (external is not None and external.is_set()))

    def _permission(self, instrument=None, *, closing=False):
        self._check_mode()
        if not self.enabled or not self._approved or not self._running():
            raise OrderNotSent("마감 청산 미승인 또는 자동주문 OFF/중지 상태입니다.")
        # The GUI's local environment/scope/stop checks also apply to exits.
        # This is not its BUY-universe/common-equity preflight: inactive,
        # verified account holdings remain eligible for close-only reduction.
        self.engine._ensure_environment(instrument, orders=True)
        if instrument is not None:
            self.service.ensure_demo(instrument)
            if closing and not self._window(instrument.market, self.clock()):
                raise OrderNotSent("해당 시장의 승인된 마감 전 청산 시간이 아닙니다.")
        if not self._running():
            raise OrderNotSent("청산 권한 검사 중 자동주문이 OFF 또는 중지되었습니다.")

    @staticmethod
    def _representative(market):
        return Instrument(market, "005930" if market is Market.DOMESTIC else "AAPL", "KRX" if market is Market.DOMESTIC else "ND")

    def _account(self, market):
        self._permission(self._representative(market))
        # TradingService.safety_account reads the entire market (not a filtered
        # ticker snapshot) and validates all pages before this stricter check.
        rows = verified_holdings(self.service.safety_account(self._representative(market)), market)
        return rows, self.clock()

    def _day_id(self, instrument, now):
        day = market_time(instrument.market, now).date().isoformat()
        watch_id = WatchItem(instrument).id
        return day, CLOSE_PREFIX + uuid5(NAMESPACE_URL, f"dockdack:demo-close-all:{day}:{watch_id}").hex

    def _existing(self, instrument):
        day, _ = self._day_id(instrument, self.clock())
        with self.store.connection() as db:
            row = db.execute("""SELECT a.* FROM lstm30_close_intents c JOIN attempts a ON a.rule_id=c.rule_id
                                WHERE c.market=? AND c.session_day=? AND c.watch_id=?""",
                             (instrument.market.value, day, WatchItem(instrument).id)).fetchone()
        return dict(row) if row else None

    def _claim(self, instrument, request, holding, reference, account_at, quote_at):
        day, rule_id = self._day_id(instrument, self.clock())
        item = WatchItem(instrument, holding.get("name", ""), 31)
        rule = TriggerRule(rule_id, item.id, TriggerKind.PRICE_GE, OrderSide.SELL, request.quantity,
                           request.price * request.quantity, reference, status="submitting")
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM lstm30_close_intents WHERE market=? AND session_day=? AND watch_id=?",
                          (instrument.market.value, day, item.id)).fetchone():
                return None
            if db.execute("SELECT 1 FROM attempts WHERE watch_id=? AND status IN ('submitting','accepted','unknown')", (item.id,)).fetchone():
                raise OrderNotSent("기존 미확정/미체결 주문이 있어 마감 청산을 전송하지 않습니다.")
            # Historical metadata only: adding an inactive row leaves the
            # normal TOP100/entry universe and its ownership guard unchanged.
            db.execute("""INSERT OR IGNORE INTO watchlist(id,market,symbol,exchange,name,days,active)
                           VALUES(?,?,?,?,?,31,0)""", (item.id, instrument.market.value, instrument.symbol, instrument.exchange, item.name))
            db.execute("INSERT INTO rules VALUES(?,?,?,?,?,?,?,?,?)", (rule.id, rule.watch_id, rule.kind.value,
                       rule.side.value, rule.quantity, str(rule.max_notional), str(reference), rule.period, rule.status))
            db.execute("INSERT INTO attempts(rule_id,watch_id,status,price,started_at) VALUES(?,?,'submitting',?,?)",
                       (rule.id, item.id, str(reference), self.clock().isoformat()))
            db.execute("INSERT INTO lstm30_close_intents VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (rule.id, item.id, instrument.market.value, day, request.quantity, str(reference), str(request.price),
                        account_at.isoformat(), quote_at.isoformat(), CLOSE_CONFIRMATION))
            self.store._insert_event(db, item.id,
                f"마감 청산 전송 의도 기록 · {rule.id} · sell {request.quantity}주 · 전체 모의계좌 매도가능 수량 · 일반 수량/금액 상한 예외",
                category="order", at=self.clock())
        return rule

    def _before_send(self, instrument, rule, request, sellable, account_at, quote_at):
        try:
            self._permission(instrument, closing=True)
            now = self.clock()
            if (request.side is not OrderSide.SELL or request.market is not instrument.market
                    or request.symbol != instrument.symbol or request.exchange != instrument.exchange
                    or type(request.quantity) is not int or request.quantity <= 0 or request.quantity > sellable
                    or request.quantity != rule.quantity or request.price is None
                    or request.order_type != ("0" if instrument.market is Market.DOMESTIC else "00")):
                raise ValueError("검증된 매도가능 수량의 지정가 SELL 요청과 다릅니다.")
            if not 0 <= (now-account_at).total_seconds() <= 15 or not 0 <= (now-quote_at).total_seconds() <= 15:
                raise ValueError("호출 대기 후 잔고/시세가 15초를 넘어 청산을 전송하지 않습니다.")
            with self.store.connection() as db:
                row = db.execute("""SELECT c.*,a.status,r.side,r.quantity AS rule_quantity,r.status AS rule_status
                                    FROM lstm30_close_intents c JOIN attempts a ON a.rule_id=c.rule_id
                                    JOIN rules r ON r.id=c.rule_id WHERE c.rule_id=?""", (rule.id,)).fetchone()
                if (not row or row["status"] != "submitting" or row["rule_status"] != "submitting"
                        or row["side"] != "sell" or row["authorisation"] != CLOSE_CONFIRMATION
                        or row["quantity"] != request.quantity or row["rule_quantity"] != request.quantity
                        or row["watch_id"] != WatchItem(instrument).id or Decimal(row["limit_price"]) != request.price
                        or row["session_day"] != market_time(instrument.market, now).date().isoformat()):
                    raise ValueError("마감 청산의 불변 승인/전송 의도 기록을 확인할 수 없습니다.")
                if db.execute("SELECT 1 FROM attempts WHERE watch_id=? AND rule_id!=? AND status IN ('submitting','accepted','unknown')",
                              (rule.watch_id, rule.id)).fetchone():
                    raise ValueError("다른 미확정/미체결 주문이 생겨 청산을 전송하지 않습니다.")
            self._permission(instrument, closing=True)
            # SQLite waits and the final local permission checks can consume
            # the remaining freshness budget after the first validation. Resolve
            # session bounds before sampling the final clock; calendar work must
            # not happen after the last time comparison.
            session = session_on(instrument.market, market_time(instrument.market, self.clock()).date())
            now = self.clock()
            if not 0 <= (now-account_at).total_seconds() <= 15 or not 0 <= (now-quote_at).total_seconds() <= 15:
                raise ValueError("최종 검증 후 잔고/시세가 15초를 넘어 청산을 전송하지 않습니다.")
            if not (session and max(session.opened, session.closed-timedelta(minutes=self.minutes_before_close)) <= now < session.closed):
                raise OrderNotSent("최종 검증 중 해당 시장의 마감 전 청산 시간이 종료되었습니다.")
            if not self._running():
                raise OrderNotSent("최종 청산 검사 중 자동주문이 OFF 또는 중지되었습니다.")
        except OrderNotSent:
            raise
        except Exception as exc:
            raise OrderNotSent(str(exc)) from exc

    def _send_one(self, target):
        instrument = Instrument(Market(target["market"]), target["symbol"], target["exchange"])
        WatchItem(instrument)
        self._permission(instrument, closing=True)
        existing = self._existing(instrument)
        if existing:
            return "ALREADY_ATTEMPTED:" + existing["status"]
        pending = self.store.attempts(WatchItem(instrument).id, pending_only=True)
        if any(row["status"] in {"submitting", "unknown"} for row in pending):
            self.engine.disarm()
            return "UNKNOWN_ORDER_REQUIRES_REVIEW"
        if pending:
            return "PENDING_ORDER"
        orders = self.service.safety_orders(instrument)
        self._permission(instrument, closing=True)
        for order in orders:
            if order.market is not instrument.market or order.symbol != instrument.symbol or _venue(order.exchange) != instrument.exchange:
                raise ValueError("미체결 조회의 종목/거래소/시장이 다릅니다.")
            if _quantity(order.remaining_quantity) > 0:
                return "BROKER_OPEN_ORDER"
        holdings, account_at = self._account(instrument.market)
        current = next((row for row in holdings if (row["symbol"], row["exchange"]) == (instrument.symbol, instrument.exchange)), None)
        if current is None:
            return "NO_HOLDING"
        sellable = current["sellable_quantity"]
        if sellable <= 0:
            return "NO_SELLABLE_QUANTITY"
        if sellable > 999_999_999:
            return "UNSUPPORTED_ORDER_QUANTITY"
        self._permission(instrument, closing=True)
        quote = self.service.quote(instrument)
        quote_at = self.clock()
        if (quote.market, quote.symbol, quote.exchange, quote.currency) != (
                instrument.market, instrument.symbol, instrument.exchange, instrument.currency):
            raise ValueError("청산 시세의 종목/거래소/통화가 다릅니다.")
        reference = positive(quote.price, "청산 참조 현재가")
        price = current_limit_price(instrument.market, OrderSide.SELL, reference)
        request = self.service.prepare(instrument, "sell", int(sellable), "limit", price)
        if (request.market, request.symbol, request.exchange, request.side, request.quantity, request.price, request.order_type) != (
                instrument.market, instrument.symbol, instrument.exchange, OrderSide.SELL, int(sellable), price,
                "0" if instrument.market is Market.DOMESTIC else "00"):
            raise ValueError("청산 주문 미리보기의 내용이 검증된 SELL 요청과 다릅니다.")
        self._permission(instrument, closing=True)
        if not 0 <= (self.clock()-account_at).total_seconds() <= 15:
            return "ACCOUNT_EXPIRED_BEFORE_INTENT"
        rule = self._claim(instrument, request, current, reference, account_at, quote_at)
        if rule is None:
            return "ALREADY_ATTEMPTED"
        try:
            self._before_send(instrument, rule, request, sellable, account_at, quote_at)
            with order_send_guard(lambda: self._before_send(instrument, rule, request, sellable, account_at, quote_at)):
                result = self.service.submit(request)
            if (not isinstance(result, OrderResult) or result.mode is not TradingMode.DEMO
                    or result.request != request or result.accepted is not True
                    or not isinstance(result.order_number, str) or not result.order_number.strip()):
                raise OrderOutcomeUnknown("마감 청산의 접수 결과를 확인할 수 없습니다.")
        except Exception as exc:
            known = (isinstance(exc, BrokerAPIError) and not isinstance(exc, OrderOutcomeUnknown)
                     and exc.status_code is not None and 200 <= exc.status_code < 500
                     and type(exc.return_code) in (int, str) and str(exc.return_code).strip().isdigit()
                     and int(exc.return_code) != 0)
            status = "not_sent" if isinstance(exc, OrderNotSent) else "rejected" if known else "unknown"
            if status in {"unknown", "rejected"}:
                self.engine.disarm()
            try:
                self.store.finish(rule.id, status, f"마감 청산 · {type(exc).__name__}")
            except Exception:
                self.engine.disarm()
                raise
            return status.upper()
        try:
            self.store.finish(rule.id, "accepted", "마감 청산 접수 · 체결 여부는 별도 확인", result.order_number)
        except Exception:
            self.engine.disarm()
            raise
        return "ACCEPTED_PENDING_FILL"

    def _attempts(self):
        with self.store.connection() as db:
            return tuple(dict(row) for row in db.execute("""SELECT c.*,a.status,a.order_number,a.started_at
                FROM lstm30_close_intents c JOIN attempts a ON a.rule_id=c.rule_id ORDER BY a.started_at,c.rule_id"""))

    def _reconcile(self):
        for row in self._attempts():
            if not self._running():
                break
            if row["status"] != "accepted":
                continue
            market = Market(row["market"])
            if row["session_day"] != market_time(market, self.clock()).date().isoformat():
                continue
            _, exchange, symbol = row["watch_id"].split(":", 2)
            instrument = Instrument(market, symbol, exchange)
            self._permission(instrument)
            executions = self.service.safety_executions(instrument)
            if not isinstance(executions, (tuple, list)) or any(not isinstance(entry, OrderExecution) for entry in executions):
                raise ValueError("청산 체결 조회의 응답 목록을 검증할 수 없습니다.")
            number = lambda value: str(value).strip().lstrip("0") or "0"
            matches = [entry for entry in executions if number(entry.order_number) == number(row["order_number"]) and entry.symbol == symbol]
            if not matches:
                continue
            if len(matches) != 1:
                raise ValueError("청산 주문번호의 체결 응답이 중복되어 확인할 수 없습니다.")
            entry = matches[0]
            side = str(getattr(entry.side, "value", entry.side)).strip().lower()
            if side not in {"sell", "매도", "-매도", "현금매도", "1"} or entry.order_quantity != Decimal(row["quantity"]):
                raise ValueError("청산 체결의 매도 방향/원주문 수량이 다릅니다.")
            filled, remaining = _quantity(entry.filled_quantity), _quantity(entry.remaining_quantity)
            if filled + remaining > row["quantity"]:
                raise ValueError("청산 체결/잔여 수량이 원주문보다 많습니다.")
            self.store.record_execution(row["rule_id"], filled_quantity=filled, remaining_quantity=remaining,
                                        fill_price=entry.fill_price, observed_at=self.clock())
            if filled == row["quantity"] and remaining == 0:
                self.store.finish(row["rule_id"], "filled", "마감 청산 전량 체결 확인")
            elif remaining == 0 and str(entry.status).strip().lower() in _CANCELLED:
                self.store.finish(row["rule_id"], "cancelled", "마감 청산 잔량 취소 확인 · 자동 재주문 안 함")

    def tick(self):
        if not self._lock.acquire(blocking=False):
            return self.status()
        try:
            if not self.enabled or not self._approved:
                return self.status()
            try:
                self._check_mode()
            except Exception:
                self.errors = [{"market": "all", "reason": "ENVIRONMENT_MISMATCH"}]
                return self.status()
            if not self._running():
                return self.status()
            self.errors = []
            self.last_checked_at = self.clock().isoformat()
            try:
                self._reconcile()
            except Exception as exc:
                self.engine.disarm()
                self.errors.append({"market": "all", "reason": f"RECONCILIATION_FAILED:{type(exc).__name__}"})
                return self.status()
            closing = self.closing_markets()
            # After close, continue verifying previously accepted same-day exits
            # and report the actual remaining holdings without placing orders.
            reporting = {Market(row["market"]) for row in self._attempts()
                         if row["session_day"] == market_time(Market(row["market"]), self.clock()).date().isoformat()}
            unsold = []
            for market in sorted(closing | reporting, key=lambda value: value.value):
                if not self._running():
                    break
                try:
                    holdings, observed = self._account(market)
                    for holding in holdings:
                        entry = {key: str(value) if isinstance(value, Decimal) else value
                                 for key, value in holding.items() if key != "name"}
                        entry["observed_at"] = observed.isoformat()
                        entry["reason"] = "NOT_PROCESSED"
                        unsold.append(entry)
                        instrument = Instrument(market, holding["symbol"], holding["exchange"])
                        try:
                            WatchItem(instrument)
                        except ValueError:
                            entry["reason"] = "UNSUPPORTED_INSTRUMENT_OR_VENUE"
                        else:
                            if holding["sellable_quantity"] <= 0:
                                entry["reason"] = "NO_SELLABLE_QUANTITY"
                            elif not self._running():
                                entry["reason"] = "ORDERS_OFF_OR_STOPPED"
                            elif market not in self.closing_markets():
                                entry["reason"] = "OUTSIDE_CLOSE_WINDOW"
                            else:
                                try:
                                    entry["reason"] = self._send_one(holding)
                                except OrderNotSent:
                                    entry["reason"] = "CLOSE_WINDOW_ENDED_OR_STOPPED"
                                except Exception as exc:
                                    self.engine.disarm()
                                    entry["reason"] = f"CLOSE_CHECK_FAILED:{type(exc).__name__}"
                                    self.errors.append({"market": market.value, "reason": entry["reason"]})
                except OrderNotSent:
                    self.errors.append({"market": market.value, "reason": "CLOSE_WINDOW_ENDED_OR_STOPPED"})
                except Exception as exc:
                    self.engine.disarm()
                    self.errors.append({"market": market.value, "reason": f"CLOSE_CHECK_FAILED:{type(exc).__name__}"})
            self.unsold = unsold
            return self.status()
        finally:
            self._lock.release()

    def status(self):
        now, markets = self.clock(), {}
        for market in Market:
            try:
                today = market_time(market, now).date()
                session = session_on(market, today)
                future = session if session and now < session.closed else next(
                    (candidate for offset in range(1, 11)
                     if (candidate := session_on(market, today + timedelta(days=offset))) is not None), None)
                markets[market.value] = {
                    "close_at": session.closed.isoformat() if session else None,
                    "starts_at": (session.closed-timedelta(minutes=self.minutes_before_close)).isoformat() if session else None,
                    "next_close_at": future.closed.isoformat() if future else None,
                    "phase": "closing" if self._window(market, now) else "closed" if session and now >= session.closed else "waiting",
                }
            except Exception:
                markets[market.value] = {"close_at": None, "starts_at": None, "next_close_at": None, "phase": "calendar_error"}
        attempts = [{key: row[key] for key in ("rule_id", "watch_id", "status", "quantity", "order_number", "session_day")}
                    for row in self._attempts()]
        return {"enabled": self.enabled and self._approved, "scope": "all_demo_holdings",
                "minutes_before_close": self.minutes_before_close, "close_only_full_sellable_override": self._approved,
                "closing_markets": sorted(market.value for market in self.closing_markets(now)),
                "markets": markets, "orders_enabled": bool(self.engine.orders_enabled),
                "unsold": list(self.unsold), "attempts": attempts, "errors": list(self.errors),
                "last_checked_at": self.last_checked_at,
                "completion_basis": "Verified fills and refreshed account holdings; accepted is not filled"}
