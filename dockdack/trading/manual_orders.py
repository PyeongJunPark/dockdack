"""Durable, single-attempt manual orders and already-fetched fill reconciliation.

The caller owns the user's final order confirmation. This module never arms an
engine, grants live permission, retries, cancels, or fabricates execution prices.
Each call submits at most once; unresolved attempts block the same instrument.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from dockdack.exceptions import (
    BrokerAPIError, LiveOrderConfirmationRequired, OrderNotSent, OrderOutcomeUnknown,
)
from dockdack.gui_service import Instrument
from dockdack.history import market_time
from dockdack.models import OrderExecution, OrderRequest, OrderResult, OrderSide, TradingMode
from dockdack.watchlist import (
    MAX_WATCH_ITEMS, TriggerKind, TriggerRule, WatchItem, positive, utc_now,
)


MANUAL_PREFIX = "manual-"


def _mode_matches(service, store):
    # Historical test doubles can omit a mode, but can only mean DEMO.
    mode = TradingMode(getattr(service, "mode", TradingMode.DEMO))
    stored = TradingMode(getattr(store, "mode", TradingMode.DEMO))
    if mode is not stored:
        raise OrderNotSent("수동 주문의 거래 환경과 매매 기록 DB가 다릅니다.")
    if mode is TradingMode.REAL or getattr(service, "storage_scope", "demo") != "demo":
        scope = getattr(service, "storage_scope", None)
        if not scope or scope == "unconfigured" or scope != getattr(store, "storage_scope", None):
            raise OrderNotSent("계좌 키/리셋 세대와 매매 기록 DB의 범위가 다릅니다.")
    return mode


def _permission(service, instrument, mode):
    guard = getattr(service, "ensure_order_permission", None)
    if callable(guard):
        guard(instrument)
    elif mode is TradingMode.REAL:
        raise OrderNotSent("실전 주문 권한을 확인할 수 없어 전송하지 않습니다.")
    else:
        legacy = getattr(service, "ensure_demo", None)
        if callable(legacy):
            legacy(instrument)


def _claim_manual(store, request, reference, now):
    item = WatchItem(Instrument(request.market, request.symbol, request.exchange))
    # The existing rule schema is reused only as immutable order metadata. This
    # rule is born 'submitting', NEVER 'ready', so it is not an executable price
    # trigger. Its manual- prefix also scopes reconciliation to manual orders.
    rule = TriggerRule(MANUAL_PREFIX + uuid4().hex, item.id, TriggerKind.PRICE_GE,
                       request.side, request.quantity, reference * request.quantity,
                       reference, status="submitting")
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM attempts WHERE watch_id=? AND status IN "
                      "('submitting','accepted','unknown')", (item.id,)).fetchone():
            raise OrderNotSent("이 종목의 미확정·미체결 주문을 먼저 확인하세요. 재주문하지 않았습니다.")
        existing = db.execute("SELECT 1 FROM watchlist WHERE id=?", (item.id,)).fetchone()
        if not existing:
            if db.execute("SELECT COUNT(*) FROM watchlist WHERE active=1").fetchone()[0] >= MAX_WATCH_ITEMS:
                raise OrderNotSent("관심종목 저장 한도에 도달해 수동 주문을 기록할 수 없습니다.")
            db.execute("INSERT INTO watchlist(id,market,symbol,exchange,name,days,active) VALUES(?,?,?,?,?,30,1)",
                       (item.id, request.market.value, request.symbol, request.exchange, ""))
        # Existing name, days and inactive membership deliberately survive.
        db.execute("INSERT INTO rules(id,watch_id,kind,side,quantity,max_notional,threshold,period,status) "
                   "VALUES(?,?,?,?,?,?,?,?,?)", (rule.id, rule.watch_id, rule.kind.value, rule.side.value,
                   rule.quantity, str(rule.max_notional), str(reference), rule.period, rule.status))
        db.execute("INSERT INTO attempts(rule_id,watch_id,status,price,started_at) VALUES(?,?,'submitting',?,?)",
                   (rule.id, item.id, str(reference), now.isoformat()))
        store._insert_event(db, item.id,
                            f"수동 주문 전송 의도 기록 · {rule.id} · {request.side.value} {request.quantity}주 · "
                            f"참조 가격 {reference} {item.instrument.currency} (체결가 아님)",
                            category="order", at=now)
    return rule


def _known_rejection(exc):
    return (isinstance(exc, BrokerAPIError) and not isinstance(exc, OrderOutcomeUnknown)
            and exc.status_code is not None and 200 <= exc.status_code < 500
            and type(exc.return_code) in (int, str)
            and str(exc.return_code).strip().isdigit() and int(exc.return_code) != 0)


def submit_manual_order(service, store, request, reference_price=None):
    """Journal an explicitly confirmed request before its single transport send.

    Returns the original OrderResult. Errors are propagated after recording the
    best-known state. If result persistence fails, the durable 'submitting'
    intent remains a blocker; the caller must not retry an ambiguous attempt.
    """
    if not isinstance(request, OrderRequest):
        raise OrderNotSent("유효한 주문 미리보기가 필요합니다.")
    instrument = Instrument(request.market, request.symbol, request.exchange)
    WatchItem(instrument)  # Validate canonical symbol, market and exchange.
    mode = _mode_matches(service, store)
    _permission(service, instrument, mode)
    if request.price is not None:
        positive(request.price, "주문 지정가")
    reference = reference_price if reference_price is not None else request.price
    if reference is None:
        quote = service.quote(instrument)
        if (quote.market, quote.symbol, quote.exchange, quote.currency) != (
                instrument.market, instrument.symbol, instrument.exchange, instrument.currency):
            raise OrderNotSent("수동 주문 참조 시세의 종목·시장·통화가 다릅니다.")
        reference = quote.price
    positive(reference, "참조 가격")
    rule = _claim_manual(store, request, reference, utc_now())
    # A stop or mode change during reference lookup must not send the order.
    try:
        if _mode_matches(service, store) is not mode:
            raise OrderNotSent("수동 주문 준비 중 거래 환경이 변경되었습니다.")
        _permission(service, instrument, mode)
    except Exception as exc:
        store.finish(rule.id, "not_sent", str(exc))
        raise
    try:
        result = service.submit(request)
        if (not isinstance(result, OrderResult) or result.mode is not mode
                or result.request != request or result.accepted is not True
                or not isinstance(result.order_number, str) or not result.order_number.strip()):
            raise OrderOutcomeUnknown("수동 주문의 접수 결과를 확인할 수 없습니다.")
    except Exception as exc:
        status = ("not_sent" if isinstance(exc, (OrderNotSent, LiveOrderConfirmationRequired)) else
                  "rejected" if _known_rejection(exc) else "unknown")
        store.finish(rule.id, status, str(exc))
        raise
    store.finish(rule.id, "accepted", f"수동 주문 접수 · {result.message}", result.order_number)
    return result


def _order_number(value):
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():
        return value.lstrip("0") or None
    return value  # Non-numeric test IDs must still match exactly.


def _side(value):
    value = str(getattr(value, "value", value)).strip().lower()
    if value in {"buy", "매수", "+매수", "현금매수", "2"}:
        return OrderSide.BUY
    if value in {"sell", "매도", "-매도", "현금매도", "1"}:
        return OrderSide.SELL
    return None  # Never borrow a correction/cancellation-child row's quantity.


def reconcile_manual_executions(store, instrument, executions):
    """Persist exact matching same-day manual fills from strict broker rows.

    This makes no API calls. An empty list proves nothing. Ambiguous duplicates,
    prior-day order numbers, unknown submission results, and mismatched sides
    are never promoted to fills. Multi-share prices remain raw evidence only;
    the performance calculator requires independent average-price provenance.
    """
    WatchItem(instrument)
    if not isinstance(executions, (tuple, list)) or any(not isinstance(row, OrderExecution) for row in executions):
        raise ValueError("검증된 체결 내역 목록이 필요합니다.")
    now = utc_now()
    today = market_time(instrument.market, now).date()
    records = [row for row in store.order_history(limit=None)
               if row["rule_id"].startswith(MANUAL_PREFIX)
               and (row["market"], row["symbol"], row["exchange"]) == (
                   instrument.market.value, instrument.symbol, instrument.exchange)]
    updated = 0
    for row in records:
        if row["status"] not in {"accepted", "filled", "cancelled"}:
            continue
        started = datetime.fromisoformat(row["started_at"])
        if started.tzinfo is None or market_time(instrument.market, started).date() != today:
            continue
        number = _order_number(row["order_number"])
        if number is None:
            continue
        same_local = [candidate for candidate in records if
                      _order_number(candidate["order_number"]) == number and
                      market_time(instrument.market, datetime.fromisoformat(candidate["started_at"])).date() == today]
        if len(same_local) != 1:
            raise ValueError("동일 주문번호의 수동 주문이 여러 개여서 체결 내역을 반영하지 않습니다.")
        matches = [entry for entry in executions if _order_number(entry.order_number) == number
                   and entry.symbol == instrument.symbol]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError("동일 주문번호의 체결 응답이 여러 개여서 체결 내역을 반영하지 않습니다.")
        execution = matches[0]
        if _side(execution.side) is not OrderSide(row["side"]) or execution.order_quantity != Decimal(row["quantity"]):
            raise ValueError("수동 주문의 체결 방향·원주문 수량이 일치하지 않습니다.")
        filled, remaining = execution.filled_quantity, execution.remaining_quantity
        if any(not isinstance(value, Decimal) or not value.is_finite() or value < 0
               for value in (filled, remaining)) or filled + remaining > Decimal(row["quantity"]):
            raise ValueError("수동 주문의 체결·미체결 수량을 확인할 수 없습니다.")
        if row["status"] == "filled" and (filled != Decimal(row["quantity"]) or remaining != 0):
            raise ValueError("기존 체결 완료 내역과 새 응답 수량이 다릅니다.")
        if row["status"] == "cancelled" and remaining != 0:
            raise ValueError("기존 취소 완료 내역과 새 응답 잔량이 다릅니다.")
        store.record_execution(row["rule_id"], filled_quantity=filled, remaining_quantity=remaining,
                               fill_price=execution.fill_price, observed_at=now)
        if row["status"] == "accepted":
            if filled == Decimal(row["quantity"]) and remaining == 0:
                store.finish(row["rule_id"], "filled", f"수동 주문번호 {row['order_number']} · {filled}주 체결 확인")
            elif str(execution.status).strip().lower() in {"취소", "취소완료", "취소확인", "취소확인완료", "cancelled", "canceled"} and remaining == 0:
                store.finish(row["rule_id"], "cancelled", f"수동 주문번호 {row['order_number']} · 잔량 취소 확인")
        updated += 1
    return updated
