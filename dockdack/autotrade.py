"""Environment-bound, explicitly armed polling and durable one-shot execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from threading import Event, Lock
from typing import Callable
from uuid import NAMESPACE_URL, uuid5
import json

from dockdack.exceptions import BrokerAPIError, OrderNotSent, OrderOutcomeUnknown
from dockdack.gui_service import TradingService
from dockdack.history import market_time, regular_session
from dockdack.history_cache import HistoryCache
from dockdack.http import order_send_guard
from dockdack.models import OrderSide, TradingMode
from dockdack.order_prices import current_common_equity_limit_price
from dockdack.signal_bridge import validate_external_rule
from dockdack.watchlist import MarketSnapshot, TriggerKind, TriggerRule, WatchItem, WatchStore, positive, utc_now


@dataclass(frozen=True)
class Signal:
    matched: bool
    reference: Decimal
    reason: str


def evaluate_trigger(rule: TriggerRule, snapshot: MarketSnapshot, now: datetime) -> Signal:
    """Level predicates, not crossovers. SMA uses only prior completed-date closes."""
    price = positive(snapshot.quote.price, "현재가")
    age = (now - snapshot.fetched_at).total_seconds()
    if not 0 <= age <= 15:
        raise ValueError("시세를 조회한 지 15초가 지났습니다. 새 시세가 필요합니다.")
    if rule.kind is TriggerKind.EXTERNAL:
        return Signal(True, price, "외부 신호 · 주문 전 수신 기록/만료/정책 재확인")
    if rule.kind in {TriggerKind.PRICE_GE, TriggerKind.PRICE_LE}:
        reference = positive(rule.threshold, "트리거 가격")
    else:
        today = market_time(snapshot.quote.market, now).date()
        completed = [b for b in snapshot.history.bars if b.day < today]
        if len(completed) < rule.period:
            raise ValueError(f"완료된 일봉이 {rule.period}개보다 적어 이동평균을 계산하지 않습니다.")
        # Suspended/delisted/stale histories must not silently drive current-price orders.
        if (today - completed[-1].day).days > 7:
            raise ValueError("최근 완료 일봉이 7일 이상 오래되었습니다. 데이터를 확인하세요.")
        reference = sum((positive(b.close, "일봉 종가") for b in completed[-rule.period:]), Decimal(0)) / rule.period
    above = rule.kind in {TriggerKind.PRICE_GE, TriggerKind.SMA_GE}
    matched = price >= reference if above else price <= reference
    return Signal(matched, reference, f"현재가 {price} {'≥' if above else '≤'} 기준 {reference:.4f}")


class AutoTrader:
    """Call poll() from one worker; enable_orders requires an explicit session opt-in.

    At-most-one submission per rule is enforced through durable SQLite claims. This is
    not an exchange-side exactly-once guarantee. Uncertain submissions remain blocked.
    """

    def __init__(self, service: TradingService, store: WatchStore, *, clock: Callable[[], datetime] = utc_now):
        self.service, self.store, self.clock = service, store, clock
        self._mode = TradingMode(getattr(service, "mode", TradingMode.DEMO))
        self._armed, self._stop = Event(), Event()
        self._poll_lock = Lock()
        self.histories = HistoryCache(store, service, lambda: self.clock())
        self._messages = {}
        self.external_only = False
        self.external_policy = None
        self.external_reader = None
        self.external_error = ""
        self.external_error_count = 0

    @property
    def orders_enabled(self) -> bool:
        return self._armed.is_set()

    def enable_orders(self, confirmation: str):
        self.disarm()
        expected = "REAL_AUTOTRADE" if self._mode is TradingMode.REAL else "DEMO_AUTOTRADE"
        if confirmation != expected:
            raise ValueError("실전 자동주문 시작 확인이 필요합니다." if self._mode is TradingMode.REAL else "모의 자동주문 시작 확인이 필요합니다.")
        self._ensure_environment(orders=True)
        if self.external_only:
            if self.external_policy is None or max(self.external_policy.max_krw, self.external_policy.max_usd) <= 0:
                raise ValueError("외부 신호 출처와 시장별 주문 상한을 먼저 설정하세요.")
        elif not any(rule.status == "ready" and rule.kind is not TriggerKind.EXTERNAL for rule in self.store.rules()):
            raise ValueError("대기 중인 트리거 규칙이 없습니다.")
        if any(a["status"] in {"submitting", "unknown"} for a in self.store.attempts(pending_only=True)):
            raise ValueError("접수 여부가 불명확한 주문이 있습니다. 주문 내역을 확인하세요.")
        for item in self.store.items():
            self._ensure_environment(item.instrument, orders=True)
        for rule in self.store.rules():
            if rule.status == "ready":
                self._reject_demo_rule(rule)
        label = "실전" if self._mode is TradingMode.REAL else "모의"
        self.store.event("SYSTEM", f"사용자 확인으로 이번 세션의 {label} 자동주문 활성화", category="system")
        self._stop.clear()
        self._armed.set()

    def disarm(self):
        self._armed.clear()

    def stop(self):
        # Never wait on an in-flight HTTP request to signal stop.
        self._armed.clear()
        self._stop.set()

    def resume_monitoring(self):
        self._stop.clear()  # Does not arm orders.

    def _ensure_environment(self, instrument=None, *, orders=False):
        """Local checks only, including when called by the final HTTP send guard."""
        try:
            selected = TradingMode(getattr(self.service, "mode", TradingMode.DEMO))
            stored = TradingMode(getattr(self.store, "mode", TradingMode.DEMO))
            if selected is not self._mode or stored is not self._mode:
                raise ValueError("서비스·매매 기록의 모의/실전 환경이 다르거나 실행 중 변경되었습니다.")
            if self._mode is TradingMode.REAL:
                scope = getattr(self.service, "storage_scope", "unconfigured")
                if scope == "unconfigured" or scope != getattr(self.store, "storage_scope", "unconfigured"):
                    raise ValueError("실전 API 키의 기록 범위와 매매 기록 저장소가 다릅니다.")
                if not getattr(self.service, "live_risk_acknowledged", False) and orders:
                    raise ValueError("이번 세션의 실전투자 위험 확인이 필요합니다.")
                if self.external_only and self.external_policy is not None and self._demo_source(self.external_policy.source_id):
                    raise ValueError("내장 모의 테스트 신호(random-demo)는 실전 주문에 연결할 수 없습니다.")
            if instrument is not None:
                check = getattr(self.service, "ensure_order_permission" if orders else "ensure_environment", None)
                if check is None:
                    if self._mode is not TradingMode.DEMO:
                        raise ValueError("실전 환경/주문 권한을 검증할 수 없는 서비스입니다.")
                    check = self.service.ensure_demo
                check(instrument)
        except Exception:
            self.disarm()
            raise

    @staticmethod
    def _demo_source(value):
        return str(value).strip().lower().replace("_", "-") == "random-demo"

    def _reject_demo_rule(self, rule):
        if self._mode is not TradingMode.REAL or rule.kind is not TriggerKind.EXTERNAL:
            return
        record = self.store.external_for_rule(rule.id)
        if record is None:
            return  # The usual external validator rejects absent provenance.
        metadata = json.loads(record["payload"])
        generated_id = uuid5(NAMESPACE_URL, f"random-demo:{record['export_id']}:{record['watch_id']}").hex
        with self.store.connection() as db:
            test_origin = db.execute("SELECT 1 FROM test_decisions WHERE export_id=? AND watch_id=?",
                                     (record["export_id"], record["watch_id"])).fetchone()
        # Check persisted origin, not just the editable UI policy/source name.
        if self._demo_source(record["source_id"]) or metadata.get("signal_id") == generated_id or test_origin:
            self.disarm()
            raise ValueError("내장 모의 테스트 신호는 출처 이름을 바꿔도 실전 주문에 사용할 수 없습니다.")

    def _message(self, key: str, symbol: str, message: str, *, category="system"):
        if self._messages.get(key) != message:
            self.store.event(symbol, message, category=category)
            self._messages[key] = message

    def snapshot(self, item: WatchItem, rules: tuple[TriggerRule, ...] = ()) -> MarketSnapshot:
        inst, now = item.instrument, self.clock()
        self._ensure_environment(inst, orders=self.orders_enabled)
        days = max([item.days] + [r.period + 1 for r in rules if r.status == "ready" and r.kind in {TriggerKind.SMA_GE, TriggerKind.SMA_LE}])
        history = self.histories.get(item, days)
        if self._stop.is_set():
            raise InterruptedError("감시가 중지되었습니다.")
        # Fetch quote last: chart pagination must not age the executable price.
        quote = self.service.quote(inst)
        snapshot = MarketSnapshot(quote, history, self.clock())
        self._validate_snapshot(item, snapshot)
        self.store.save_snapshot(item, snapshot)
        self.store.event(item.id,
                         f"시세/차트 조회 완료 · 현재가 {quote.price} {inst.currency} · 일봉 {len(history.bars)}개",
                         category="monitor")
        return snapshot

    @staticmethod
    def _validate_snapshot(item: WatchItem, snapshot: MarketSnapshot):
        inst = item.instrument
        for data in (snapshot.quote, snapshot.history):
            if (data.market, data.symbol, data.exchange, data.currency) != (inst.market, inst.symbol, inst.exchange, inst.currency):
                raise ValueError("시세/일봉의 종목·거래소·통화가 관심종목과 다릅니다.")
        positive(snapshot.quote.price, "현재가")
        bars = snapshot.history.bars
        if not bars or any(a.day >= b.day for a, b in zip(bars, bars[1:])):
            raise ValueError("일봉이 비어 있거나 거래일 정렬/중복에 문제가 있습니다.")
        age = (market_time(inst.market, snapshot.fetched_at).date() - bars[-1].day).days
        if not 0 <= age <= 7:
            raise ValueError("최근 일봉이 7일 이상 오래되었거나 미래 날짜입니다. 재조회 후 확인하세요.")

    def _reconcile(self, item: WatchItem):
        pending = self.store.attempts(item.id, pending_only=True)
        if any(a["status"] in {"submitting", "unknown"} for a in pending):
            self.disarm()
            raise ValueError("접수 여부 확인 필요 · 자동주문을 껐습니다. 영웅문 주문 내역을 확인하세요.")
        if not pending:
            return
        executions = self.service.safety_executions(item.instrument)
        from dockdack.manual_orders import MANUAL_PREFIX, reconcile_manual_executions
        if any(a["rule_id"].startswith(MANUAL_PREFIX) for a in pending):
            reconcile_manual_executions(self.store, item.instrument, executions)
        rules = {rule.id: rule for rule in self.store.rules(item.id)}
        for attempt in pending:
            if attempt["rule_id"].startswith(MANUAL_PREFIX):
                continue
            # Today's endpoint cannot prove a previous day's fill; keep it blocked for manual review.
            started = datetime.fromisoformat(attempt["started_at"])
            if market_time(item.instrument.market, started).date() != market_time(item.instrument.market, self.clock()).date():
                continue
            for execution in executions:
                if execution.order_number != attempt["order_number"] or execution.symbol != item.instrument.symbol:
                    continue
                rule = rules[attempt["rule_id"]]
                filled, remaining = execution.filled_quantity, execution.remaining_quantity
                if not filled.is_finite() or not remaining.is_finite() or filled < 0 or remaining < 0:
                    raise ValueError("체결 수량을 확인할 수 없습니다.")
                if execution.order_quantity != rule.quantity:
                    raise ValueError("체결 내역의 원주문 수량이 저장된 주문과 다릅니다.")
                self.store.record_execution(rule.id, filled_quantity=filled, remaining_quantity=remaining,
                                            fill_price=execution.fill_price, observed_at=self.clock())
                if filled == rule.quantity and remaining == 0:
                    self.store.finish(rule.id, "filled", f"주문번호 {execution.order_number} · {filled}주 체결 확인")
                    break
                if "취소" in execution.status and remaining == 0:
                    self.store.finish(rule.id, "cancelled", f"주문번호 {execution.order_number} · 잔량 취소 확인")
                    break

    def _preflight(self, item: WatchItem, rule: TriggerRule, snapshot: MarketSnapshot):
        inst = item.instrument
        self._ensure_environment(inst, orders=True)
        self.service.ensure_common_equity(inst)
        if self.store.attempts(item.id, pending_only=True):
            raise ValueError("이 종목의 이전 주문이 미확정/미체결 상태입니다.")
        orders = self.service.safety_orders(inst)
        if self._stop.is_set():
            raise InterruptedError("사용자 중지 요청")
        for order in orders:
            if not order.remaining_quantity.is_finite() or order.remaining_quantity < 0:
                raise ValueError("미체결 잔량을 확인할 수 없습니다.")
            if order.remaining_quantity > 0:
                raise ValueError("미체결 주문이 있어 추가 자동주문을 차단합니다.")
        account = self.service.safety_account(inst)
        if self._stop.is_set():
            raise InterruptedError("사용자 중지 요청")
        if account.market is not inst.market or account.currency != inst.currency:
            raise ValueError("잔고의 시장/통화가 주문과 다릅니다.")
        positions = [p for p in account.positions if p.symbol == inst.symbol]
        for p in positions:
            if p.market is not inst.market or p.currency != inst.currency or not p.quantity.is_finite() or not p.sellable_quantity.is_finite():
                raise ValueError("보유 수량/통화를 확인할 수 없습니다.")
            if not 0 <= p.sellable_quantity <= p.quantity:
                raise ValueError("보유 수량과 매도 가능 수량이 일치하지 않습니다.")
        if rule.side is OrderSide.BUY and any(p.quantity > 0 for p in positions):
            raise ValueError("이미 보유한 종목의 추가 자동매수는 지원하지 않습니다.")
        if rule.side is OrderSide.SELL and sum((p.sellable_quantity for p in positions), Decimal(0)) < rule.quantity:
            raise ValueError("매도 가능 수량이 부족합니다. 공매도는 지원하지 않습니다.")
        # Account/open-order calls take time: recheck both the trigger and limit using a fresh quote.
        fresh = MarketSnapshot(self.service.quote(inst), snapshot.history, self.clock())
        self._validate_snapshot(item, fresh)
        if not evaluate_trigger(rule, fresh, self.clock()).matched:
            raise ValueError("주문 직전 재조회한 가격에서는 트리거 조건이 성립하지 않습니다.")
        notional = fresh.quote.price * rule.quantity
        if notional > rule.max_notional:
            raise ValueError("예상 주문금액이 규칙의 상한을 넘습니다.")
        if rule.side is OrderSide.BUY:
            available = account.available_to_order
            # Leave 1% headroom for fees; the broker still makes the final funds check.
            if available is None or not available.is_finite() or available < notional * Decimal("1.01"):
                raise ValueError("주문가능금액이 불명확하거나 부족합니다 (1% 여유 포함).")
        metadata = validate_external_rule(self.store, rule, self.external_policy, self.clock()) if rule.kind is TriggerKind.EXTERNAL else {}
        if "min_sell_price" in metadata and fresh.quote.price < Decimal(metadata["min_sell_price"]):
            raise ValueError("매도 직전 현재가가 신호의 최소 매도가에 미달합니다.")
        if "cost_profit_pct" in metadata or "cost_loss_pct" in metadata:
            held_positions = [p for p in positions if p.quantity > 0]
            if not held_positions or any(not p.average_price.is_finite() or p.average_price <= 0 for p in held_positions):
                raise ValueError("매도 직전 평균 매입가를 확인할 수 없습니다.")
            total = sum((p.quantity for p in held_positions), Decimal(0))
            average = sum((p.average_price*p.quantity for p in held_positions), Decimal(0))/total
            if "cost_profit_pct" in metadata and fresh.quote.price < average*(1+Decimal(metadata["cost_profit_pct"])/100):
                raise ValueError("매도 직전 실제 평균 매입가 대비 수익률 조건에 미달합니다.")
            if "cost_loss_pct" in metadata and fresh.quote.price > average*(1-Decimal(metadata["cost_loss_pct"])/100):
                raise ValueError("매도 직전 실제 평균 매입가 대비 손절 조건이 성립하지 않습니다.")
        kind = metadata.get("order_type", "limit")
        price = None if kind == "market" else current_common_equity_limit_price(inst.market, rule.side, fresh.quote.price)
        if price is not None and price * rule.quantity > rule.max_notional:
            raise ValueError("가격 단위에 맞춘 지정가 주문금액이 규칙의 상한을 넘습니다.")
        request = self.service.prepare(inst, rule.side.value, rule.quantity, kind, price)
        if (request.market, request.symbol, request.exchange, request.side, request.quantity, request.price) != (
                inst.market, inst.symbol, inst.exchange, rule.side, rule.quantity, price) or request.order_type not in ({"3"} if kind == "market" else {"0", "00"}):
            raise ValueError("주문 미리보기와 트리거의 종목·수량·가격이 다릅니다.")
        return request, fresh

    def _execute(self, item: WatchItem, rule: TriggerRule, snapshot: MarketSnapshot) -> bool:
        self._validate_rule(rule)
        request, fresh = self._preflight(item, rule, snapshot)
        # A newer HOLD/SELL decision may have arrived during account/quote requests.
        self._read_external()
        self._validate_rule(rule)
        self._ensure_environment(item.instrument, orders=True)
        if self._stop.is_set() or not self.orders_enabled or not regular_session(item.instrument.market, self.clock()):
            return False
        if (self.clock() - fresh.fetched_at).total_seconds() > 15:
            raise ValueError("주문 직전 시세가 오래되어 전송하지 않습니다.")
        if not self.store.claim(rule, fresh.quote.price, self.clock()):
            return False
        if request.price is not None and request.price != fresh.quote.price:
            self.store.event(item.id, f"현재가 지정가 가격 단위 적용 · 수량 {rule.quantity}주 · 참조 시세 {fresh.quote.price} → 주문 지정가 {request.price} {item.instrument.currency} (체결가 아님)", category="order")
        if self._stop.is_set() or not self.orders_enabled:
            self.store.finish(rule.id, "not_sent", "사용자 중지 요청으로 전송하지 않음")
            return True
        try:
            self._validate_rule(rule)
        except Exception as exc:
            self.store.finish(rule.id, "not_sent", str(exc))
            return True
        try:
            # The shared HTTP pacer may wait after preflight. Revalidate locally
            # after that wait, immediately before its one transport send.
            with order_send_guard(lambda: self._before_order_send(item, rule, fresh)):
                result = self.service.submit(request)
            if result.mode is not self._mode or not result.accepted or not result.order_number:
                raise OrderOutcomeUnknown("선택한 환경의 주문 접수 결과를 확인할 수 없습니다.")
        except OrderNotSent as exc:
            try:
                self.store.finish(rule.id, "not_sent", str(exc))
            except Exception:
                self.disarm()
                raise
        except Exception as exc:
            known_rejection = (isinstance(exc, BrokerAPIError) and not isinstance(exc, OrderOutcomeUnknown)
                               and exc.status_code is not None and 200 <= exc.status_code < 500
                               and type(exc.return_code) in (int, str) and str(exc.return_code).strip().isdigit()
                               and int(exc.return_code) != 0)
            status = "rejected" if known_rejection else "unknown"
            if not known_rejection:
                self.disarm()
            try:
                self.store.finish(rule.id, status, str(exc))
            except Exception:
                self.disarm()
                raise
        else:
            try:
                self.store.finish(rule.id, "accepted", result.message, result.order_number)
            except Exception:
                # Intent stays 'submitting' on disk; no second send after a journal failure.
                self.disarm()
                raise
        return True

    def _before_order_send(self, item, rule, fresh):
        """Final paced-send guard: local file/DB checks only, never broker I/O."""
        try:
            if self._stop.is_set() or not self.orders_enabled:
                raise ValueError("사용자 중지/OFF 요청으로 전송하지 않음")
            self._ensure_environment(item.instrument, orders=True)
            self._read_external()
            self._validate_rule(rule)
            with self.store.connection() as db:
                claimed = db.execute("""SELECT r.status,w.active FROM rules r
                                        JOIN watchlist w ON w.id=r.watch_id WHERE r.id=?""",
                                     (rule.id,)).fetchone()
                if not claimed or claimed["status"] != "submitting" or not claimed["active"]:
                    raise ValueError("주문 전송 의도 또는 관심종목 상태가 변경되어 전송하지 않음")
                if rule.kind is TriggerKind.EXTERNAL:
                    # Ingestion supersedes only READY rules, not this already
                    # claimed SUBMITTING intent. A newer HOLD must still stop it.
                    latest = db.execute("""SELECT newer.rule_id FROM external_signals newer
                                           JOIN external_signals current
                                             ON newer.source_id=current.source_id AND newer.watch_id=current.watch_id
                                           WHERE current.rule_id=? AND newer.status!='expired'
                                           ORDER BY newer.generated_at DESC LIMIT 1""", (rule.id,)).fetchone()
                    if not latest or latest["rule_id"] != rule.id:
                        raise ValueError("더 최근의 외부 매매/HOLD 신호가 도착하여 이전 주문을 전송하지 않음")
            now = self.clock()
            if not regular_session(item.instrument.market, now):
                raise ValueError("정규장이 종료되었거나 장 상태를 확인할 수 없어 전송하지 않음")
            # Include time spent on final local checks/calendar access as well.
            if not 0 <= (self.clock() - fresh.fetched_at).total_seconds() <= 15:
                raise ValueError("호출 대기 후 시세가 15초를 초과했거나 미래 시각이므로 전송하지 않음")
            if self._stop.is_set() or not self.orders_enabled:
                raise ValueError("사용자 중지/OFF 요청으로 전송하지 않음")
        except Exception as exc:
            raise OrderNotSent(str(exc) or type(exc).__name__) from exc

    def _validate_rule(self, rule):
        self._ensure_environment()
        self._reject_demo_rule(rule)
        if rule.kind is TriggerKind.EXTERNAL:
            if not self.external_only:
                raise ValueError("외부 신호 모드가 꺼져 있습니다.")
            validate_external_rule(self.store, rule, self.external_policy, self.clock())
        elif self.external_only:
            raise ValueError("외부 신호 모드에서는 수동 트리거를 실행하지 않습니다.")

    def _read_external(self):
        if self.external_only and self.external_reader is not None:
            try:
                self.external_reader()
            except Exception as exc:
                self.external_error_count += 1
                self.external_error = str(exc) or type(exc).__name__
                self.disarm()
                self._message("external:file", "SYSTEM", f"외부 신호 읽기 실패 · 자동주문 OFF: {exc}")
            else:
                self.external_error = ""
                self._messages.pop("external:file", None)
        self.store.expire_external(self.clock())

    def poll(self, progress=None, checkpoint=None, on_snapshot=None) -> dict[str, MarketSnapshot | Exception]:
        if not self._poll_lock.acquire(blocking=False):
            return {}
        results = {}
        try:
            items = {item.id: item for item in self.store.items()}
            remaining, seen_external, sent = list(items), set(), set()
            # Bounded even if a producer continuously publishes new decisions.
            for _ in range(len(items) + 500):
                if self._stop.is_set():
                    break
                if checkpoint is not None and checkpoint():
                    items = {item.id: item for item in self.store.items()}
                    results = {key: value for key, value in results.items() if key in items}
                    remaining = [key for key in items if key not in results]
                self._read_external()
                priority = next((r for r in self.store.rules() if self.external_only
                                 and r.kind is TriggerKind.EXTERNAL and r.status == "ready"
                                 and r.id not in seen_external and r.watch_id in items), None)
                if priority:
                    item = items[priority.watch_id]
                    if item.id in remaining:
                        remaining.remove(item.id)
                elif remaining:
                    item = items[remaining.pop(0)]
                else:
                    break
                try:
                    rules = self.store.rules(item.id)
                    snapshot = self.snapshot(item, rules)
                    results[item.id] = snapshot
                    self._reconcile(item)
                    if on_snapshot is not None and not self._stop.is_set():
                        try:
                            on_snapshot(item, snapshot)
                        except Exception:
                            self.disarm()
                            raise
                        self._read_external()
                        rules = self.store.rules(item.id)
                    seen_external.update(r.id for r in rules if r.kind is TriggerKind.EXTERNAL)
                    for rule in rules:
                        if rule.status != "ready" or self._stop.is_set():
                            continue
                        if (rule.kind is TriggerKind.EXTERNAL) != self.external_only:
                            continue
                        try:
                            self._validate_rule(rule)
                            signal = evaluate_trigger(rule, snapshot, self.clock())
                            self._message(rule.id + ":signal", item.id,
                                          f"{rule.description} · {'조건 충족' if signal.matched else '대기'}", category="signal")
                            if not signal.matched or not self.orders_enabled or item.id in sent:
                                continue
                            if not regular_session(item.instrument.market, self.clock()):
                                self._message(rule.id + ":gate", item.id, "정규장 시간이 아니므로 자동주문하지 않음", category="signal")
                                continue
                            if self._execute(item, rule, snapshot):
                                sent.add(item.id)
                                break  # At most one submission per symbol per poll.
                        except Exception as exc:
                            self._message(rule.id + ":gate", item.id, f"자동주문 보류: {exc}", category="signal")
                except Exception as exc:
                    results[item.id] = exc
                    self._message(item.id + ":error", item.id, f"조회/확인 실패: {exc}", category="monitor")
                if progress is not None:
                    progress((item.id, results[item.id], len(results), len(items)))
            return results
        except Exception:
            self.disarm()
            raise
        finally:
            self._poll_lock.release()

    def run_forever(self, interval_seconds: int = 30):
        """Headless generator; consuming it polls until stop(). Monitoring is off by default."""
        if type(interval_seconds) is not int or interval_seconds < 15:
            raise ValueError("조회 간격은 15초 이상의 정수여야 합니다.")
        while not self._stop.is_set():
            yield self.poll()
            if self._stop.wait(interval_seconds):
                break
