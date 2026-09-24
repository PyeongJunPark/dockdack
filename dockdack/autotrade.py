"""Environment-bound, explicitly armed polling and durable one-shot execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from threading import Event, Lock
from typing import Callable
from uuid import NAMESPACE_URL, uuid5, uuid4
import json

from requests.exceptions import ConnectionError as TransportConnectionError, Timeout as TransportTimeout

from dockdack.exceptions import BrokerAPIError, OrderNotSent, OrderOutcomeUnknown
from dockdack.gui_service import TradingService
from dockdack.history import DailyHistory, market_time, regular_session
from dockdack.history_cache import HistoryCache
from dockdack.http import _is_rate_limit, order_send_guard
from dockdack.models import Market, OrderSide, TradingMode
from dockdack.gui_service import Instrument
from dockdack.execution_policy import allocation_quantity, holding_exit_targets
from dockdack.order_prices import current_common_equity_limit_price
from dockdack.signal_bridge import mark1_prototype_origin, validate_external_rule
from dockdack.watchlist import MarketSnapshot, TriggerKind, TriggerRule, WatchItem, WatchStore, positive, utc_now


@dataclass(frozen=True)
class Signal:
    matched: bool
    reference: Decimal
    reason: str


def _transient_poll_failure(error: Exception) -> bool:
    """Recognize read/checkpoint outages, never infer recovery from arbitrary errors.

    The caller aborts this sweep and leaves retries to its normal timer. This
    does not retry an order, re-arm an already disabled engine, or forgive a
    missing/corrupt journal, changed environment, or unknown order outcome.
    """
    if isinstance(error, (OrderNotSent, OrderOutcomeUnknown)):
        return False
    network_errors = (TimeoutError, ConnectionError, TransportTimeout, TransportConnectionError)
    if isinstance(error, network_errors):
        return True
    if not isinstance(error, BrokerAPIError):
        return False
    status = error.status_code
    if status in (401, 403):
        return False
    if type(status) is int and (status == 429 or 500 <= status < 600):
        return True
    if type(status) is int and _is_rate_limit(status, {
        "return_code": error.return_code, "return_msg": str(error),
    }):
        return True
    # The HTTP adapter wraps transport failures without a response/code.
    return status is None and error.return_code is None and isinstance(error.__cause__, network_errors)


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
        self.external_sources = {}
        self.external_source_errors = {}
        self.source_validators = {}
        # Opt-in virtual inventory. Broker positions remain aggregate; each
        # prototype owns only its confirmed fills and its own exit allocation.
        self.prototype_lots_enabled = False
        self._lot_sellable_checks = {}
        self.session_only_poll = True
        self.enable_holdings_exits = False
        self.equity_buy_percent = None
        self.isolated_symbol_errors = False
        self.us_retry_attempts = 3
        self.us_failure_cooldown_seconds = 300
        self.holding_caps = {Market.DOMESTIC: Decimal("10000000"), Market.US: Decimal("10000")}

    def configure_external_sources(self, sources):
        """Replace extra readers while OFF; the legacy primary reader remains supported."""
        if self.orders_enabled:
            raise ValueError("외부 신호 연결 변경 전에 자동주문을 OFF 하세요.")
        entries = {}
        for policy, reader in sources:
            if policy.source_id in entries or not callable(reader):
                raise ValueError("신호 출처는 중복 없이 읽기 함수와 함께 등록하세요.")
            if self._mode is TradingMode.REAL and self._demo_source(policy.source_id):
                raise ValueError("내장 모의 신호를 실전에 연결할 수 없습니다.")
            entries[policy.source_id] = (policy, reader)
        self.external_sources = entries
        self.external_source_errors = {}

    def configure_source_validators(self, validators):
        """Install trusted, local-only execution checks; never arms monitoring/orders.

        Each callback receives (item, rule, snapshot, actual_limit_price,
        stage=...). It must not perform broker I/O, including at final_send.
        Wire payloads cannot provide or replace executable callbacks.
        """
        if self.orders_enabled:
            raise ValueError("신호 주문 검증 연결 변경 전에 자동주문을 OFF 하세요.")
        entries = dict(validators)
        if any(not isinstance(source, str) or not source or not callable(callback)
               for source, callback in entries.items()):
            raise ValueError("신호 출처별 로컬 주문 검증 함수를 등록하세요.")
        if self._mode is TradingMode.REAL and any(self._demo_source(source) for source in entries):
            raise ValueError("내장 모의 신호 검증기를 실전에 연결할 수 없습니다.")
        self.source_validators = entries

    def holding_exit_targets(self, position):
        return holding_exit_targets(self.store, position, prototype_lots=self.prototype_lots_enabled)

    def _prototype_source_for(self, rule):
        if not self.prototype_lots_enabled:
            return None
        return self.store.prototype_rule_source(rule.id)

    def _sent_key(self, item, rule):
        source = self._prototype_source_for(rule)
        if source is None:
            return item.id
        allocation = self.store.prototype_sell_allocation(rule.id)
        return (item.id, "lot", allocation["lot_id"]) if allocation else (item.id, "model", source)

    def _compatible_lot_pending(self, rule, pending):
        """Only known accepted orders for a different strategy may coexist."""
        source = self._prototype_source_for(rule)
        if source is None or pending["status"] != "accepted":
            return False
        other = self.store.prototype_rule_source(pending["rule_id"])
        if not other:
            return False
        from dockdack.signal_bridge import prototype_family
        return prototype_family(source) != prototype_family(other)

    def _validate_lot_inventory(self, item, rule, positions):
        """Match durable virtual lots against the fresh aggregate broker position."""
        if self._prototype_source_for(rule) is None:
            return None
        if any(p.exchange != item.instrument.exchange for p in positions):
            raise ValueError("모델별 보유분과 브로커 거래소가 일치하지 않습니다.")
        quantity = sum((p.quantity for p in positions), Decimal(0))
        sellable = sum((p.sellable_quantity for p in positions), Decimal(0))
        inventory = self.store.prototype_inventory(item.id, broker_quantity=quantity, broker_sellable=sellable)
        if not inventory["reconciled"]:
            raise ValueError("모델별 체결 수량과 실제 보유수량 대조 실패: " + " · ".join(inventory["issues"]))
        from dockdack.signal_bridge import prototype_family
        family = prototype_family(self._prototype_source_for(rule))
        own = [lot for lot in inventory["lots"] if lot["strategy_id"] == family.id]
        if rule.side is OrderSide.BUY:
            if any(lot["quantity_remaining"] > 0 for lot in own):
                raise ValueError("이 모델은 이미 해당 종목을 보유하고 있어 추가 매수하지 않습니다.")
            return inventory
        allocation = self.store.prototype_sell_allocation(rule.id)
        lot = next((lot for lot in own if allocation and lot["lot_id"] == allocation["lot_id"]), None)
        if lot is None or lot["average_price"] is None or lot["quantity_remaining"] < rule.quantity:
            raise ValueError("매도 대상 모델의 확정 체결 보유분을 확인할 수 없습니다.")
        # This rule's reserved shares are included in quantity_reserved_sell.
        other_reserved = max(Decimal(0), lot["quantity_reserved_sell"] - Decimal(allocation["quantity"]))
        # The broker can lag an accepted order or already subtract it. Reserve
        # outstanding shares across ALL model lots either way, as with pending
        # BUY cash: double reservation may defer a sell but never reuse a shared
        # account limit. This ready rule's own allocation is already reserved.
        shared_reserved = sum((row["quantity_reserved_sell"] for row in inventory["lots"]), Decimal(0))
        other_shared_reserved = max(Decimal(0), shared_reserved - Decimal(allocation["quantity"]))
        if (lot["quantity_remaining"] - other_reserved < rule.quantity
                or sellable - other_shared_reserved < rule.quantity):
            raise ValueError("모델별 미체결 매도 예약 또는 실제 매도 가능 수량이 부족합니다.")
        self._lot_sellable_checks[rule.id] = (sellable, self.clock())
        while len(self._lot_sellable_checks) > 500:
            self._lot_sellable_checks.pop(next(iter(self._lot_sellable_checks)))
        return lot

    def _validate_lot_final(self, item, rule):
        if self._prototype_source_for(rule) is None:
            return
        inventory = self.store.prototype_inventory(item.id)
        if not inventory["reconciled"]:
            raise ValueError("모델별 체결 기록을 확인할 수 없어 전송하지 않습니다.")
        if any(attempt["rule_id"] != rule.id and not self._compatible_lot_pending(rule, attempt)
               for attempt in self.store.attempts(item.id, pending_only=True)):
            raise ValueError("같은 모델 또는 출처 불명의 미확정 주문이 생겨 전송하지 않습니다.")
        if rule.side is OrderSide.BUY:
            from dockdack.signal_bridge import prototype_family
            family = prototype_family(self._prototype_source_for(rule))
            if any(lot["strategy_id"] == family.id and lot["quantity_remaining"] > 0 for lot in inventory["lots"]):
                raise ValueError("같은 모델의 기존 보유 체결이 확인되어 추가 매수하지 않습니다.")
        if rule.side is OrderSide.SELL:
            allocation = self.store.prototype_sell_allocation(rule.id)
            lot = next((lot for lot in inventory["lots"] if allocation and lot["lot_id"] == allocation["lot_id"]), None)
            if (lot is None or lot["average_price"] is None or lot["quantity_remaining"] < rule.quantity
                    or lot["quantity_reserved_sell"] > lot["quantity_remaining"]):
                raise ValueError("모델별 매도 예약/확정 체결 수량이 바뀌어 전송하지 않습니다.")
            checked = self._lot_sellable_checks.pop(rule.id, None)
            reserved = sum((row["quantity_reserved_sell"] for row in inventory["lots"]), Decimal(0))
            if (checked is None or not 0 <= (self.clock() - checked[1]).total_seconds() <= 15
                    or reserved > checked[0]):
                raise ValueError("모델 전체 매도 예약이 확인된 계좌 매도가능수량을 초과하거나 검증이 만료되었습니다.")

    def _lot_cash_account(self, account, rule):
        """Conservatively reserve unfilled prototype buys across this market.

        Some broker snapshots already deduct those orders; deliberately reserving
        again may defer a buy but cannot make an acknowledged order look free.
        """
        if rule.side is not OrderSide.BUY or self._prototype_source_for(rule) is None:
            return account
        reserved = Decimal(0)
        for row in self.store.order_history(limit=None):
            if (row["market"] != account.market.value or row["side"] != "buy"
                    or row["status"] not in {"accepted", "submitting", "unknown"}):
                continue
            if self.store.prototype_rule_source(row["rule_id"]) is None:
                continue
            remaining = max(Decimal(0), Decimal(row["quantity"]) - Decimal(row["filled_quantity"] or "0"))
            reserved += remaining * positive(Decimal(row["reference_price"]), "미체결 매수 예약 단가") * Decimal("1.01")
        available = account.available_to_order
        if isinstance(available, Decimal) and available.is_finite():
            return replace(account, available_to_order=max(Decimal(0), available - reserved))
        return account

    def _policy_for(self, rule):
        record = self.store.external_for_rule(rule.id)
        if record and self.external_policy is not None and record["source_id"] == self.external_policy.source_id:
            return self.external_policy
        if record and record["source_id"] in self.external_sources:
            return self.external_sources[record["source_id"]][0]
        return self.external_policy

    @staticmethod
    def _holding_rule(rule):
        return rule.id.startswith("holding-exit-") and rule.side is OrderSide.SELL

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
            policies = [value[0] for value in self.external_sources.values()] + ([self.external_policy] if self.external_policy else [])
            if not policies or not any(max(policy.max_krw, policy.max_usd) > 0 for policy in policies):
                raise ValueError("외부 신호 출처와 시장별 주문 상한을 먼저 설정하세요.")
            if any(self._mark1_source(policy.source_id) and policy.source_id not in self.source_validators for policy in policies):
                raise ValueError("mark1 prototype 모의 트리거의 신뢰된 모델 주문 검증 연결이 필요합니다.")
        elif not self.enable_holdings_exits and not any(rule.kind is not TriggerKind.EXTERNAL for rule in self.store.rules(statuses=("ready",))):
            raise ValueError("대기 중인 트리거 규칙이 없습니다.")
        if not self.isolated_symbol_errors and any(a["status"] in {"submitting", "unknown"} for a in self.store.attempts(pending_only=True)):
            raise ValueError("접수 여부가 불명확한 주문이 있습니다. 주문 내역을 확인하세요.")
        for item in self.store.items():
            self._ensure_environment(item.instrument, orders=True)
        for rule in self.store.rules(statuses=("ready",)):
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
                policies = [value[0] for value in self.external_sources.values()] + ([self.external_policy] if self.external_policy else [])
                if self.external_only and any(self._demo_source(policy.source_id) for policy in policies):
                    raise ValueError("내장 모의 테스트/연구 신호는 실전 주문에 연결할 수 없습니다.")
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
        return (str(value).strip().lower().replace("_", "-") == "random-demo"
                or AutoTrader._mark1_source(value))

    @staticmethod
    def _mark1_source(value):
        return mark1_prototype_origin(value)

    @classmethod
    def _mark1_origin(cls, record, metadata):
        if not record:
            return False
        return (mark1_prototype_origin(record["source_id"], metadata)
                or cls._mark1_source(metadata.get("origin_strategy", ""))
                or cls._mark1_source(metadata.get("strategy_id", "")))

    def _reject_demo_rule(self, rule):
        if self._mode is not TradingMode.REAL:
            return
        if self._holding_rule(rule):
            if self.store.prototype_sell_allocation(rule.id) is not None:
                self.disarm()
                raise ValueError("prototype 모델별 보유분 청산은 모의 환경만 허용합니다.")
            saved = self.store.exit_targets(rule.watch_id)
            record = self.store.external_for_rule(saved["rule_id"]) if saved else None
            metadata = json.loads(record["payload"]) if record else {}
            if saved and (self._mark1_source(saved["source"]) or self._mark1_origin(record, metadata)):
                self.disarm()
                raise ValueError("mark1 prototype 보유분 청산은 모의 환경만 허용합니다.")
            return
        if rule.kind is not TriggerKind.EXTERNAL:
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
        if self._demo_source(record["source_id"]) or self._mark1_origin(record, metadata) or metadata.get("signal_id") == generated_id or test_origin:
            self.disarm()
            raise ValueError("내장 모의 테스트 신호는 출처 이름을 바꿔도 실전 주문에 사용할 수 없습니다.")

    def _validate_source_execution(self, item, rule, fresh, actual_limit_price, *, stage):
        if rule.kind is not TriggerKind.EXTERNAL:
            return
        record = self.store.external_for_rule(rule.id)
        if not record:
            raise ValueError("외부 주문의 출처 기록이 없습니다.")
        metadata = json.loads(record["payload"])
        mark1 = self._mark1_origin(record, metadata)
        validator = self.source_validators.get(record["source_id"])
        if mark1:
            from dockdack.signal_bridge import prototype_family
            if prototype_family(record["source_id"], metadata) is None:
                raise ValueError("모델의 원본 매수 신호 출처를 확인할 수 없습니다.")
            if self._mode is not TradingMode.DEMO:
                self.disarm()
                raise ValueError("mark1 prototype은 모의 트리거 전용이며 실전 전송을 허용하지 않습니다.")
            if not self._mark1_source(record["source_id"]) or validator is None:
                raise ValueError("mark1 prototype의 원본 출처와 신뢰된 모델 주문 검증 연결이 필요합니다.")
            if actual_limit_price is None:
                raise ValueError("mark1 prototype은 검증 가능한 현재가 지정가 주문만 허용합니다.")
        if validator is not None:
            validator(item, rule, fresh, actual_limit_price, stage=stage)

    def _message(self, key: str, symbol: str, message: str, *, category="system"):
        if self._messages.get(key) != message:
            self.store.event(symbol, message, category=category)
            self._messages[key] = message
            if len(self._messages) > 4096:
                for stale in list(self._messages)[:512]:
                    self._messages.pop(stale, None)

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
            if not self.isolated_symbol_errors:
                self.disarm()
            raise ValueError("접수 여부 확인 필요 · 해당 종목 추가 주문 격리. 영웅문 주문 내역을 확인하세요.")
        if not pending:
            return
        executions = self.service.safety_executions(item.instrument)
        from dockdack.manual_orders import MANUAL_PREFIX, _side, reconcile_manual_executions
        from dockdack.fill_recovery import normalized_order_number
        if any(a["rule_id"].startswith(MANUAL_PREFIX) for a in pending):
            reconcile_manual_executions(self.store, item.instrument, executions)
        rules = {rule.id: rule for rule in self.store.rules(item.id, include_inactive=True, statuses=("accepted", "submitting", "unknown"))}
        for attempt in pending:
            if attempt["rule_id"].startswith(MANUAL_PREFIX):
                continue
            # Today's endpoint cannot prove a previous day's fill; keep it blocked for manual review.
            started = datetime.fromisoformat(attempt["started_at"])
            if market_time(item.instrument.market, started).date() != market_time(item.instrument.market, self.clock()).date():
                continue
            matches = [execution for execution in executions if execution.symbol == item.instrument.symbol
                       and normalized_order_number(execution.order_number) == normalized_order_number(attempt["order_number"])]
            if len(matches) > 1:
                raise ValueError("동일 주문번호의 체결 응답이 여러 개여서 반영하지 않습니다.")
            for execution in matches:
                rule = rules[attempt["rule_id"]]
                filled, remaining = execution.filled_quantity, execution.remaining_quantity
                if not filled.is_finite() or not remaining.is_finite() or filled < 0 or remaining < 0:
                    raise ValueError("체결 수량을 확인할 수 없습니다.")
                if execution.order_quantity != rule.quantity or _side(execution.side) is not rule.side:
                    raise ValueError("체결 내역의 원주문 수량·방향이 저장된 주문과 다릅니다.")
                if filled + remaining > rule.quantity:
                    raise ValueError("체결·미체결 합계가 원주문 수량을 넘습니다.")
                self.store.record_execution(rule.id, filled_quantity=filled, remaining_quantity=remaining,
                                            fill_price=execution.fill_price, observed_at=self.clock())
                if filled == rule.quantity and remaining == 0:
                    self.store.finish(rule.id, "filled", f"주문번호 {execution.order_number} · {filled}주 체결 확인")
                    break
                if str(execution.status).strip().lower() in {"취소", "취소완료", "취소확인", "취소확인완료", "cancelled", "canceled"} and remaining == 0:
                    self.store.finish(rule.id, "cancelled", f"주문번호 {execution.order_number} · 잔량 취소 확인")
                    break
                self._message(attempt["rule_id"] + ":unfilled", item.id,
                              f"{'부분체결' if filled > 0 else '미체결'} · 주문번호 {execution.order_number} · 체결 {filled}주 / 잔량 {remaining}주 · 중복 재주문하지 않음", category="order")
                break
            else:
                self._message(attempt["rule_id"] + ":unfilled", item.id,
                              f"접수 후 체결 확인 대기 · 주문번호 {attempt['order_number']} · 이번 조회에 주문 행 없음(체결 실패 확정 아님), 중복 재주문하지 않음", category="order")

    def _preflight(self, item: WatchItem, rule: TriggerRule, snapshot: MarketSnapshot):
        self._lot_sellable_checks.pop(rule.id, None)
        inst = item.instrument
        self._ensure_environment(inst, orders=True)
        self.service.ensure_common_equity(inst)
        pending = self.store.attempts(item.id, pending_only=True)
        if any(not self._compatible_lot_pending(rule, attempt) for attempt in pending):
            raise ValueError("이 종목의 이전 주문이 미확정/미체결 상태입니다.")
        orders = self.service.safety_orders(inst)
        if self._stop.is_set():
            raise InterruptedError("사용자 중지 요청")
        for order in orders:
            if not order.remaining_quantity.is_finite() or order.remaining_quantity < 0:
                raise ValueError("미체결 잔량을 확인할 수 없습니다.")
            if order.remaining_quantity > 0:
                from dockdack.fill_recovery import normalized_order_number
                known = [attempt for attempt in pending
                         if normalized_order_number(attempt["order_number"]) == normalized_order_number(order.order_number)]
                if (len(known) != 1 or not self._compatible_lot_pending(rule, known[0])
                        or order.symbol != inst.symbol):
                    raise ValueError("미체결 주문이 있어 추가 자동주문을 차단합니다.")
                from dockdack.manual_orders import _side
                with self.store.connection() as db:
                    original = db.execute("SELECT side,quantity FROM rules WHERE id=?", (known[0]["rule_id"],)).fetchone()
                if (not original or order.market is not inst.market or order.exchange != inst.exchange
                        or _side(order.side).value != original["side"]
                        or order.order_quantity != original["quantity"]
                        or not order.filled_quantity.is_finite() or order.filled_quantity < 0
                        or order.filled_quantity + order.remaining_quantity > order.order_quantity):
                    raise ValueError("다른 모델 미체결 주문의 종목·방향·수량 기록이 일치하지 않습니다.")
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
        lot_inventory = self._validate_lot_inventory(item, rule, positions)
        account = self._lot_cash_account(account, rule)
        if rule.side is OrderSide.BUY and lot_inventory is None and any(p.quantity > 0 for p in positions):
            raise ValueError("이미 보유한 종목의 추가 자동매수는 지원하지 않습니다.")
        if rule.side is OrderSide.SELL and sum((p.sellable_quantity for p in positions), Decimal(0)) < rule.quantity:
            raise ValueError("매도 가능 수량이 부족합니다. 공매도는 지원하지 않습니다.")
        # Account/open-order calls take time: recheck both the trigger and limit using a fresh quote.
        fresh = MarketSnapshot(self.service.quote(inst), snapshot.history, self.clock())
        if self._holding_rule(rule):
            self._validate_quote(item, fresh.quote)
        else:
            self._validate_snapshot(item, fresh)
        if not evaluate_trigger(rule, fresh, self.clock()).matched:
            raise ValueError("주문 직전 재조회한 가격에서는 트리거 조건이 성립하지 않습니다.")
        if (self.enable_holdings_exits and rule.side is OrderSide.SELL and
                (rule.kind is TriggerKind.EXTERNAL or self._holding_rule(rule))):
            held = [position for position in positions if position.quantity > 0]
            total = sum((position.quantity for position in held), Decimal(0))
            average = sum((position.average_price * position.quantity for position in held), Decimal(0)) / total
            targets = lot_inventory if lot_inventory is not None else self.holding_exit_targets(replace(held[0], average_price=average))
            upper, lower = targets["take_profit_price"], targets["stop_loss_price"]
            if upper is None or lower is None:
                raise ValueError("매도 직전 보유종목 목표가격/평균매입가를 확인할 수 없습니다.")
            if not (fresh.quote.price >= upper or fresh.quote.price <= lower):
                raise ValueError("매도 직전 현재가가 보유종목 상방·하방 목표가격에 도달하지 않았습니다.")
        metadata = validate_external_rule(self.store, rule, self._policy_for(rule), self.clock()) if rule.kind is TriggerKind.EXTERNAL else {}
        kind = metadata.get("order_type", "limit")
        price = None if kind == "market" else current_common_equity_limit_price(inst.market, rule.side, fresh.quote.price)
        if rule.side is OrderSide.BUY and self.equity_buy_percent is not None:
            policy = self._policy_for(rule) if rule.kind is TriggerKind.EXTERNAL else None
            cap = min(rule.max_notional, policy.cap(inst.market)) if policy else rule.max_notional
            quantity = allocation_quantity(account, price or fresh.quote.price, self.equity_buy_percent, cap,
                                           policy.max_quantity if policy else 999_999_999)
            rule = self.store.size_ready_rule(rule, quantity)
            self.store.event(item.id, f"비중 매수 수량 산정 · 예수금+보유평가액의 {self.equity_buy_percent}% · {quantity}주 · 주문상한/가용액 1% 여유 적용", category="order")
        if "take_profit_price" in metadata and not Decimal(metadata["stop_loss_price"]) < fresh.quote.price < Decimal(metadata["take_profit_price"]):
            raise ValueError("매수 직전 현재가가 하방·상방 목표가격 사이에 있지 않습니다.")
        notional = fresh.quote.price * rule.quantity
        if notional > rule.max_notional:
            raise ValueError("예상 주문금액이 규칙의 상한을 넘습니다.")
        if rule.side is OrderSide.BUY:
            available = account.available_to_order
            # Leave 1% headroom for fees; the broker still makes the final funds check.
            if available is None or not available.is_finite() or available < notional * Decimal("1.01"):
                raise ValueError("주문가능금액이 불명확하거나 부족합니다 (1% 여유 포함).")
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
        if price is not None and price * rule.quantity > rule.max_notional:
            raise ValueError("가격 단위에 맞춘 지정가 주문금액이 규칙의 상한을 넘습니다.")
        self._validate_source_execution(item, rule, fresh, price, stage="preflight")
        request = self.service.prepare(inst, rule.side.value, rule.quantity, kind, price)
        if (request.market, request.symbol, request.exchange, request.side, request.quantity, request.price) != (
                inst.market, inst.symbol, inst.exchange, rule.side, rule.quantity, price) or request.order_type not in ({"3"} if kind == "market" else {"0", "00"}):
            raise ValueError("주문 미리보기와 트리거의 종목·수량·가격이 다릅니다.")
        return request, fresh, rule

    def _execute(self, item: WatchItem, rule: TriggerRule, snapshot: MarketSnapshot) -> bool:
        if item.instrument.market is Market.US:
            remaining = self.store.rejection_cooldown_remaining(item.id, rule.id, self.clock(), seconds=self.us_failure_cooldown_seconds)
            if remaining:
                self._message("rejection-cooldown:" + item.id, item.id,
                              f"미국 주문 거절 후 해당 종목 재시도 대기 · 최대 {self.us_failure_cooldown_seconds}초 · 새 신호도 즉시 재주문하지 않음", category="order")
                return False
        attempted = False
        for _ in range(self.us_retry_attempts if item.instrument.market is Market.US else 1):
            if self._stop.is_set() or not self.orders_enabled:
                break
            attempted = self._execute_once(item, rule, snapshot) or attempted
            if item.instrument.market is not Market.US or self._stop.is_set() or not self.orders_enabled:
                break
            next_rule = self.store.retry_rule(rule, maximum=self.us_retry_attempts)
            if next_rule is None:
                break  # accepted, partially filled, unknown and cancellation-pending never retry.
            rule = next_rule
        return attempted

    def _execute_once(self, item: WatchItem, rule: TriggerRule, snapshot: MarketSnapshot) -> bool:
        self._validate_rule(rule)
        request, fresh, rule = self._preflight(item, rule, snapshot)
        # A newer HOLD/SELL decision may have arrived during account/quote requests.
        self._read_external()
        self._validate_rule(rule)
        self._ensure_environment(item.instrument, orders=True)
        if self._stop.is_set() or not self.orders_enabled or not regular_session(item.instrument.market, self.clock()):
            return False
        if (self.clock() - fresh.fetched_at).total_seconds() > 15:
            raise ValueError("주문 직전 시세가 오래되어 전송하지 않습니다.")
        claim_options = {"prototype_lots": True} if self._prototype_source_for(rule) is not None else {}
        if not self.store.claim(rule, fresh.quote.price, self.clock(), **claim_options):
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
            if not known_rejection and not self.isolated_symbol_errors:
                self.disarm()
            try:
                self.store.finish(rule.id, status, str(exc))
            except Exception:
                self.disarm()
                raise
        else:
            try:
                self.store.finish(rule.id, "accepted", result.message, result.order_number)
                if rule.side is OrderSide.BUY:
                    record = self.store.external_for_rule(rule.id)
                    metadata = json.loads(record["payload"]) if record else {}
                    if self._mark1_origin(record, metadata) and self.prototype_lots_enabled:
                        # Confirmed execution snapshots create the virtual lot.
                        # An acknowledgement is never inventory or a fill price.
                        pass
                    elif self._mark1_origin(record, metadata):
                        # Persist strategy ownership, NOT a promised fill price.
                        # The shared holding resolver derives mark1 boundaries
                        # from the broker's actual average cost on each read.
                        reference = request.price or fresh.quote.price
                        from dockdack.signal_bridge import prototype_family
                        family = prototype_family(record["source_id"], metadata)
                        if family is None:
                            raise ValueError("매수 모델 출처를 확인할 수 없습니다.")
                        self.store.set_exit_targets(item.id, reference * (1 + family.take_profit), reference * (1 - family.stop_loss),
                                                    source=record["source_id"], rule_id=rule.id, now=self.clock())
                    elif "take_profit_price" in metadata:
                        self.store.set_exit_targets(item.id, Decimal(metadata["take_profit_price"]), Decimal(metadata["stop_loss_price"]),
                                                    source=record["source_id"], rule_id=rule.id, now=self.clock())
                    else:
                        self.store.clear_exit_targets(item.id)
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
                if not claimed or claimed["status"] != "submitting" or (not claimed["active"] and not self._holding_rule(rule)):
                    raise ValueError("주문 전송 의도 또는 관심종목 상태가 변경되어 전송하지 않음")
                if rule.kind is TriggerKind.EXTERNAL:
                    # Ingestion supersedes only READY rules, not this already
                    # claimed SUBMITTING intent. A newer HOLD must still stop it.
                    record = self.store.external_for_rule(rule.id)
                    latest = db.execute("""SELECT newer.rule_id FROM external_signals newer
                                           JOIN external_signals current
                                             ON newer.source_id=current.source_id AND newer.watch_id=current.watch_id
                                           WHERE current.rule_id=? AND newer.status!='expired'
                                           ORDER BY newer.generated_at DESC LIMIT 1""", (record["rule_id"] if record else rule.id,)).fetchone()
                    if not latest or not record or latest["rule_id"] != record["rule_id"]:
                        raise ValueError("더 최근의 외부 매매/HOLD 신호가 도착하여 이전 주문을 전송하지 않음")
            metadata = json.loads(record["payload"]) if rule.kind is TriggerKind.EXTERNAL and record else {}
            actual_limit = (None if metadata.get("order_type", "limit") == "market" else
                            current_common_equity_limit_price(item.instrument.market, rule.side, fresh.quote.price))
            self._validate_source_execution(item, rule, fresh, actual_limit, stage="final_send")
            self._validate_lot_final(item, rule)
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
            record = self.store.external_for_rule(rule.id)
            if record and record["source_id"] in self.external_source_errors:
                raise ValueError("해당 신호 출처 오류가 해소될 때까지 해당 출처의 주문을 보류합니다.")
            validate_external_rule(self.store, rule, self._policy_for(rule), self.clock())
        elif self.external_only and not self._holding_rule(rule):
            raise ValueError("외부 신호 모드에서는 수동 트리거를 실행하지 않습니다.")

    def _read_external(self):
        readers = dict(self.external_sources)
        if self.external_reader is not None:
            readers[self.external_policy.source_id if self.external_policy else "legacy-reader"] = (self.external_policy, self.external_reader)
        if self.external_only:
            for source, (_, reader) in readers.items():
                message_key = "external:file" if source == "legacy-reader" else "external:file:" + source
                try:
                    reader()
                except Exception as exc:
                    self.external_error_count += 1
                    self.external_error = str(exc) or type(exc).__name__
                    self.external_source_errors[source] = self.external_error
                    if not self.isolated_symbol_errors:
                        self.disarm()
                    self._message(message_key, "SYSTEM", f"외부 신호 읽기 실패 · {source} 격리: {exc}")
                else:
                    self.external_source_errors.pop(source, None)
                    self._messages.pop(message_key, None)
        self.external_error = " · ".join(error if source == "legacy-reader" else f"{source}: {error}"
                                         for source, error in self.external_source_errors.items())
        self.store.expire_external(self.clock())

    @staticmethod
    def _validate_quote(item, quote):
        inst = item.instrument
        if (quote.market, quote.symbol, quote.exchange, quote.currency) != (inst.market, inst.symbol, inst.exchange, inst.currency):
            raise ValueError("현재가의 종목·거래소·통화가 보유종목과 다릅니다.")
        positive(quote.price, "보유종목 현재가")

    def _lot_holdings_exits(self, item, position, snapshot, targets, sent):
        """Sell each confirmed lot against its own basis, not the broker average."""
        if not targets.get("reconciled"):
            raise ValueError("모델별 보유 대조 실패: " + " · ".join(targets.get("issues", ())))
        for lot in targets["lots"]:
            if self._stop.is_set():
                return
            upper, lower = lot["take_profit_price"], lot["stop_loss_price"]
            if upper is None or lower is None:
                continue
            price = snapshot.quote.price
            hit = TriggerKind.PRICE_GE if price >= upper else TriggerKind.PRICE_LE if price <= lower else None
            self._message("holding-lot:" + lot["lot_id"], item.id,
                          f"{lot['model_title']} 분리 매도 감시 · 체결평균 {lot['average_price']} · 현재가 {price} · 상방 {upper} / 하방 {lower} · {'매도 조건 충족' if hit else '대기'}",
                          category="monitor")
            key = (item.id, "lot", lot["lot_id"])
            if not hit or not self.orders_enabled or item.id in sent or key in sent:
                continue
            cap = self.holding_caps.get(position.market, Decimal(0))
            if not isinstance(cap, Decimal) or not cap.is_finite() or cap <= 0:
                continue
            unit = current_common_equity_limit_price(position.market, OrderSide.SELL, price)
            quantity = min(int(lot["sellable_quantity"]), int(position.sellable_quantity), int(cap / unit))
            if quantity < 1:
                continue
            rule = TriggerRule("holding-exit-" + uuid4().hex, item.id, hit, OrderSide.SELL, quantity, cap,
                               upper if hit is TriggerKind.PRICE_GE else lower)
            self.store.save_holding_rule(item, rule)
            try:
                self.store.reserve_prototype_sell(rule.id, lot["lot_id"], quantity)
                if self._execute(item, rule, snapshot):
                    sent.add(key)
            except Exception as exc:
                self._message("holding-lot-error:" + lot["lot_id"], item.id,
                              f"{lot['model_title']} 분리 매도 보류: {exc}", category="signal")
            finally:
                self.store.pause_rule(rule.id)

    def _holdings_pass(self, sent, *, checkpoint=None, progress=None):
        """Account holdings are SELL candidates even when absent from the watchlist.

        No chart/history calls; every position gets a fresh quote and preflight
        rechecks both the price condition and actual sellable shares before send.
        """
        def report(phase, market=None, *, completed=0, total=0, position=None, error=None):
            # Keep UI callback failures outside the per-position/broker guards.
            # Progress is observational: it must never cause a silent retry.
            if progress is not None:
                payload = {"phase": phase, "market": market, "completed": completed, "total": total,
                           "symbol": position.symbol if position is not None else "",
                           "name": position.name if position is not None else ""}
                if error is not None:
                    payload["error"] = str(error) or type(error).__name__
                progress(("holdings_progress", payload))

        if progress is not None:
            progress(("phase", "보유종목 매도 조건 점검"))
        pass_completed = pass_total = 0
        for market, symbol, exchange in ((Market.DOMESTIC, "005930", "KRX"), (Market.US, "AAPL", "ND")):
            if self._stop.is_set():
                return
            if not regular_session(market, self.clock()):
                report("market_closed", market)
                continue
            report("account", market)
            account_error = None
            try:
                representative = Instrument(market, symbol, exchange)
                self._ensure_environment(representative)
                account = self.service.safety_account(representative)
                if account.market is not market or account.currency != representative.currency:
                    raise ValueError("보유종목 조회의 시장·통화가 다릅니다.")
            except Exception as exc:
                self._message("holdings:" + market.value, "SYSTEM", f"보유종목 매도 감시 조회 실패 · {market.value}: {exc}", category="monitor")
                account_error = exc
            if account_error is not None:
                report("market_error", market, error=account_error)
                continue
            completed, total = 0, len(account.positions)
            pass_total += total
            for position in account.positions:
                if self._stop.is_set():
                    return
                if checkpoint is not None:
                    checkpoint()
                report("checking", market, completed=completed, total=total, position=position)
                rule = position_error = None
                try:
                    if (position.market is not market or position.currency != account.currency or
                        not position.quantity.is_finite() or not position.sellable_quantity.is_finite() or
                        not 0 <= position.sellable_quantity <= position.quantity):
                        raise ValueError("보유종목 시장·수량을 확인할 수 없습니다.")
                    if position.quantity <= 0:
                        continue
                    inst = Instrument(market, position.symbol, position.exchange)
                    item = WatchItem(inst, position.name)
                    quote = self.service.quote(inst)
                    self._validate_quote(item, quote)
                    # Empty history explicitly means quote-only, never fabricated OHLC.
                    snapshot = MarketSnapshot(quote, DailyHistory(market, inst.symbol, inst.exchange, inst.currency, 0, ()), self.clock())
                    self._reconcile(item)
                    targets = self.holding_exit_targets(position)
                    upper, lower = targets["take_profit_price"], targets["stop_loss_price"]
                    if progress is not None:
                        progress(("holding_quote", {"watch_id": item.id, "instrument": inst,
                                                   "quote": quote, "position": position, "targets": targets}))
                    if "lots" in targets:
                        self._lot_holdings_exits(item, position, snapshot, targets, sent)
                        continue
                    if upper is None or lower is None:
                        raise ValueError("보유종목 목표가격/평균매입가를 확인할 수 없습니다.")
                    hit = TriggerKind.PRICE_GE if quote.price >= upper else TriggerKind.PRICE_LE if quote.price <= lower else None
                    self._message("holding:" + item.id, item.id,
                                  f"보유종목 매도 감시 · 현재가 {quote.price} · 상방 {upper} / 하방 {lower} · {'매도 조건 충족' if hit else '대기'}", category="monitor")
                    if not hit or not self.orders_enabled or item.id in sent or self.store.attempts(item.id, pending_only=True):
                        continue
                    if market is Market.US and self.store.rejection_cooldown_remaining(item.id, "", self.clock(), seconds=self.us_failure_cooldown_seconds):
                        self._message("rejection-cooldown:" + item.id, item.id,
                                      f"미국 주문 거절 후 해당 종목 재시도 대기 · 최대 {self.us_failure_cooldown_seconds}초 · 새 신호도 즉시 재주문하지 않음", category="order")
                        continue
                    cap = self.holding_caps.get(market, Decimal(0))
                    if not isinstance(cap, Decimal) or not cap.is_finite() or cap <= 0:
                        continue
                    unit = current_common_equity_limit_price(market, OrderSide.SELL, quote.price)
                    quantity = min(int(position.sellable_quantity), int(cap / unit))
                    if quantity < 1:
                        raise ValueError("매도 가능 정수 수량/주문 상한이 1주에 미달합니다.")
                    rule = TriggerRule("holding-exit-" + uuid4().hex, item.id, hit, OrderSide.SELL, quantity, cap,
                                       upper if hit is TriggerKind.PRICE_GE else lower)
                    self.store.save_holding_rule(item, rule)
                    if self._execute(item, rule, snapshot):
                        sent.add(item.id)
                except Exception as exc:
                    position_error = exc
                    self._message("holding-error:" + position.symbol, position.symbol, f"보유종목 매도 보류: {exc}", category="signal")
                finally:
                    if rule is not None:
                        self.store.pause_rule(rule.id)  # Failed preflight creates no lingering sell candidate.
                    completed += 1
                    pass_completed += 1
                    report("checked", market, completed=completed, total=total, position=position, error=position_error)
            if self._stop.is_set():
                return
            report("market_complete", market, completed=completed, total=total)
        if not self._stop.is_set():
            report("complete", completed=pass_completed, total=pass_total)

    def poll(self, progress=None, checkpoint=None, on_snapshot=None) -> dict[str, MarketSnapshot | Exception]:
        if not self._poll_lock.acquire(blocking=False):
            return {}
        results = {}
        try:
            items = {item.id: item for item in self.store.items()
                     if not self.session_only_poll or regular_session(item.instrument.market, self.clock())}
            deferred_sells = []
            remaining, seen_external, sent = list(items), set(), set()
            # Bounded even if a producer continuously publishes new decisions.
            for _ in range(len(items) + 500):
                if self._stop.is_set():
                    break
                if checkpoint is not None and checkpoint():
                    items = {item.id: item for item in self.store.items()
                             if not self.session_only_poll or regular_session(item.instrument.market, self.clock())}
                    results = {key: value for key, value in results.items() if key in items}
                    remaining = [key for key in items if key not in results]
                self._read_external()
                priority = next((r for r in self.store.rules(statuses=("ready",)) if self.external_only
                                 and r.kind is TriggerKind.EXTERNAL and r.status == "ready"
                                 and r.id not in seen_external and r.watch_id in items
                                 and (not self.enable_holdings_exits or r.watch_id in remaining)), None)
                if priority:
                    item = items[priority.watch_id]
                    if item.id in remaining:
                        remaining.remove(item.id)
                elif remaining:
                    item = items[remaining.pop(0)]
                else:
                    break
                try:
                    rules = self.store.rules(item.id, statuses=("ready",))
                    if self.session_only_poll and not regular_session(item.instrument.market, self.clock()):
                        continue
                    if progress is not None:
                        progress(("watch_progress", {"market": item.instrument.market,
                                                     "symbol": item.instrument.symbol, "name": item.name,
                                                     "completed": len(results), "total": len(items)}))
                    snapshot = self.snapshot(item, rules)
                    results[item.id] = snapshot
                    self._reconcile(item)
                    if on_snapshot is not None and not self._stop.is_set():
                        try:
                            on_snapshot(item, snapshot)
                        except Exception:
                            if not self.isolated_symbol_errors:
                                self.disarm()
                            raise
                        self._read_external()
                        rules = self.store.rules(item.id, statuses=("ready",))
                    seen_external.update(r.id for r in rules if r.kind is TriggerKind.EXTERNAL)
                    for rule in rules:
                        if rule.status != "ready" or self._stop.is_set():
                            continue
                        if (rule.kind is TriggerKind.EXTERNAL) != self.external_only:
                            continue
                        if self.enable_holdings_exits and rule.side is OrderSide.SELL:
                            deferred_sells.append((item, rule, snapshot))
                            continue
                        try:
                            self._validate_rule(rule)
                            signal = evaluate_trigger(rule, snapshot, self.clock())
                            self._message(rule.id + ":signal", item.id,
                                          f"{rule.description} · {'조건 충족' if signal.matched else '대기'}", category="signal")
                            sent_key = self._sent_key(item, rule)
                            if not signal.matched or not self.orders_enabled or item.id in sent or sent_key in sent:
                                continue
                            if not regular_session(item.instrument.market, self.clock()):
                                self._message(rule.id + ":gate", item.id, "정규장 시간이 아니므로 자동주문하지 않음", category="signal")
                                continue
                            if self._execute(item, rule, snapshot):
                                sent.add(sent_key)
                                if not self.prototype_lots_enabled or self._prototype_source_for(rule) is None:
                                    break  # Legacy/manual policies remain one order per symbol.
                        except Exception as exc:
                            self._message(rule.id + ":gate", item.id, f"자동주문 보류: {exc}", category="signal")
                except Exception as exc:
                    results[item.id] = exc
                    self._message(item.id + ":error", item.id, f"조회/확인 실패: {exc}", category="monitor")
                if progress is not None:
                    progress((item.id, results[item.id], len(results), len(items)))
            for item, rule, snapshot in deferred_sells:
                if self._stop.is_set() or not self.orders_enabled or item.id in sent:
                    continue
                try:
                    if regular_session(item.instrument.market, self.clock()) and self._execute(item, rule, snapshot):
                        sent.add(item.id)
                except Exception as exc:
                    self._message(rule.id + ":gate", item.id, f"자동매도 보류: {exc}", category="signal")
            if self.enable_holdings_exits and not self._stop.is_set():
                self._holdings_pass(sent, checkpoint=checkpoint, progress=progress)
            return results
        except Exception as exc:
            # Auxiliary account/ranking/notification checkpoints run outside
            # the per-symbol guard. A confirmed transient read outage must not
            # silently revoke the user's ON state. Unexpected/data-integrity
            # errors still fail closed; explicit earlier OFF is never undone.
            if not self.isolated_symbol_errors or not _transient_poll_failure(exc):
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
