"""Bounded, read-only broker history recovery for the existing local order ledger.

Runs inside the GUI's single broker worker. It never changes order permission,
attempt status, or submits/cancels an order. Unknown fill prices stay unknown.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal

from dockdack.history import market_time
from dockdack.models import ExecutionHistoryRecord, Market, OrderSide, TradingMode
from dockdack.watchlist import utc_now


def normalized_order_number(value: str) -> str:
    if not isinstance(value, str) or not value.strip().isascii() or not value.strip().isdigit():
        raise ValueError("숫자 주문번호를 확인할 수 없습니다.")
    result = value.strip().lstrip("0")
    if not result:
        raise ValueError("유효한 주문번호가 없습니다.")
    return result


def _positive(value) -> bool:
    try:
        number = Decimal(value)
    except (ValueError, TypeError, ArithmeticError):
        return False
    return number.is_finite() and number > 0


class _EnvironmentMismatch(ValueError):
    """Never catch a ledger identity error as a recoverable broker-data error."""


def _scope(row):
    market = Market(row["market"])
    started = datetime.fromisoformat(row["started_at"])
    if started.tzinfo is None:
        raise ValueError("저장된 주문 시각에 시간대가 없습니다.")
    return market, market_time(market, started).date()


def _local_key(row):
    market, day = _scope(row)
    return (market, day, normalized_order_number(row["order_number"]), row["symbol"],
            OrderSide(row["side"]), Decimal(row["quantity"]))


class FillRecovery:
    """At most four date/market history queries per refresh, paced by the service.

    All scans have a 60-second minimum. Each queried group waits 300 seconds for
    automatic retries, or 60 seconds for an explicit manual refresh. Failed
    attempts count too; saved timestamps preserve the minimum across restarts.
    """

    def __init__(self, service, store, *, clock=utc_now):
        self.service, self.store, self.clock = service, store, clock
        self._mode = TradingMode(getattr(service, "mode", TradingMode.DEMO))
        self._storage_scope = getattr(store, "storage_scope", "demo" if self._mode is TradingMode.DEMO else "unconfigured")
        # An unconfigured REAL selection may render an empty dashboard. It must
        # not query a broker or touch recovery evidence until keys are bound.
        self._check_environment(allow_unconfigured=True)
        self._last_scan = None
        self._attempted = {}
        self._status = {"state": "idle", "last_checked_at": None, "checked_groups": 0,
                        "enriched": 0, "unresolved": 0, "errors": {}, "message": "체결가 보완 조회 대기"}

    def status(self):
        return {**self._status, "errors": dict(self._status["errors"])}

    def _check_environment(self, *, allow_unconfigured=False):
        try:
            selected = TradingMode(getattr(self.service, "mode", TradingMode.DEMO))
            stored = TradingMode(getattr(self.store, "mode", TradingMode.DEMO))
        except (TypeError, ValueError) as exc:
            raise _EnvironmentMismatch("체결가 보완의 거래 환경을 확인할 수 없습니다.") from exc
        if selected is not self._mode or stored is not self._mode:
            raise _EnvironmentMismatch("체결가 보완 서비스와 매매 기록의 모의/실전 환경이 다릅니다.")
        if self._mode is TradingMode.REAL:
            scope = getattr(self.service, "storage_scope", "unconfigured")
            if scope != self._storage_scope or scope != getattr(self.store, "storage_scope", "unconfigured"):
                raise _EnvironmentMismatch("체결가 보완 서비스와 실전 매매 기록의 계좌 키 범위가 다릅니다.")
            if not allow_unconfigured and (not scope or scope == "unconfigured"):
                raise _EnvironmentMismatch("실전 API 키 범위가 미설정되어 체결가 보완 조회를 차단합니다.")

    @staticmethod
    def _candidate(row):
        if _positive(row["fill_price"]):
            return False
        return (row["status"] in {"filled", "accepted"}
                or row["status"] == "cancelled" and _positive(row["filled_quantity"]))

    @staticmethod
    def _validate_records(records, market, day):
        if not isinstance(records, (tuple, list)):
            raise ValueError("전체 체결 내역 배열을 확인할 수 없습니다.")
        for record in records:
            if (not isinstance(record, ExecutionHistoryRecord) or record.market is not market
                    or record.order_date != day or not isinstance(record.side, OrderSide)
                    or record.currency != ("KRW" if market is Market.DOMESTIC else "USD")):
                raise ValueError("체결 내역의 조회 시장/날짜/통화/방향이 다릅니다.")
            normalized_order_number(record.order_number)
            for value in (record.order_quantity, record.filled_quantity, record.remaining_quantity):
                if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                    raise ValueError("체결 내역 수량을 확인할 수 없습니다.")
            if record.order_quantity <= 0 or record.filled_quantity + record.remaining_quantity > record.order_quantity:
                raise ValueError("체결 내역 수량이 원주문 수량과 다릅니다.")
            if not record.symbol or record.exchange not in ({"KRX"} if market is Market.DOMESTIC else {"", "ND", "NY", "NA"}):
                raise ValueError("체결 내역의 종목/거래소를 확인할 수 없습니다.")

    def _save(self, row, status, message, **kwargs):
        self._check_environment()
        self.store.record_fill_recovery(row["rule_id"], status=status, message=message,
                                        checked_at=self.clock(), **kwargs)

    def _recover(self, row, records, local_index):
        key = _local_key(row)
        matching = [record for record in records if
                    (record.market, record.order_date, normalized_order_number(record.order_number),
                     record.symbol, record.side, record.order_quantity) == key
                    and (not record.exchange or record.exchange == row["exchange"])]
        if not matching:
            self._save(row, "not_found", "조회일의 주문번호·종목·방향·수량이 일치하는 체결 내역 없음")
            return False
        if len(matching) != 1:
            self._save(row, "ambiguous", "동일 주문의 복수/상충 내역으로 체결가 미확인")
            return False
        record = matching[0]
        locals_matching = [other for other in local_index[key]
                           if not record.exchange or other["exchange"] == record.exchange]
        if len(locals_matching) != 1:
            self._save(row, "ambiguous", "동일 주문번호의 로컬 후보가 여러 개여서 거래소/주문 일치 미확인")
            return False
        metadata = {"source_api": record.source_api, "price_basis": record.price_basis,
                    "order_date": record.order_date.isoformat(), "order_time": record.order_time,
                    "fill_time": record.fill_time, "reported_fill_price": record.reported_fill_price}
        if record.original_order_number.strip("0 "):
            self._save(row, "ambiguous", "정정·취소 원주문 연결이 있는 내역은 자동 체결가 보완하지 않음", **metadata)
            return False
        if record.market is Market.US:
            # ust21150 does not return an independently identified order date,
            # and its query date timezone is undocumented. order_date is only
            # our query scope: equality cannot prove that a recycled order
            # number belongs to the local New York trading date. Even a known
            # venue / single-share price cannot bridge that missing evidence.
            metadata["price_basis"] = "date_scope_unverified"
            self._save(row, "ambiguous",
                       "미국 주문일자의 시간대 근거 미확인 · 조회 날짜만으로 실제 주문일을 확정할 수 없어 "
                       "체결가·체결수량을 기존 주문에 반영하지 않음", **metadata)
            return False
        previous_qty = (Decimal(row["filled_quantity"]) if row["filled_quantity"] is not None
                        else Decimal(row["quantity"]) if row["status"] == "filled" else Decimal(0))
        if record.filled_quantity < previous_qty:
            self._save(row, "quantity_conflict", "기존 확인 체결 수량보다 적은 응답 · 기존 기록 유지", **metadata)
            return False
        # The dated APIs expose an execution price without proven multi-fill
        # VWAP semantics. Only a single filled share proves its unit cost.
        usable = (record.price_basis == "single_share" and record.order_quantity == record.filled_quantity == 1
                  and isinstance(record.fill_price, Decimal) and _positive(record.fill_price))
        status = "enriched" if usable else "price_unknown"
        message = (f"주문번호 {row['order_number']} · 1주 체결가 {record.fill_price} {record.currency} 확인"
                   if usable else "체결 수량 조회 완료 · 유효 체결가/다수 체결 평균가 근거 미확인")
        self._save(row, status, message, **metadata, filled_quantity=record.filled_quantity,
                   remaining_quantity=record.remaining_quantity, fill_price=record.fill_price if usable else None)
        return usable

    def refresh_due(self, force=False, stopped=lambda: False):
        # Before throttling, DB scans, error persistence or any broker I/O.
        self._check_environment()
        now = self.clock()
        if stopped() or self._last_scan is not None and (now-self._last_scan).total_seconds() < 60:
            return self.status()
        self._last_scan = now
        rows = self.store.order_history(limit=None)
        candidates = [row for row in rows if self._candidate(row)]
        groups, local_index, errors = defaultdict(list), defaultdict(list), {}
        for row in rows:
            try:
                local_index[_local_key(row)].append(row)
            except (ValueError, TypeError, ArithmeticError):
                pass  # Non-candidates cannot be used to corroborate a fill.
        for row in candidates:
            try:
                group = _scope(row)
                _local_key(row)
                groups[group].append(row)
                if row["recovery_checked_at"]:
                    persisted = datetime.fromisoformat(row["recovery_checked_at"])
                    if persisted.tzinfo is None:
                        raise ValueError("저장된 체결가 조회 시각에 시간대가 없습니다.")
                    if group not in self._attempted or persisted > self._attempted[group]:
                        self._attempted[group] = persisted
            except (ValueError, TypeError, ArithmeticError) as exc:
                errors[row["rule_id"]] = str(exc)
                self._save(row, "error", str(exc))
        minimum = 60 if force else 300
        oldest = datetime.min.replace(tzinfo=timezone.utc)
        due = [key for key in groups if key not in self._attempted or (now-self._attempted[key]).total_seconds() >= minimum]
        due.sort(key=lambda key: (self._attempted.get(key, oldest), -key[1].toordinal(), key[0].value))
        checked = enriched = 0
        for market, day in due[:4]:
            if stopped():
                break
            group = (market, day)
            self._attempted[group] = self.clock()
            checked += 1
            try:
                self._check_environment()
                records = self.service.execution_history(market, day)
                self._check_environment()
                self._validate_records(records, market, day)
                for row in groups[group]:
                    if stopped():
                        break
                    try:
                        enriched += bool(self._recover(row, records, local_index))
                    except (ValueError, TypeError, ArithmeticError) as exc:
                        errors[row["rule_id"]] = str(exc)
                        self._save(row, "error", str(exc))
            except _EnvironmentMismatch:
                raise  # A different account's response must not alter this ledger.
            except Exception as exc:
                message = str(exc)[:500] or type(exc).__name__
                errors[f"{market.value}:{day.isoformat()}"] = message
                for row in groups[group]:
                    self._save(row, "error", f"체결가 조회 실패 · {message}")
            self._attempted[group] = self.clock()
        unresolved = len(candidates)-enriched
        state = "error" if errors else "complete" if not unresolved else "partial" if checked else "waiting"
        self._status = {"state": state, "last_checked_at": self.clock(), "checked_groups": checked,
                        "enriched": enriched, "unresolved": unresolved, "errors": errors,
                        "message": f"체결가 보완 {enriched}건 · 미확인 {unresolved}건 · 날짜/시장 조회 {checked}회"}
        return self.status()
