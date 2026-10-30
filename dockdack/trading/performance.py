"""Gross realized P/L for the application's *complete local* order ledger.

This is not account-wide broker performance: purchases made elsewhere, corporate
actions, fees and taxes are not known here. Supply every historical order, not
the paginated/display subset. Input order must be chronological for records with
identical ``started_at`` values; sorting by timestamp is otherwise stable.

Only executed quantities participate. Missing acquisition prices remain unknown
FIFO lots, and an uncovered or partly unknown sale has no invented total P/L.
An order's reference/current price is deliberately never used as a fill price.
Only fully executed one-share orders can use a price without average-price
provenance. Multi-share/partial-order prices require verified average evidence
bound to exactly the current cumulative filled quantity and price.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any


ZERO = Decimal("0")
HUNDRED = Decimal("100")
_EXECUTION_STATES = {"filled", "accepted", "cancelled", "reviewed"}
_AVERAGE_PRICE_BASES = {"broker_average", "weighted_fills"}
_REASONS = {
    "buy": "매수는 실현손익 계산 대상이 아닙니다.",
    "not_filled": "확인된 체결 수량이 없습니다.",
    "missing_fill_price": "실제 매도 체결가가 확인되지 않았습니다.",
    "missing_buy_price": "대응하는 매수의 실제 체결가가 확인되지 않았습니다.",
    "missing_buy_history": "매도 수량에 대응하는 과거 매수 기록이 부족합니다.",
    "duplicate_rule_id": "중복 주문 ID 때문에 거래 순서와 수량을 확정할 수 없습니다.",
    "missing_rule_id": "주문 ID가 없어 거래를 식별할 수 없습니다.",
    "invalid_time": "주문 시각이 유효하지 않아 거래 순서를 확정할 수 없습니다.",
    "invalid_quantity": "체결 수량이 유효하지 않거나 주문 수량과 충돌합니다.",
    "invalid_instrument": "시장·거래소·종목·통화 정보가 불완전합니다.",
    "invalid_side": "매수·매도 구분을 확인할 수 없습니다.",
    "unverified_average_price": "다주 주문의 누적 평균 체결가 근거가 확인되지 않았습니다.",
    "invalid_lot_allocation": "모델별 매도와 원매수 연결이 올바르지 않습니다.",
}


def _text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if number.is_finite() else None


def _positive(value: Any) -> Decimal | None:
    number = _decimal(value)
    return number if number is not None and number > ZERO else None


def _time(value: Any) -> datetime | None:
    try:
        moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if moment.tzinfo is None or moment.utcoffset() is None:
            return None
        return moment.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def _quantity(record: Mapping[str, Any]) -> tuple[Decimal, bool, str | None]:
    status = _text(record.get("status"))
    if status not in _EXECUTION_STATES:
        return ZERO, False, None
    requested = _positive(record.get("quantity"))
    raw = record.get("filled_quantity")
    inferred = raw is None and status == "filled"
    if raw is None and not inferred:
        return ZERO, False, None
    executed = requested if inferred else _decimal(raw)
    if (requested is None or executed is None or executed < ZERO
            or executed > requested or (status == "filled" and executed != requested)):
        return ZERO, inferred, "invalid_quantity"
    raw_remaining = record.get("remaining_quantity")
    if raw_remaining is not None:
        remaining = _decimal(raw_remaining)
        if (remaining is None or remaining < ZERO or executed + remaining > requested
                or (status == "filled" and remaining != ZERO)):
            return ZERO, inferred, "invalid_quantity"
    return executed, inferred, None


def _execution_price(record: Mapping[str, Any], quantity: Decimal) -> tuple[Decimal | None, str | None]:
    """Never turn an unverified last-execution unit price into cumulative VWAP."""
    if quantity <= ZERO:
        return None, "not_filled"
    price = _positive(record.get("fill_price"))
    if price is None:
        return None, "missing_fill_price"
    if _positive(record.get("quantity")) == quantity == Decimal("1"):
        return price, None
    basis = _text(record.get("price_basis") or record.get("recovery_price_basis"))
    # Provenance alone may be stale after more shares fill. Require the
    # evidence to identify this exact cumulative quantity AND price snapshot.
    if (basis in _AVERAGE_PRICE_BASES
            and _positive(record.get("price_basis_quantity")) == quantity
            and _positive(record.get("price_basis_price")) == price):
        return price, None
    return None, "unverified_average_price"


def _metric(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": row["id"], "status": "not_applicable", "side": row["side"],
        "market": row["key"][0], "currency": row["key"][3],
        "filled_quantity": row["quantity"], "quantity_inferred": row["inferred"],
        "fill_price": row["price"], "effective_fill_price": row["price"],
        "price_reason_code": row["price_reason"],
        "price_reason": ("실제 체결가가 확인되지 않았습니다." if row["price_reason"] == "missing_fill_price"
                         else _REASONS.get(row["price_reason"], "")),
        "proceeds": None, "cost_basis": None,
        "realized_profit": None, "return_pct": None,
        "matched_quantity": ZERO, "unmatched_quantity": ZERO,
        "allocations": (),
        "reason_code": "buy" if row["side"] == "buy" else "not_filled",
        "reason": _REASONS["buy" if row["side"] == "buy" else "not_filled"],
        "basis": "local_fifo", "gross": True,
    }


def realized_performance(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate per-sell FIFO results without API, database or filesystem I/O.

    Returns ``by_rule_id`` metrics and ``summaries`` keyed by ``(market,
    currency)``. Amounts/quantities are Decimal; unknown values are None. A
    summary's ``known_*`` fields cover ONLY fully attributable sales, and
    ``complete`` is false if any executed sale in that market is unknown.

    Filled legacy orders can infer their full requested quantity, but never
    their execution price. Partial accepted/cancelled/reviewed orders use only
    explicit confirmed filled_quantity. Manual review must not erase a prior
    execution, but a reviewed status never establishes a fill by itself. Other
    states are not execution evidence. Snapshot records
    must be one cumulative record per order, not incremental executions.
    Executed records require a positive requested quantity and consistent
    finite, nonnegative filled/remaining quantities. A filled status must mean
    the full requested quantity with no remaining shares; absent legacy
    remaining_quantity is allowed, but an explicit contradictory value is not.

    ``effective_fill_price`` (also ``fill_price``) is safe for P/L/display;
    ``price_reason`` explains an unusable raw price. Without provenance, only
    requested quantity == filled quantity == 1 is accepted. Other orders
    require ``price_basis`` (or ``recovery_price_basis``) equal to
    ``broker_average`` or ``weighted_fills``, plus ``price_basis_quantity`` and
    ``price_basis_price`` matching the current cumulative filled quantity and
    raw fill_price exactly. Callers must provide these only from verified
    broker-average or weighted-execution evidence, not merely label a price.

    Duplicate IDs, invalid chronology/quantities or instrument identity fail
    closed for the affected instrument, including subsequent sales. Unknown
    cost lots are consumed normally so their quantity cannot be reused later.
    Unmatched sales do not borrow a later purchase to fabricate their basis.

    Each sell's ``allocations`` is an additive, per-acquisition breakdown using
    the exact same matching decisions as its total. An unknown total leaves
    every allocation unknown (even a seemingly usable subset); callers must
    not silently recover optimistic partial profits from incomplete evidence.
    """
    raw_records = list(records)
    ids = Counter(_text(record.get("rule_id")) for record in raw_records)
    rows, warnings = [], []
    poisoned: dict[tuple[str, ...], str] = {}
    for index, record in enumerate(raw_records):
        original_id = _text(record.get("rule_id"))
        rule_id = original_id or f"__missing_rule_id__:{index}"
        key = tuple(_text(record.get(name)) for name in ("market", "exchange", "symbol", "currency"))
        side = _text(record.get("side"))
        quantity, inferred, quantity_error = _quantity(record)
        price, price_reason = _execution_price(record, quantity)
        if quantity_error:
            price_reason = quantity_error
        moment = _time(record.get("started_at"))
        # Ignore unexecuted attempts when judging ledger reliability; an
        # invalid rejected order cannot change the cost of an actual holding.
        issues = []
        participating = quantity > ZERO or quantity_error is not None
        if participating:
            if not original_id:
                issues.append("missing_rule_id")
            elif ids[original_id] > 1:
                issues.append("duplicate_rule_id")
            if not all(key):
                issues.append("invalid_instrument")
            if side not in {"buy", "sell"}:
                issues.append("invalid_side")
            if moment is None:
                issues.append("invalid_time")
            if quantity_error:
                issues.append(quantity_error)
            if side == 'sell' and record.get('prototype_lot_id') and record.get('prototype_buy_rule_id') and record['prototype_lot_id'] != record['prototype_buy_rule_id']:
                issues.append('invalid_lot_allocation')
        if issues:
            poisoned.setdefault(key, issues[0])
            warnings.append({"rule_id": rule_id, "reason_codes": tuple(issues)})
        close_plan = []
        try:
            offset, seen = ZERO, set()
            for allocation in record.get("close_allocations", ()):
                lot_id = _text(allocation["lot_id"])
                amount = Decimal(allocation["quantity"])
                start = Decimal(allocation["fill_offset"])
                if (not original_id.startswith("close-") or side != "sell" or not lot_id or lot_id in seen
                        or not amount.is_finite() or amount <= 0 or amount != amount.to_integral_value()
                        or not start.is_finite() or start != offset):
                    raise ValueError("invalid close allocation")
                seen.add(lot_id)
                close_plan.append((lot_id, min(amount, max(ZERO, quantity - start))))
                offset += amount
            if close_plan and offset > Decimal(str(record.get("quantity", 0))):
                raise ValueError("close allocation exceeds request")
        except (ValueError, TypeError, KeyError, ArithmeticError):
            issues.append("invalid_lot_allocation")
            poisoned.setdefault(key, "invalid_lot_allocation")
            warnings.append({"rule_id": rule_id, "reason_codes": ("invalid_lot_allocation",)})
            close_plan = []
        rows.append({"id": rule_id, "key": key, "side": side, "quantity": quantity,
                     "inferred": inferred, "price": price, "price_reason": price_reason,
                     "time": moment, "index": index, "issues": issues,
                     "allocation": _text(record.get('prototype_buy_rule_id') or record.get('prototype_lot_id')) if side == 'sell' else '',
                     "close_plan": close_plan,
                     "duplicate": bool(original_id and ids[original_id] > 1)})

    # A partially identified historical buy might belong to an otherwise
    # well-identified sale. Do not make it disappear merely because one key
    # component was absent in a legacy/corrupt record.
    incomplete_keys = {row["key"] for row in rows if "invalid_instrument" in row["issues"]}
    for partial in incomplete_keys:
        for row in rows:
            if all(not value or value == candidate for value, candidate in zip(partial, row["key"])):
                poisoned.setdefault(row["key"], "invalid_instrument")

    rows.sort(key=lambda row: (row["time"] or datetime.max.replace(tzinfo=timezone.utc), row["index"]))
    lots: dict[tuple[str, ...], deque] = defaultdict(deque)
    by_rule_id: dict[str, dict[str, Any]] = {}
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        # A duplicated cumulative snapshot is not a second execution. Its
        # instrument has already been poisoned, so no profit can leak through.
        if row["id"] in by_rule_id:
            continue
        metric = by_rule_id[row["id"]] = _metric(row)
        quantity, key = row["quantity"], row["key"]
        if quantity <= ZERO and not row["issues"]:
            continue
        if row["side"] == "buy":
            if not row["duplicate"] and quantity > ZERO:
                lots[key].append([quantity, row["price"], row['id']])
            if row["issues"]:
                metric.update(status="unknown", reason_code=row["issues"][0],
                              reason=_REASONS[row["issues"][0]])
            continue
        if row["side"] != "sell":
            if row["issues"]:
                metric.update(status="unknown", reason_code=row["issues"][0],
                              reason=_REASONS[row["issues"][0]])
            continue

        remaining, known_cost, missing_price = quantity, ZERO, False
        allocations = []
        if row['allocation'] or row['close_plan']:
            metric['basis'] = 'strategy_lot'
        if not row["duplicate"]:
            if row['close_plan']:
                # Cumulative close fills reduce explicitly captured model lots
                # first. Only the unallocated account residual follows FIFO.
                candidates = [(lot, cap) for lot_id, cap in row['close_plan']
                              for lot in lots[key] if lot[2] == lot_id and cap > ZERO]
                residual = quantity - sum((cap for _, cap in row['close_plan']), ZERO)
                allocated_ids = {lot_id for lot_id, _ in row['close_plan']}
                for lot in lots[key]:
                    if lot[2] in allocated_ids or residual <= ZERO:
                        continue
                    cap = min(residual, lot[0])
                    candidates.append((lot, cap))
                    residual -= cap
            else:
                candidates = [(lot, remaining) for lot in lots[key]
                              if not row['allocation'] or lot[2] == row['allocation']]
            for lot, cap in candidates:
                if remaining <= ZERO:
                    break
                used = min(remaining, lot[0], cap)
                if used <= ZERO:
                    continue
                allocations.append({"buy_rule_id": lot[2], "quantity": used,
                                    "cost_basis": used * lot[1] if lot[1] is not None else None})
                if lot[1] is None:
                    missing_price = True
                else:
                    known_cost += used * lot[1]
                lot[0] -= used
                remaining -= used
                if lot[0] == ZERO:
                    lots[key].remove(lot)
        metric["matched_quantity"] = quantity - remaining
        metric["unmatched_quantity"] = remaining
        if row["price"] is not None and quantity > ZERO:
            metric["proceeds"] = row["price"] * quantity
        reason = poisoned.get(key)
        if reason is None and remaining > ZERO:
            reason = "missing_buy_history"
        if reason is None and missing_price:
            reason = "missing_buy_price"
        if reason is None:
            metric["cost_basis"] = known_cost
            if row["price"] is None:
                reason = row["price_reason"] or "missing_fill_price"
        if reason is None and quantity > ZERO and known_cost > ZERO:
            profit = metric["proceeds"] - known_cost
            metric.update(status="known", realized_profit=profit,
                          return_pct=profit / known_cost * HUNDRED,
                          reason_code=None, reason="")
        else:
            reason = reason or "invalid_quantity"
            metric.update(status="unknown", reason_code=reason, reason=_REASONS[reason])

        if remaining > ZERO:
            allocations.append({"buy_rule_id": None, "quantity": remaining, "cost_basis": None})
        for allocation in allocations:
            known = metric["status"] == "known"
            proceeds = row["price"] * allocation["quantity"] if known else None
            allocation.update(status=metric["status"],
                              proceeds=proceeds,
                              cost_basis=allocation["cost_basis"] if known else None,
                              realized_profit=proceeds - allocation["cost_basis"] if known else None,
                              reason_code=metric["reason_code"])
        metric["allocations"] = tuple(allocations)

        summary_key = (key[0], key[3])
        summary = summaries.setdefault(summary_key, {
            "market": key[0], "currency": key[3], "basis": "local_fifo", "gross": True,
            "known_sell_count": 0, "unknown_sell_count": 0,
            "known_quantity": ZERO, "unknown_quantity": ZERO,
            "known_realized_profit": None, "known_cost_basis": None,
            "known_proceeds": None, "known_return_pct": None,
            "realized_profit": None, "return_pct": None, "complete": True,
        })
        if row['allocation'] or row['close_plan']:
            summary['basis'] = 'local_fifo_or_strategy_lot'
        if metric["status"] == "known":
            summary["known_sell_count"] += 1
            summary["known_quantity"] += quantity
            for target, source in (("known_realized_profit", "realized_profit"),
                                   ("known_cost_basis", "cost_basis"), ("known_proceeds", "proceeds")):
                summary[target] = (summary[target] or ZERO) + metric[source]
        else:
            summary["unknown_sell_count"] += 1
            summary["unknown_quantity"] += quantity
            summary["complete"] = False

    for summary in summaries.values():
        if summary["known_cost_basis"] is not None and summary["known_cost_basis"] > ZERO:
            summary["known_return_pct"] = summary["known_realized_profit"] / summary["known_cost_basis"] * HUNDRED
        if summary["complete"]:
            summary["realized_profit"] = summary["known_realized_profit"]
            summary["return_pct"] = summary["known_return_pct"]
    return {
        "by_rule_id": by_rule_id, "summaries": summaries, "warnings": tuple(warnings),
        "basis": "local_fifo_or_strategy_lot" if any(row['allocation'] or row['close_plan'] for row in rows) else "local_fifo", "gross": True,
        "description": "전체 로컬 주문 기록 · 모델 매도는 지정 매수분, 일반 매도는 FIFO · 수수료·세금 제외 · 계좌 전체 수익률 아님",
        "complete": not warnings and all(summary["complete"] for summary in summaries.values()),
    }
