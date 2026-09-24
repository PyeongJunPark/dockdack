"""Fill-based virtual strategy inventory over an account-bound durable ledger.

These are accounting lots, not separate broker positions. Acknowledgements never
create shares, cumulative execution snapshots are replayed rather than appended,
and an unallocated sell or aggregate account discrepancy requires review.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation


ZERO = Decimal(0)
PENDING = frozenset({"submitting", "accepted", "unknown"})
RESERVED = PENDING | {"ready"}


def _aware_time(value):
    """Ledger chronology is an absolute instant, never raw ISO text ordering."""
    try:
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            return None
        return stamp.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def number(value, *, default=None):
    if value is None:
        return default
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("전략별 체결 수량/가격 형식이 올바르지 않습니다.") from None
    if not result.is_finite() or result < 0:
        raise ValueError("전략별 체결 수량/가격은 유한한 음이 아닌 값이어야 합니다.")
    return result


def row_family(row):
    """Resolve the persisted BUY, not a GUI selection or supplied friendly title."""
    from dockdack.signal_bridge import prototype_record_family
    if not row.get("external_payload"):
        return None
    return prototype_record_family({
        "source_id": row.get("external_source_id"),
        "signal_id": row.get("external_signal_id"),
        "payload": row.get("external_payload"),
        "decision": row.get("external_decision"),
        "watch_id": row.get("external_watch_id"),
    }, watch_id=row["watch_id"], action="buy")


def verified_average(row, filled):
    price = number(row.get("fill_price"))
    if not price or not filled:
        return None
    # Existing broker snapshots can report a last fill, not a weighted average.
    if number(row["quantity"]) == filled == 1:
        return price
    if (row.get("recovery_price_basis") in {"broker_average", "weighted_fills"}
            and number(row.get("price_basis_quantity")) == filled
            and number(row.get("price_basis_price")) == price):
        return price
    return None


def project_prototype_inventory(rows, allocations, *, mode="demo", scope="demo"):
    """Pure idempotent projection. Return lots and issues; never guess missing fills."""
    rows = tuple(rows)
    timestamps = {row["rule_id"]: _aware_time(row.get("started_at")) for row in rows}
    rows = tuple(sorted(rows, key=lambda row: timestamps[row["rule_id"]] or datetime.max.replace(tzinfo=timezone.utc)))
    allocations = tuple(allocations)
    by_rule = {row["rule_id"]: row for row in rows}
    lots, issues = {}, []
    for row in rows:
        if row["side"] != "buy":
            continue
        try:
            family = row_family(row)
        except ValueError:
            issues.append(f"{row['rule_id']}: 매수 모델 출처 불일치")
            continue
        if family is None:
            continue
        filled = number(row.get("filled_quantity"), default=ZERO)
        ordered = number(row["quantity"])
        problems = []
        if timestamps[row["rule_id"]] is None and (filled > ZERO or row["status"] in PENDING | {"filled", "reviewed"}):
            problems.append("매수 체결 기록의 시각/시간대 미확인")
        if filled > ordered or filled != filled.to_integral_value():
            problems.append("매수 체결수량 불일치")
        if row["status"] == "filled" and row.get("filled_quantity") is None:
            problems.append("체결 완료의 수량 근거 없음")
        elif row["status"] == "filled" and filled != ordered:
            problems.append("전체 체결 상태와 실제 누적 수량 불일치")
        if row["status"] == "reviewed" and row.get("filled_quantity") is None:
            problems.append("수동 확인된 주문의 체결 수량 근거 없음")
        average = verified_average(row, filled)
        lots[row["rule_id"]] = {
            "lot_id": row["rule_id"], "buy_rule_id": row["rule_id"],
            "buy_order_number": row.get("order_number", ""),
            "watch_id": row["watch_id"], "mode": mode, "storage_scope": scope,
            "source_id": row["external_source_id"], "strategy_id": family.id,
            "model_title": family.title, "buy_signal_id": row["external_signal_id"],
            "buy_status": row["status"], "buy_pending": row["status"] in PENDING,
            "filled_quantity": filled, "quantity_sold": ZERO,
            "quantity_remaining": filled, "quantity_reserved_sell": ZERO,
            "available_quantity": filled, "average_price": average,
            "take_profit_price": average * (1 + family.take_profit) if average else None,
            "stop_loss_price": average * (1 - family.stop_loss) if average else None,
            "buy_started_at": row["started_at"], "issues": problems,
        }
    mapped = set()
    bulk_ranges = {}
    allocation_kinds = {}
    for allocation in allocations:
        rule_id = allocation["rule_id"]
        mapped.add(rule_id)
        lot = lots.get(allocation["lot_id"])
        if lot is None:
            issues.append(f"{rule_id}: 원매수 체결 로트 없음")
            continue
        row = by_rule.get(rule_id)
        quantity = number(allocation["quantity"])
        captured_quantity = number(allocation.get("buy_filled_quantity"))
        captured_average = number(allocation.get("buy_average_price"))
        if (captured_quantity is not None and captured_quantity != lot["filled_quantity"]
                or captured_average is not None and captured_average != lot["average_price"]):
            lot["issues"].append("매도 배분 이후 원매수 체결 근거 변경 · 확인 필요")
        if (allocation.get("watch_id", lot["watch_id"]) != lot["watch_id"]
                or allocation.get("side", "sell") != "sell"):
            lot["issues"].append("매도 배분 종목/방향 불일치")
            continue
        if quantity <= 0 or quantity != quantity.to_integral_value():
            lot["issues"].append("매도 배분수량 불일치")
            continue
        bulk = allocation.get("bulk", False)
        if rule_id in allocation_kinds and allocation_kinds[rule_id] != bool(bulk):
            issues.append(f"{rule_id}: 일반 매도와 마감 매도 배분 혼합")
        allocation_kinds[rule_id] = bool(bulk)
        ordered = number(allocation.get("order_quantity")) if bulk else quantity
        offset = number(allocation.get("fill_offset")) if bulk else ZERO
        if bulk:
            if (ordered is None or offset is None or ordered <= 0
                    or ordered != ordered.to_integral_value() or offset != offset.to_integral_value()
                    or offset + quantity > ordered):
                lot["issues"].append("마감 매도 전체수량/체결 배분구간 불일치")
                continue
            previous = bulk_ranges.setdefault(rule_id, [])
            if any(total != ordered or other_lot == lot["lot_id"]
                   or max(start, offset) < min(end, offset + quantity)
                   for start, end, total, other_lot in previous):
                issues.append(f"{rule_id}: 마감 매도 체결 배분구간 중복/불일치")
            previous.append((offset, offset + quantity, ordered, lot["lot_id"]))
        filled = number(row.get("filled_quantity"), default=ZERO) if row else ZERO
        status = row["status"] if row else allocation.get("rule_status", "ready")
        if row and (filled > ZERO or status in RESERVED | {"filled", "reviewed"}):
            sell_time, buy_time = timestamps[rule_id], timestamps[lot["buy_rule_id"]]
            if sell_time is None:
                lot["issues"].append("매도 체결 기록의 시각/시간대 미확인")
            elif buy_time is not None and sell_time < buy_time:
                lot["issues"].append("원매수보다 이른 매도 체결 기록 · 시각 확인 필요")
        if row and (row["side"] != "sell" or row["watch_id"] != lot["watch_id"]
                    or number(row["quantity"]) != ordered):
            lot["issues"].append("매도 주문과 로트 배분 불일치")
        if filled > ordered or filled != filled.to_integral_value():
            lot["issues"].append("매도 체결수량이 배분 초과")
        if row and status == "filled" and row.get("filled_quantity") is None:
            lot["issues"].append("매도 완료의 체결수량 근거 없음")
        elif row and status == "filled" and filled != ordered:
            lot["issues"].append("매도 전체 체결 상태와 실제 누적 수량 불일치")
        if row and status == "reviewed" and row.get("filled_quantity") is None:
            lot["issues"].append("수동 확인된 매도의 체결 수량 근거 없음")
        if bulk:
            # One cumulative closing order may cover several model lots and
            # manual residual shares. Attribute only this captured FIFO slice.
            filled = min(quantity, max(ZERO, filled - offset))
        lot["quantity_sold"] += filled
        if status in RESERVED:
            lot["quantity_reserved_sell"] += max(ZERO, quantity - filled)
    for row in rows:
        if row["side"] != "sell" or row["rule_id"] in mapped:
            continue
        sell_time = timestamps[row["rule_id"]]
        related = [lot for lot in lots.values() if lot["watch_id"] == row["watch_id"]
                   and (sell_time is None or timestamps[lot["buy_rule_id"]] is None
                        or timestamps[lot["buy_rule_id"]] <= sell_time)]
        if related and (number(row.get("filled_quantity"), default=ZERO) > 0
                        or row["status"] in PENDING | {"filled", "reviewed"}):
            issues.append(f"{row['rule_id']}: 매수 모델에 배분되지 않은 매도/확인 필요")
    for lot in lots.values():
        remaining = lot["filled_quantity"] - lot["quantity_sold"]
        available = remaining - lot["quantity_reserved_sell"]
        if remaining < 0 or available < 0:
            lot["issues"].append("모델별 보유량 초과 매도/예약")
        if remaining > 0 and lot["average_price"] is None:
            lot["issues"].append("모델별 가중평균 체결가 미확인")
        lot["quantity_remaining"] = max(ZERO, remaining)
        # Without individual fill events, a later BUY fill would change the
        # residual cost after a sale. Wait for this BUY's definitive completion.
        lot["available_quantity"] = ZERO if lot["buy_pending"] else max(ZERO, available)
        lot["issues"] = tuple(dict.fromkeys(lot["issues"]))
        issues.extend(f"{lot['lot_id']}: {issue}" for issue in lot["issues"])
    return {"lots": tuple(lots.values()), "issues": tuple(dict.fromkeys(issues)),
            "has_prototype_history": bool(lots or issues),
            "expected_quantity": sum((lot["quantity_remaining"] for lot in lots.values()), ZERO)}


def reconcile_inventory(inventory, *, broker_quantity=None, broker_sellable=None):
    """Exact aggregate ownership required; manual residuals are never assigned."""
    issues = list(inventory["issues"])
    quantity = number(broker_quantity)
    sellable = number(broker_sellable)
    if quantity is not None and quantity != inventory["expected_quantity"]:
        issues.append("브로커 전체 보유량과 모델별 체결 보유량 불일치 · 수동/외부 거래 확인 필요")
    if sellable is not None and (quantity is None or sellable > quantity):
        issues.append("브로커 매도가능수량 근거 불일치")
    return {**inventory, "broker_quantity": quantity, "broker_sellable": sellable,
            "issues": tuple(dict.fromkeys(issues)), "reconciled": not issues}
