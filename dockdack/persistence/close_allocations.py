"""Account-wide closing allocation policy; never changes model inference.

Known original BUY lots are reduced oldest-first by cumulative SELL fills.
Untracked account residuals come after those lots. This is an explicit virtual
accounting convention, not a claim that the broker separates strategy shares.
"""
from decimal import Decimal
from datetime import datetime, timezone


def plan_close_allocations(store, db, watch_id, quantity, broker_quantity):
    inventory = store._prototype_inventory(db, watch_id)
    # A missing cost need not stop an account-wide close. Unknown ownership or
    # previously unallocated manual sales must not be silently repaired here.
    if any("모델별 가중평균 체결가 미확인" not in issue for issue in inventory["issues"]):
        return ()
    if inventory["expected_quantity"] > broker_quantity:
        return ()
    ordered = []
    for lot in inventory["lots"]:
        try:
            moment = datetime.fromisoformat(lot["buy_started_at"])
            if moment.tzinfo is None or moment.utcoffset() is None:
                return ()
            ordered.append((moment.astimezone(timezone.utc), lot["lot_id"], lot))
        except (ValueError, TypeError, OverflowError):
            return ()
    left, offset, result = Decimal(quantity), Decimal(0), []
    for _, _, lot in sorted(ordered, key=lambda row: row[:2]):
        if lot["buy_pending"] or lot["quantity_reserved_sell"]:
            return ()
        used = min(left, lot["quantity_remaining"])
        if used > 0:
            result.append((lot["lot_id"], str(used), str(offset), str(lot["filled_quantity"]),
                           str(lot["average_price"]) if lot["average_price"] is not None else None))
            offset += used
            left -= used
    return tuple(result)


def persist_close_allocations(db, rule_id, plan, now):
    for lot_id, quantity, offset, bought, average in plan:
        db.execute("INSERT INTO close_lot_allocations VALUES(?,?,?,?,?,?,?)",
                   (rule_id, lot_id, quantity, offset, now.isoformat(), bought, average))
