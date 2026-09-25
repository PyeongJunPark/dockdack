"""Model-attributed, gross realized performance from one complete account ledger.

This is an idempotent projection of durable cumulative execution snapshots, not
a balance tracker, backtest, or write-a-second-profit-counter scheme. Callers
must pass the complete history from exactly one environment/account scope.
There is no brokerage, database, filesystem, or model inference I/O here.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from dockdack.models import TradingMode
from dockdack.trading.performance import realized_performance
from dockdack.trading.strategy_lots import row_family


ZERO = Decimal("0")
MODELS = (("mark1-prototype", "mark1 prototype"),
          ("mark1-1-prototype", "mark1.1 prototype"),
          ("mark1-2-prototype", "mark1.2 prototype"))
MARKETS = (("domestic", "KRW"), ("us", "USD"))
UNASSIGNED = "unassigned"
DESCRIPTION = (
    "현재 계정 장부의 확인된 체결 기반 누적 실현 성과 · 원매수 모델별 배분 · "
    "매도된 매수원금 대비 손익(수익률 단순 평균 아님) · 수수료·세금 제외 · "
    "미매도 보유분·다른 계정·장부 밖 거래 제외 · 계좌 전체 수익률/백테스트 아님"
)


def _empty(strategy_id: str, title: str, market: str, currency: str) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id, "model_title": title, "market": market, "currency": currency,
        "known_sell_count": 0, "unknown_sell_count": 0, "executed_sell_count": 0,
        "known_quantity": ZERO, "unknown_quantity": ZERO,
        "known_cost_basis": None, "known_proceeds": None, "known_realized_profit": None,
        "known_return_pct": None, "realized_profit": None, "return_pct": None,
        "complete": True, "attribution_complete": strategy_id != UNASSIGNED,
        "status": "no_sales", "gross": True,
    }


def _moment(value: Any) -> datetime | None:
    try:
        stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        return stamp.astimezone(timezone.utc) if stamp.tzinfo is not None and stamp.utcoffset() is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def model_realized_performance(records: Iterable[Mapping[str, Any]], *, mode: TradingMode | str,
                               performance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Project current-account execution returns, separated by model/market/currency.

    Decimal quantities and amounts are never FX-converted or averaged across
    trades. ``return_pct`` equals total realized gross profit / sold acquisition
    cost * 100, and is None for no sales or an incomplete model bucket. The
    ``known_*`` fields expose the fully evidenced subset without presenting it
    as the whole. Open positions have no realized return, including 0%.

    Identity comes only from the matched BUY's validated persisted signal, not
    the sell's label, current model selection, or a same-symbol guess. Manual
    FIFO and end-of-day allocation use the shared execution accounting engine.
    Missing/malformed provenance appears in an explicit unassigned group; it
    is never silently dropped or credited to a selected model. Prototype BUY
    signals are currently DEMO-only, so those may not become REAL performance.

    A collector may pass ``performance=realized_performance(records)`` computed
    from the EXACT SAME complete snapshot to avoid a second FIFO replay. This
    is an internal already-validated projection, never external user data or a
    cached projection from another revision/account. Omitting it calculates it.
    """
    selected_mode = TradingMode(mode)
    records = tuple(records)
    performance = realized_performance(records) if performance is None else performance
    warnings = list(performance["warnings"])
    ids = Counter(str(row.get("rule_id") or "").strip() for row in records)
    origins = {}
    for record in records:
        if str(getattr(record.get("side"), "value", record.get("side"))) != "buy":
            continue
        rule_id = str(record.get("rule_id") or "").strip()
        if not rule_id or ids[rule_id] != 1:
            continue
        try:
            family = row_family(record)
        except (ValueError, KeyError, TypeError):
            family = None
            warnings.append({"rule_id": rule_id, "reason_codes": ("invalid_model_origin",)})
        if family is not None and selected_mode is TradingMode.REAL:
            family = None
            warnings.append({"rule_id": rule_id, "reason_codes": ("model_environment_mismatch",)})
        origins[rule_id] = family

    buckets = {(model, market, currency): _empty(model, title, market, currency)
               for model, title in MODELS for market, currency in MARKETS}
    seen_known, seen_unknown = {}, {}
    executed_sales, unassigned_sales, incomplete_sales = set(), set(), set()
    for sell_id, metric in performance["by_rule_id"].items():
        if metric["side"] != "sell" or metric["status"] == "not_applicable":
            continue
        executed_sales.add(sell_id)
        if metric["status"] != "known":
            incomplete_sales.add(sell_id)
        allocations = metric["allocations"] or ({
            "buy_rule_id": None, "quantity": metric["filled_quantity"],
            "status": "unknown", "cost_basis": None, "proceeds": None, "realized_profit": None,
        },)
        for allocation in allocations:
            family = origins.get(allocation["buy_rule_id"])
            strategy_id, title = ((family.id, family.title) if family else (UNASSIGNED, "미확인 / 수동·외부"))
            key = (strategy_id, metric["market"], metric["currency"])
            row = buckets.setdefault(key, _empty(strategy_id, title, metric["market"], metric["currency"]))
            if family is None:
                unassigned_sales.add(sell_id)
            known = allocation["status"] == "known"
            selected = seen_known if known else seen_unknown
            selected.setdefault(key, set()).add(sell_id)
            row["known_quantity" if known else "unknown_quantity"] += allocation["quantity"]
            if known:
                for field in ("cost_basis", "proceeds", "realized_profit"):
                    row["known_" + field] = (row["known_" + field] or ZERO) + allocation[field]

    for key, row in buckets.items():
        row["known_sell_count"] = len(seen_known.get(key, ()))
        row["unknown_sell_count"] = len(seen_unknown.get(key, ()))
        row["executed_sell_count"] = len(seen_known.get(key, set()) | seen_unknown.get(key, set()))
        row["complete"] = not row["unknown_sell_count"]
        cost = row["known_cost_basis"]
        if cost is not None and cost > ZERO:
            row["known_return_pct"] = row["known_realized_profit"] / cost * Decimal("100")
        if not row["complete"]:
            row["status"] = "incomplete"
        elif row["known_sell_count"]:
            row["status"] = "known"
            row["realized_profit"] = row["known_realized_profit"]
            row["return_pct"] = row["known_return_pct"]

    if unassigned_sales:
        warnings.append({"rule_id": None, "reason_codes": ("unassigned_model_origin",),
                         "sell_count": len(unassigned_sales)})
    moments = sorted(stamp for record in records if (stamp := _moment(record.get("started_at"))) is not None)
    return {
        "mode": selected_mode.value, "rows": tuple(buckets.values()), "gross": True,
        "complete": performance["complete"] and not warnings and not unassigned_sales,
        "warnings": tuple(warnings), "description": DESCRIPTION,
        "coverage": {
            "ledger_order_count": len(records), "executed_sell_count": len(executed_sales),
            "unassigned_sell_count": len(unassigned_sales), "incomplete_sell_count": len(incomplete_sales),
            "first_order_at": moments[0].isoformat() if moments else None,
            "last_order_at": moments[-1].isoformat() if moments else None,
        },
    }
