"""Read-only, market-separated daily summaries of the complete local ledger.

Days are exchange-local *order* days, not asserted execution dates. The store
has cumulative fill snapshots and observation times, not an incremental fill
ledger; spreading those snapshots over inferred execution days would fabricate
cash flows. In particular, a recovery HHMMSS or US broker order date alone does
not establish the timezone/date of an execution.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from dockdack.performance import realized_performance


ZERO = Decimal("0")
MARKETS = {"domestic": ("KRW", "Asia/Seoul"), "us": ("USD", "America/New_York")}
DATE_DESCRIPTION = "주문일 기준: 한국은 서울, 미국은 뉴욕 날짜 · 정확한 체결일별 집계가 아닙니다."
RETURN_DESCRIPTION = "매도 실현 수익률 = 확인된 실현손익 ÷ 해당 매수원가 · 모델 매도는 지정 매수분, 일반 매도는 FIFO · 계좌 전체 일수익률 아님"
_INVALID_CASH = {"duplicate_rule_id", "missing_rule_id", "invalid_quantity", "invalid_instrument", "invalid_side"}


def _text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def order_day(record: Mapping[str, Any]) -> tuple[date | None, str]:
    """Return a timezone-explicit local order day; never use observed_at."""
    market = _text(record.get("market"))
    if market not in MARKETS:
        return None, "시장 미확인"
    try:
        raw = record.get("started_at")
        moment = raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw))
        if moment.tzinfo is None or moment.utcoffset() is None:
            return None, "주문 시각의 시간대 미확인"
        local = moment.astimezone(ZoneInfo(MARKETS[market][1]))
        return local.date(), local.strftime("%H:%M:%S")
    except (TypeError, ValueError, OverflowError):
        return None, "주문 시각 미확인"


def empty_day(market: str, day: date) -> dict[str, Any]:
    if market not in MARKETS:
        raise ValueError("지원하지 않는 매매일지 시장입니다.")
    return {
        "market": market, "currency": MARKETS[market][0], "day": day,
        "date_basis": "market_local_order_date", "gross": True,
        "order_count": 0, "pending_count": 0, "rejected_count": 0,
        "other_count": 0, "invalid_count": 0,
        "buy_count": 0, "sell_count": 0, "buy_quantity": ZERO, "sell_quantity": ZERO,
        "known_buy_count": 0, "known_sell_amount_count": 0,
        "unknown_buy_count": 0, "unknown_sell_amount_count": 0,
        "known_buy_amount": ZERO, "known_sell_amount": ZERO,
        "buy_amount": ZERO, "sell_amount": ZERO,
        "known_profit_count": 0, "unknown_profit_count": 0,
        "known_realized_profit": None, "known_cost_basis": None,
        "known_return_pct": None, "realized_profit": None, "return_pct": None,
        "complete": True, "rows": [],
    }


def daily_trade_journal(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate known executed cash and FIFO sell returns without any I/O.

    Supply one cumulative snapshot per order and ALL historical rows. Prior
    days' buys must remain available to cost today's sales. Unknown totals are
    None; known subtotals are separately named and must be labelled partial.
    Zero is used only for an actually empty category, never an unknown price.
    Accepted/rejected/unknown attempts alone are never counted as cash flows.
    """
    ledger = tuple(dict(record) for record in records)
    performance = realized_performance(ledger)
    id_counts = Counter(_text(row.get("rule_id")) for row in ledger)
    warnings_by_id = {item["rule_id"]: set(item["reason_codes"]) for item in performance["warnings"]}
    days, rows, undated = {}, [], []
    seen = set()
    for index, record in enumerate(ledger):
        rule_id = _text(record.get("rule_id")) or f"__missing_rule_id__:{index}"
        # One duplicated cumulative snapshot is not another trade. All copies
        # remain invalid; retain one visible row rather than doubling the cash.
        if rule_id in seen:
            continue
        seen.add(rule_id)
        metric = performance["by_rule_id"].get(rule_id, {})
        market, currency = _text(record.get("market")), _text(record.get("currency"))
        day, local_time = order_day(record)
        issues = set(warnings_by_id.get(rule_id, ()))
        if id_counts[_text(record.get("rule_id"))] > 1:
            issues.add("duplicate_rule_id")
        if market not in MARKETS or currency != MARKETS.get(market, (None,))[0]:
            issues.add("invalid_instrument")
        invalid = bool(issues & _INVALID_CASH)
        quantity = metric.get("filled_quantity", ZERO)
        price = metric.get("effective_fill_price")
        amount = quantity * price if quantity > ZERO and price is not None and not invalid else None
        has_fill = quantity > ZERO
        side = _text(record.get("side"))
        row = {
            **record, "rule_id": rule_id, "market": market, "currency": currency,
            "day": day, "local_order_time": local_time,
            "date_basis": "market_local_order_date", "has_fill": has_fill,
            "confirmed_quantity": quantity, "effective_fill_price": price if not invalid else None,
            "amount": amount, "metric": metric, "invalid": invalid,
            "issues": tuple(sorted(issues)),
        }
        rows.append(row)
        if day is None or market not in MARKETS or currency != MARKETS[market][0]:
            undated.append(row)
            continue
        summary = days.setdefault((market, day), empty_day(market, day))
        summary["rows"].append(row)
        summary["order_count"] += 1
        status = _text(record.get("status"))
        if status in {"submitting", "accepted", "unknown"}:
            summary["pending_count"] += 1
        elif status == "rejected":
            summary["rejected_count"] += 1
        elif status != "filled":
            summary["other_count"] += 1
        if invalid:
            summary["invalid_count"] += 1
            summary["complete"] = False
            # A contradictory executed-quantity snapshot is not evidence of
            # zero cash. Mark that side unknown even when the validator cannot
            # produce a usable positive quantity.
            if not has_fill and status in {"filled", "accepted", "cancelled"} and side in {"buy", "sell"}:
                count_key = "buy" if side == "buy" else "sell_amount"
                summary[f"unknown_{count_key}_count"] += 1
                if side == "sell":
                    summary["unknown_profit_count"] += 1
        if not has_fill or side not in {"buy", "sell"}:
            continue
        summary[f"{side}_count"] += 1
        if not invalid:
            summary[f"{side}_quantity"] += quantity
        count_key = "buy" if side == "buy" else "sell_amount"
        if amount is None:
            summary[f"unknown_{count_key}_count"] += 1
            summary["complete"] = False
        else:
            summary[f"known_{count_key}_count"] += 1
            summary[f"known_{side}_amount"] += amount
        if side == "sell":
            if metric.get("status") == "known" and not invalid:
                summary["known_profit_count"] += 1
                for key in ("realized_profit", "cost_basis"):
                    summary[f"known_{key}"] = (summary[f"known_{key}"] or ZERO) + metric[key]
            else:
                summary["unknown_profit_count"] += 1
                summary["complete"] = False
    for summary in days.values():
        for side, count_key in (("buy", "buy"), ("sell", "sell_amount")):
            summary[f"{side}_amount"] = (None if summary[f"unknown_{count_key}_count"]
                                          else summary[f"known_{side}_amount"])
        cost = summary["known_cost_basis"]
        if cost is not None and cost > ZERO:
            summary["known_return_pct"] = summary["known_realized_profit"] / cost * Decimal("100")
        if not summary["unknown_profit_count"]:
            summary["realized_profit"] = summary["known_realized_profit"]
            summary["return_pct"] = summary["known_return_pct"]
        summary["rows"] = tuple(summary["rows"])
    return {
        "days": days, "rows": tuple(rows), "undated": tuple(undated),
        "performance": performance, "date_description": DATE_DESCRIPTION,
        "return_description": RETURN_DESCRIPTION,
        "description": "전체 로컬 주문 장부 · 주문일 기준 · 수수료·세금 제외 · 시장·통화별 분리",
    }
