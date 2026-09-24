"""Pure v0.0 allocation and holding-exit calculations (no broker calls)."""

from decimal import Decimal, ROUND_FLOOR

from dockdack.models import Market, TradingMode


def account_equity_cash(account):
    """Choose a cash basis without hiding a negative broker cash balance.

    A negative current KR cash value can coexist with positive post-settlement
    funds. Only the broker's explicitly identified kt00001 D+2 estimate is used
    in that case; never substitute buying power, an untrusted raw value, or USD
    figures. This does not change cash on the snapshot or authorize an order.
    """
    cash = account.cash
    if not isinstance(cash, Decimal) or not cash.is_finite():
        raise ValueError("시장별 예수금을 확인해야 비중 매수할 수 있습니다.")
    if cash >= 0:
        return cash
    settled = getattr(account, "cash_d2", None)
    if (account.market is Market.DOMESTIC and account.currency == "KRW"
            and getattr(account, "cash_settlement_source", "") == "kt00001:d2_entra"
            and isinstance(settled, Decimal) and settled.is_finite() and settled >= 0):
        return settled
    raise ValueError("현재 예수금이 음수이고 검증된 국내 D+2 추정예수금이 없어 비중 매수를 보류합니다.")


def account_equity(account):
    """Same-market cash basis plus holdings, before hypothetical sale fees/tax."""
    values = (account_equity_cash(account), account.total_evaluation)
    if any(not isinstance(value, Decimal) or not value.is_finite() or value < 0 for value in values):
        raise ValueError("시장별 예수금과 보유 평가금액을 모두 확인해야 비중 매수할 수 있습니다.")
    total = sum(values, Decimal(0))
    if total <= 0:
        raise ValueError("시장별 계좌 평가금액(예수금 + 보유 평가금액)이 0입니다.")
    return total


def allocation_quantity(account, price, percent, cap, quantity_cap=999_999_999):
    if not isinstance(percent, Decimal) or not percent.is_finite() or not 0 < percent <= 100:
        raise ValueError("1회 매수 비중은 0 초과 100 이하 %여야 합니다.")
    if not isinstance(price, Decimal) or not price.is_finite() or price <= 0:
        raise ValueError("현재 주문 단가를 확인할 수 없습니다.")
    available = account.available_to_order
    if not isinstance(available, Decimal) or not available.is_finite() or available < 0:
        raise ValueError("주문가능금액을 확인할 수 없습니다.")
    budget = min(account_equity(account) * percent / 100, cap, available / Decimal("1.01"))
    if account.cash < 0:
        # A positive buying-power figure may include margin. When relying on
        # post-settlement cash, do not spend beyond that verified cash estimate.
        budget = min(budget, account_equity_cash(account) / Decimal("1.01"))
    quantity = min(quantity_cap, int((budget / price).to_integral_value(rounding=ROUND_FLOOR)))
    if quantity < 1:
        raise ValueError("설정한 평가금액 비중·주문 상한·주문가능금액으로 1주를 살 수 없습니다.")
    return quantity


def holding_exit_targets(store, position, *, prototype_lots=False):
    """UI/engine shared targets; approved fallback applies only without saved targets."""
    from dockdack.gui_service import Instrument
    from dockdack.watchlist import instrument_key
    key = instrument_key(Instrument(position.market, position.symbol, position.exchange))
    saved = store.exit_targets(key)
    if prototype_lots:
        from dockdack.signal_bridge import mark1_prototype_origin
        import json
        original = store.external_for_rule(saved.get("rule_id", "")) if saved else None
        original_payload = json.loads(original["payload"]) if original else {}
        marked_prototype = bool(saved and (mark1_prototype_origin(saved.get("source"))
                                  or (original and mark1_prototype_origin(original["source_id"], original_payload))))
        inventory = store.prototype_inventory(key, broker_quantity=position.quantity,
                                               broker_sellable=position.sellable_quantity)
        if marked_prototype and not inventory.get("has_prototype_history"):
            inventory = {**inventory, "reconciled": False,
                         "issues": (*inventory["issues"], "모델 소유 표식만 있고 확정 체결 장부가 없습니다.")}
        if marked_prototype or inventory.get("has_prototype_history") or inventory["lots"]:
            lots = []
            ledger_available = sum((lot['available_quantity'] for lot in inventory['lots']), Decimal(0))
            for lot in inventory["lots"]:
                if lot["quantity_remaining"] <= 0:
                    continue
                lots.append({**lot, "quantity": lot["quantity_remaining"],
                             "sellable_quantity": min(lot["available_quantity"], position.sellable_quantity) if inventory["reconciled"] else Decimal(0),
                             "broker_sellable_quantity": position.sellable_quantity,
                             "sellable_is_shared": position.sellable_quantity < ledger_available,
                             "source": lot["model_title"], "buy_rule_id": lot["lot_id"],
                             "take_profit_price": lot["take_profit_price"] if inventory["reconciled"] else None,
                             "stop_loss_price": lot["stop_loss_price"] if inventory["reconciled"] else None})
            return {"watch_id": key, "lots": tuple(lots), "reconciled": inventory["reconciled"],
                    "issues": inventory["issues"], "take_profit_price": None, "stop_loss_price": None,
                    "source": "모델별 분리 보유" if inventory["reconciled"] else "모델별 체결 대조 확인 필요"}
    from dockdack.signal_bridge import mark1_prototype_origin, prototype_family, prototype_record_family
    import json
    record = store.external_for_rule(saved.get("rule_id", "")) if saved is not None else None
    try:
        metadata = json.loads(record["payload"]) if record else {}
    except (TypeError, ValueError) as exc:
        raise ValueError("원본 매수 신호 기록을 읽을 수 없습니다.") from exc
    mark1 = saved is not None and (mark1_prototype_origin(saved.get("source"))
                                  or (record and mark1_prototype_origin(record["source_id"], metadata)))
    if mark1:
        # A saved bracket identifies a strategy-owned purchase. Recalculate
        # against the broker's actual cost, not the earlier candidate quote.
        # Do not change legacy or unrelated holdings' absolute/default targets.
        if TradingMode(store.mode) is not TradingMode.DEMO:
            raise ValueError("mark1 prototype 보유분 청산은 모의 환경만 허용합니다.")
        family = prototype_record_family(record, watch_id=key, action="buy")
        saved_family = prototype_family(saved.get("source"))
        if family is None or (saved_family is not None and saved_family != family):
            raise ValueError("mark1 prototype 보유분의 원본 매수 신호 기록을 확인할 수 없습니다.")
        provenance = {"model_id": family.id, "model_title": family.title,
                      "buy_source_id": record["source_id"], "buy_signal_id": record["signal_id"],
                      "buy_rule_id": saved["rule_id"]}
        average = position.average_price
        if not isinstance(average, Decimal) or not average.is_finite() or average <= 0:
            return {**saved, **provenance, "take_profit_price": None, "stop_loss_price": None,
                    "source": family.title + " · 실제 평균매입가 확인 필요"}
        profit_pct = format((family.take_profit * 100).normalize(), "f")
        loss_pct = format((family.stop_loss * 100).normalize(), "f")
        return {**saved, **provenance, "take_profit_price": average * (1 + family.take_profit),
                "stop_loss_price": average * (1 - family.stop_loss),
                "source": f"{family.title} · 실제 평균매입가 +{profit_pct}% / -{loss_pct}%"}
    if saved is not None:
        return saved
    average = position.average_price
    if not isinstance(average, Decimal) or not average.is_finite() or average <= 0:
        return {"take_profit_price": None, "stop_loss_price": None, "source": "매입가 확인 필요", "watch_id": key}
    return {"take_profit_price": average * Decimal("1.01"), "stop_loss_price": average * Decimal("0.992"),
            "source": "평균매입가 +1% / -0.8% (기본)", "watch_id": key}
