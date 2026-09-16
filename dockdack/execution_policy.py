"""Pure v0.0 allocation and holding-exit calculations (no broker calls)."""

from decimal import Decimal, ROUND_FLOOR

from dockdack.models import Market


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


def holding_exit_targets(store, position):
    """UI/engine shared targets; approved fallback applies only without saved targets."""
    from dockdack.gui_service import Instrument
    from dockdack.watchlist import instrument_key
    key = instrument_key(Instrument(position.market, position.symbol, position.exchange))
    saved = store.exit_targets(key)
    if saved is not None:
        return saved
    average = position.average_price
    if not isinstance(average, Decimal) or not average.is_finite() or average <= 0:
        return {"take_profit_price": None, "stop_loss_price": None, "source": "매입가 확인 필요", "watch_id": key}
    return {"take_profit_price": average * Decimal("1.01"), "stop_loss_price": average * Decimal("0.992"),
            "source": "평균매입가 +1% / -0.8% (기본)", "watch_id": key}
