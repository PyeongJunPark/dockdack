"""Price + model-direction rules. Returns signals; does not place orders."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal


TAKE_PROFIT_RATE = Decimal("0.01")


def decimal_value(value, name: str, *, allow_zero: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not number.is_finite() or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return number


@dataclass(frozen=True)
class TradeSignal:
    action: Literal["BUY", "SELL", "HOLD"]
    reason: str
    unrealized_profit_pct: Decimal | None

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "unrealized_profit_pct": (
                str(self.unrealized_profit_pct) if self.unrealized_profit_pct is not None else None
            ),
        }


def evaluate_signal(
    *,
    current_price,
    previous_close=None,
    predicted_direction: str | None = None,
    position_quantity=0,
    average_entry_price=None,
) -> TradeSignal:
    """Apply the user's three rules in priority order.

    1. An existing long position at >=1% gross price profit -> SELL.
    2. UP and current price < previous trading close -> BUY.
    3. NOT_UP and current price > previous trading close, with holdings -> SELL.

    Equal prices / unmatched rules -> HOLD. BUY may also occur while holding;
    this is a directional signal, not a position-size or additional-order decision.
    The caller must supply a prediction for the session being evaluated.
    """
    price = decimal_value(current_price, "current_price")
    quantity = decimal_value(position_quantity, "position_quantity", allow_zero=True)
    profit = None
    if quantity > 0:
        average = decimal_value(average_entry_price, "average_entry_price")
        profit = (price / average - 1) * 100
        # Compare Decimal prices directly so exactly +1% is included.
        # Taking profit does not depend on model availability or previous close.
        if price >= average * (1 + TAKE_PROFIT_RATE):
            return TradeSignal("SELL", "TAKE_PROFIT_1PCT", profit)
    if previous_close is None or predicted_direction is None:
        return TradeSignal("HOLD", "PREDICTION_OR_REFERENCE_UNAVAILABLE", profit)
    previous = decimal_value(previous_close, "previous_close")
    if predicted_direction not in {"UP", "NOT_UP"}:
        raise ValueError("predicted_direction must be UP or NOT_UP")
    if price < previous and predicted_direction == "UP":
        return TradeSignal("BUY", "UP_BELOW_PREVIOUS_CLOSE", profit)
    if price > previous and predicted_direction == "NOT_UP":
        if quantity > 0:
            return TradeSignal("SELL", "NOT_UP_ABOVE_PREVIOUS_CLOSE", profit)
        return TradeSignal("HOLD", "NO_POSITION_TO_SELL", profit)
    return TradeSignal("HOLD", "NO_RULE_MATCH", profit)
