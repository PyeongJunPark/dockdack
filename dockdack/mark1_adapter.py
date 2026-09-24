"""Mark1 daily-barrier research signals, using the existing DEMO-only wire.

Thirty completed OHLCV bars plus the current entry-price query form 31 tokens.
The last token never contains today's eventual high, low, close or volume.
The learned event is a conservative whole-day proxy, not intraday first-touch
probability. Producing a signal does not enable or submit an order.
"""

from datetime import date
from decimal import Decimal
from zoneinfo import ZoneInfo

from dockdack.lstm30_adapter import LSTM30SignalProducer, _market_calendar, completed_bars, number
from dockdack.market_schedule import EXTRA_CLOSURES
from dockdack.models import Market


SOURCE_ID = "mark1-daily-barrier"
TARGET_BASIS = "daily high >= entry * 1.01 AND daily low > entry * 0.991; intraday ordering unverified"
STRATEGY_NOTICE = (
    "매수: 보정 성공확률 50% 초과 (> 50%, 50%는 대기) · "
    "매도: 평균 매수가 대비 +1% 익절 / -0.9% 손절 · "
    "일봉 기반 보수적 추정 / 장중 선후관계 미검증"
)


def decide_position(*, current_price, quantity, sellable_quantity, average_price=None, prediction=None):
    """Cost exits take priority; model probability only controls a flat entry."""
    price = number(current_price, "current price")
    qty = number(quantity, "quantity", zero=True)
    sellable = number(sellable_quantity, "sellable quantity", zero=True)
    if qty != qty.to_integral_value() or sellable != sellable.to_integral_value() or sellable > qty:
        raise ValueError("Invalid position quantities")
    if qty:
        average = number(average_price, "average price")
        if price >= average * Decimal("1.01"):
            return ({"action": "sell", "reason": "TAKE_PROFIT_1PCT", "cost_profit_pct": "1"}
                    if sellable else {"action": "hold", "reason": "NO_SELLABLE_POSITION"})
        if price <= average * Decimal("0.991"):
            return ({"action": "sell", "reason": "STOP_LOSS_0_9PCT", "cost_loss_pct": "0.9"}
                    if sellable else {"action": "hold", "reason": "NO_SELLABLE_POSITION"})
        return {"action": "hold", "reason": "POSITION_INSIDE_EXIT_BOUNDS"}
    if prediction is None:
        return {"action": "hold", "reason": "PREDICTION_UNAVAILABLE"}
    probability = number(prediction.get("probability_success"), "probability", zero=True)
    threshold = number(prediction.get("buy_threshold"), "buy threshold")
    success = prediction.get("predicts_success")
    if probability > 1 or threshold != Decimal("0.5") or type(success) is not bool:
        raise ValueError("Mark1 requires a finite probability and its fixed strict > 0.5 threshold")
    if success != (probability > threshold):
        raise ValueError("Model decision conflicts with strict probability > 0.5")
    return ({"action": "buy", "reason": "PREDICTED_DAILY_BARRIER_SUCCESS"}
            if success else {"action": "hold", "reason": "BELOW_OR_EQUAL_BUY_THRESHOLD"})


def consecutive_completed_bars(stock, now):
    """Prevent inference from silently treating missing trading days as adjacent."""
    bars = completed_bars(stock, now)
    market = Market(stock["market"])
    zone = ZoneInfo("Asia/Seoul" if market is Market.DOMESTIC else "America/New_York")
    today = now.astimezone(zone).date()
    dates = [date.fromisoformat(row["date"]) for row in stock["bars"]
             if date.fromisoformat(row["date"]) < today and row.get("is_current_day") is not True][-30:]
    # The scheduler's calendar begins only on December 1 of the previous year,
    # which is too short for 30 completed sessions early in January.
    calendar = _market_calendar(market.value, today.year)
    expected = [stamp.date() for stamp in calendar.sessions
                if stamp.date() < today and (market, stamp.date()) not in EXTRA_CLOSURES][-30:]
    if len(expected) != 30 or dates != expected:
        raise ValueError("Mark1 requires 30 consecutive completed exchange sessions without gaps")
    if any(row[4] != int(row[4]) for row in bars):
        raise ValueError("Mark1 requires integral share volumes")
    return bars


class Mark1SignalProducer(LSTM30SignalProducer):
    """Reuse expiry, immutable IDs, freshness, position and sizing safeguards."""

    source_id = SOURCE_ID
    position_decision = staticmethod(decide_position)
    input_bars = staticmethod(consecutive_completed_bars)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any(not isinstance(row, dict) or not isinstance(row.get("payload"), dict)
               or row["payload"].get("source_id") != self.source_id for row in self.state.values()):
            raise ValueError("Mark1 requires its own decision state; do not reuse a mark0 runtime")

    def model_prediction(self, predictor, bars, price):
        return predictor.predict(bars, current_price=price), {
            "target_basis": TARGET_BASIS,
            "reference_price": str(price),
            "input_completed_bars": 30,
            "input_tokens": 31,
            "intraday_path_verified": False,
            "model_name": predictor.metadata.get("variant", predictor.metadata.get("model_name", predictor.metadata.get("architecture", "mark1"))),
            "strategy_notice": STRATEGY_NOTICE,
        }
