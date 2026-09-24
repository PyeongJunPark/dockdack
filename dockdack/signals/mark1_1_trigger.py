"""mark1.1 prototype in the ordinary, manually armed DEMO trading GUI.

Uses the frozen +0.5%/-0.4% predictor. This module does not arm orders or
change the existing mark1 prototype's saved model or strategy-owned holdings.
"""
from decimal import Decimal

from dockdack.lstm30_adapter import number
from dockdack.mark1_adapter import decide_position as flat_decision
from dockdack.mark1_trigger import Mark1DemoSignalProducer, Mark1TriggerBridge


SOURCE_ID = "mark1-1-prototype-demo-trigger"
STRATEGY_ID = "mark1-1-prototype"
TITLE = "mark1.1 prototype"
STRATEGY_NOTICE = "매수 확률 > 50% · 평균매수가 +0.5% 익절 / -0.4% 손절 · 모의 전용"
RISK_NOTICE = (
    "연구 검증 미통과 · 재사용 과거 평가 국내 2신호/미국 123신호 · 비용 후 손실 · "
    "잔여 기업행사 데이터 위험 · 장중 선후관계 미검증"
)


def decide_position(*, current_price, quantity, sellable_quantity, average_price=None, prediction=None):
    price = number(current_price, "current price")
    qty = number(quantity, "quantity", zero=True)
    sellable = number(sellable_quantity, "sellable quantity", zero=True)
    if qty != qty.to_integral_value() or sellable != sellable.to_integral_value() or sellable > qty:
        raise ValueError("Invalid position quantities")
    if qty:
        average = number(average_price, "average price")
        if price >= average * Decimal("1.005"):
            return ({"action": "sell", "reason": "TAKE_PROFIT_0_5PCT", "cost_profit_pct": "0.5"}
                    if sellable else {"action": "hold", "reason": "NO_SELLABLE_POSITION"})
        if price <= average * Decimal("0.996"):
            return ({"action": "sell", "reason": "STOP_LOSS_0_4PCT", "cost_loss_pct": "0.4"}
                    if sellable else {"action": "hold", "reason": "NO_SELLABLE_POSITION"})
        return {"action": "hold", "reason": "POSITION_INSIDE_EXIT_BOUNDS"}
    return flat_decision(current_price=price, quantity=0, sellable_quantity=0, prediction=prediction)


class Mark11DemoSignalProducer(Mark1DemoSignalProducer):
    source_id = SOURCE_ID
    strategy_id = STRATEGY_ID
    title = TITLE
    strategy_notice = STRATEGY_NOTICE
    risk_notice = RISK_NOTICE
    take_profit_pct = .5
    stop_loss_pct = .4
    take_multiplier = Decimal("1.005")
    stop_multiplier = Decimal("0.996")
    strict_model_identity = True
    position_decision = staticmethod(decide_position)


class Mark11TriggerBridge(Mark1TriggerBridge):
    source_id = SOURCE_ID
    strategy_id = STRATEGY_ID
    title = TITLE
    strategy_notice = STRATEGY_NOTICE
    risk_notice = RISK_NOTICE
    producer_type = Mark11DemoSignalProducer
    bundle_directory = "models/mark1_1_prototype"
    state_filename = "exchange/mark1-1-trigger-decisions-v1.json"

    def _new_predictor(self, market):
        from dockdack.mark1_1_prototype_inference import Mark11PrototypePredictor
        return Mark11PrototypePredictor(self.bundle_root, market)
