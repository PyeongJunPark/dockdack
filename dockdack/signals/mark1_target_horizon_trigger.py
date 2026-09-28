"""Isolated DEMO signals for sealed daily target/horizon research models.

The worker computes research-only DEMO candidates. A candidate can become an
order only through the GUI's explicit permission, strategy-lot ownership, and
a matching timed-exit policy; no model directly sends a broker order.
"""
from __future__ import annotations

from functools import lru_cache
import math
import re

from dockdack.history import market_time
from dockdack.lstm30_adapter import number
from dockdack.mark1_adapter import Mark1SignalProducer, consecutive_completed_bars
from dockdack.models import OrderSide
from dockdack.signals.mark1_trigger import Mark1DemoSignalProducer, Mark1TriggerBridge


# Lightweight, independent of NumPy/PyTorch so discovery does not load models.
MODEL_SPECS = {
    "mark1-23-prototype": (20, 10, 3.),
    "mark1-24-prototype": (30, 20, 3.),
    "mark1-25-prototype": (20, 20, 4.),
    "mark1-26-prototype": (20, 20, 2.),
    "mark1-27-prototype": (20, 10, 3.),
    "mark1-28-prototype": (10, 10, 4.),
}
MODEL_IDS = tuple(MODEL_SPECS)
TARGET = "daily_first_high_target_touch_else_Hth_close_no_stop_proxy"
SCORE_SCOPE = "next_open_proxy_not_intraday_verified"
RISK_NOTICE = (
    "완료 일봉과 현재가 질의 · 다음 거래일 관측 시가 기준 학습 · 장중 조회 후 고가 선후/"
    "지정가 체결/수익성 미검증 · 손절 없음 · 모의 연구 전용"
)
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def target_horizon_decision(*, current_price, quantity, sellable_quantity,
                            average_price=None, prediction=None):
    """Only a flat, validated model candidate can issue BUY; exits are lot-owned."""
    number(current_price, "current price")
    held = number(quantity, "quantity", zero=True)
    sellable = number(sellable_quantity, "sellable quantity", zero=True)
    if (held != held.to_integral_value() or sellable != sellable.to_integral_value()
            or sellable > held):
        raise ValueError("Invalid target/horizon position quantities")
    if held:
        number(average_price, "average price")
        return {"action": "hold", "reason": "POSITION_EXIT_MANAGED_BY_GUI"}
    if prediction is None:
        return {"action": "hold", "reason": "PREDICTION_UNAVAILABLE"}
    return ({"action": "buy", "reason": "TARGET_HORIZON_CANDIDATE"}
            if prediction["predicts_success"] else
            {"action": "hold", "reason": "TARGET_HORIZON_POLICY_NOT_MET"})


def require_exit_schedule(model_id: str, horizon: int):
    """A candidate cannot become a BUY until its lot has a matching timed exit."""
    from dockdack.trading.model_exit_schedule import ModelExitSchedule, model_exit_schedule

    schedule = model_exit_schedule(model_id)
    if (not isinstance(schedule, ModelExitSchedule)
            or schedule.sessions_after_fill != horizon - 1
            or schedule.timing != "preclose"):
        raise ValueError("Target/horizon DEMO BUY lacks its matching model exit schedule")
    return schedule


@lru_cache(maxsize=None)
def target_horizon_bridge_type(model_id: str):
    if model_id not in MODEL_SPECS:
        raise ValueError("Unknown Mark1 target/horizon model")
    lookback, horizon, target_pct = MODEL_SPECS[model_id]
    title = model_id.replace("mark1-", "mark1.").replace("-prototype", " prototype")
    source_id = model_id + "-demo-trigger"
    notice = (f"완료 {lookback}일봉 + 현재가 · {horizon}거래 세션 안에 +{target_pct:g}%"
              " 대리점수와 예상 순수익이 모두 기준 충족할 때만 모의 매수 후보 · 손절 없음")

    class Producer(Mark1DemoSignalProducer):
        strategy_id = model_id
        strategy_notice = notice
        risk_notice = RISK_NOTICE
        position_decision = staticmethod(target_horizon_decision)

        @classmethod
        def validate_prediction(cls, prediction):
            """Reject changed research semantics before any BUY/HOLD is emitted."""
            if not isinstance(prediction, dict):
                raise ValueError("Target/horizon prediction is missing")
            numeric = ("probability_success", "expected_net_return",
                       "candidate_entry_price", "candidate_take_price")
            if any(type(prediction.get(key)) not in (int, float)
                   or not math.isfinite(prediction[key]) for key in numeric):
                raise ValueError("Target/horizon prediction contains a nonfinite score or price")
            probability = prediction["probability_success"]
            net = prediction["expected_net_return"]
            entry = prediction["candidate_entry_price"]
            take = prediction["candidate_take_price"]
            candidate = probability >= .5 and net > 0
            if (prediction.get("strategy_id") != model_id
                    or prediction.get("title") != title
                    or prediction.get("target") != TARGET
                    or prediction.get("score_scope") != SCORE_SCOPE
                    or prediction.get("horizon_sessions") != horizon
                    or prediction.get("take_profit_pct") != target_pct
                    or prediction.get("stop_loss_pct", False) is not None
                    or prediction.get("candidate_stop_price", False) is not None
                    or prediction.get("probability_stop", False) is not None
                    or prediction.get("buy_threshold") != .5
                    or prediction.get("policy_threshold") != .5
                    or prediction.get("research_only") is not True
                    or prediction.get("research_qualified") is not False
                    or prediction.get("deployment_allowed") is not False
                    or prediction.get("intraday_path_verified") is not False
                    or not isinstance(prediction.get("bundle_manifest_sha256"), str)
                    or not _HASH.fullmatch(prediction["bundle_manifest_sha256"])
                    or not 0 <= probability <= 1 or entry <= 0 or take <= entry
                    or not math.isclose(take, entry * (1 + target_pct / 100), rel_tol=1e-10)
                    or type(prediction.get("predicts_success")) is not bool
                    or prediction["predicts_success"] != candidate
                    or prediction.get("selected_research") is not candidate):
                raise ValueError("Target/horizon model identity, policy, or risk limits differ")

        @classmethod
        def checked_model_query(cls, predictor, bars, price):
            metadata = getattr(predictor, "metadata", {})
            if (metadata.get("strategy_id") != cls.strategy_id
                    or metadata.get("lookback") != lookback
                    or metadata.get("horizon_sessions") != horizon
                    or metadata.get("take_profit_pct") != target_pct
                    or metadata.get("research_only") is not True
                    or metadata.get("deployment_allowed") is not False
                    or metadata.get("intraday_path_verified") is not False):
                raise ValueError("Target/horizon checkpoint metadata differs")
            prediction = predictor.predict(bars, current_price=price)
            cls.validate_prediction(prediction)
            if (prediction.get("market") != metadata.get("market")
                    or prediction["candidate_entry_price"] != float(price)
                    or prediction["bundle_manifest_sha256"] != metadata.get("bundle_manifest_sha256")):
                raise ValueError("Target/horizon prediction query or checkpoint differs")
            return prediction

        def model_prediction(self, predictor, bars, price):
            prediction = self.checked_model_query(predictor, bars, price)
            return prediction, {
                "target_basis": "next-session observed OPEN; first daily HIGH target touch else Hth CLOSE",
                "reference_price": str(price), "input_completed_bars": 30,
                "effective_lookback": lookback, "horizon_sessions": horizon,
                "intraday_path_verified": False,
                "strategy_notice": self.strategy_notice, "risk_notice": self.risk_notice,
                "execution_scope": "demo_manual_gui_permission_only",
            }

        def _decision(self, stock, charts, now):
            # Skip Mark1DemoSignalProducer's mandatory +1%/-0.9% BUY bracket.
            decision, detail = Mark1SignalProducer._decision(self, stock, charts, now)
            if decision["action"] == "buy":
                try:
                    require_exit_schedule(self.strategy_id, horizon)
                except ValueError as exc:
                    decision = {"action": "hold", "reason": "EXIT_SCHEDULE_UNAVAILABLE"}
                    detail = {**detail, "error": str(exc)}
                else:
                    prediction = detail["prediction"]
                    decision = {**decision, "strategy_id": self.strategy_id,
                                "model_title": self.title,
                                "model_manifest_sha256": prediction["bundle_manifest_sha256"]}
            return decision, {
                **detail, "title": self.title, "strategy_id": self.strategy_id,
                "strategy_notice": self.strategy_notice, "risk_notice": self.risk_notice,
                "research_only": True, "research_qualified": False,
                "deployment_allowed": False, "intraday_path_verified": False,
            }

    class Bridge(Mark1TriggerBridge):
        producer_type = Producer
        strategy_id = model_id
        strategy_notice = notice
        risk_notice = RISK_NOTICE
        bundle_directory = "models/mark1_target_horizon_v1"
        state_filename = "exchange/" + model_id + "-decisions-v1.json"

        def _new_predictor(self, market):
            from dockdack.mark1_target_horizon_inference import MarkTargetHorizonPredictor

            return MarkTargetHorizonPredictor(self.bundle_root, market, model_id)

        def validate_execution(self, item, rule, fresh_snapshot, actual_limit_price,
                               *, stage="preflight"):
            """Re-infer at both fresh quote and rounded limit, without broker I/O."""
            self._ensure_demo()
            if stage not in {"preflight", "final_send"} or rule.side is not OrderSide.BUY:
                raise ValueError("Target/horizon model permits only DEMO BUY rechecks")
            require_exit_schedule(self.strategy_id, horizon)
            record = self.window.store.external_for_rule(rule.id)
            if record is not None:
                from dockdack.signal_bridge import prototype_record_family

                family = prototype_record_family(record, watch_id=item.id, action="buy")
                if (record["source_id"] != self.source_id or family is None
                        or family.id != self.strategy_id):
                    raise ValueError("Target/horizon BUY source or model differs")
            engine = self.window.engine
            engine._validate_snapshot(item, fresh_snapshot)
            now = engine.clock()
            if not 0 <= (now - fresh_snapshot.fetched_at).total_seconds() <= 15:
                raise ValueError("Target/horizon recheck quote is stale")
            predictor = self.predictors.get(item.instrument.market.value)
            if predictor is None or getattr(predictor, "metadata", {}).get("market") != item.instrument.market.value:
                raise ValueError("Target/horizon market checkpoint is unavailable")
            today = market_time(item.instrument.market, now).date()
            history = fresh_snapshot.history.bars[-item.days:]
            stock = {"market": item.instrument.market.value,
                     "complete": len(history) >= 30, "available_days": len(history),
                     "bars": [{"date": bar.day.isoformat(), "is_current_day": bar.day == today,
                               **{key: str(getattr(bar, key))
                                  for key in ("open", "high", "low", "close", "volume")}}
                              for bar in history]}
            bars = consecutive_completed_bars(stock, now)
            if actual_limit_price is None:
                raise ValueError("Target/horizon DEMO BUY requires a rechecked limit price")
            prices = [number(fresh_snapshot.quote.price, "current price")]
            limit = number(actual_limit_price, "limit price")
            if limit != prices[0]:
                prices.append(limit)
            for price in prices:
                prediction = self.producer_type.checked_model_query(predictor, bars, price)
                if not prediction["predicts_success"]:
                    raise ValueError("Target/horizon policy fails at the fresh quote or limit")
            self._ensure_demo()
            if not 0 <= (engine.clock() - fresh_snapshot.fetched_at).total_seconds() <= 15:
                raise ValueError("Target/horizon recheck quote expired during inference")

    for kind in (Producer, Bridge):
        kind.source_id = source_id
        kind.title = title
    Producer.__name__ = "MarkTargetHorizon" + model_id.split("-")[1] + "Producer"
    Bridge.__name__ = "MarkTargetHorizon" + model_id.split("-")[1] + "Bridge"
    return Bridge
