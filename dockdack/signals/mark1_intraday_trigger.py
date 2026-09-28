"""Ten independent DEMO-only daily-proxy query sources (MK1.13–MK1.22)."""
from __future__ import annotations

from functools import lru_cache

from dockdack.signals.mark1_trigger import Mark1DemoSignalProducer, Mark1TriggerBridge


# Keep GUI-visible identities lightweight. Importing the trained feature
# modules here would require NumPy and PyTorch merely to open a GUI-only app.
MODEL_IDS = tuple(f"mark1-{number}-prototype" for number in range(13, 23))
RISK_NOTICE = (
    "완료 30일봉 + 현재가 질의 · 다음 날 관측 시가에서 정의한 일봉 전체 대리사건 학습 · "
    "장중 진입 후 경로/수익성 미검증 · 모의 전용"
)


@lru_cache(maxsize=None)
def intraday_bridge_type(model_id: str):
    if model_id not in MODEL_IDS:
        raise ValueError("Unknown MK1 daily-proxy model")
    variant = model_id.replace("mark1-", "mark1.").replace("-prototype", "")
    title = variant + " prototype"
    source_id = model_id + "-demo-trigger"
    notice = "독립 일봉 대리점수 > 0.5일 때만 모의 매수 후보 · 해당 모델 매수분 +1% / −0.9%"

    class Producer(Mark1DemoSignalProducer):
        strict_model_identity = True
        strategy_id = model_id
        strategy_notice = notice
        risk_notice = RISK_NOTICE

        @classmethod
        def validate_prediction(cls, prediction):
            super().validate_prediction(prediction)
            if (prediction.get("target") !=
                    "daily_open_to_whole_session_take_only_1pct_without_0.9pct_stop"
                    or prediction.get("score_scope") != "daily_open_whole_session_proxy_not_intraday"
                    or prediction.get("intraday_path_verified") is not False
                    or prediction.get("research_only") is not True
                    or prediction.get("research_qualified") is not False
                    or prediction.get("deployment_allowed") is not False):
                raise ValueError("MK1 daily-proxy research identity or risk limits differ")

        def model_prediction(self, predictor, bars, price):
            prediction, detail = super().model_prediction(predictor, bars, price)
            detail.update(input_completed_bars=30, input_features=8, input_tokens=None,
                          target_basis="next session observed OPEN whole-session +1%/−0.9% proxy",
                          intraday_path_verified=False, strategy_notice=self.strategy_notice,
                          risk_notice=self.risk_notice)
            return prediction, detail

    class Bridge(Mark1TriggerBridge):
        producer_type = Producer
        bundle_directory = ("models/mark1_intraday" if int(variant.split(".")[1]) < 18
                            else "models/mark1_intraday_extra")
        state_filename = "exchange/" + model_id + "-decisions-v1.json"

        def _new_predictor(self, market):
            from dockdack.mark1_intraday_inference import MarkIntradayPredictor
            return MarkIntradayPredictor(self.bundle_root, market, variant)

    for kind in (Producer, Bridge):
        kind.source_id = source_id
        kind.strategy_id = model_id
        kind.title = title
        kind.strategy_notice = notice
        kind.risk_notice = RISK_NOTICE
    Producer.__name__ = "MarkIntraday" + variant.split(".")[1] + "Producer"
    Bridge.__name__ = "MarkIntraday" + variant.split(".")[1] + "Bridge"
    return Bridge
