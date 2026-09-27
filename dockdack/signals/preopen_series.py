"""DEMO-only identities for separately frozen Mark1.n pre-open models.

Each model has its own source, child, output file, threshold and strategy lot.
The research score is not a broker permission or a calibrated win probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from dockdack.signals.mark1_4_external import Mark14ExternalFeed, Mark14TriggerBridge


@dataclass(frozen=True)
class PreopenModelSpec:
    model_id: str
    bundle_directory: str
    method: str
    output: str
    horizon: str = "당일 종료형 · 모델 보유분만 장마감 5분 전 기간 매도"

    @property
    def title(self) -> str:
        return self.model_id.replace("mark1-", "mark1.").replace("-prototype", " prototype")

    @property
    def source_id(self) -> str:
        return self.model_id + "-demo-trigger"

    @property
    def strategy_notice(self) -> str:
        return f"완료 30일봉 · {self.method} · 개장 10분 전 상위 후보 동결 · {self.horizon}"

    @property
    def risk_notice(self) -> str:
        return f"{self.output} · 화면 값은 실제 체결 수익률/적중률이 아님 · 모의 전용"


PREOPEN_MODELS = {
    "mark1-3-prototype": PreopenModelSpec("mark1-3-prototype", "models/mark1_3",
                                           "8개 완료봉 특징·CUDA 학습 MLP 순수익 회귀", "예상 비용 후 순수익률 %"),
    "mark1-5-prototype": PreopenModelSpec("mark1-5-prototype", "models/mark1_series",
                                           "LSTM 순환망 순수익 회귀", "예상 순수익률 %"),
    "mark1-6-prototype": PreopenModelSpec("mark1-6-prototype", "models/mark1_series",
                                           "30×5 OHLCV 영상 CNN", "양의 순수익 추정 점수"),
    "mark1-7-prototype": PreopenModelSpec("mark1-7-prototype", "models/mark1_series",
                                           "같은 날 종목 전체 attention 상대순위", "무단위 순위 점수"),
    "mark1-8-prototype": PreopenModelSpec("mark1-8-prototype", "models/mark1_8",
                                           "비용·하방 위험 직접 최적화 점수+투자비중망", "순위 점수와 자산 비중"),
    "mark1-9-prototype": PreopenModelSpec("mark1-9-prototype", "models/mark1_9",
                                           "앙상블 평균−불확실성 점수망", "보수적 순수익 점수"),
    "mark1-10-prototype": PreopenModelSpec("mark1-10-prototype", "models/mark1_10",
                                            "최악 연도·40bp 견고성 유전탐색", "무단위 진화 점수"),
    "mark1-11-prototype": PreopenModelSpec(
        "mark1-11-prototype", "models/mark1_horizons",
        "E4 동결 유전망 · 3거래 세션 보유", "무단위 동결 점수",
        "매수 체결 세션=1일째 · 3번째 거래 세션 장중 기간 매도 · 종가 백테스트와 다름"),
    "mark1-12-prototype": PreopenModelSpec(
        "mark1-12-prototype", "models/mark1_horizons",
        "E4 동결 유전망 · 5거래 세션 보유", "무단위 동결 점수",
        "매수 체결 세션=1일째 · 5번째 거래 세션 장중 기간 매도 · 종가 백테스트와 다름"),
}


@lru_cache(maxsize=None)
def preopen_bridge_type(model_id: str):
    spec = PREOPEN_MODELS.get(model_id)
    if spec is None:
        raise ValueError("Unknown pre-open model")
    return type(
        "Preopen" + model_id.replace("-", "_") + "Bridge",
        (Mark14TriggerBridge,),
        {"source_id": spec.source_id, "strategy_id": spec.model_id,
         "title": spec.title, "strategy_notice": spec.strategy_notice,
         "risk_notice": spec.risk_notice,
         "bundle_directory": spec.bundle_directory,
         "state_filename": "exchange/" + model_id + "-external-decisions-v1.json"},
    )


class PreopenExperimentalFeed(Mark14ExternalFeed):
    def __init__(self, window, model_id, policy, output_path, **kwargs):
        spec = PREOPEN_MODELS.get(model_id)
        if spec is None:
            raise ValueError("Unknown pre-open model")
        self.expected_model_id = model_id
        self.strategy_notice = spec.strategy_notice
        self.risk_notice = spec.risk_notice
        super().__init__(window, model_id, policy, output_path, **kwargs)


def load_preopen_predictor(model_id: str, bundle_root, market: str):
    """The child loads one SHA-checked trained model, never fits on live bars."""
    if model_id == "mark1-3-prototype":
        from dockdack.mark1_3_preopen import Mark13Predictor
        return Mark13Predictor(bundle_root, market)
    if model_id in {"mark1-5-prototype", "mark1-6-prototype", "mark1-7-prototype"}:
        from dockdack.mark1_series_inference import MarkSeriesPredictor
        return MarkSeriesPredictor(bundle_root, market,
                                   model_id.replace("mark1-", "mark1.").replace("-prototype", ""))
    if model_id == "mark1-8-prototype":
        from dockdack.signals.preopen_predictors import Mark18PreopenPredictor
        return Mark18PreopenPredictor(bundle_root, market)
    if model_id == "mark1-9-prototype":
        from dockdack.mark1_special_inference import Mark19Predictor
        return Mark19Predictor(bundle_root, market)
    if model_id == "mark1-10-prototype":
        from dockdack.mark1_special_inference import Mark110Predictor
        return Mark110Predictor(bundle_root, market)
    if model_id in {"mark1-11-prototype", "mark1-12-prototype"}:
        from dockdack.mark1_horizons_inference import MarkHorizonPredictor
        return MarkHorizonPredictor(
            bundle_root, market,
            model_id.replace("mark1-", "mark1.").replace("-prototype", ""),
        )
    raise ValueError("Unknown pre-open model")
