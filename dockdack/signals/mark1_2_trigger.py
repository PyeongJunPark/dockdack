"""Unqualified Mark1.2 neural research model on the manually armed DEMO wire.

The sealed neural bundle is not changed or promoted by this adapter.  This
module has no broker or order-starting authority; the ordinary GUI and its
existing preflight/final-send checks retain that responsibility.
"""
from __future__ import annotations

import copy
from decimal import Decimal
import math
import re

import numpy as np

from dockdack.lstm30_adapter import number
from dockdack.signals.mark1_trigger import Mark1DemoSignalProducer, Mark1TriggerBridge


SOURCE_ID = "mark1-2-prototype-demo-trigger"
STRATEGY_ID = "mark1-2-prototype"
TITLE = "mark1.2 prototype"
STRATEGY_NOTICE = "매수 추정 확률 > 50% · 평균매수가 +1% 익절 / -0.9% 손절 · 모의 전용"
RISK_NOTICE = (
    "연구 검증 미통과 · 2025+ 과거 비용 반영 백테스트 국내/미국 모두 손실 · "
    "장중 진입 이후의 고가/저가 순서 미검증 · 추정 확률은 실제 성공률이 아님"
)


class Mark12PrototypePredictor:
    """Translate sealed ``predict_proba`` into the existing price-aware wire.

    Deliberately use the saved CPU-FP32 ensemble and its original calibration;
    no fitting, threshold optimization or qualification happens in the GUI.
    """

    def __init__(self, bundle_root, market, *, predictor=None):
        from dockdack.mark1_2_inference import Predictor, SEMANTICS

        if market not in {"domestic", "us"}:
            raise ValueError("mark1.2 시장은 국내/미국만 지원합니다.")
        self._predictor = predictor if predictor is not None else Predictor(bundle_root, market)
        metadata = self._predictor.metadata
        if (not isinstance(metadata, dict) or metadata.get("market") != market
                or metadata.get("title") != TITLE or metadata.get("semantics") != SEMANTICS
                or metadata.get("research_only") is not True
                or metadata.get("research_qualified") is not False
                or metadata.get("deployment_allowed") is not False
                or metadata.get("intraday_path_verified") is not False
                or not isinstance(metadata.get("bundle_manifest_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", metadata["bundle_manifest_sha256"]) is None):
            raise ValueError("mark1.2 저장 모델의 시장·정책·연구 제한을 확인할 수 없습니다.")
        self._metadata = {**copy.deepcopy(metadata), "strategy_id": STRATEGY_ID,
                          "model_name": metadata["architecture"],
                          "version": "20260924-v1"}

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def predict(self, bars, current_price=None):
        from dockdack.mark1_2_inference import SEMANTICS

        price = number(current_price, "current price")
        try:
            history = np.asarray(bars, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("mark1.2는 완료된 30개 숫자 OHLCV 봉이 필요합니다.") from exc
        if history.shape != (30, 5) or not np.isfinite(history).all():
            raise ValueError("mark1.2는 완료된 30개 숫자 OHLCV 봉이 필요합니다.")
        entry = float(price)
        if not math.isfinite(entry) or entry <= 0:
            raise ValueError("mark1.2 현재가가 유한한 양수가 아닙니다.")
        probabilities = np.asarray(self._predictor.predict_proba(history, np.array([entry], dtype=np.float64)))
        if (probabilities.shape != (1,) or probabilities.dtype.kind not in "fiu"
                or not np.isfinite(probabilities).all()):
            raise ValueError("mark1.2 모델이 유효한 확률을 반환하지 않았습니다.")
        success = float(probabilities[0])
        if not 0 <= success <= 1:
            raise ValueError("mark1.2 모델 확률이 0~1 범위를 벗어났습니다.")
        selected = success > .5
        return {"title": TITLE, "strategy_id": STRATEGY_ID,
                "version": self._metadata["version"], "market": self._metadata["market"],
                "model_name": self._metadata["model_name"],
                "probability_success": success, "probability_stop": None,
                "predicts_success": selected, "selected_research": selected,
                "buy_threshold": .5, "policy_threshold": .5, "stop_probability_cap": 1.,
                "take_profit_pct": 1., "stop_loss_pct": .9,
                "candidate_entry_price": entry,
                "candidate_take_price": float(price * Decimal("1.01")),
                "candidate_stop_price": float(price * Decimal("0.991")),
                "target": SEMANTICS["target"],
                "entry_matches_evaluated_type": False, "inference_device": "cpu",
                "bundle_manifest_sha256": self._metadata["bundle_manifest_sha256"],
                "ensemble_method": SEMANTICS["ensemble"],
                "warnings": list(self._metadata.get("warnings", ())),
                "research_only": True, "research_qualified": False,
                "deployment_allowed": False, "intraday_path_verified": False}


class Mark12DemoSignalProducer(Mark1DemoSignalProducer):
    source_id = SOURCE_ID
    strategy_id = STRATEGY_ID
    title = TITLE
    strategy_notice = STRATEGY_NOTICE
    risk_notice = RISK_NOTICE
    strict_model_identity = True

    def model_prediction(self, predictor, bars, price):
        prediction, detail = super().model_prediction(predictor, bars, price)
        detail.update(input_features=18, input_tokens=31,
                      feature_matrix="30 completed OHLCV bars + 1 candidate price; 31 x 18 features")
        return prediction, detail

    @classmethod
    def validate_prediction(cls, prediction):
        from dockdack.mark1_2_inference import SEMANTICS

        super().validate_prediction(prediction)
        if (prediction.get("title") != TITLE or prediction.get("target") != SEMANTICS["target"]
                or prediction.get("research_only") is not True
                or prediction.get("research_qualified") is not False
                or prediction.get("deployment_allowed") is not False
                or prediction.get("intraday_path_verified") is not False):
            raise ValueError("mark1.2 연구 전용 모델 식별자 또는 검증 상태가 다릅니다.")


class Mark12TriggerBridge(Mark1TriggerBridge):
    source_id = SOURCE_ID
    strategy_id = STRATEGY_ID
    title = TITLE
    strategy_notice = STRATEGY_NOTICE
    risk_notice = RISK_NOTICE
    producer_type = Mark12DemoSignalProducer
    bundle_directory = "models/mark1_2_prototype"
    state_filename = "exchange/mark1-2-trigger-decisions-v1.json"

    def _new_predictor(self, market):
        return Mark12PrototypePredictor(self.bundle_root, market)
