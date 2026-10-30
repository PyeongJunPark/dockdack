"""Lightweight identities for the daily-trained minute prototypes.

This catalog contains display and provenance metadata only. Importing it never
loads model weights, a broker, or an order sender. In particular, selection in
the desktop is not permission to turn a research result into a trade.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinuteResearchSpec:
    model_id: str
    title: str
    kind: str
    method: str
    output: str
    bundle_name: str | None = None
    architecture: str | None = None
    variant: str | None = None
    lookback: int | None = None
    horizon_bars: int | None = None

    @property
    def source_id(self) -> str:
        return self.model_id + ("-demo-trigger" if self.kind == "transfer"
                                else "-demo-research")


_TRANSFER_SETTINGS = (
    (20, 3, "daily-proxy-domestic-20x3-segment-safe-20260930"),
    (20, 6, "daily-proxy-domestic-20x6-segment-safe-20260930"),
    (30, 12, "daily-proxy-domestic-30x12-segment-safe-20260930"),
)
_ARCHITECTURES = (("linear", "선형망"), ("conv", "합성곱망"), ("gru", "GRU"))
_HEDGE_VARIANTS = (
    ("residual_z", "지수 대비 잔차"),
    ("sector_relative", "업종 상대 하락"),
    ("turnover_vwap", "거래대금·VWAP"),
    ("atr_drop", "변동폭 대비 하락"),
    ("gap_relative", "시가 갭 상대 하락"),
    ("volume_shock_reversal", "거래량 급증·반등"),
    ("range_recovery", "당일 범위 회복"),
)


def _build_specs() -> tuple[MinuteResearchSpec, ...]:
    specs = []
    number = 29
    for lookback, horizon, bundle in _TRANSFER_SETTINGS:
        for architecture, label in _ARCHITECTURES:
            model_id = f"mark1-{number}-prototype"
            specs.append(MinuteResearchSpec(
                model_id, f"mark1.{number} prototype", "transfer",
                f"완료 일봉만 학습 · 실제 5분봉 {lookback}개 추론 · {label}",
                f"{horizon}개 5분봉 보유 수익 방향 대리값 · 분봉 수익성 미검증",
                bundle_name=bundle, architecture=architecture,
                lookback=lookback, horizon_bars=horizon))
            number += 1
    for variant, label in _HEDGE_VARIANTS:
        model_id = f"mark1-{number}-prototype"
        specs.append(MinuteResearchSpec(
            model_id, f"mark1.{number} prototype", "hedge",
            f"실제 5분봉 · {label} 조건 · 인버스 두 다리 연구(주문 보류)",
            "주식 매수 + 시장지수 −1배 인버스 ETF 매수의 모의 손익",
            variant=variant))
        number += 1
    return tuple(specs)


MINUTE_RESEARCH_SPECS = _build_specs()
MINUTE_RESEARCH_BY_ID = {spec.model_id: spec for spec in MINUTE_RESEARCH_SPECS}
MINUTE_RESEARCH_IDS = tuple(MINUTE_RESEARCH_BY_ID)
MINUTE_TRANSFER_IDS = tuple(spec.model_id for spec in MINUTE_RESEARCH_SPECS if spec.kind == "transfer")
MINUTE_HEDGE_IDS = tuple(spec.model_id for spec in MINUTE_RESEARCH_SPECS if spec.kind == "hedge")
