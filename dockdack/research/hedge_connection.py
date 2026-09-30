"""Read-only connection of minute/inverse-ETF research to application status.

This is deliberately *not* a trading signal producer. Historical data-gate
``pass`` means an out-of-sample paper trade existed, not that the strategy is
qualified. No report currently contains a deployable threshold, and two-leg
stock/ETF execution cannot be made atomic by this adapter. Every connection
therefore abstains and exposes no order intent, including in DEMO mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import re

from dockdack.minute_hedge_research import Variant
from dockdack.models import TradingMode


MAX_REPORT_BYTES = 2_000_000
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SYMBOL = re.compile(r"[0-9]{6}\Z")
_REASON = re.compile(r"[A-Za-z0-9_.; -]{1,256}\Z")
_TITLES = {
    Variant.RESIDUAL_Z: "시장 대비 잔차",
    Variant.SECTOR_RELATIVE: "업종 대비 하락",
    Variant.TURNOVER_VWAP: "거래대금 VWAP",
    Variant.ATR_DROP: "ATR 낙폭",
    Variant.GAP_RELATIVE: "전일 갭 대비",
    Variant.VOLUME_SHOCK_REVERSAL: "거래량 급증 반전",
    Variant.RANGE_RECOVERY: "저점 이탈 회복",
}


@dataclass(frozen=True, slots=True)
class HedgeConnection:
    """Display/status contract; ``order_intents`` can never contain an order."""

    model_id: str
    title: str
    variant: Variant
    state: str
    reason: str
    research_gate: str | None
    research_reason: str | None
    stock_symbol: str | None
    inverse_etf_symbol: str | None
    trading_mode: TradingMode

    @property
    def order_eligible(self) -> bool:
        return False

    @property
    def order_intents(self) -> tuple:
        return ()


def _abstain_all(reason: str, mode: TradingMode) -> tuple[HedgeConnection, ...]:
    return tuple(HedgeConnection(f"minute-hedge-{variant.value}", _TITLES[variant],
                                 variant, "abstain", reason, None, None, None, None, mode)
                 for variant in Variant)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _read_report(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("hedge report exceeds size limit")
    report = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                        parse_constant=_invalid_constant)
    if not isinstance(report, dict):
        raise ValueError("hedge report must be an object")
    return report


def _validate_report(report: dict) -> dict[Variant, dict]:
    if (report.get("schema") != "dockdack.minute_hedge_research.v1"
            or report.get("market") != "domestic"
            or report.get("hedge_leg") != "long_inverse_etf"
            or report.get("benchmark_index_code") != "201"
            or report.get("inverse_etf_symbol") != "114800"
            or report.get("actual_broker_trades") is not False
            or not isinstance(report.get("stock_symbol"), str)
            or _SYMBOL.fullmatch(report["stock_symbol"]) is None
            or report["stock_symbol"] == "114800"):
        raise ValueError("unsupported research report or non-DEMO hedge")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict) or not {"stock", "index", "etf"} <= inputs.keys():
        raise ValueError("missing stock/index/ETF DEMO data provenance")
    for key, symbol in (("stock", report["stock_symbol"]), ("index", "201"), ("etf", "114800")):
        row = inputs[key]
        if (not isinstance(row, dict) or row.get("source") != "kiwoom_rest_demo_minute_chart"
                or row.get("symbol") != symbol or not isinstance(row.get("sha256"), str)
                or _SHA256.fullmatch(row["sha256"]) is None):
            raise ValueError("stock/index/ETF data are not verified DEMO collector inputs")
    rows = report.get("candidates")
    if not isinstance(rows, list) or len(rows) != len(Variant):
        raise ValueError("all seven candidate statuses are required")
    found: dict[Variant, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid candidate status")
        try:
            variant = Variant(row.get("variant"))
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown hedge variant") from exc
        if variant in found:
            raise ValueError("duplicate hedge variant")
        gate = row.get("data_gate")
        count = row.get("out_of_sample_trades")
        pnl = row.get("out_of_sample_net_pnl")
        if (gate not in {"pass", "abstain"} or row.get("deployment_allowed") is not False
                or row.get("pass_means_profitable") is not False
                or type(count) is not int or count < 0
                or (pnl is not None and (type(pnl) not in {int, float} or not isfinite(pnl)))
                or (gate == "pass" and count == 0)
                or not isinstance(row.get("reason"), str)
                or _REASON.fullmatch(row["reason"]) is None):
            raise ValueError("research result is not an abstaining, non-deployable status")
        found[variant] = row
    if set(found) != set(Variant):
        raise ValueError("all seven known hedge variants are required")
    return found


def load_hedge_connections(report_path: Path | str | None, *,
                           mode: TradingMode) -> tuple[HedgeConnection, ...]:
    """Expose seven research statuses without ever creating an order signal.

    The source report is optional because ``outputs/`` is intentionally ignored
    by Git. Missing, malformed, unsupported or REAL-mode data fail closed for
    *all* seven variants. A historical ``data_gate='pass'`` remains abstain.
    """
    if not isinstance(mode, TradingMode):
        raise TypeError("mode must be an explicit TradingMode")
    if mode is not TradingMode.DEMO:
        return _abstain_all("demo_only", mode)
    if report_path is None:
        return _abstain_all("report_missing", mode)
    try:
        report = _read_report(Path(report_path))
        rows = _validate_report(report)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return _abstain_all("report_unverified", mode)
    return tuple(HedgeConnection(
        model_id=f"minute-hedge-{variant.value}", title=_TITLES[variant], variant=variant,
        state="abstain", reason="historical_research_only_no_qualified_threshold",
        research_gate=rows[variant]["data_gate"], research_reason=rows[variant]["reason"],
        stock_symbol=report["stock_symbol"], inverse_etf_symbol="114800", trading_mode=mode,
    ) for variant in Variant)
