"""Fail-closed, read-only bridge for daily-to-five-minute *paper* signals.

Importing this module is cheap: PyTorch and the research evaluator are loaded
only when ``evaluate_transfer_model`` is called.  Stock/exchange identities
come from each verified bundle manifest, so the current single-stock bundles
do not silently generalize to other watchlist symbols. A paper BUY is never
an order authorization; these bundles have no validated exit policy, and the
current trained bundles have no selected buy threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType, SimpleNamespace
from typing import Mapping
from zoneinfo import ZoneInfo


class PaperAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class TransferModelConfig:
    model_id: str
    bundle_name: str
    architecture: str
    lookback: int
    horizon_bars: int
    market: str = "domestic"


def _configs() -> Mapping[str, TransferModelConfig]:
    variants = (
        ("transfer-005930-causal-20260929", 20, 3),
        ("transfer-005930-20x6-causal-20260929", 20, 6),
        ("transfer-005930-30x12-causal-20260929", 30, 12),
    )
    result = {}
    number = 29
    for bundle, lookback, horizon in variants:
        for architecture in ("linear", "conv", "gru"):
            model_id = f"mark1-{number}-prototype"
            result[model_id] = TransferModelConfig(
                model_id, bundle, architecture, lookback, horizon)
            number += 1
    return MappingProxyType(result)


TRANSFER_CONFIGS = _configs()
TRANSFER_MODEL_IDS = tuple(TRANSFER_CONFIGS)


@dataclass(frozen=True)
class MinuteTransferPaperSignal:
    model_id: str
    market: str
    exchange: str
    symbol: str
    action: PaperAction
    reason: str
    probability_proxy: float | None = None
    validation_threshold: float | None = None
    last_broker_bar_label: str | None = None
    manifest_sha256: str | None = None
    minute_sha256: str | None = None

    @property
    def order_eligible(self) -> bool:
        """Never promote a research-only paper candidate to an order."""
        return False

    def as_dict(self) -> dict:
        """Data-only diagnostics suitable for a model worker's JSON reply."""
        return {
            "model_id": self.model_id, "market": self.market,
            "exchange": self.exchange, "symbol": self.symbol,
            "paper_action": self.action.value, "order_eligible": False,
            "trading_mode": "demo", "reason": self.reason,
            "probability_proxy": self.probability_proxy,
            "validation_threshold": self.validation_threshold,
            "last_broker_bar_label": self.last_broker_bar_label,
            "manifest_sha256": self.manifest_sha256,
            "minute_sha256": self.minute_sha256,
        }


def bundle_path_for_model(model_id: str, *, research_root: str | Path | None = None) -> Path:
    config = TRANSFER_CONFIGS[model_id]
    if research_root is None:
        from dockdack.runtime_paths import app_home
        research_root = app_home() / "outputs" / "minute-research"
    return Path(research_root) / config.bundle_name


def evaluate_transfer_model(
    model_id: str,
    minute_jsonl: str | Path,
    *,
    market: str,
    exchange: str,
    symbol: str,
    now: datetime | None = None,
    research_root: str | Path | None = None,
    trading_mode: str = "demo",
) -> MinuteTransferPaperSignal:
    """Evaluate one model using the same verified, fresh paper path as the CLI.

    No broker, account, order, or signal-file API is imported.  The delegated
    evaluator validates the DEMO collector receipt/hash, current open exchange
    session, complete contiguous bars, trained identity, and future session.
    Its ephemeral report is removed before this function returns.  Any data or
    dependency error abstains instead of crossing the execution boundary.
    """
    config = TRANSFER_CONFIGS[model_id]

    def abstain(reason: str) -> MinuteTransferPaperSignal:
        return MinuteTransferPaperSignal(model_id, market, exchange, symbol,
                                         PaperAction.ABSTAIN, reason)

    if trading_mode != "demo":
        return abstain("DEMO-only paper bridge")
    if market != config.market:
        return abstain("market differs from the trained adaptation market")
    try:
        # Lazy import matters for ordinary GUI startup when research extras are
        # intentionally absent.  The CLI remains the single validator for
        # source authenticity, timing, bundle hashes, and model identity.
        from examples.paper_minute_signal import run as paper_run
        from dockdack.research.minute_transfer import sha256_file

        bundle = bundle_path_for_model(model_id, research_root=research_root)
        as_of = now or datetime.now(timezone.utc)
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            return abstain("paper clock must be timezone-aware")
        with TemporaryDirectory(prefix="dockdack-paper-minute-") as directory:
            args = SimpleNamespace(
                bundle=bundle, minute=Path(minute_jsonl), market=market,
                exchange=exchange, symbol=symbol,
                session=as_of.astimezone(ZoneInfo("Asia/Seoul")).date().isoformat(),
                output=Path(directory) / "paper.json",
            )
            report = paper_run(args, now=as_of)
        prediction = report["models"][config.architecture]
        manifest_path = bundle / "manifest.json"
        manifest_sha256 = sha256_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (report.get("format") != "dockdack-minute-transfer-paper-v1"
                or report.get("market") != market
                or report.get("exchange") != exchange
                or report.get("symbol") != symbol
                or report.get("safety", {}).get("paper_only") is not True
                or report["safety"].get("deployment_allowed") is not False
                or report["safety"].get("order_routing_connected") is not False
                or prediction.get("market") != market
                or prediction.get("exchange") != exchange
                or prediction.get("symbol") != symbol
                or prediction.get("deployment_allowed") is not False
                or report.get("model_bundle", {}).get("manifest_sha256") != manifest_sha256
                or manifest.get("study", {}).get("lookback") != config.lookback
                or manifest["study"].get("horizon_bars") != config.horizon_bars
                or {"exchange": exchange, "symbol": symbol}
                   not in manifest["study"].get("minute_adaptation_identities", [])):
            return abstain("paper report identity or safety metadata mismatch")
        threshold = prediction["validation_threshold"]
        candidate = prediction["paper_candidate"]
        if threshold is None and candidate:
            return abstain("threshold-less model returned an impossible paper candidate")
        action = PaperAction.BUY if threshold is not None and candidate is True else PaperAction.ABSTAIN
        reason = ("validated paper buy candidate; no exit policy or order authorization"
                  if action is PaperAction.BUY else
                  "validation selected no buy threshold" if threshold is None else
                  "paper probability did not reach the validation threshold")
        return MinuteTransferPaperSignal(
            model_id, market, exchange, symbol, action, reason,
            probability_proxy=prediction["probability_proxy"],
            validation_threshold=threshold,
            last_broker_bar_label=report["last_broker_bar_label"],
            manifest_sha256=manifest_sha256,
            minute_sha256=report["source"]["minute_sha256"],
        )
    except Exception as exc:
        # This is a diagnostic path, not an order path.  Never let a corrupt or
        # stale research artifact turn into a default BUY/SELL decision.
        return abstain(f"paper input unavailable or unsafe: {type(exc).__name__}: {str(exc)[:180]}")
