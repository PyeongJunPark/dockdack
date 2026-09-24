"""Portable, research-only CPU inference for the separately retrained 0.5/0.4 model.

This module does not instantiate brokers, call APIs, open GUIs, or read training
caches/databases. DockDack's base package dependencies are still required. A
positive prediction is a daily-bar research event, never permission to order.
The original mark1 prototype and its +1%/-0.9% contract remain independent.
"""
from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import hashlib
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import platform
import re

import numpy as np

from . import mark1_0504_data, mark1_metrics, mark1_selective_features, mark1_selective_models
from .mark1_0504_data import FEATURE_NAMES, features_from_history
from .mark1_metrics import calibrated_probability
from .mark1_selective_models import CLASS_NAMES, load_model, predict_raw


TITLE = "mark1 0.5/0.4 research"
BUNDLE_VERSION = "20260920-v1"
OWNER = "dockdack.mark1_0504"
SCHEMA_VERSION = 1
MARKET_MODELS = {"domestic": "cat_joint6", "us": "cat_binary8"}
SEEDS = (42, 43, 44)
TARGET = "daily_high_ge_entry_0_5pct_and_low_gt_entry_minus_0_4pct_conservative"
PATH_LIMITATION = (
    "Whole-session conservative daily-bar event: high >= entry*1.005 and "
    "low > entry*0.996. Both-touch is failure. This is not a verified "
    "first-touch probability after an arbitrary intraday entry."
)
SEMANTICS = {
    "lookback": 30, "bar_columns": ["open", "high", "low", "close", "volume"],
    "target": TARGET, "class_names": list(CLASS_NAMES), "both_touch": "stop_first_failure",
    "take_profit_pct": .5, "stop_loss_pct": .4, "buy_threshold": .5,
    "buy_comparison": "strict_greater_than", "ensemble": "mean_raw_logits_then_platt",
    "historical_representation": "validate_original_then_float32_raw_cache_parity",
    "candidate_entry_representation": "float64",
    "path_limitation": PATH_LIMITATION,
}
RISK_FLAGS = {
    "research_only": True, "deployment_allowed": False, "research_qualified": False,
    "intraday_path_verified": False, "known_data_quality_issues": True,
}
WARNINGS = [
    "This separately retrained 0.5/0.4 bundle is research-only and grants no broker or order permission.",
    "Daily OHLC cannot recover intraday barrier order. Both-touch counts as stop first; whole-session labels do not validate an arbitrary intraday entry's remaining path.",
    "The cleaned historical source is not certified corporate-action-clean. Training exclusions are documented in provenance, not a repair of the original database.",
    "2025+ results overlap earlier research evaluation periods and are not a new untouched test.",
    "Tighter profit/stop barriers do not guarantee more buy signals or better net returns; trading costs and slippage matter.",
]
RUNTIME_MODULES = {
    "barrier_features": mark1_0504_data,
    "base_features": mark1_selective_features,
    "calibration": mark1_metrics,
    "native_backend": mark1_selective_models,
}


def sha256_file(path: Path) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or linked bundle file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_code_hashes() -> dict:
    # The wrapper itself owns the historical float32 conversion, logit
    # aggregation and strict threshold, so dependencies alone are insufficient.
    return {**{name: sha256_file(Path(module.__file__)) for name, module in RUNTIME_MODULES.items()},
            "inference": sha256_file(Path(__file__))}


def _read(path: Path) -> dict:
    sha256_file(path)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(result, allow_nan=False)
    except (OSError, ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid bundle JSON: {path.name}") from error
    if not isinstance(result, dict):
        raise ValueError(f"Bundle JSON must be an object: {path.name}")
    return result


def _same(left, right) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _checked_file(root: Path, relative: str, digest: str) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or Path(relative).is_absolute() or any(part in ("", ".", "..") for part in relative.split("/"))
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("Unsafe artifact path or invalid SHA256")
    candidate = root / relative
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError("Linked bundle paths are not supported")
    if not candidate.resolve().is_relative_to(root.resolve()) or sha256_file(candidate) != digest:
        raise ValueError("Bundle artifact checksum mismatch")
    return candidate


def _versions() -> dict:
    try:
        versions = {"python": platform.python_version(), "numpy": np.__version__,
                    "catboost": package_metadata.version("catboost")}
    except package_metadata.PackageNotFoundError as error:
        raise ImportError("mark1 0.5/0.4 requires the optional CatBoost CPU inference dependency") from error
    bounds = {"python": ((3, 10), (4, 0)), "numpy": ((1, 24), (3, 0)), "catboost": ((1, 2), (2, 0))}
    for name, version in versions.items():
        matched = re.match(r"^(\d+)\.(\d+)", version)
        parsed = tuple(map(int, matched.groups())) if matched else ()
        if not matched or not bounds[name][0] <= parsed < bounds[name][1]:
            raise ValueError(f"Unsupported 0.5/0.4 inference {name} version: {version}")
    return versions


def _calibrator(value):
    if (not isinstance(value, dict) or value.get("method") != "platt_monotone"
            or value.get("weighted") is not False or type(value.get("fit_samples")) is not int
            or value["fit_samples"] <= 0 or type(value.get("slope")) not in (int, float)
            or type(value.get("bias")) not in (int, float)):
        raise ValueError("Invalid unweighted frozen Platt calibration")
    calibrated_probability([0.], value)


class HalfPercentPredictor:
    """A checksummed three-seed CPU ensemble, without any execution authority."""

    buy_threshold = .5

    def __init__(self, bundle_root, market):
        if market not in MARKET_MODELS:
            raise ValueError("market must be domestic or us")
        root = Path(bundle_root).absolute()
        if root.is_symlink() or not root.is_dir():
            raise ValueError("0.5/0.4 bundle root must be an existing, unlinked directory")
        digest_path = root / "manifest.sha256"
        sha256_file(digest_path)
        manifest_digest = digest_path.read_text(encoding="ascii").strip()
        manifest = _read(_checked_file(root, "manifest.json", manifest_digest))
        if (manifest.get("owner") != OWNER or type(manifest.get("schema_version")) is not int
                or manifest["schema_version"] != SCHEMA_VERSION or manifest.get("title") != TITLE
                or manifest.get("version") != BUNDLE_VERSION or manifest.get("completed") is not True
                or not _same(manifest.get("semantics"), SEMANTICS)
                or not _same(manifest.get("risk_flags"), RISK_FLAGS)
                or manifest.get("warnings") != WARNINGS
                or manifest.get("feature_names") != list(FEATURE_NAMES)
                or len(FEATURE_NAMES) != 184 or manifest.get("feature_count") != 184
                or not _same(manifest.get("runtime_code_sha256"), runtime_code_hashes())):
            raise ValueError("0.5/0.4 feature, code, target or risk contract mismatch")
        markets = manifest.get("markets")
        if not isinstance(markets, dict) or set(markets) != set(MARKET_MODELS):
            raise ValueError("Both 0.5/0.4 markets must be included")
        market_manifests = {}
        for item_market, reference in markets.items():
            if not isinstance(reference, dict) or reference.get("path") != f"{item_market}/manifest.json":
                raise ValueError("0.5/0.4 market manifest path mismatch")
            item = _read(_checked_file(root, reference["path"], reference.get("sha256")))
            if (item.get("market") != item_market or item.get("model_name") != MARKET_MODELS[item_market]
                    or item.get("seeds") != list(SEEDS) or not isinstance(item.get("fold"), str)
                    or not item["fold"] or not _same(item.get("risk_flags"), RISK_FLAGS)):
                raise ValueError("0.5/0.4 market/model/seed/risk contract mismatch")
            market_manifests[item_market] = item
        selected = market_manifests[market]
        calibration, policy = selected.get("calibration"), selected.get("policy")
        if not isinstance(calibration, dict) or set(calibration) != {"success", "stop"}:
            raise ValueError("0.5/0.4 requires success and optional stop calibration")
        _calibrator(calibration["success"])
        if market == "domestic":
            _calibrator(calibration["stop"])
        elif calibration["stop"] is not None:
            raise ValueError("Binary US model cannot invent a stop probability")
        if not _same(policy, {"threshold": .5, "stop_probability_cap": 1.}):
            raise ValueError("Policy differs from the fixed >50% research contract")
        members = selected.get("members")
        if not isinstance(members, list) or len(members) != len(SEEDS):
            raise ValueError("0.5/0.4 requires exactly three native model members")
        versions = _versions()
        models = []
        for seed, member in zip(SEEDS, members):
            if (not isinstance(member, dict) or type(member.get("seed")) is not int or member["seed"] != seed
                    or member.get("path") != f"{market}/seed{seed}.cbm"
                    or member.get("sidecar_path") != f"{market}/seed{seed}.cbm.json"):
                raise ValueError("0.5/0.4 member order/path mismatch")
            native = _checked_file(root, member["path"], member.get("sha256"))
            sidecar = _checked_file(root, member["sidecar_path"], member.get("sidecar_sha256"))
            sidecar_metadata = _read(sidecar)
            if (sidecar_metadata.get("feature_count") != 184
                    or sidecar_metadata.get("model_sha256") != member["sha256"]):
                raise ValueError("Native member feature/checksum contract mismatch")
            models.append(load_model(selected["model_name"], native))
        self._models = tuple(models)
        self.market, self.model_name = market, selected["model_name"]
        self._calibration, self._policy = copy.deepcopy(calibration), copy.deepcopy(policy)
        self._metadata = {
            "title": TITLE, "model_name": self.model_name, "architecture": self.model_name,
            "market": market, "version": BUNDLE_VERSION, "bundle_version": BUNDLE_VERSION,
            "bundle_manifest_sha256": manifest_digest,
            "market_manifest_sha256": markets[market]["sha256"],
            "seeds": list(SEEDS), "fold": selected["fold"], "feature_count": 184,
            "buy_threshold": .5, "policy": copy.deepcopy(policy), "target": TARGET,
            "take_profit_pct": .5, "stop_loss_pct": .4,
            "historical_representation": SEMANTICS["historical_representation"],
            "candidate_entry_representation": SEMANTICS["candidate_entry_representation"],
            "path_limitation": PATH_LIMITATION, "warnings": list(WARNINGS),
            "runtime_versions": versions, "training_versions": copy.deepcopy(manifest.get("training_versions")),
            "research_results": copy.deepcopy(selected.get("research_results")),
            "source": copy.deepcopy(selected.get("source")), **RISK_FLAGS,
        }

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def predict(self, bars, current_price=None):
        """30 completed OHLCV bars + virtual entry; never target-day HLCV."""
        if isinstance(current_price, (bool, np.bool_)):
            raise ValueError("current_price must be a finite positive numeric scalar")
        try:
            value = Decimal(str(current_price))
            quote = float(value)
        except (InvalidOperation, TypeError, ValueError, OverflowError) as error:
            raise ValueError("current_price must be a finite positive numeric scalar") from error
        if not value.is_finite() or value <= 0 or not math.isfinite(quote * 1.005) or quote <= 0:
            raise ValueError("current_price must be a finite positive numeric scalar")
        historical = np.asarray(bars)
        if historical.shape != (30, 5) or historical.dtype.kind not in "fiu":
            raise ValueError("Exactly 30 numeric completed OHLCV bars are required")
        # Training's immutable raw window bank stores OHLCV in float32, while
        # target OPEN / virtual query prices are float64. Match that input
        # representation before recomputing the sixteen historical barriers:
        # exact decimal touches can straddle the boundary after float32 storage.
        # Validate BEFORE conversion so rounding cannot disguise invalid OHLC.
        entries = np.array([quote], dtype=np.float64)
        mark1_selective_features._validate(historical[None], entries, True)
        with np.errstate(over="ignore", invalid="ignore"):
            stored_history = historical.astype(np.float32)
        features = features_from_history(stored_history[None], entries, validate=True)
        raw = [predict_raw(model, self.model_name, features) for model in self._models]
        has_stop = self.model_name == "cat_joint6"
        for member in raw:
            if (not isinstance(member, dict) or set(member) != {"success_logits", "stop_logits"}
                    or np.asarray(member["success_logits"]).shape != (1,)
                    or not np.isfinite(member["success_logits"]).all()
                    or (has_stop and (np.asarray(member["stop_logits"]).shape != (1,)
                                     or not np.isfinite(member["stop_logits"]).all()))
                    or (not has_stop and member["stop_logits"] is not None)):
                raise ValueError("Native ensemble returned invalid raw logits")
        mean = math.fsum(float(item["success_logits"][0]) for item in raw) / len(SEEDS)
        success = float(calibrated_probability([mean], self._calibration["success"])[0])
        stop = None
        if has_stop:
            stop_mean = math.fsum(float(item["stop_logits"][0]) for item in raw) / len(SEEDS)
            stop = float(calibrated_probability([stop_mean], self._calibration["stop"])[0])
        selected = success > .5 and (stop is None or stop <= self._policy["stop_probability_cap"])
        return {
            "title": TITLE, "version": BUNDLE_VERSION, "market": self.market,
            "model_name": self.model_name, "probability_success": success, "probability_stop": stop,
            "predicts_success": bool(selected), "selected_research": bool(selected),
            "buy_threshold": .5, "policy_threshold": self._policy["threshold"],
            "stop_probability_cap": self._policy["stop_probability_cap"],
            "take_profit_pct": .5, "stop_loss_pct": .4, "candidate_entry_price": quote,
            "candidate_take_price": quote * 1.005, "candidate_stop_price": quote * .996,
            "target": TARGET, "path_limitation": PATH_LIMITATION,
            "entry_matches_evaluated_type": False, "inference_device": "cpu",
            "bundle_manifest_sha256": self._metadata["bundle_manifest_sha256"],
            "ensemble_method": "mean_raw_logits_then_platt", "warnings": list(WARNINGS), **RISK_FLAGS,
        }
