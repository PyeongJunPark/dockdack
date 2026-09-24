"""Portable CPU probabilities for ``mark1 prototype``; never order permission.

Only the checksummed bundle and the three small local inference dependencies
are needed. Training directories, caches, databases, Torch and brokers are not
loaded. Daily-bar labels do not verify the remaining path after an intraday buy.
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

from . import mark1_metrics, mark1_selective_features, mark1_selective_models
from .mark1_metrics import calibrated_probability
from .mark1_selective_features import FEATURE_NAMES, features_from_history
from .mark1_selective_models import CLASS_NAMES, load_model, predict_raw


TITLE = "mark1 prototype"
BUNDLE_VERSION = "20260916-v1"
OWNER = "dockdack.mark1_prototype"
SCHEMA_VERSION = 1
MARKET_MODELS = {"domestic": "cat_joint6", "us": "cat_binary8"}
SEEDS = (42, 43, 44)
TARGET = "daily_high_ge_entry_1pct_and_low_gt_entry_minus_0_9pct_conservative"
PATH_LIMITATION = (
    "Whole-session conservative daily-bar event: high >= entry*1.01 and "
    "low > entry*0.991. Both-touch is failure. This is not a verified "
    "first-touch probability after an arbitrary intraday entry."
)
SEMANTICS = {
    "lookback": 30, "bar_columns": ["open", "high", "low", "close", "volume"],
    "target": TARGET, "class_names": list(CLASS_NAMES), "both_touch": "stop_first_failure",
    "take_profit_pct": 1.0, "stop_loss_pct": .9, "buy_threshold": .5,
    "buy_comparison": "strict_greater_than", "ensemble": "mean_raw_logits_then_platt",
    "path_limitation": PATH_LIMITATION,
}
RISK_FLAGS = {
    "research_only": True, "deployment_allowed": False, "research_qualified": False,
    "intraday_path_verified": False, "known_data_quality_issues": True,
}
WARNINGS = [
    "Both selected ensembles failed the frozen research qualification; this is not a profitable or deployable model.",
    "The US selected model produced no >50% signals in the reused evaluation; zero trades are not evidence of trading skill.",
    "The clean-20260916-v1 source still contains confirmed US split/reverse-split price-basis discontinuities (FCEL, BNED, BBSI); no correction or retraining was performed.",
    "Domestic data has not been certified free of corporate-action issues. Known-data-quality warnings apply to this bundle as a whole.",
    "Daily OHLC cannot recover intraday barrier order. Both-touch counts as stop first; whole-session labels do not validate an arbitrary intraday entry's remaining path.",
    "2025+ results were reused during research and are not a new independent test.",
    "Probabilities and candidate signals are research diagnostics only; this bundle grants no broker or order permission.",
]
RUNTIME_MODULES = {
    "features": mark1_selective_features,
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
    return {name: sha256_file(Path(module.__file__)) for name, module in RUNTIME_MODULES.items()}


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
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or Path(relative).is_absolute() or any(part in (".", "..") for part in relative.split("/"))
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("Unsafe artifact path or invalid SHA256")
    candidate = root / relative
    if any(part.is_symlink() for part in (candidate, *candidate.parents) if part != root.parent):
        raise ValueError("Linked bundle paths are not supported")
    if not candidate.resolve().is_relative_to(root.resolve()) or sha256_file(candidate) != digest:
        raise ValueError("Bundle artifact checksum mismatch")
    return candidate


def _versions() -> dict:
    try:
        versions = {"python": platform.python_version(), "numpy": np.__version__,
                    "catboost": package_metadata.version("catboost")}
    except package_metadata.PackageNotFoundError as error:
        raise ImportError("mark1 prototype requires the optional CatBoost CPU inference dependency") from error
    bounds = {"python": ((3, 10), (4, 0)), "numpy": ((1, 24), (3, 0)), "catboost": ((1, 2), (2, 0))}
    for name, version in versions.items():
        matched = re.match(r"^(\d+)\.(\d+)", version)
        parsed = tuple(map(int, matched.groups())) if matched else ()
        if not matched or not bounds[name][0] <= parsed < bounds[name][1]:
            raise ValueError(f"Unsupported prototype inference {name} version: {version}")
    return versions


def _calibrator(value):
    if (not isinstance(value, dict) or value.get("method") != "platt_monotone"
            or value.get("weighted") is not False or type(value.get("fit_samples")) is not int
            or value["fit_samples"] <= 0 or type(value.get("slope")) not in (int, float)
            or type(value.get("bias")) not in (int, float)):
        raise ValueError("Invalid unweighted frozen Platt calibration")
    calibrated_probability([0.], value)


class PrototypePredictor:
    """Three native CatBoost members cached on CPU, with portable provenance."""

    buy_threshold = .5

    def __init__(self, bundle_root, market):
        if market not in MARKET_MODELS:
            raise ValueError("market must be domestic or us")
        root = Path(bundle_root).absolute()
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Prototype bundle root must be an existing, unlinked directory")
        digest_path = root / "manifest.sha256"
        sha256_file(digest_path)
        manifest_digest = digest_path.read_text(encoding="ascii").strip()
        manifest_path = _checked_file(root, "manifest.json", manifest_digest)
        manifest = _read(manifest_path)
        if (manifest.get("owner") != OWNER or type(manifest.get("schema_version")) is not int
                or manifest["schema_version"] != SCHEMA_VERSION or manifest.get("title") != TITLE
                or manifest.get("version") != BUNDLE_VERSION or manifest.get("completed") is not True
                or not _same(manifest.get("semantics"), SEMANTICS)
                or not _same(manifest.get("risk_flags"), RISK_FLAGS)
                or manifest.get("warnings") != WARNINGS
                or manifest.get("feature_names") != list(FEATURE_NAMES)
                or len(FEATURE_NAMES) != 184 or manifest.get("feature_count") != 184
                or not _same(manifest.get("runtime_code_sha256"), runtime_code_hashes())):
            raise ValueError("Prototype feature, code, target or risk contract mismatch")
        markets = manifest.get("markets")
        if not isinstance(markets, dict) or set(markets) != set(MARKET_MODELS):
            raise ValueError("Both prototype markets must be included")
        market_manifests = {}
        for item_market, reference in markets.items():
            if not isinstance(reference, dict) or reference.get("path") != f"{item_market}/manifest.json":
                raise ValueError("Prototype market manifest path mismatch")
            market_path = _checked_file(root, reference["path"], reference.get("sha256"))
            item = _read(market_path)
            if (item.get("market") != item_market or item.get("model_name") != MARKET_MODELS[item_market]
                    or item.get("fold") != "walk_2024" or item.get("seeds") != list(SEEDS)
                    or not _same(item.get("risk_flags"), RISK_FLAGS)):
                raise ValueError("Prototype market/model/seed/risk contract mismatch")
            market_manifests[item_market] = item
        selected = market_manifests[market]
        calibration, policy = selected.get("calibration"), selected.get("policy")
        if not isinstance(calibration, dict) or set(calibration) != {"success", "stop"}:
            raise ValueError("Prototype requires success and optional stop calibration")
        _calibrator(calibration["success"])
        if market == "domestic":
            _calibrator(calibration["stop"])
        elif calibration["stop"] is not None:
            raise ValueError("Binary US model cannot invent a stop probability")
        # This version exports exactly the final locked policies, not a new grid.
        if not _same(policy, {"threshold": .5, "stop_probability_cap": 1.}):
            raise ValueError("Policy differs from the final locked prototype version")
        members = selected.get("members")
        if not isinstance(members, list) or len(members) != len(SEEDS):
            raise ValueError("Prototype requires exactly three native model members")
        versions = _versions()
        models = []
        for seed, member in zip(SEEDS, members):
            if (not isinstance(member, dict) or type(member.get("seed")) is not int or member["seed"] != seed
                    or member.get("path") != f"{market}/seed{seed}.cbm"
                    or member.get("sidecar_path") != f"{market}/seed{seed}.cbm.json"):
                raise ValueError("Prototype member order/path mismatch")
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
            "seeds": list(SEEDS), "fold": "walk_2024", "feature_count": 184,
            "buy_threshold": .5, "policy": copy.deepcopy(policy), "target": TARGET,
            "take_profit_pct": 1.0, "stop_loss_pct": .9,
            "path_limitation": PATH_LIMITATION, "warnings": list(WARNINGS),
            "runtime_versions": versions, "training_versions": copy.deepcopy(manifest.get("training_versions")),
            "research_results": copy.deepcopy(selected.get("research_results")), **RISK_FLAGS,
        }

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def predict(self, bars, current_price=None):
        """Accept 30 completed OHLCV bars; return research diagnostics only."""
        if isinstance(current_price, (bool, np.bool_)):
            raise ValueError("current_price must be a finite positive numeric scalar")
        try:
            value = Decimal(str(current_price))
            quote = float(value)
        except (InvalidOperation, TypeError, ValueError, OverflowError) as error:
            raise ValueError("current_price must be a finite positive numeric scalar") from error
        if not value.is_finite() or value <= 0 or not math.isfinite(quote * 1.01) or quote <= 0:
            raise ValueError("current_price must be a finite positive numeric scalar")
        historical = np.asarray(bars)
        if historical.shape != (30, 5) or historical.dtype.kind not in "fiu":
            raise ValueError("Exactly 30 numeric completed OHLCV bars are required")
        features = features_from_history(historical[None], np.array([quote]), validate=True)
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
        mean = math.fsum(float(item["success_logits"][0]) for item in raw) / 3
        success = float(calibrated_probability([mean], self._calibration["success"])[0])
        stop = None
        if has_stop:
            stop_mean = math.fsum(float(item["stop_logits"][0]) for item in raw) / 3
            stop = float(calibrated_probability([stop_mean], self._calibration["stop"])[0])
        selected = (success > .5 and success > self._policy["threshold"]
                    and (stop is None or stop <= self._policy["stop_probability_cap"]))
        return {
            "title": TITLE, "version": BUNDLE_VERSION, "market": self.market,
            "model_name": self.model_name, "probability_success": success, "probability_stop": stop,
            "predicts_success": bool(selected), "selected_research": bool(selected),
            "buy_threshold": .5, "policy_threshold": self._policy["threshold"],
            "stop_probability_cap": self._policy["stop_probability_cap"],
            "take_profit_pct": 1.0, "stop_loss_pct": .9, "candidate_entry_price": quote,
            "candidate_take_price": quote * 1.01, "candidate_stop_price": quote * .991,
            "target": TARGET, "path_limitation": PATH_LIMITATION,
            "entry_matches_evaluated_type": False, "inference_device": "cpu",
            "bundle_manifest_sha256": self._metadata["bundle_manifest_sha256"],
            "ensemble_method": "mean_raw_logits_then_platt", "warnings": list(WARNINGS), **RISK_FLAGS,
        }
