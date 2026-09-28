"""Sealed, DEMO-only inference for Mark1.23-28 daily target/horizon models.

Only completed daily bars and one current quote enter the model. This module
does not fetch prices, contact a broker, create an order, or infer an intraday
path from daily highs. Scores are research proxies, not validated fill odds.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from dockdack.mark1_target_horizon_models import build_model, sequence_features
from dockdack.research_artifacts import sha256_file


EXPECTED_SPECS = {
    "mark1-23-prototype": (20, 10, 3., "gru"),
    "mark1-24-prototype": (30, 20, 3., "mlp"),
    "mark1-25-prototype": (20, 20, 4., "gru"),
    "mark1-26-prototype": (20, 20, 2., "linear"),
    "mark1-27-prototype": (20, 10, 3., "cnn"),
    "mark1-28-prototype": (10, 10, 4., "mlp"),
}
MODEL_IDS = tuple(EXPECTED_SPECS)
MARKETS = ("domestic", "us")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_FILENAME = re.compile(r"(?:domestic|us)-mark1-(?:2[3-8])-prototype\.pt\Z")
RISK_FLAGS = {"research_only": True, "research_qualified": False,
              "deployment_allowed": False, "intraday_path_verified": False}


def _path(path: Path) -> Path:
    absolute = Path(path).absolute()
    if any(item.is_symlink() or getattr(item, "is_junction", lambda: False)()
           for item in (absolute, *absolute.parents)):
        raise ValueError("Linked Mark1 target/horizon model paths are forbidden")
    return absolute


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate model manifest key")
        result[key] = value
    return result


def _read_manifest(root: Path) -> tuple[dict, str]:
    root = _path(root)
    manifest_file, seal = root / "manifest.json", root / "manifest.sha256"
    if (not root.is_dir() or not manifest_file.is_file() or not seal.is_file()
            or manifest_file.stat().st_size > 128_000 or seal.stat().st_size > 128):
        raise ValueError("Missing or oversized target/horizon bundle manifest")
    digest = seal.read_text(encoding="ascii").strip()
    if not _HEX.fullmatch(digest) or sha256_file(manifest_file) != digest:
        raise ValueError("Target/horizon bundle manifest seal mismatch")
    data = json.loads(manifest_file.read_text(encoding="utf-8"),
                      object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(
                          ValueError("Nonfinite bundle manifest")))
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or data.get("owner") != "mark1_target_horizon_v1"
            or data.get("research_only") is not True
            or data.get("deployment_allowed") is not False
            or data.get("intraday_path_verified") is not False
            or data.get("trading_mode") != "demo"
            or data.get("feature_contract") !=
                "completed_ohlcv_plus_query_log_gap_v2_6_channels"
            or data.get("training_query") != "observed_next_session_open"
            or data.get("proposed_runtime_query") != "current_intraday_price"
            or data.get("feature_module_sha256") != sha256_file(
                Path(__file__).with_name("mark1_target_horizon_models.py"))
            or not isinstance(data.get("warnings"), list)
            or len(data["warnings"]) < 3
            or not isinstance(data.get("specs"), dict)
            or set(data["specs"]) != set(MODEL_IDS)
            or not isinstance(data.get("markets"), dict)
            or set(data["markets"]) != set(MARKETS)):
        raise ValueError("Target/horizon bundle identity or risk contract mismatch")
    expected = {"manifest.json", "manifest.sha256"}
    for model_id, spec in data["specs"].items():
        if (not isinstance(spec, dict) or spec.get("lookback") not in (10, 20, 30)
                or spec.get("horizon") not in (5, 10, 20)
                or type(spec.get("target_pct")) not in (int, float)
                or not 0 < spec["target_pct"] < 100
                or spec.get("family") not in ("linear", "mlp", "cnn", "gru")
                or not isinstance(spec.get("markets"), dict)
                or set(spec["markets"]) != set(MARKETS)):
            raise ValueError("Target/horizon model specification mismatch")
        if (spec["lookback"], spec["horizon"], spec["target_pct"], spec["family"]) != EXPECTED_SPECS[model_id]:
            raise ValueError("Target/horizon model identity or rule changed")
        for market in MARKETS:
            item = spec["markets"][market]
            name = f"{market}-{model_id}.pt"
            if (not isinstance(item, dict) or item.get("weights") != name
                    or not _FILENAME.fullmatch(name)
                    or not isinstance(item.get("weights_sha256"), str)
                    or not _HEX.fullmatch(item["weights_sha256"])):
                raise ValueError("Target/horizon model checkpoint receipt mismatch")
            for field in ("normalization_center", "normalization_scale"):
                try:
                    values = np.asarray(item[field], dtype=np.float64)
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("Missing model feature normalization") from exc
                if (values.shape != (6,) or not np.isfinite(values).all()
                        or (field.endswith("scale") and np.any(values <= 0))):
                    raise ValueError("Invalid model feature normalization")
            try:
                temperature = float(item["temperature"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Missing model calibration") from exc
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("Invalid model calibration")
            expected.add(name)
    actual = {file.relative_to(root).as_posix() for file in root.rglob("*")
              if file.is_file()}
    if actual != expected or any(not item.is_file() or item.is_symlink()
                                 for item in root.iterdir()):
        raise ValueError("Unexpected or missing target/horizon bundle files")
    return data, digest


class MarkTargetHorizonPredictor:
    """Query one sealed research model with completed OHLCV and current price."""

    def __init__(self, bundle_root, market: str, model_id: str):
        if market not in MARKETS or model_id not in MODEL_IDS:
            raise ValueError("Unsupported Mark1 target/horizon market or model")
        root = _path(Path(bundle_root))
        manifest, manifest_sha256 = _read_manifest(root)
        spec = manifest["specs"][model_id]
        member = spec["markets"][market]
        weights = root / member["weights"]
        if (not weights.is_file() or weights.stat().st_size > 16 * 1024 * 1024
                or sha256_file(weights) != member["weights_sha256"]):
            raise ValueError("Target/horizon checkpoint checksum mismatch")
        payload = torch.load(weights, map_location="cpu", weights_only=True)
        if (not isinstance(payload, dict)
                or set(payload) != {"state_dict", "family", "lookback"}
                or payload["family"] != spec["family"]
                or payload["lookback"] != spec["lookback"]):
            raise ValueError("Target/horizon checkpoint identity mismatch")
        state = payload["state_dict"]
        if (not isinstance(state, dict) or not state
                or any(not isinstance(value, torch.Tensor)
                       or value.dtype != torch.float32
                       or value.device.type != "cpu"
                       or not bool(torch.isfinite(value).all())
                       for value in state.values())):
            raise ValueError("Target/horizon checkpoint tensor mismatch")
        self._model = build_model(spec["family"], spec["lookback"]).cpu().eval()
        self._model.load_state_dict(state, strict=True)
        self._center = np.asarray(member["normalization_center"], dtype=np.float32)
        self._scale = np.asarray(member["normalization_scale"], dtype=np.float32)
        self._temperature = float(member["temperature"])
        self._lookback = int(spec["lookback"])
        self._horizon = int(spec["horizon"])
        self._target_pct = float(spec["target_pct"])
        self._metadata = {
            "title": model_id.replace("-", ".", 1).replace("-prototype", " prototype"),
            "strategy_id": model_id, "market": market, "model_id": model_id,
            "model_name": f"daily target/horizon {model_id}",
            "bundle_manifest_sha256": manifest_sha256,
            "family": spec["family"], "lookback": self._lookback,
            "horizon_sessions": self._horizon, "take_profit_pct": self._target_pct,
            "warnings": list(manifest["warnings"]), **RISK_FLAGS,
        }

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    @torch.inference_mode()
    def predict(self, bars, current_price=None):
        try:
            price = float(current_price)
            raw = np.asarray(bars, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Completed bars and current price must be numeric") from exc
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Current price must be finite and positive")
        if raw.shape != (30, 5):
            raise ValueError("Exactly 30 completed daily OHLCV bars are required")
        features = sequence_features(raw[-self._lookback:][None, ...],
                                     self._lookback, np.asarray([price]))
        normalized = np.clip((features - self._center) / self._scale,
                             -8.0, 8.0).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError("Nonfinite target/horizon model input")
        output = self._model(torch.from_numpy(normalized)).cpu().numpy()[0]
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite target/horizon model output")
        logit, expected_net_return = map(float, output)
        probability = 1.0 / (1.0 + math.exp(-max(-40., min(40., logit / self._temperature))))
        candidate = probability >= .5 and expected_net_return > 0
        return {
            "title": self._metadata["title"], "strategy_id": self._metadata["strategy_id"],
            "market": self._metadata["market"], "model_name": self._metadata["model_name"],
            "probability_success": probability, "probability_stop": None,
            "expected_net_return": expected_net_return,
            "predicts_success": candidate, "selected_research": candidate,
            "buy_threshold": .5, "policy_threshold": .5,
            "candidate_entry_price": price,
            "candidate_take_price": price * (1 + self._target_pct / 100),
            "candidate_stop_price": None,
            "take_profit_pct": self._target_pct, "stop_loss_pct": None,
            "horizon_sessions": self._horizon,
            "target": "daily_first_high_target_touch_else_Hth_close_no_stop_proxy",
            "score_scope": "next_open_proxy_not_intraday_verified",
            "bundle_manifest_sha256": self._metadata["bundle_manifest_sha256"],
            "warnings": list(self._metadata["warnings"]), **RISK_FLAGS,
        }
