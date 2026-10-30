"""Hash-checked, offline inference for ten separately trained daily-proxy models.

The DEMO child process receives only completed chart bars and a current-price
query. Loading or scoring here cannot contact the broker or submit an order.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from dockdack.mark1_intraday_models import ARCHITECTURES, VARIANTS, build_model, feature_matrix
from dockdack import mark1_intraday_extra_models as extra_models


MARKETS = ("domestic", "us")
TARGET = "daily_open_to_whole_session_take_only_1pct_without_0.9pct_stop"
RISK_FLAGS = {"research_only": True, "research_qualified": False,
              "deployment_allowed": False, "intraday_path_verified": False}
EFFECTIVE_LOOKBACK = {"mark1.13": 21, "mark1.14": 30, "mark1.15": 30,
                      "mark1.16": 21, "mark1.17": 30}
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _family(variant):
    if variant in VARIANTS:
        from dockdack import mark1_intraday_models as module
        return ("dockdack.mark1_intraday", module, VARIANTS,
                ARCHITECTURES, EFFECTIVE_LOOKBACK, build_model, feature_matrix)
    if variant in extra_models.VARIANTS:
        return ("dockdack.mark1_intraday_extra", extra_models,
                extra_models.VARIANTS, extra_models.ARCHITECTURES,
                extra_models.EFFECTIVE_LOOKBACK,
                extra_models.build_model, extra_models.feature_matrix)
    raise ValueError("Unknown MK1 daily-proxy variant")


def _plain_path(path: Path) -> Path:
    absolute = Path(path).absolute()
    if any(item.is_symlink() or getattr(item, "is_junction", lambda: False)()
           for item in (absolute, *absolute.parents)):
        raise ValueError("Linked MK1 model paths are forbidden")
    return absolute


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _plain_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path, variant: str):
    owner, feature_module, variants, architectures, lookbacks, _, _ = _family(variant)
    root = _plain_path(root)
    seal = root / "manifest.sha256"
    manifest_file = root / "manifest.json"
    if (not root.is_dir() or not seal.is_file() or seal.stat().st_size > 128
            or not manifest_file.is_file() or manifest_file.stat().st_size > 128 * 1024):
        raise ValueError("Missing or oversized MK1 model manifest")
    digest = seal.read_text(encoding="ascii").strip()
    if not _HEX.fullmatch(digest) or _sha256(manifest_file) != digest:
        raise ValueError("MK1 model manifest seal mismatch")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate MK1 manifest key")
            result[key] = value
        return result
    data = json.loads(manifest_file.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite manifest")))
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or data.get("owner") != owner
            or data.get("source_kind") != "verified_clean_daily_cache"
            or data.get("target") != TARGET or data.get("history_bars") != 30
            or data.get("entry_training_basis") != "target_session_observed_open_only"
            or data.get("query_inference_basis") != "current_price_relative_to_last_completed_daily_close"
            or data.get("threshold") != .5 or data.get("take_profit_pct") != 1.
            or data.get("stop_loss_pct") != .9 or data.get("risk_flags") != RISK_FLAGS
            or not isinstance(data.get("warnings"), list) or len(data["warnings"]) < 3
            or not isinstance(data.get("variants"), dict)
            or set(data["variants"]) != set(variants)
            or not isinstance(data.get("markets"), dict)
            or set(data["markets"]) != set(MARKETS)):
        raise ValueError("MK1 model semantics or risk flags mismatch")
    if data.get("runtime_feature_code_sha256") != _sha256(Path(feature_module.__file__)):
        raise ValueError("MK1 feature code differs from sealed training version")
    expected = {"manifest.json", "manifest.sha256"}
    for market in MARKETS:
        info = data["markets"][market]
        if (not isinstance(info, dict) or info.get("market") != market
                or type(info.get("sample_count")) is not int or info["sample_count"] < 1
                or any(not isinstance(info.get(key), str) or not _HEX.fullmatch(info[key])
                       for key in ("source_database_sha256", "source_cache_sha256", "source_json_sha256"))):
            raise ValueError("MK1 market training provenance mismatch")
    for member_variant in variants:
        info = data["variants"][member_variant]
        if (not isinstance(info, dict)
                or info.get("model_id") != member_variant.replace("mark1.", "mark1-") + "-prototype"
                or tuple(info.get("feature_names", ())) != variants[member_variant]
                or (tuple(info.get("architecture", ())) if isinstance(architectures[member_variant], tuple)
                    else info.get("architecture")) != architectures[member_variant]
                or info.get("effective_lookback") != lookbacks[member_variant]
                or not isinstance(info.get("models"), dict)
                or set(info["models"]) != set(MARKETS)):
            raise ValueError("MK1 variant identity or features mismatch")
        for market in MARKETS:
            member = info["models"][market]
            name = f"{market}-{member_variant.replace('.', '_')}.pt"
            if (not isinstance(member, dict) or member.get("path") != name
                    or not isinstance(member.get("sha256"), str)
                    or not _HEX.fullmatch(member["sha256"])
                    or type(member.get("train_examples")) is not int
                    or member["train_examples"] < 1
                    or type(member.get("train_positive")) is not int
                    or not 0 <= member["train_positive"] <= member["train_examples"]
                    or type(member.get("epochs")) is not int or member["epochs"] < 1
                    or type(member.get("seed")) is not int):
                raise ValueError("MK1 member training receipt mismatch")
            expected.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != expected or any(not p.is_file() or p.is_symlink() for p in root.iterdir()):
        raise ValueError("Unexpected or missing MK1 bundle files")
    return data, digest


class MarkIntradayPredictor:
    """Daily-proxy score queried at a current price, DEMO identity only."""

    def __init__(self, bundle_root, market: str, variant: str):
        if market not in MARKETS:
            raise ValueError("Unknown MK1 market or variant")
        _, _, variants, architectures, _, model_builder, feature_fn = _family(variant)
        root = _plain_path(Path(bundle_root))
        manifest, digest = _read_manifest(root, variant)
        member = manifest["variants"][variant]["models"][market]
        path = root / member["path"]
        if (not path.is_file() or path.stat().st_size > 16 * 1024 * 1024
                or _sha256(path) != member["sha256"]):
            raise ValueError("MK1 trained checkpoint checksum mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(payload, dict)
                or set(payload) != {"variant", "market", "target", "feature_names",
                                    "architecture", "mean", "scale", "state_dict"}
                or payload["variant"] != variant or payload["market"] != market
                or payload["target"] != TARGET
                or tuple(payload["feature_names"]) != variants[variant]
                or (tuple(payload["architecture"]) if isinstance(architectures[variant], tuple)
                    else payload["architecture"]) != architectures[variant]):
            raise ValueError("MK1 trained checkpoint identity mismatch")
        for key in ("mean", "scale"):
            tensor = payload[key]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != (8,)
                    or tensor.dtype != torch.float32 or tensor.device.type != "cpu"
                    or not bool(torch.isfinite(tensor).all())
                    or (key == "scale" and not bool((tensor > 0).all()))):
                raise ValueError("MK1 feature standardization mismatch")
        self._mean = payload["mean"]
        self._scale = payload["scale"]
        self._model = model_builder(variant).cpu().eval()
        self._feature_fn = feature_fn
        state = payload["state_dict"]
        if (not isinstance(state, dict) or not state
                or any(not isinstance(value, torch.Tensor)
                       or value.dtype != torch.float32 or value.device.type != "cpu"
                       or not bool(torch.isfinite(value).all()) for value in state.values())):
            raise ValueError("MK1 model weights must be finite CPU float32 tensors")
        self._model.load_state_dict(state, strict=True)
        self._metadata = {
            "title": variant + " prototype",
            "strategy_id": variant.replace("mark1.", "mark1-") + "-prototype",
            "market": market, "variant": variant, "model_name": "daily-proxy " + variant,
            "bundle_manifest_sha256": digest, "version": "20260928-v1",
            "target": TARGET, "feature_names": list(variants[variant]),
            "warnings": list(manifest["warnings"]), **RISK_FLAGS,
        }

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    @torch.inference_mode()
    def predict(self, bars, current_price=None):
        try:
            price = float(current_price)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Current price must be finite and positive") from exc
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Current price must be finite and positive")
        raw = self._feature_fn(bars, np.array([price]), self._metadata["variant"])
        normalized = np.clip((raw - self._mean.numpy()) / self._scale.numpy(), -8., 8.)
        if not np.isfinite(normalized).all():
            raise ValueError("Nonfinite normalized MK1 query")
        score = float(torch.sigmoid(self._model(torch.from_numpy(normalized))).item())
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("MK1 model returned invalid score")
        selected = score > .5
        return {
            "title": self._metadata["title"],
            "strategy_id": self._metadata["strategy_id"],
            "version": self._metadata["version"], "market": self._metadata["market"],
            "model_name": self._metadata["model_name"],
            "probability_success": score, "probability_stop": None,
            "predicts_success": selected, "selected_research": selected,
            "buy_threshold": .5, "policy_threshold": .5, "stop_probability_cap": 1.,
            "take_profit_pct": 1., "stop_loss_pct": .9,
            "candidate_entry_price": price,
            "candidate_take_price": price * 1.01,
            "candidate_stop_price": price * .991,
            "target": TARGET, "score_scope": "daily_open_whole_session_proxy_not_intraday",
            "bundle_manifest_sha256": self._metadata["bundle_manifest_sha256"],
            "warnings": list(self._metadata["warnings"]), **RISK_FLAGS,
        }
