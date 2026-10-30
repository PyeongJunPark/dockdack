"""Frozen, offline Mark1.4 E5-domestic / E1-US daily-window inference.

The score is a research signal about a hypothetical next-session open-to-close
trade. This module has no broker, quote, order, or portfolio dependency. A
threshold crossing is only a candidate; it is not a probability or an order.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import torch

from . import mark1_4_evolution, mark1_4_followup_models
from .mark1_4_evolution import normalize_windows
from .mark1_4_followup_models import FEATURE_NAMES, _SmallScorer, extract_engineered_features


TITLE = "mark1.4"
MARKETS = ("domestic", "us")
TARGET = "next_session_open_to_same_session_close_after_20bp_roundtrip_cost"
_MANIFEST_SHA256 = "7f75d25bbc33ebd2f96205182f415ec1f3f343acefe8a24f54fe21baed4005b3"
_EXPECTED = {
    "domestic": {
        "experiment": "e5", "variant": "e5_features", "seed": 43,
        "coverage": 0.02, "threshold": 0.451765888929366,
        "metric": "predicted_net_return_percent",
        "model_sha256": "439205fcd1262539c5449754f40fc5446f1a45689217f78481a49c8b57cbf36f",
        "report_sha256": "082bbac668eaafd13b5c17060b81e9805902d9f5a28f472d76621365e442019d",
        "architecture": "20-to-16-to-8-to-1 tanh MLP",
    },
    "us": {
        "experiment": "e1", "variant": "e1_rank", "seed": 41,
        "coverage": 0.005, "threshold": 0.6831178557872781,
        "metric": "unscaled_pairwise_ranking_score",
        "model_sha256": "d90143d017e7c6c7d2b159364462e1280c8e337e77b1dde4d0c4dd6329c107b9",
        "report_sha256": "bbbd8d92d4698b0a2f7d7715ee1bbf3c978f8047a74fac9ab993bde2cc505d9a",
        "architecture": "150-to-8-to-1 tanh same-session pairwise ranker",
    },
}
_RESEARCH_CODE_SHA256 = {
    "mark1_4_followup_models.py": "4ae7a7746e7616e5b835418f47d404c9944f2d8f9917252c324202fa374e094c",
    "mark1_4_evolution.py": "81b17fc137cf011d03a117efe6392c4bfb424cfb38d5ab37b56116ad565cd834",
}
_BUNDLE_FILES = {
    "manifest.json", "manifest.sha256",
    "domestic-seed43-model.json", "us-seed41-model.json",
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SYMBOL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,31}\Z")
_EXCHANGES = {"domestic": frozenset({"KRX"}),
              "us": frozenset({"ND", "NY", "NA"})}


def _sha256(path: Path, *, max_bytes: int) -> str:
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"Missing or oversized Mark1.4 artifact: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_pairs(items: list[tuple[str, object]]) -> dict:
    data = {}
    for key, value in items:
        if key in data:
            raise ValueError("Duplicate Mark1.4 JSON field")
        data[key] = value
    return data


def _read_json(path: Path, *, max_bytes: int) -> dict:
    _sha256(path, max_bytes=max_bytes)
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError("Mark1.4 artifact must be a JSON object")
    return value


def _check_code() -> None:
    for name, module in (("mark1_4_followup_models.py", mark1_4_followup_models),
                         ("mark1_4_evolution.py", mark1_4_evolution)):
        if _sha256(Path(module.__file__), max_bytes=256 * 1024) != _RESEARCH_CODE_SHA256[name]:
            raise ValueError(f"Mark1.4 scoring code changed: {name}")


def _numeric_vector(value, length: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid Mark1.4 {name}") from exc
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"Invalid Mark1.4 {name}")
    return array


def _validated_window_batch(windows, count: int) -> np.ndarray:
    try:
        raw = np.asarray(windows, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Mark1.4 requires numeric completed OHLCV bars") from exc
    if (raw.shape != (count, 30, 5) or not np.isfinite(raw).all()
            or np.any(raw[:, :, :4] <= 0) or np.any(raw[:, :, 4] < 0)
            or np.any(raw[:, :, 1] < raw[:, :, [0, 2, 3]].max(axis=2))
            or np.any(raw[:, :, 2] > raw[:, :, [0, 1, 3]].min(axis=2))):
        raise ValueError("Mark1.4 requires exactly 30 valid, completed OHLCV bars per symbol")
    return raw


class Mark14Predictor:
    """Read the sealed research bundle and score members of its fixed universe."""

    def __init__(self, bundle_root: str | Path, market: str):
        if market not in MARKETS:
            raise ValueError("Mark1.4 market must be domestic or us")
        root = Path(bundle_root).absolute()
        if (not root.is_dir() or root.is_symlink()
                or any(path.is_symlink() for path in root.iterdir())):
            raise ValueError("Mark1.4 bundle is missing or contains linked artifacts")
        if {path.name for path in root.iterdir()} != _BUNDLE_FILES:
            raise ValueError("Mark1.4 bundle file set is incomplete or unexpected")
        _check_code()
        seal_path = root / "manifest.sha256"
        if not seal_path.is_file() or seal_path.stat().st_size > 128:
            raise ValueError("Missing Mark1.4 manifest seal")
        seal = seal_path.read_text(encoding="ascii").strip()
        if (seal != _MANIFEST_SHA256 or _HEX.fullmatch(seal) is None
                or _sha256(root / "manifest.json", max_bytes=128 * 1024) != seal):
            raise ValueError("Mark1.4 manifest checksum mismatch")
        manifest = _read_json(root / "manifest.json", max_bytes=128 * 1024)
        if (manifest.get("schema_version") != 1 or manifest.get("title") != TITLE
                or manifest.get("research_only") is not True
                or manifest.get("deployment_allowed") is not False
                or manifest.get("lookback") != 30
                or manifest.get("bar_columns") != ["open", "high", "low", "close", "volume"]
                or manifest.get("entry") != "next_session_open"
                or manifest.get("exit") != "same_session_close"
                or manifest.get("cost_bps") != 20.0
                or manifest.get("threshold_comparison") != "strict_greater_than"
                or manifest.get("research_code_sha256") != _RESEARCH_CODE_SHA256
                or not isinstance(manifest.get("markets"), dict)
                or set(manifest["markets"]) != set(MARKETS)):
            raise ValueError("Mark1.4 bundle identity or semantics mismatch")
        for item_market in MARKETS:
            item = manifest["markets"][item_market]
            expected = _EXPECTED[item_market]
            name = f"{item_market}-seed{expected['seed']}-model.json"
            if (not isinstance(item, dict) or item.get("experiment") != expected["experiment"]
                    or item.get("variant") != expected["variant"]
                    or item.get("seed") != expected["seed"]
                    or item.get("score_metric") != expected["metric"]
                    or item.get("target_train_candidate_coverage") != expected["coverage"]
                    or item.get("frozen_numeric_score_threshold") != expected["threshold"]
                    or item.get("model_file") != name
                    or item.get("model_sha256") != expected["model_sha256"]
                    or item.get("source_report_sha256") != expected["report_sha256"]
                    or item.get("catalog_point_in_time") is not False
                    or not isinstance(item.get("selected_symbols"), list)
                    or len(item["selected_symbols"]) != 100
                    or _sha256(root / name, max_bytes=256 * 1024) != expected["model_sha256"]):
                raise ValueError(f"Mark1.4 {item_market} model/policy provenance mismatch")
            pairs = [(entry.get("symbol"), entry.get("exchange"))
                     for entry in item["selected_symbols"] if isinstance(entry, dict)]
            if (len(pairs) != 100 or len(set(pairs)) != 100
                    or any(not isinstance(symbol, str) or not symbol
                           or not isinstance(exchange, str) or not exchange
                           for symbol, exchange in pairs)):
                raise ValueError(f"Mark1.4 {item_market} universe is invalid")
        item = manifest["markets"][market]
        expected = _EXPECTED[market]
        model_info = _read_json(root / item["model_file"], max_bytes=256 * 1024)
        if (model_info.get("variant") != expected["variant"]
                or model_info.get("seed") != expected["seed"]
                or model_info.get("score_formula") != expected["metric"]
                or model_info.get("architecture") != expected["architecture"]
                or model_info.get("research_only") is not True
                or model_info.get("deployment_allowed") is not False
                or model_info.get("score_uses_only_completed_30_bars") is not True
                or model_info.get("selection_or_calibration_in_this_module") is not False
                or model_info.get("cost_bps") != 20.0
                or not str(model_info.get("train_first_target", "")).startswith("2018-")
                or not str(model_info.get("train_last_target", "")).startswith("2020-")
                or model_info.get("feature_names") != (list(FEATURE_NAMES) if market == "domestic" else None)):
            raise ValueError("Mark1.4 saved model contract mismatch")
        feature_count = 20 if market == "domestic" else 150
        scaler = model_info.get("feature_scaler")
        if (not isinstance(scaler, dict)
                or scaler.get("fitted_on") != "observed training rows only"
                or scaler.get("post_scale_clip") != [-6.0, 6.0]):
            raise ValueError("Mark1.4 feature scaler provenance mismatch")
        mean = _numeric_vector(scaler.get("mean"), feature_count, "scaler mean")
        std = _numeric_vector(scaler.get("std"), feature_count, "scaler std")
        if np.any(std <= 0):
            raise ValueError("Mark1.4 feature scaler has nonpositive standard deviation")
        model = _SmallScorer(feature_count, expected["variant"])
        state = model_info.get("model_state")
        if not isinstance(state, dict) or set(state) != set(model.state_dict()):
            raise ValueError("Mark1.4 saved model weights are incomplete")
        tensors = {}
        for name, reference in model.state_dict().items():
            try:
                values = np.asarray(state[name], dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Mark1.4 saved model weights are invalid") from exc
            if values.shape != tuple(reference.shape) or not np.isfinite(values).all():
                raise ValueError("Mark1.4 saved model weights have wrong shape or values")
            tensors[name] = torch.from_numpy(values.copy())
        model.load_state_dict(tensors, strict=True)
        self._model = model.cpu().eval()
        self._mean = mean
        self._std = std
        self._market = market
        self._metric = expected["metric"]
        self._threshold = expected["threshold"]
        self._seed = expected["seed"]
        self._coverage = expected["coverage"]
        self._experiment = expected["experiment"]
        self._universe = frozenset((row["symbol"], row["exchange"])
                                   for row in item["selected_symbols"])
        self._metadata = {
            "title": TITLE, "version": "1.4", "market": market,
            "experiment": self._experiment, "seed": self._seed,
            "score_metric": self._metric,
            "score_unit": "percent" if market == "domestic" else "arbitrary_rank_score",
            "target_train_candidate_coverage": self._coverage,
            "frozen_numeric_score_threshold": self._threshold,
            "threshold_comparison": "strict_greater_than",
            "lookback": 30, "target": TARGET, "cost_bps": 20.0,
            "selected_universe_size": 100,
            "current_universe_selected_externally": True,
            "out_of_training_universe_allowed": True,
            "catalog_point_in_time": False,
            "research_only": True, "research_qualified": False,
            "deployment_allowed": False,
            "bundle_manifest_sha256": seal,
            "source_report_sha256": expected["report_sha256"],
            "model_sha256": expected["model_sha256"],
        }

    @property
    def metadata(self) -> dict:
        return copy.deepcopy(self._metadata)

    def score_many(self, windows, symbols: Sequence[tuple[str, str]]) -> list[dict]:
        """Score registered symbols from exactly 30 already completed daily bars.

        ``symbols`` holds exact ``(symbol, exchange)`` market identities.
        The current TOP100 may differ from the frozen training 100; the
        difference is reported for each score. Dates/completion and current
        TOP100 membership must be checked by the caller first.
        """
        try:
            identities = tuple(tuple(pair) for pair in symbols)
        except (TypeError, ValueError) as exc:
            raise ValueError("Mark1.4 requires (symbol, exchange) identities") from exc
        if (not identities or any(len(pair) != 2
                                  or not isinstance(pair[0], str)
                                  or _SYMBOL.fullmatch(pair[0]) is None
                                  or not isinstance(pair[1], str)
                                  or pair[1] not in _EXCHANGES[self._market]
                                  for pair in identities)
                or len(set(identities)) != len(identities)):
            raise ValueError("Mark1.4 symbol/exchange identity is invalid or repeated")
        raw = _validated_window_batch(windows, len(identities))
        if self._market == "domestic":
            features = extract_engineered_features(raw)
        else:
            features = normalize_windows(raw).reshape(len(raw), -1)
        scaled = np.clip((features - self._mean) / self._std, -6.0, 6.0).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Mark1.4 feature transform produced nonfinite values")
        with torch.inference_mode():
            scores = self._model(torch.from_numpy(scaled)).squeeze(1).numpy()
        if scores.shape != (len(identities),) or not np.isfinite(scores).all():
            raise ValueError("Mark1.4 model produced invalid scores")
        return [
            {
                "title": TITLE, "version": "1.4", "market": self._market,
                "symbol": symbol, "exchange": exchange,
                "experiment": self._experiment, "seed": self._seed,
                "score": float(score), "score_metric": self._metric,
                "score_unit": "percent" if self._market == "domestic" else "arbitrary_rank_score",
                "frozen_numeric_score_threshold": self._threshold,
                "above_frozen_threshold": bool(score > self._threshold),
                "target_train_candidate_coverage": self._coverage,
                "scored_symbol_count": len(identities),
                "selected_universe_size": 100,
                "in_training_universe": (symbol, exchange) in self._universe,
                "out_of_training_universe": (symbol, exchange) not in self._universe,
                "target": TARGET, "cost_bps": 20.0,
                "price_independent": True,
                "research_only": True, "research_qualified": False,
                "deployment_allowed": False,
                "bundle_manifest_sha256": self._metadata["bundle_manifest_sha256"],
            }
            for (symbol, exchange), score in zip(identities, scores)
        ]

    def score_one(self, window, symbol: str, exchange: str) -> dict:
        return self.score_many([window], [(symbol, exchange)])[0]

    def predict(self, bars, *, symbol: str, exchange: str, current_price=None) -> dict:
        """GUI-friendly alias; current quote never enters this daily-bar model."""
        return self.score_one(bars, symbol, exchange)
