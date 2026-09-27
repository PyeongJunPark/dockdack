"""Sealed Mark1.3 pre-open scorer; training and brokerage are never imported here.

The eight inputs use only 30 completed daily bars. A score is a neural estimate
of next-session OPEN-to-CLOSE net return in percentage points, *not* a win
probability or a guaranteed executable return. The bundle remains research-only
and the ordinary DEMO gate owns any subsequent order decision.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import torch

from dockdack import mark1_3_research
from dockdack.mark1_3_research import DailyNetRegressor, FEATURE_NAMES, features_through_t


MARKETS = frozenset({"domestic", "us"})
NUMERIC_GUARD_PERCENT = 0.0001
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SYMBOL = re.compile(r"[A-Za-z0-9.\-]+\Z")
_MANIFEST_SHA256 = "6e51ddf466a810ee6a0213fccfa11d4b6b0ca3a4ed2e15e416d19f47375ad48f"


def _sha(path: Path, *, maximum: int) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError(f"Missing, linked or oversized Mark1.3 artifact: {path.name}")
    return sha256(path.read_bytes()).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate Mark1.3 artifact key")
        result[key] = value
    return result


def _json(path: Path, *, maximum: int) -> dict:
    _sha(path, maximum=maximum)
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    if not isinstance(value, dict):
        raise ValueError("Mark1.3 artifact must be an object")
    return value


def _array(value, shape, name) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid Mark1.3 {name}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid Mark1.3 {name}")
    return result


class Mark13Predictor:
    """CPU inference from one SHA-pinned GPU-trained model per market."""

    def __init__(self, bundle_root: str | Path, market: str):
        if market not in MARKETS:
            raise ValueError("Unknown Mark1.3 market")
        root = Path(bundle_root).absolute()
        expected_files = {"manifest.json", "manifest.sha256", "domestic.json", "us.json"}
        if (not root.is_dir() or root.is_symlink()
                or {path.name for path in root.iterdir()} != expected_files
                or any(path.is_symlink() for path in root.iterdir())):
            raise ValueError("Incomplete or linked Mark1.3 model bundle")
        seal = (root / "manifest.sha256").read_text(encoding="ascii").strip()
        if (_HEX.fullmatch(seal) is None or seal != _MANIFEST_SHA256
                or _sha(root / "manifest.json", maximum=64_000) != seal):
            raise ValueError("Mark1.3 manifest seal mismatch")
        manifest = _json(root / "manifest.json", maximum=64_000)
        if (manifest.get("schema_version") != 1
                or manifest.get("model_id") != "mark1-3-prototype"
                or manifest.get("lookback") != 30
                or manifest.get("features") != list(FEATURE_NAMES)
                or manifest.get("training_device") != "cuda"
                or manifest.get("entry") != "next_session_open_proxy"
                or manifest.get("exit") != "same_session_close_proxy"
                or manifest.get("cost_bps") != 20
                or manifest.get("research_only") is not True
                or manifest.get("deployment_allowed") is not False
                or set(manifest.get("markets", {})) != MARKETS):
            raise ValueError("Mark1.3 manifest contract mismatch")
        source = Path(mark1_3_research.__file__)
        if _sha(source, maximum=128_000) != manifest.get("feature_and_architecture_sha256"):
            raise ValueError("Mark1.3 feature or architecture code changed")
        entry = manifest["markets"][market]
        if (not isinstance(entry, dict) or entry.get("file") != market + ".json"
                or _sha(root / entry["file"], maximum=256_000) != entry.get("sha256")):
            raise ValueError("Mark1.3 market model checksum mismatch")
        artifact = _json(root / entry["file"], maximum=256_000)
        if (artifact.get("market") != market or artifact.get("model_id") != "mark1-3-prototype"
                or artifact.get("features") != list(FEATURE_NAMES)
                or artifact.get("trained_on") != "cuda"
                or artifact.get("threshold") != NUMERIC_GUARD_PERCENT):
            raise ValueError("Mark1.3 market model identity mismatch")
        self._mean = _array(artifact.get("mean"), (len(FEATURE_NAMES),), "mean")
        self._scale = _array(artifact.get("scale"), (len(FEATURE_NAMES),), "scale")
        if np.any(self._scale <= 0):
            raise ValueError("Mark1.3 scale must be positive")
        model = DailyNetRegressor()
        saved = artifact.get("state_dict")
        if not isinstance(saved, dict) or set(saved) != set(model.state_dict()):
            raise ValueError("Mark1.3 network state is incomplete")
        state = {}
        for name, reference in model.state_dict().items():
            state[name] = torch.from_numpy(_array(saved[name], tuple(reference.shape), name).copy())
        model.load_state_dict(state, strict=True)
        self._model = model.cpu().eval()
        self._market = market
        self._seal = seal
        self._selected = frozenset((row["symbol"], row["exchange"])
                                   for row in artifact.get("training_symbols", ()))
        self._metadata = {
            "model_id": "mark1-3-prototype", "title": "mark1.3 prototype",
            "market": market, "version": "1.3", "score_metric": "predicted_t_plus_1_net_open_to_close_return_percent",
            "score_unit": "percent", "score_is_calibrated_probability": False,
            "frozen_numeric_score_threshold": NUMERIC_GUARD_PERCENT,
            "numeric_guard_percent": NUMERIC_GUARD_PERCENT,
            "threshold_comparison": "strict_greater_than",
            "lookback": 30, "cost_bps": 20, "training_device": "cuda",
            "research_only": True, "research_qualified": False, "deployment_allowed": False,
            "bundle_manifest_sha256": seal,
        }

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def score_many(self, windows, symbols: Sequence[tuple[str, str]]) -> list[dict]:
        try:
            identities = tuple(tuple(pair) for pair in symbols)
            raw = np.asarray(windows, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid Mark1.3 candidate batch") from exc
        if (not identities or raw.shape != (len(identities), 30, 5)
                or len(set(identities)) != len(identities)
                or any(len(pair) != 2 or not isinstance(pair[0], str)
                       or _SYMBOL.fullmatch(pair[0]) is None
                       or not isinstance(pair[1], str) or not pair[1]
                       for pair in identities)
                or not np.isfinite(raw).all() or np.any(raw[:, :, :4] <= 0)
                or np.any(raw[:, :, 4] < 0)
                or np.any(raw[:, :, 1] < raw[:, :, [0, 2, 3]].max(axis=2))
                or np.any(raw[:, :, 2] > raw[:, :, [0, 1, 3]].min(axis=2))):
            raise ValueError("Mark1.3 needs unique symbols and exactly 30 valid completed OHLCV bars")
        features = features_through_t(raw)
        scaled = np.clip((features - self._mean) / self._scale, -8, 8).astype(np.float32)
        with torch.inference_mode():
            scores = self._model(torch.from_numpy(scaled)).numpy()
        if scores.shape != (len(identities),) or not np.isfinite(scores).all():
            raise ValueError("Mark1.3 produced invalid scores")
        return [{
            "title": "mark1.3 prototype", "version": "1.3", "market": self._market,
            "symbol": symbol, "exchange": exchange, "score": float(score),
            "score_metric": self._metadata["score_metric"], "score_unit": "percent",
            "score_is_calibrated_probability": False,
            "frozen_numeric_score_threshold": NUMERIC_GUARD_PERCENT,
            "above_frozen_threshold": bool(score > NUMERIC_GUARD_PERCENT),
            "in_training_universe": (symbol, exchange) in self._selected,
            "out_of_training_universe": (symbol, exchange) not in self._selected,
            "price_independent": True, "research_only": True,
            "research_qualified": False, "deployment_allowed": False,
            "bundle_manifest_sha256": self._seal,
        } for (symbol, exchange), score in zip(identities, scores)]

    def score_one(self, window, symbol: str, exchange: str) -> dict:
        return self.score_many([window], [(symbol, exchange)])[0]
