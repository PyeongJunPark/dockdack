"""Sealed Mark1.9 uncertainty and Mark1.10 robust-GA research inference.

Both methods score 30 *completed* daily bars for hypothetical next-session
open-to-close selection. A threshold crossing is not an order or a calibrated
chance of profit. This module has no broker, account, or database dependency.
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

from dockdack import mark1_4_evolution, mark1_4_followup_models, mark1_4_sparse
from dockdack.mark1_4_evolution import normalize_windows
from dockdack.mark1_4_followup_models import _SmallScorer
from dockdack.mark1_4_sparse import score_sparse_genome


VERSIONS = ("1.9", "1.10")
MARKETS = ("domestic", "us")
_SEALS = {
    "1.9": "208b05fb8b4423c843138956a1942f24d4879962aec1a3a9c03fd85842a6f5ac",
    "1.10": "c23970f24009c83bb494ce87804587874faa26ad8909f0b89bae5c2103e59a41",
}
_EXPECTED = {
    "1.9": {
        "domestic": (42, -1.2470741236209881),
        "us": (43, -0.8052668985724445),
    },
    "1.10": {
        "domestic": (41, 1.9375118432461458),
        "us": (41, 2.0943155987308897),
    },
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SYMBOL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,31}\Z")
_EXCHANGES = {"domestic": frozenset({"KRX"}),
              "us": frozenset({"ND", "NY", "NA"})}


def _sha(path: Path, limit: int = 4 * 1024 * 1024) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
        raise ValueError(f"Missing, linked, or oversized model artifact: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_json(path: Path) -> dict:
    _sha(path)

    def collect(pairs):
        item = {}
        for key, value in pairs:
            if key in item:
                raise ValueError(f"Duplicate JSON field in {path.name}")
            item[key] = value
        return item

    data = json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=collect,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"Nonfinite JSON value: {value}")))
    if not isinstance(data, dict):
        raise ValueError("Expected model JSON object")
    return data


def _windows(values, count: int) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Numeric completed OHLCV windows required") from exc
    if (raw.shape != (count, 30, 5) or not np.isfinite(raw).all()
            or np.any(raw[:, :, :4] <= 0) or np.any(raw[:, :, 4] < 0)
            or np.any(raw[:, :, 1] < raw[:, :, [0, 2, 3]].max(axis=2))
            or np.any(raw[:, :, 2] > raw[:, :, [0, 1, 3]].min(axis=2))):
        raise ValueError("Expected 30 valid, completed OHLCV bars per symbol")
    return raw


class MarkSpecialPredictor:
    def __init__(self, bundle_root: str | Path, market: str, version: str):
        if version not in VERSIONS or market not in MARKETS:
            raise ValueError("Expected Mark1.9/1.10 domestic or us")
        root = Path(bundle_root).absolute()
        if not root.is_dir() or root.is_symlink() or any(
                item.is_symlink() for item in root.iterdir()):
            raise ValueError("Bundle folder missing or linked")
        seed, threshold = _EXPECTED[version][market]
        suffix = "model.json" if version == "1.9" else "champion.npz"
        expected_files = {"manifest.json", "manifest.sha256"}
        expected_files.update(f"{name}-seed{_EXPECTED[version][name][0]}-{suffix}"
                              for name in MARKETS)
        expected_files.update(f"{name}-audit.npz" for name in MARKETS)
        if {path.name for path in root.iterdir()} != expected_files:
            raise ValueError("Incomplete or unexpected bundle file set")
        seal = (root / "manifest.sha256").read_text(encoding="ascii").strip()
        if (seal != _SEALS[version] or _HEX.fullmatch(seal) is None or
                _sha(root / "manifest.json") != seal):
            raise ValueError("Model manifest seal mismatch")
        manifest = _unique_json(root / "manifest.json")
        if (manifest.get("schema_version") != 1 or
                manifest.get("title") != f"mark{version}" or
                manifest.get("research_only") is not True or
                manifest.get("deployment_allowed") is not False or
                manifest.get("lookback") != 30 or
                manifest.get("bar_columns") != ["open", "high", "low", "close", "volume"] or
                manifest.get("entry") != "next_session_open" or
                manifest.get("exit") != "same_session_close" or
                manifest.get("cost_bps") != 20.0 or
                manifest.get("max_positions") != 10 or
                manifest.get("threshold_comparison") != "strict_greater_than" or
                set(manifest.get("markets", {})) != set(MARKETS)):
            raise ValueError("Model bundle contract mismatch")
        modules = {"mark1_4_evolution.py": mark1_4_evolution,
                   "mark1_4_followup_models.py": mark1_4_followup_models,
                   "mark1_4_sparse.py": mark1_4_sparse}
        if manifest.get("research_code_sha256") != {
                name: _sha(Path(module.__file__), limit=256 * 1024)
                for name, module in modules.items()}:
            raise ValueError("Frozen research scorer code changed")
        for name in MARKETS:
            entry = manifest["markets"][name]
            selected_seed, selected_threshold = _EXPECTED[version][name]
            model_name = f"{name}-seed{selected_seed}-{suffix}"
            audit_name = f"{name}-audit.npz"
            pairs = [(row.get("symbol"), row.get("exchange")) for row in
                     entry.get("selected_symbols", []) if isinstance(row, dict)]
            if (entry.get("seed") != selected_seed or
                    entry.get("threshold") != selected_threshold or
                    entry.get("model_file") != model_name or
                    entry.get("audit_file") != audit_name or
                    entry.get("research_only") is not True or
                    entry.get("deployment_allowed") is not False or
                    entry.get("catalog_point_in_time") is not False or
                    len(pairs) != 100 or len(set(pairs)) != 100 or
                    _sha(root / model_name) != entry.get("model_sha256") or
                    _sha(root / audit_name) != entry.get("audit_sha256")):
                raise ValueError(f"Invalid Mark{version} {name} frozen policy")
        entry = manifest["markets"][market]
        self._version, self._market, self._threshold = version, market, threshold
        self._seed = seed
        self._metric = entry["score_metric"]
        self._unit = entry["score_unit"]
        self._universe = frozenset((row["symbol"], row["exchange"])
                                   for row in entry["selected_symbols"])
        if version == "1.9":
            artifact = _unique_json(root / entry["model_file"])
            if (artifact.get("variant") != "e2_uncertainty" or
                    artifact.get("seed") != seed or
                    artifact.get("score_formula") != "mean_minus_one_sigma" or
                    artifact.get("research_only") is not True or
                    artifact.get("deployment_allowed") is not False):
                raise ValueError("Mark1.9 model metadata mismatch")
            scaler = artifact["feature_scaler"]
            self._mean = np.asarray(scaler["mean"], dtype=np.float32)
            self._std = np.asarray(scaler["std"], dtype=np.float32)
            if (self._mean.shape != (150,) or self._std.shape != (150,) or
                    not np.isfinite(self._mean).all() or
                    not np.isfinite(self._std).all() or np.any(self._std <= 0)):
                raise ValueError("Mark1.9 feature scaler invalid")
            model = _SmallScorer(150, "e2_uncertainty")
            reference = model.state_dict()
            state = artifact["model_state"]
            if set(state) != set(reference):
                raise ValueError("Mark1.9 network weights incomplete")
            tensors = {}
            for key, template in reference.items():
                values = np.asarray(state[key], dtype=np.float32)
                if values.shape != tuple(template.shape) or not np.isfinite(values).all():
                    raise ValueError("Mark1.9 network weight invalid")
                tensors[key] = torch.from_numpy(values.copy())
            model.load_state_dict(tensors, strict=True)
            self._model = model.cpu().eval()
        else:
            with np.load(root / entry["model_file"], allow_pickle=False) as saved:
                genome = np.asarray(saved["genome"], dtype=np.float32)
                frozen = float(saved["frozen_threshold"])
            if (genome.shape != (1826,) or not np.isfinite(genome).all() or
                    frozen != threshold):
                raise ValueError("Mark1.10 frozen genome invalid")
            self._genome = genome
        with np.load(root / entry["audit_file"], allow_pickle=False) as audit:
            windows = np.asarray(audit["windows"], dtype=np.float32)
            expected = np.asarray(audit["scores"], dtype=np.float32)
        if (windows.ndim != 3 or windows.shape[1:] != (30, 5) or
                expected.shape != (len(windows),)):
            raise ValueError("Model audit sample malformed")
        observed, _, _ = self._score(_windows(windows, len(windows)))
        if not np.allclose(observed, expected, atol=3e-6, rtol=1e-6):
            raise ValueError("Frozen scorer failed saved-vector audit")
        self._metadata = {
            "title": f"mark{version}", "version": version, "market": market,
            "method": ("Gaussian mean minus one sigma" if version == "1.9" else
                       "worst-year 40bp robust sparse genetic network"),
            "seed": seed, "score_metric": self._metric,
            "score_unit": self._unit,
            "frozen_numeric_score_threshold": threshold,
            "threshold_comparison": "strict_greater_than",
            "lookback": 30,
            "target": "hypothetical next-session open to same-session close",
            "cost_bps": 20.0, "max_positions": 10,
            "selected_universe_size": 100,
            "current_universe_selected_externally": True,
            "out_of_training_universe_allowed": True,
            "catalog_point_in_time": False,
            "research_only": True, "research_qualified": False,
            "deployment_allowed": False,
            "manifest_sha256": seal,
            "bundle_manifest_sha256": seal,
            "source_report_sha256": entry["source_report_sha256"],
        }

    def _score(self, windows: np.ndarray):
        if self._version == "1.10":
            scores = score_sparse_genome(windows, self._genome,
                                         device="cpu")
            return scores, None, None
        features = normalize_windows(windows).reshape(len(windows), 150)
        scaled = np.clip((features - self._mean) / self._std, -6, 6).astype(np.float32)
        with torch.inference_mode():
            output = self._model(torch.from_numpy(scaled)).numpy()
        mean = output[:, 0].astype(np.float32)
        sigma = np.exp(np.clip(output[:, 1], -3, 3)).astype(np.float32)
        return (mean - sigma).astype(np.float32), mean, sigma

    @property
    def metadata(self) -> dict:
        return copy.deepcopy(self._metadata)

    def score_many(self, windows, symbols: Sequence[tuple[str, str]]) -> list[dict]:
        """Score registered market identities; caller verifies t-bar finality."""
        try:
            identities = tuple(tuple(pair) for pair in symbols)
        except (TypeError, ValueError) as exc:
            raise ValueError("Expected (symbol, exchange) identities") from exc
        if (not identities or len(set(identities)) != len(identities) or
                any(len(pair) != 2 or not isinstance(pair[0], str) or
                    _SYMBOL.fullmatch(pair[0]) is None or
                    pair[1] not in _EXCHANGES[self._market]
                    for pair in identities)):
            raise ValueError("Invalid or duplicate market identity")
        raw = _windows(windows, len(identities))
        score, mean, sigma = self._score(raw)
        if score.shape != (len(identities),) or not np.isfinite(score).all():
            raise ValueError("Nonfinite model score")
        output = []
        for index, (symbol, exchange) in enumerate(identities):
            row = {
                "title": f"mark{self._version}", "version": self._version,
                "market": self._market, "symbol": symbol, "exchange": exchange,
                "experiment": "e2" if self._version == "1.9" else "e3",
                "seed": self._seed, "score": float(score[index]),
                "score_metric": self._metric, "score_unit": self._unit,
                "frozen_numeric_score_threshold": self._threshold,
                "above_frozen_threshold": bool(score[index] > self._threshold),
                "in_frozen_universe": (symbol, exchange) in self._universe,
                "in_training_universe": (symbol, exchange) in self._universe,
                "out_of_training_universe": (symbol, exchange) not in self._universe,
                "scored_symbol_count": len(identities),
                "selected_universe_size": 100,
                "target": self._metadata["target"], "cost_bps": 20.0,
                "price_independent": True,
                "research_only": True, "research_qualified": False,
                "deployment_allowed": False,
                "bundle_manifest_sha256": self._metadata["bundle_manifest_sha256"],
            }
            if mean is not None and sigma is not None:
                row["predicted_mean_net_percent"] = float(mean[index])
                row["predicted_sigma_percent"] = float(sigma[index])
            output.append(row)
        return output

    def score_one(self, window, symbol: str, exchange: str) -> dict:
        return self.score_many([window], [(symbol, exchange)])[0]

    def predict(self, bars, *, symbol: str, exchange: str,
                current_price=None) -> dict:
        """GUI-friendly alias; current quote never enters this 30-bar model."""
        return self.score_one(bars, symbol, exchange)


class Mark19Predictor(MarkSpecialPredictor):
    def __init__(self, bundle_root: str | Path, market: str):
        super().__init__(bundle_root, market, "1.9")


class Mark110Predictor(MarkSpecialPredictor):
    def __init__(self, bundle_root: str | Path, market: str):
        super().__init__(bundle_root, market, "1.10")
