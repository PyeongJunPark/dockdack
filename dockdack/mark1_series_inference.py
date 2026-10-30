"""Sealed, broker-free CPU inference for selected CUDA Mark1.5--1.7 models.

Each score estimates a hypothetical next-session open-to-close opportunity.
This research bundle failed its 2022 development comparison; threshold
crossing is a *candidate* and never, by itself, an order authorization.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence
import zipfile

import numpy as np
import torch

from . import mark1_4_evolution, mark1_series_models
from .mark1_4_evolution import normalize_windows
from .mark1_series_models import (
    SCORE_UNITS, VARIANTS, _CrossSectionScorer, _ImageScorer, _RecurrentScorer,
)


TITLE = "mark1-series"
MARKETS = ("domestic", "us")
_MANIFEST_SHA256 = "5c08bf4ea52d2ed0ef4a7cb0be45b2c970c39fe423a79b3657be869c586f791e"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SYMBOL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,31}\Z")
_EXCHANGES = {"domestic": frozenset({"KRX"}),
              "us": frozenset({"ND", "NY", "NA"})}
_ARCHITECTURES = {
    "mark1.5": "LSTM 30x5 sequence regressor",
    "mark1.6": "30x5 price-volume tile 2D CNN binary net-return classifier",
    "mark1.7": "same-session cross-sectional attention pairwise ranker",
}


def _sha256(path: Path, *, max_bytes: int) -> str:
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"Missing or oversized Mark1 series artifact: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate Mark1 series JSON field")
        value[key] = item
    return value


def _read_json(path: Path, *, max_bytes: int) -> dict:
    _sha256(path, max_bytes=max_bytes)
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    json.dumps(result, allow_nan=False)
    if not isinstance(result, dict):
        raise ValueError("Mark1 series artifact must be a JSON object")
    return result


def _number(value, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError("Invalid Mark1 series numeric metadata")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid Mark1 series numeric metadata") from exc
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError("Nonfinite Mark1 series numeric metadata")
    return result


def _check_source_code(manifest: dict) -> None:
    expected = manifest.get("scoring_code_sha256")
    modules = {
        "mark1_series_models.py": mark1_series_models,
        "mark1_4_evolution.py": mark1_4_evolution,
    }
    if not isinstance(expected, dict) or set(expected) != set(modules):
        raise ValueError("Mark1 series scoring code provenance is incomplete")
    for name, module in modules.items():
        checksum = expected[name]
        if (_HEX.fullmatch(checksum) is None or
                _sha256(Path(module.__file__), max_bytes=256 * 1024) != checksum):
            raise ValueError(f"Mark1 series scoring code changed: {name}")


def _validate_symbols(symbols, *, market: str) -> tuple[tuple[str, str], ...]:
    try:
        identities = tuple(tuple(pair) for pair in symbols)
    except (TypeError, ValueError) as exc:
        raise ValueError("Mark1 series requires (symbol, exchange) identities") from exc
    if (not identities or len(set(identities)) != len(identities)
            or any(len(pair) != 2 or not isinstance(pair[0], str)
                   or _SYMBOL.fullmatch(pair[0]) is None
                   or not isinstance(pair[1], str)
                   or pair[1] not in _EXCHANGES[market]
                   for pair in identities)):
        raise ValueError("Invalid or repeated Mark1 series symbol/exchange identity")
    return identities


def _validate_windows(windows, count: int) -> np.ndarray:
    try:
        raw = np.asarray(windows, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Mark1 series requires numeric completed OHLCV bars") from exc
    if (raw.shape != (count, 30, 5) or not np.isfinite(raw).all()
            or np.any(raw[:, :, :4] <= 0) or np.any(raw[:, :, 4] < 0)
            or np.any(raw[:, :, 1] < raw[:, :, [0, 2, 3]].max(axis=2))
            or np.any(raw[:, :, 2] > raw[:, :, [0, 1, 3]].min(axis=2))):
        raise ValueError("Mark1 series requires exactly 30 valid, completed OHLCV bars")
    return raw


def _load_state(path: Path, model: torch.nn.Module) -> None:
    # Validate the ZIP index before NumPy decompresses any member. A compact
    # checkpoint can otherwise claim arbitrarily large uncompressed arrays.
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (not members or len(members) > 64
                or sum(member.file_size for member in members) > 4 * 1024 * 1024
                or any(member.file_size > 1024 * 1024 or not member.filename.endswith(".npy")
                       or "/" in member.filename or "\\" in member.filename
                       for member in members)):
            raise ValueError("Mark1 series checkpoint structure is invalid")
    with np.load(path, allow_pickle=False) as saved:
        reference = model.state_dict()
        if set(saved.files) != set(reference):
            raise ValueError("Mark1 series checkpoint weights are incomplete")
        state = {}
        for name, expected in reference.items():
            array = np.asarray(saved[name])
            if (array.shape != tuple(expected.shape) or array.dtype != np.float32
                    or not np.isfinite(array).all()):
                raise ValueError("Mark1 series checkpoint shape, dtype or values are invalid")
            state[name] = torch.from_numpy(array.copy())
    model.load_state_dict(state, strict=True)


class Mark1SeriesPredictor:
    """Offline single-variant scorer; caller validates dates and candidate list.

    Mark1.7 requires *all* eligible candidates from one target session passed
    together in one score_many call. Its score changes with the batch, so it
    deliberately has no single-symbol inference method.
    """

    def __init__(self, bundle_root: str | Path, market: str, variant: str):
        version = variant
        if market not in MARKETS or version not in VARIANTS:
            raise ValueError("Unknown Mark1 series market or version")
        root = Path(bundle_root).absolute()
        if (not root.is_dir() or root.is_symlink()
                or any(path.is_symlink() for path in root.iterdir())):
            raise ValueError("Mark1 series bundle is missing or contains linked artifacts")
        seal_path = root / "manifest.sha256"
        if not seal_path.is_file() or seal_path.stat().st_size > 128:
            raise ValueError("Missing Mark1 series manifest seal")
        seal = seal_path.read_text(encoding="ascii").strip()
        if (seal != _MANIFEST_SHA256 or _HEX.fullmatch(seal) is None
                or _sha256(root / "manifest.json", max_bytes=128 * 1024) != seal):
            raise ValueError("Mark1 series manifest checksum mismatch")
        manifest = _read_json(root / "manifest.json", max_bytes=128 * 1024)
        if (manifest.get("schema_version") != 1 or manifest.get("title") != TITLE
                or manifest.get("versions") != list(VARIANTS)
                or manifest.get("lookback") != 30
                or manifest.get("bar_columns") != ["open", "high", "low", "close", "volume"]
                or manifest.get("entry") != "next_session_open"
                or manifest.get("exit") != "same_session_close"
                or manifest.get("cost_bps") != 20.0
                or manifest.get("threshold_comparison") != "strict_greater_than"
                or manifest.get("training_device") != "cuda"
                or manifest.get("research_only") is not True
                or manifest.get("research_qualified") is not False
                or manifest.get("deployment_allowed") is not False
                or not isinstance(manifest.get("markets"), dict)
                or set(manifest["markets"]) != set(MARKETS)):
            raise ValueError("Mark1 series bundle identity or semantics mismatch")
        _check_source_code(manifest)
        files = {"manifest.json", "manifest.sha256"}
        for item_market in MARKETS:
            item = manifest["markets"][item_market]
            if (not isinstance(item, dict) or item.get("catalog_point_in_time") is not False
                    or item.get("selected_universe_size") != 100
                    or not isinstance(item.get("selected_symbols"), list)
                    or len(item["selected_symbols"]) != 100
                    or not isinstance(item.get("models"), dict)
                    or set(item["models"]) != set(VARIANTS)):
                raise ValueError(f"Invalid Mark1 series market metadata: {item_market}")
            universe = _validate_symbols(
                [(row.get("symbol"), row.get("exchange"))
                 for row in item["selected_symbols"] if isinstance(row, dict)],
                market=item_market,
            )
            if len(universe) != 100:
                raise ValueError("Mark1 series training universe is incomplete")
            reference_file = f"{item_market}-reference.npz"
            if (item.get("reference_file") != reference_file
                    or _HEX.fullmatch(item.get("reference_sha256", "")) is None
                    or _sha256(root / reference_file, max_bytes=2 * 1024 * 1024)
                    != item["reference_sha256"]):
                raise ValueError("Mark1 series reference vectors are invalid")
            files.add(reference_file)
            for item_version in VARIANTS:
                spec = item["models"][item_version]
                if not isinstance(spec, dict):
                    raise ValueError("Mark1 series selected model metadata is invalid")
                seed = spec.get("seed")
                expected_stem = f"{item_market}-{item_version}-seed{seed}"
                model_file = f"{expected_stem}-model.json"
                weights_file = f"{expected_stem}-weights.npz"
                if (type(seed) is not int or seed < 0
                        or spec.get("strategy_id") != f"mark1-{item_version.split('.')[1]}-prototype"
                        or spec.get("signal_phase") != "preopen"
                        or spec.get("exit_after_sessions") != 0
                        or spec.get("exit_timing") != "preclose"
                        or type(spec.get("cpu_gpu_audit_candidate_rows")) is not int
                        or spec["cpu_gpu_audit_candidate_rows"] < 100
                        or type(spec.get("cpu_gpu_observed_threshold_flips")) is not int
                        or spec["cpu_gpu_observed_threshold_flips"] < 0
                        or spec.get("score_unit") != SCORE_UNITS[item_version]
                        or spec.get("model_file") != model_file
                        or spec.get("weights_file") != weights_file
                        or any(_HEX.fullmatch(spec.get(field, "")) is None for field in
                               ("model_sha256", "weights_sha256", "source_report_sha256"))
                        or _sha256(root / model_file, max_bytes=256 * 1024)
                        != spec["model_sha256"]
                        or _sha256(root / weights_file, max_bytes=256 * 1024)
                        != spec["weights_sha256"]
                        or not 0 < _number(spec.get("target_train_candidate_coverage")) < 1
                        or _number(spec.get("development_2022_net_return_after_20bps")) >= 0
                        or type(spec.get("development_2022_exact_fills")) is not int
                        or spec["development_2022_exact_fills"] < 1):
                    raise ValueError("Mark1 series model/policy provenance mismatch")
                _number(spec.get("frozen_numeric_score_threshold"))
                if (_number(spec.get("cpu_gpu_observed_max_abs_score_delta")) < 0
                        or _number(spec.get("cpu_numeric_guard_margin"), positive=True)
                        <= _number(spec["cpu_gpu_observed_max_abs_score_delta"])):
                    raise ValueError("Mark1 series CPU numeric guard is invalid")
                files.update((model_file, weights_file))
        if {path.name for path in root.iterdir()} != files:
            raise ValueError("Mark1 series bundle file set is incomplete or unexpected")
        item = manifest["markets"][market]
        spec = item["models"][version]
        info = _read_json(root / spec["model_file"], max_bytes=256 * 1024)
        if (info.get("variant") != version or info.get("seed") != spec["seed"]
                or info.get("score_unit") != SCORE_UNITS[version]
                or info.get("architecture") != _ARCHITECTURES[version]
                or info.get("device") != "cuda" or info.get("cost_bps") != 20.0
                or info.get("research_only") is not True
                or info.get("deployment_allowed") is not False
                or info.get("checkpoint_file") != spec["weights_file"]
                or info.get("checkpoint_sha256") != spec["weights_sha256"]
                or not str(info.get("train_first_target", "")).startswith("2018-")
                or not str(info.get("train_last_target", "")).startswith("2020-")):
            raise ValueError("Mark1 series saved model contract mismatch")
        scaler = info.get("feature_scaler")
        if (not isinstance(scaler, dict)
                or scaler.get("fitted_on") != "observed training rows only"
                or scaler.get("post_scale_clip") != [-6, 6]):
            raise ValueError("Mark1 series feature scaler provenance mismatch")
        mean = np.asarray(scaler.get("mean"), dtype=np.float64)
        std = np.asarray(scaler.get("std"), dtype=np.float64)
        if (mean.shape != (5,) or std.shape != (5,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or np.any(std <= 0)):
            raise ValueError("Mark1 series feature scaler is invalid")
        hidden = info.get("hidden")
        if type(hidden) is not int or hidden < 8 or hidden > 256 or hidden % 4:
            raise ValueError("Mark1 series hidden width is invalid")
        if version == "mark1.5":
            if info.get("recurrent_cell") != "lstm":
                raise ValueError("Mark1.5 recurrent cell identity changed")
            model: torch.nn.Module = _RecurrentScorer(hidden, "lstm")
        elif version == "mark1.6":
            model = _ImageScorer(hidden)
        else:
            model = _CrossSectionScorer(hidden)
        _load_state(root / spec["weights_file"], model)
        self._model = model.cpu().eval()
        self._mean = mean
        self._std = std
        self._market = market
        self._version = version
        self._threshold = _number(spec["frozen_numeric_score_threshold"])
        self._numeric_guard_margin = _number(spec["cpu_numeric_guard_margin"], positive=True)
        self._coverage = _number(spec["target_train_candidate_coverage"])
        self._seed = spec["seed"]
        self._universe = frozenset((row["symbol"], row["exchange"])
                                   for row in item["selected_symbols"])
        self._metadata = {
            "title": TITLE, "version": version, "market": market,
            "strategy_id": spec["strategy_id"],
            "signal_phase": "preopen",
            "exit_after_sessions": 0,
            "exit_timing": "preclose",
            "seed": self._seed, "score_metric": SCORE_UNITS[version],
            "score_unit": SCORE_UNITS[version],
            "frozen_numeric_score_threshold": self._threshold,
            "cpu_numeric_guard_margin": self._numeric_guard_margin,
            "cpu_gpu_observed_max_abs_score_delta": spec["cpu_gpu_observed_max_abs_score_delta"],
            "cpu_gpu_observed_threshold_flips": spec["cpu_gpu_observed_threshold_flips"],
            "cpu_gpu_audit_candidate_rows": spec["cpu_gpu_audit_candidate_rows"],
            "numeric_guard_basis": "observed full-history CPU versus saved CUDA score delta; not a future guarantee",
            "target_train_candidate_coverage": self._coverage,
            "threshold_comparison": "strict_greater_than",
            "lookback": 30,
            "target": "next_session_open_to_same_session_close_after_20bp_roundtrip_cost",
            "cost_bps": 20.0,
            "requires_same_session_full_cross_section": version == "mark1.7",
            "score_is_calibrated_probability": False,
            "selected_universe_size": 100,
            "current_universe_selected_externally": True,
            "out_of_training_universe_allowed": True,
            "catalog_point_in_time": False,
            "development_2022_net_return_after_20bps": spec["development_2022_net_return_after_20bps"],
            "research_only": True, "research_qualified": False,
            "deployment_allowed": False,
            "bundle_manifest_sha256": seal,
            "source_report_sha256": spec["source_report_sha256"],
            "model_sha256": spec["model_sha256"],
            "weights_sha256": spec["weights_sha256"],
        }

    @property
    def metadata(self) -> dict:
        return copy.deepcopy(self._metadata)

    def score_many(self, windows, symbols: Sequence[tuple[str, str]]) -> list[dict]:
        """Score one prevalidated same-date candidate set of completed t bars.

        The caller must verify session dates, completed status, and that all
        current TOP100 candidates are present. Mark1.7 scores are contextual:
        splitting its batch or adding/removing symbols changes every score.
        """
        identities = _validate_symbols(symbols, market=self._market)
        if self._version == "mark1.7" and len(identities) < 2:
            raise ValueError("Mark1.7 requires a complete same-date cross-section")
        raw = _validate_windows(windows, len(identities))
        normalized = normalize_windows(raw)
        values = np.clip((normalized - self._mean) / self._std, -6, 6).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Mark1 series feature transform produced nonfinite values")
        with torch.inference_mode():
            if self._version == "mark1.7":
                batched = torch.from_numpy(values[None, ...])
                padding = torch.zeros((1, len(identities)), dtype=torch.bool)
                scores = self._model(batched, padding)[0].numpy()
            else:
                scores = self._model(torch.from_numpy(values)).numpy()
                if self._version == "mark1.6":
                    scores = torch.sigmoid(torch.from_numpy(scores)).numpy()
        if scores.shape != (len(identities),) or not np.isfinite(scores).all():
            raise ValueError("Mark1 series model produced invalid scores")
        output = []
        for (symbol, exchange), score in zip(identities, scores):
            raw_above = bool(score > self._threshold)
            uncertain = bool(abs(float(score) - self._threshold) <= self._numeric_guard_margin)
            guarded_above = raw_above and not uncertain
            output.append({
                "title": TITLE, "version": self._version, "market": self._market,
                "strategy_id": self._metadata["strategy_id"],
                "signal_phase": "preopen",
                "exit_after_sessions": 0,
                "exit_timing": "preclose",
                "symbol": symbol, "exchange": exchange,
                "seed": self._seed, "score": float(score),
                "score_metric": SCORE_UNITS[self._version],
                "score_unit": SCORE_UNITS[self._version],
                "score_is_calibrated_probability": False,
                "frozen_numeric_score_threshold": self._threshold,
                "cpu_numeric_guard_margin": self._numeric_guard_margin,
                "raw_above_frozen_threshold": raw_above,
                "uncertain_numeric_boundary": uncertain,
                "above_frozen_threshold": guarded_above,
                "signal_decision": ("HOLD_NUMERIC_BOUNDARY" if uncertain else
                                    "BUY_CANDIDATE" if guarded_above else "HOLD"),
                "target_train_candidate_coverage": self._coverage,
                "scored_symbol_count": len(identities),
                "requires_same_session_full_cross_section": self._version == "mark1.7",
                "in_training_universe": (symbol, exchange) in self._universe,
                "out_of_training_universe": (symbol, exchange) not in self._universe,
                "target": self._metadata["target"], "cost_bps": 20.0,
                "price_independent": True,
                "research_only": True, "research_qualified": False,
                "deployment_allowed": False,
                "bundle_manifest_sha256": self._metadata["bundle_manifest_sha256"],
            })
        return output

    def score_one(self, window, symbol: str, exchange: str) -> dict:
        if self._version == "mark1.7":
            raise ValueError("Mark1.7 has no standalone single-symbol score")
        return self.score_many([window], [(symbol, exchange)])[0]


# The runtime adapter uses this concise public name. Keep the longer name for
# previously written research checks; both resolve to the identical class.
MarkSeriesPredictor = Mark1SeriesPredictor
