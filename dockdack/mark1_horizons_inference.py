"""Sealed, broker-free inference for frozen E4 H3/H5 horizon research.

Mark1.11 and Mark1.12 intentionally reuse the *same* market-specific v2
genome and threshold. They differ only in intended holding period/cadence.
Their historical comparison assumed Hth-session CLOSE, not an intraday exit.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from . import mark1_4_evolution, mark1_4_sparse
from .mark1_4_evolution import GENOME_SIZE
from .mark1_4_sparse import score_sparse_genome


TITLE = "mark1-horizons"
VARIANTS = {"mark1.11": 3, "mark1.12": 5}
MARKETS = ("domestic", "us")
_MANIFEST_SHA256 = "fa622b4bbe504a6a523c7a5bdf3416cba327cdb7d4c348b92e698b97214340dc"
# Rounded up beyond observed full-history CPU-vs-GPU maximum score deviations.
_NUMERIC_MARGIN = {"domestic": 0.000005, "us": 0.000002}
_OBSERVED_MAX_DELTA = {"domestic": 0.0000026226043701171875,
                       "us": 0.0000011920928955078125}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SYMBOL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,31}\Z")
_EXCHANGES = {"domestic": frozenset({"KRX"}),
              "us": frozenset({"ND", "NY", "NA"})}


def _sha256(path: Path, *, max_bytes: int) -> str:
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"Missing or oversized Mark1 horizon artifact: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate Mark1 horizon JSON field")
        value[key] = item
    return value


def _read_json(path: Path, *, max_bytes: int) -> dict:
    _sha256(path, max_bytes=max_bytes)
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    json.dumps(result, allow_nan=False)
    if not isinstance(result, dict):
        raise ValueError("Mark1 horizon artifact must be a JSON object")
    return result


def _number(value) -> float:
    if isinstance(value, bool):
        raise ValueError("Invalid Mark1 horizon numeric metadata")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid Mark1 horizon numeric metadata") from exc
    if not np.isfinite(result):
        raise ValueError("Nonfinite Mark1 horizon numeric metadata")
    return result


def _identities(symbols, market: str) -> tuple[tuple[str, str], ...]:
    try:
        pairs = tuple(tuple(pair) for pair in symbols)
        distinct = len(set(pairs)) == len(pairs)
    except (TypeError, ValueError) as exc:
        raise ValueError("Mark1 horizon requires (symbol, exchange) identities") from exc
    if (not pairs or not distinct
            or any(len(pair) != 2 or not isinstance(pair[0], str)
                   or _SYMBOL.fullmatch(pair[0]) is None
                   or not isinstance(pair[1], str)
                   or pair[1] not in _EXCHANGES[market]
                   for pair in pairs)):
        raise ValueError("Invalid or repeated Mark1 horizon symbol/exchange identity")
    return pairs


def _windows(windows, count: int) -> np.ndarray:
    try:
        raw = np.asarray(windows, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Mark1 horizon requires numeric completed OHLCV bars") from exc
    if (raw.shape != (count, 30, 5) or not np.isfinite(raw).all()
            or np.any(raw[:, :, :4] <= 0) or np.any(raw[:, :, 4] < 0)
            or np.any(raw[:, :, 1] < raw[:, :, [0, 2, 3]].max(axis=2))
            or np.any(raw[:, :, 2] > raw[:, :, [0, 1, 3]].min(axis=2))):
        raise ValueError("Mark1 horizon requires exactly 30 valid, completed OHLCV bars")
    return raw


class MarkHorizonPredictor:
    """Score a current candidate batch; no horizon fills or orders are inferred."""

    def __init__(self, bundle_root: str | Path, market: str, variant: str):
        if market not in MARKETS or variant not in VARIANTS:
            raise ValueError("Unknown Mark1 horizon market or variant")
        root = Path(bundle_root).absolute()
        if (not root.is_dir() or root.is_symlink()
                or any(path.is_symlink() for path in root.iterdir())):
            raise ValueError("Mark1 horizon bundle is missing or contains linked artifacts")
        seal_path = root / "manifest.sha256"
        if not seal_path.is_file() or seal_path.stat().st_size > 128:
            raise ValueError("Missing Mark1 horizon manifest seal")
        seal = seal_path.read_text(encoding="ascii").strip()
        if (seal != _MANIFEST_SHA256 or _HEX.fullmatch(seal) is None
                or _sha256(root / "manifest.json", max_bytes=128 * 1024) != seal):
            raise ValueError("Mark1 horizon manifest checksum mismatch")
        manifest = _read_json(root / "manifest.json", max_bytes=128 * 1024)
        if (manifest.get("schema_version") != 1 or manifest.get("title") != TITLE
                or manifest.get("variants") != list(VARIANTS)
                or manifest.get("lookback") != 30
                or manifest.get("bar_columns") != ["open", "high", "low", "close", "volume"]
                or manifest.get("entry") != "next_session_open"
                or manifest.get("backtest_exit") != "Hth_scheduled_session_close_entry_session_is_day_1"
                or manifest.get("backtest_entry_stride_sessions") != "equals_horizon_no_overlap"
                or manifest.get("runtime_exit") != "Hth_scheduled_session_intraday_not_backtest_equivalent"
                or manifest.get("cost_bps") != 20.0
                or manifest.get("research_only") is not True
                or manifest.get("research_qualified") is not False
                or manifest.get("deployment_allowed") is not False
                or manifest.get("new_posthoc_model_not_e4_selected_winner") is not True
                or manifest.get("seed_selection") !=
                "highest_original_2018_2021_train_fitness_only_among_41_42_43"
                or not isinstance(manifest.get("markets"), dict)
                or set(manifest["markets"]) != set(MARKETS)):
            raise ValueError("Mark1 horizon bundle identity or semantics mismatch")
        code = manifest.get("scoring_code_sha256")
        modules = {"mark1_4_sparse.py": mark1_4_sparse,
                   "mark1_4_evolution.py": mark1_4_evolution}
        if not isinstance(code, dict) or set(code) != set(modules):
            raise ValueError("Mark1 horizon scoring code provenance is incomplete")
        for name, module in modules.items():
            if (_HEX.fullmatch(code[name]) is None
                    or _sha256(Path(module.__file__), max_bytes=256 * 1024) != code[name]):
                raise ValueError(f"Mark1 horizon scoring code changed: {name}")
        files = {"manifest.json", "manifest.sha256"}
        for item_market in MARKETS:
            item = manifest["markets"][item_market]
            if (not isinstance(item, dict) or item.get("catalog_point_in_time") is not False
                    or item.get("selected_seed") != 43
                    or not isinstance(item.get("selected_symbols"), list)
                    or len(item["selected_symbols"]) != 100
                    or not isinstance(item.get("champions"), dict)
                    or set(item["champions"]) != {"41", "42", "43"}
                    or not isinstance(item.get("models"), dict)
                    or set(item["models"]) != set(VARIANTS)):
                raise ValueError(f"Invalid Mark1 horizon market metadata: {item_market}")
            universe = _identities(
                [(row.get("symbol"), row.get("exchange"))
                 for row in item["selected_symbols"] if isinstance(row, dict)], item_market)
            if len(universe) != 100:
                raise ValueError("Mark1 horizon frozen universe is incomplete")
            reference_file = f"{item_market}-reference.npz"
            if (item.get("reference_file") != reference_file
                    or _HEX.fullmatch(item.get("reference_sha256", "")) is None
                    or _sha256(root / reference_file, max_bytes=2 * 1024 * 1024)
                    != item["reference_sha256"]):
                raise ValueError("Mark1 horizon reference vectors are invalid")
            files.add(reference_file)
            fitnesses = {}
            for seed in (41, 42, 43):
                info = item["champions"][str(seed)]
                file = f"{item_market}-seed{seed}-champion.json"
                if (not isinstance(info, dict) or info.get("seed") != seed
                        or info.get("champion_file") != file
                        or _HEX.fullmatch(info.get("champion_sha256", "")) is None
                        or _HEX.fullmatch(info.get("genome_sha256", "")) is None
                        or _HEX.fullmatch(info.get("source_report_sha256", "")) is None
                        or _sha256(root / file, max_bytes=256 * 1024)
                        != info["champion_sha256"]):
                    raise ValueError("Mark1 horizon champion provenance mismatch")
                fitnesses[seed] = _number(info.get("train_fitness"))
                _number(info.get("frozen_numeric_score_threshold"))
                files.add(file)
            if max((fitnesses[seed], -seed) for seed in fitnesses) != (fitnesses[43], -43):
                raise ValueError("Mark1 horizon seed choice is not train-only maximum")
            for name, horizon in VARIANTS.items():
                spec = item["models"][name]
                if (not isinstance(spec, dict)
                        or spec.get("strategy_id") !=
                        ("mark1-11-prototype" if horizon == 3 else "mark1-12-prototype")
                        or spec.get("horizon_sessions") != horizon
                        or spec.get("backtest_entry_stride_sessions") != horizon
                        or spec.get("backtest_exit_ordinal_offset_from_entry") != horizon - 1
                        or spec.get("runtime_exit_timing") != "Hth_scheduled_session_intraday"
                        or spec.get("runtime_not_backtest_equivalent") is not True
                        or spec.get("selected_seed") != 43
                        or spec.get("score_metric") != "frozen_v2_sparse_neural_rank_score"
                        or spec.get("frozen_numeric_score_threshold") !=
                        item["champions"]["43"]["frozen_numeric_score_threshold"]):
                    raise ValueError("Mark1 horizon model timing or threshold mismatch")
        if {path.name for path in root.iterdir()} != files:
            raise ValueError("Mark1 horizon bundle file set is incomplete or unexpected")
        item = manifest["markets"][market]
        spec = item["models"][variant]
        info = item["champions"]["43"]
        champion = _read_json(root / info["champion_file"], max_bytes=256 * 1024)
        genome = np.asarray(champion.get("genome"), dtype=np.float32)
        if (champion.get("model") != "mark1-4-v2-sparse-random-neural-evolution"
                or champion.get("research_only") is not True
                or champion.get("deployment_allowed") is not False
                or champion.get("genome_size") != GENOME_SIZE
                or genome.shape != (GENOME_SIZE,) or not np.isfinite(genome).all()
                or hashlib.sha256(genome.tobytes()).hexdigest() != info["genome_sha256"]
                or _number(champion.get("frozen_train_numeric_threshold")) !=
                spec["frozen_numeric_score_threshold"]
                or champion.get("threshold_provenance", {}).get("source") != "train_scores_only"
                or champion["threshold_provenance"].get("validation_or_test_used") is not False):
            raise ValueError("Mark1 horizon selected genome is invalid")
        self._genome = genome
        self._market = market
        self._variant = variant
        self._horizon = VARIANTS[variant]
        self._threshold = _number(spec["frozen_numeric_score_threshold"])
        self._margin = _NUMERIC_MARGIN[market]
        self._universe = frozenset((row["symbol"], row["exchange"])
                                   for row in item["selected_symbols"])
        self._metadata = {
            "title": TITLE, "version": variant, "market": market,
            "strategy_id": spec["strategy_id"],
            "seed": 43,
            "score_metric": spec["score_metric"],
            "score_unit": "arbitrary_rank_score",
            "frozen_numeric_score_threshold": self._threshold,
            "cpu_numeric_guard_margin": self._margin,
            "cpu_gpu_observed_max_abs_score_delta": _OBSERVED_MAX_DELTA[market],
            "horizon_sessions": self._horizon,
            "entry_session_is_day_1": True,
            "backtest_entry_stride_sessions": self._horizon,
            "backtest_exit_ordinal_offset_from_entry": self._horizon - 1,
            "backtest_exit_timing": "Hth_scheduled_session_close",
            "runtime_exit_timing": "Hth_scheduled_session_intraday",
            "runtime_not_backtest_equivalent": True,
            "validation_2022_net_return_after_20bps": spec["validation_2022_net_return_after_20bps"],
            "validation_2022_exact_fills": spec["validation_2022_exact_fills"],
            "research_only": True, "research_qualified": False,
            "deployment_allowed": False,
            "bundle_manifest_sha256": seal,
            "champion_sha256": info["champion_sha256"],
            "source_e4_report_sha256": manifest["source_e4_report_sha256"],
        }

    @property
    def metadata(self) -> dict:
        return copy.deepcopy(self._metadata)

    def score_many(self, windows, symbols: Sequence[tuple[str, str]]) -> list[dict]:
        """Score completed t windows; caller verifies dates/current candidate set."""
        pairs = _identities(symbols, self._market)
        raw = _windows(windows, len(pairs))
        scores = score_sparse_genome(raw, self._genome, device="cpu")
        if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
            raise ValueError("Mark1 horizon scorer produced invalid scores")
        output = []
        for (symbol, exchange), score in zip(pairs, scores):
            raw_above = bool(score > self._threshold)
            uncertain = bool(abs(float(score) - self._threshold) <= self._margin)
            safe_above = raw_above and not uncertain
            output.append({
                "title": TITLE, "version": self._variant, "market": self._market,
                "strategy_id": self._metadata["strategy_id"],
                "symbol": symbol, "exchange": exchange,
                "seed": 43, "score": float(score),
                "score_metric": self._metadata["score_metric"],
                "score_unit": "arbitrary_rank_score",
                "frozen_numeric_score_threshold": self._threshold,
                "raw_above_frozen_threshold": raw_above,
                "uncertain_numeric_boundary": uncertain,
                "above_frozen_threshold": safe_above,
                "signal_decision": ("HOLD_NUMERIC_BOUNDARY" if uncertain else
                                    "BUY_CANDIDATE" if safe_above else "HOLD"),
                "cpu_numeric_guard_margin": self._margin,
                "horizon_sessions": self._horizon,
                "entry_session_is_day_1": True,
                "backtest_entry_stride_sessions": self._horizon,
                "backtest_exit_ordinal_offset_from_entry": self._horizon - 1,
                "backtest_exit_timing": "Hth_scheduled_session_close",
                "runtime_exit_timing": "Hth_scheduled_session_intraday",
                "runtime_not_backtest_equivalent": True,
                "scored_symbol_count": len(pairs),
                "in_training_universe": (symbol, exchange) in self._universe,
                "out_of_training_universe": (symbol, exchange) not in self._universe,
                "research_only": True, "research_qualified": False,
                "deployment_allowed": False,
                "bundle_manifest_sha256": self._metadata["bundle_manifest_sha256"],
            })
        return output

    def score_one(self, window, symbol: str, exchange: str) -> dict:
        return self.score_many([window], [(symbol, exchange)])[0]
