"""Read-only MK1.2 inputs and on-demand candidate-price augmentation.

The frozen source contract is never rewritten. Quarantine removes approved
windows in memory, not bars in the original bank or rows in a database. Labels
describe the whole-session +1%/-0.9% event; both-touch is a stop-first failure,
not evidence about the order of intraday transactions.
"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import sqlite3

import numpy as np
import torch

from .mark1_data import LOOKBACK, TARGET, Mark1Dataset, barrier_outcomes
from .mark1_deep_data import class_targets
from .mark1_deep_models import features_from_history, success_logit
from .research_artifacts import (ArtifactResolver, assert_no_wal, cache_key,
                                 read_json, sha256_file)
from .research_compat import load_frozen_cache_compatible


FACTORS = (1., .99, .995, 1.005, 1.01)
QUARANTINE = {"domestic": (), "us": ("FCEL", "BNED", "BBSI", "SONY")}
RECEIPT_PATH = "docs/audits/2026-09-24/research-fix-verification.json"
_CHUNK = 131_072


def _integers(value, name, *, length=None, upper=None):
    array = np.asarray(value)
    if (array.ndim != 1 or array.dtype.kind not in "iu"
            or (length is not None and len(array) != length)
            or (array.size and (np.any(array < 0)
                               or np.any(array > np.iinfo(np.int64).max)
                               or (upper is not None and np.any(array >= upper))))):
        raise ValueError(f"Invalid {name}: nonnegative in-range integer vector required")
    return array.astype(np.int64, copy=False)


def _validate_dataset(dataset):
    bars, ohlc = np.asarray(dataset.bars), np.asarray(dataset.target_ohlc)
    if (bars.ndim != 2 or bars.shape[1] != 5 or bars.dtype.kind not in "iuf"
            or ohlc.ndim != 2 or ohlc.shape[1] != 4 or ohlc.dtype.kind not in "iuf"):
        raise ValueError("Expected numeric raw [B,5] OHLCV and target [N,4] OHLC")
    count = len(ohlc)
    starts = _integers(dataset.starts, "starts", length=count)
    if starts.size and (len(bars) < LOOKBACK or np.any(starts > len(bars) - LOOKBACK)):
        raise ValueError("Historical window extends beyond the raw bar bank")
    dates = np.asarray(dataset.target_dates)
    if dates.ndim != 1 or dates.dtype.kind not in "iu" or len(dates) != count:
        raise ValueError("Target dates must be an integer vector aligned with windows")
    _integers(dataset.symbol_ids, "symbol IDs", length=count)
    # Bounded temporary arrays: never materialize every 30-bar window.
    for array in (bars, ohlc):
        for first in range(0, len(array), _CHUNK):
            chunk = array[first:first + _CHUNK]
            prices = chunk[:, :4]
            if (not np.isfinite(chunk).all() or np.any(prices <= 0)
                    or np.any(prices[:, 1] < prices.max(axis=1))
                    or np.any(prices[:, 2] > prices.min(axis=1))
                    or (chunk.shape[1] == 5 and np.any(chunk[:, 4] < 0))):
                raise ValueError("Invalid historical or target OHLCV")
            with np.errstate(over="ignore", under="ignore"):
                fp32 = chunk.astype(np.float32, copy=False)
            if not np.isfinite(fp32).all() or np.any(fp32[:, :4] <= 0):
                raise ValueError("OHLCV is not representable as finite positive FP32 prices")


def _quarantine(dataset, market):
    """Apply the predeclared symbol exclusion to every split without rebasing bars."""
    _validate_dataset(dataset)
    symbols = dataset.manifest.get("symbols")
    if not isinstance(symbols, list):
        raise ValueError("Frozen symbol identity inventory is missing")
    by_id = {}
    for row in symbols:
        if (not isinstance(row, dict) or type(row.get("symbol_id")) is not int
                or row["symbol_id"] < 0 or not isinstance(row.get("symbol"), str)
                or not row["symbol"] or row["symbol_id"] in by_id):
            raise ValueError("Ambiguous frozen symbol identity inventory")
        by_id[row["symbol_id"]] = row
    if not set(np.unique(dataset.symbol_ids)).issubset(by_id):
        raise ValueError("Window symbol ID is absent from the frozen inventory")
    excluded = [row for row in symbols if row["symbol"] in QUARANTINE[market]]
    excluded_ids = [row["symbol_id"] for row in excluded]
    keep = ~np.isin(dataset.symbol_ids, excluded_ids)
    original_indices = np.flatnonzero(keep).astype(np.int64)
    mapping = np.full(len(keep), -1, dtype=np.int64)
    mapping[original_indices] = np.arange(len(original_indices), dtype=np.int64)
    splits = {}
    for name, values in dataset.splits.items():
        indices = _integers(values, f"{name} indices", upper=len(keep))
        if np.any(indices[1:] <= indices[:-1]):
            raise ValueError("Frozen split indices must be unique and increasing")
        splits[name] = mapping[indices[keep[indices]]]
    experiment = {
        "target": TARGET, "lookback": LOOKBACK, "both_touch": "stop_first_failure",
        "quarantine_policy": {"market": market, "symbols": list(QUARANTINE[market]),
                              "scope": "all_periods_all_splits", "raw_bars_retained": True,
                              "database_modified": False},
        "raw_samples": len(keep), "eligible_samples": len(original_indices),
        "raw_bar_count": len(dataset.bars),
        "quarantine": [{"symbol": row["symbol"], "symbol_id": row["symbol_id"],
                        "excluded_samples": int(np.count_nonzero(dataset.symbol_ids == row["symbol_id"]))}
                       for row in excluded],
        "original_indices_sha256": hashlib.sha256(original_indices.astype("<i8", copy=False).tobytes()).hexdigest(),
        "cached_labels_not_used": True,
        "limitation": "Other corporate-action issues and survivorship bias are not certified absent",
    }
    manifest = deepcopy(dataset.manifest)
    manifest["symbols"] = [deepcopy(row) for row in symbols if row["symbol_id"] not in excluded_ids]
    manifest["selected_symbols"] = len(manifest["symbols"])
    manifest["mark1_2"] = deepcopy(experiment)
    fields = ({name: getattr(dataset, name)[keep]
               for name in ("starts", "target_dates", "symbol_ids", "target_ohlc")}
              if excluded_ids else {})
    return replace(dataset, **fields, splits=splits, manifest=manifest), experiment


def load_dataset(market, workspace):
    """Load a content-receipted frozen cache, then quarantine windows in memory.

    Returns ``(Mark1Dataset, unchanged_source_contract, receipt)``. The receipt
    records the relocated physical DB path under ``source.physical_path`` and
    the new exclusion contract under ``experiment``. No cache is generated.
    """
    if market not in QUARANTINE:
        raise ValueError("market must be domestic or us")
    resolver = ArtifactResolver(Path(workspace))
    source_path = resolver.location(f"outputs/mark1/selective-20260916/{market}/source.json")
    verification_path = resolver.location(RECEIPT_PATH)
    source_digest, verification_digest = sha256_file(source_path), sha256_file(verification_path)
    source = read_json(source_path)
    verification = read_json(verification_path)
    known = verification.get("read_only_database_and_cache_checks", {}).get(market, {})
    if (source.get("market") != market or source.get("target") != TARGET
            or known.get("database_sha256") != source.get("database_sha256")
            or known.get("cache_key") != cache_key(source)
            or known.get("source_contract_unchanged") is not True
            or known.get("read_before_after_verified") is not True):
        raise ValueError("Explicit frozen source/cache receipt does not match this source contract")
    dataset, unchanged_source, receipt = load_frozen_cache_compatible(
        source, market, resolver.location("outputs/mark1/cache"), resolver=resolver,
        expected_cache_sha256=known.get("cache_sha256"))
    if dataset.manifest.get("market") != market:
        raise ValueError("Cached dataset market differs from the frozen source contract")
    dataset, experiment = _quarantine(dataset, market)
    if (sha256_file(source_path) != source_digest
            or sha256_file(verification_path) != verification_digest):
        raise RuntimeError("Source contract or explicit content receipt changed while loading")
    receipt = {**receipt, "experiment": experiment,
               "source_json_sha256": source_digest, "verification_receipt_sha256": verification_digest,
               "verification_receipt_path": str(verification_path)}
    return dataset, unchanged_source, receipt


def read_sessions(database):
    """Read the frozen calendar without creating or updating a SQLite file."""
    path = Path(database).resolve(strict=True)
    assert_no_wal(path)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute("SELECT session_date FROM sessions ORDER BY ordinal").fetchall()
    assert_no_wal(path)
    dates = []
    for (value,) in rows:
        day = np.datetime64(value, "D")
        if np.isnat(day) or str(day) != value:
            raise ValueError("Invalid session date")
        dates.append(int(day.astype(np.int64)))
    result = np.asarray(dates, dtype=np.int64)
    if not len(result) or np.any(result[1:] <= result[:-1]):
        raise ValueError("Sessions must be nonempty, unique and increasing in ordinal order")
    return result


class BuildAugmentedBank:
    """One FP32 raw bar bank, small split metadata, and batch-only features.

    ``classes`` is [N,F] int64 computed by the existing float64 label function.
    The factors and tensors are on ``device``; dates/symbols/outcomes remain
    NumPy arrays. No [all windows,30,5] or augmented feature bank is allocated.
    """

    def __init__(self, dataset, splits, device, factors=FACTORS):
        _validate_dataset(dataset)
        multipliers = np.asarray(factors, dtype=np.float64)
        if (multipliers.ndim != 1 or not len(multipliers) or multipliers[0] != 1.
                or not np.isfinite(multipliers).all() or np.any(multipliers <= 0)
                or len(np.unique(multipliers)) != len(multipliers)):
            raise ValueError("Unique positive factors starting with actual-open 1.0 are required")
        with np.errstate(over="ignore", under="ignore"):
            fp32_factors = multipliers.astype(np.float32)
        if (not np.isfinite(fp32_factors).all() or np.any(fp32_factors <= 0)
                or len(np.unique(fp32_factors)) != len(fp32_factors)):
            raise ValueError("Factors must remain positive, distinct and finite in FP32")
        self.dataset, self.device = dataset, torch.device(device)
        prepared = {}
        seen = np.zeros(len(dataset.starts), dtype=bool)
        for name, values in splits.items():
            indices = _integers(values, f"{name} indices", upper=len(dataset.starts))
            if np.any(indices[1:] <= indices[:-1]):
                raise ValueError("Split indices must be unique and increasing")
            if np.any(seen[indices]):
                raise ValueError("An event cannot belong to more than one split")
            seen[indices] = True
            for first in range(0, len(indices), _CHUNK):
                entries = dataset.target_ohlc[indices[first:first + _CHUNK], 0].astype(np.float32)
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    candidates = entries[:, None] * fp32_factors[None, :]
                if not np.isfinite(candidates).all() or np.any(candidates <= 0):
                    raise ValueError("Candidate entries must remain finite and positive in FP32")
            prepared[name] = indices.copy()
        del seen
        # Explicitly budget only resident raw data/metadata; leave at least half
        # available CUDA memory for the model, activations, and transient batches.
        resident_bytes = dataset.bars.size * 4 + sum(
            len(values) * (8 + 4 + 8 * len(multipliers)) for values in prepared.values())
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            if resident_bytes > free // 2:
                raise MemoryError("Raw bank would consume over half of available CUDA memory")
        self.bars = torch.tensor(dataset.bars, dtype=torch.float32, device=self.device)
        self.offsets = torch.arange(LOOKBACK, device=self.device)
        self.factors = torch.tensor(fp32_factors, dtype=torch.float32, device=self.device)
        self.factor_values = tuple(float(value) for value in multipliers)
        self.parts = {}
        for name, indices in prepared.items():
            ohlc = dataset.target_ohlc[indices]
            outcomes = barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], ohlc[:, 0])
            classes = torch.empty((len(indices), len(multipliers)), dtype=torch.int64, device=self.device)
            for first in range(0, len(indices), _CHUNK):
                stop = first + _CHUNK
                classes[first:stop] = torch.tensor(class_targets(ohlc[first:stop], multipliers),
                                                   dtype=torch.int64, device=self.device)
            # Training features intentionally use the frozen FP32 raw history;
            # all barrier decisions above retain float64 target arithmetic.
            self.parts[name] = {
                "indices": indices, "ohlc": ohlc, "dates": dataset.target_dates[indices],
                "symbols": dataset.symbol_ids[indices], "outcomes": outcomes,
                "starts": torch.tensor(dataset.starts[indices], dtype=torch.int64, device=self.device),
                "entries": torch.tensor(ohlc[:, 0], dtype=torch.float32, device=self.device),
                "classes": classes,
            }

    def _features(self, part, index, factor_indices, device):
        index = torch.as_tensor(index, device=self.device)
        if index.ndim != 1 or index.dtype not in (torch.int32, torch.int64) or not len(index):
            raise ValueError("A nonempty one-dimensional integer batch index is required")
        starts = torch.index_select(part["starts"], 0, index)
        history = self.bars[starts[:, None] + self.offsets[None, :]].to(device)
        entries = torch.index_select(part["entries"], 0, index).to(device)
        if factor_indices is not None:
            factor_indices = torch.as_tensor(factor_indices, device=self.device)
            if factor_indices.shape != index.shape or factor_indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("One integer factor index per batch entry is required")
            entries = entries * torch.index_select(self.factors, 0, factor_indices).to(device)
        return features_from_history(history, entries)

    def features(self, part, index, factor_indices=None):
        """Construct only this batch, changing the candidate query, never bars."""
        return self._features(part, index, factor_indices, self.device)

    @torch.inference_mode()
    def logits(self, model, split, batch_size=2048, device=None, factor_index=0):
        """FP32 raw success logits for one candidate factor, in split order.

        Explicit CPU inference also constructs features on CPU. The model's
        original single device and every module's train/eval state are restored
        even on failure. Mixed-device or non-FP32 models are rejected unchanged.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(factor_index) is not int or not 0 <= factor_index < len(self.factor_values):
            raise ValueError("factor_index is outside the declared augmentation factors")
        part = self.parts[split]
        tensors = [*model.parameters(), *model.buffers()]
        devices = {value.device for value in tensors}
        if len(devices) > 1 or any(value.is_floating_point() and value.dtype != torch.float32 for value in tensors):
            raise ValueError("Inference requires a single-device FP32 model")
        original_device = next(iter(devices), torch.device("cpu"))
        inference_device = torch.device(device) if device is not None else self.device
        states = [(module, module.training) for module in model.modules()]
        result = np.empty(len(part["indices"]), dtype=np.float32)
        try:
            model.to(inference_device)
            model.eval()
            for first in range(0, len(result), batch_size):
                stop = min(first + batch_size, len(result))
                index = torch.arange(first, stop, device=self.device)
                factors = torch.full_like(index, factor_index)
                with torch.autocast(inference_device.type, enabled=False):
                    features = self._features(part, index, factors, inference_device)
                    logits = success_logit(model(features).float())
                result[first:stop] = logits.cpu().numpy()
            if not np.isfinite(result).all():
                raise ValueError("Non-finite inference")
            return result
        finally:
            try:
                model.to(original_device)
            finally:
                for module, training in states:
                    module.training = training
