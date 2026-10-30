"""Opt-in bounded-memory cache/feature storage, separate from sealed trainers.

Existing NPZ/features and model formulas are untouched. Completed new stores
are immutable, content-verified .npy mappings; interrupted staging is retained.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from typing import Callable
import zipfile

import numpy as np

from dockdack.research_artifacts import (ArtifactResolver, cache_key, read_json,
                                       sha256_file, write_new_json)


ARRAY_NAMES = ("bars", "starts", "target_dates", "symbol_ids", "target_ohlc",
               "split_train", "split_tune", "split_calibration", "split_selection", "split_test")
MEMBERS = frozenset(name + ".npy" for name in (*ARRAY_NAMES, "manifest", "cache_config"))


def _new_stage(destination: Path) -> tuple[Path, Path]:
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to overwrite an existing research artifact")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError("Existing unlinked artifact parent required")
    return destination, Path(tempfile.mkdtemp(prefix=f".{destination.name}-building-", dir=destination.parent))


def _close_mmap(array: np.ndarray) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _array_info(path: Path) -> dict:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if array.dtype.hasobject:
            raise ValueError("Object arrays are forbidden")
        return {"sha256": sha256_file(path), "shape": list(array.shape), "dtype": str(array.dtype), "bytes": path.stat().st_size}
    finally:
        _close_mmap(array)


def unpack_frozen_cache(source_contract: dict, market: str, cache_dir: Path,
                        destination: Path, *, resolver: ArtifactResolver,
                        expected_cache_sha256: str) -> dict:
    """Stream NPZ members to a NEW mmap store; never materialize the whole bank."""
    destination = resolver.location(str(destination))
    database = resolver.database(source_contract, market)
    cache = Path(cache_dir) / f"{market}-{cache_key(source_contract)}.npz"
    if not cache.is_file() or cache.is_symlink() or sha256_file(cache) != expected_cache_sha256:
        raise ValueError("Frozen NPZ must match an explicit prior content receipt")
    with zipfile.ZipFile(cache) as archive:
        names = archive.namelist()
        if len(names) != len(MEMBERS) or set(names) != MEMBERS:
            raise ValueError("Unexpected, duplicate or unsafe NPZ members")
        # Only the two tiny JSON-bearing members are loaded; numerical arrays stream.
        if archive.getinfo("cache_config.npy").file_size > 1024 * 1024:
            raise ValueError("Oversized cache contract")
        with archive.open("cache_config.npy") as stream:
            contract = json.loads(str(np.load(stream, allow_pickle=False).item()))
        if contract != source_contract:
            raise ValueError("Frozen cache logical contract mismatch")
        destination, stage = _new_stage(destination)
        files = {}
        for name in names:
            with archive.open(name) as incoming, (stage / name).open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            files[name] = _array_info(stage / name)
    resolver.recheck(database, database=True)
    if sha256_file(cache) != expected_cache_sha256:
        raise RuntimeError("NPZ changed during streaming conversion")
    manifest = {"format": "dockdack-mmap-v1", "completed": True, "source_contract": source_contract,
                "source_database": database.receipt(), "source_cache_sha256": expected_cache_sha256,
                "cache_key": cache_key(source_contract), "files": files,
                "logical_identity_unchanged": True, "model_math_changed": False}
    write_new_json(stage / "completed.json", manifest)
    stage.rename(destination)
    return manifest


def load_mmap_cache(directory: Path, source_contract: dict) -> SimpleNamespace:
    """Verify immutable bytes before mapping; the caller owns returned mappings."""
    directory = Path(directory).resolve()
    manifest = read_json(directory / "completed.json")
    if (manifest.get("format") != "dockdack-mmap-v1" or manifest.get("completed") is not True
            or manifest.get("source_contract") != source_contract or set(manifest.get("files", {})) != MEMBERS):
        raise ValueError("Incomplete or mismatched mmap cache")
    arrays = {}
    try:
        for name, expected in manifest["files"].items():
            path = directory / name
            if path.is_symlink() or _array_info(path) != expected:
                raise ValueError("Changed mmap array")
            arrays[name[:-4]] = np.load(path, mmap_mode="r", allow_pickle=False)
        if json.loads(str(arrays["cache_config"].item())) != source_contract:
            raise ValueError("Embedded frozen contract differs")
        result = SimpleNamespace(**{name: arrays[name] for name in ARRAY_NAMES[:5]},
            splits={name[6:]: arrays[name] for name in ARRAY_NAMES[5:]},
            manifest=json.loads(str(arrays["manifest"].item())))
        _close_mmap(arrays["manifest"])
        _close_mmap(arrays["cache_config"])
        return result
    except BaseException:
        for value in arrays.values():
            _close_mmap(value)
        raise


def write_feature_chunks(dataset, indices, destination: Path, *,
                         feature_function: Callable, feature_count: int,
                         feature_identity: dict, batch_size: int = 8192) -> dict:
    """Write each original-function batch straight to disk, with no whole-fold allocation.

    Callers supply immutable input/source identities in feature_identity. This
    generic storage primitive is not a replacement training/evaluation protocol.
    """
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices < 0)
            or np.any(indices >= len(dataset.starts)) or type(batch_size) is not int or batch_size < 1
            or type(feature_count) is not int or feature_count < 1 or not len(indices)):
        raise ValueError("Nonempty valid indices and positive dimensions required")
    if not isinstance(feature_identity, dict) or not feature_identity:
        raise ValueError("An explicit feature/input identity is required")
    destination, stage = _new_stage(destination)
    target = stage / "features.npy"
    output = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(len(indices), feature_count))
    offsets = np.arange(30)
    try:
        for first in range(0, len(indices), batch_size):
            picked = indices[first:first + batch_size]
            starts = np.asarray(dataset.starts[picked])
            if np.any(starts < 0) or np.any(starts + 29 >= len(dataset.bars)):
                raise ValueError("History index is outside packed bars")
            history = dataset.bars[starts[:, None] + offsets[None, :]]
            features = np.asarray(feature_function(history, dataset.target_ohlc[picked, 0]), dtype=np.float32)
            if features.shape != (len(picked), feature_count) or not np.isfinite(features).all():
                raise ValueError("Feature function returned an invalid batch")
            output[first:first + len(picked)] = features
        output.flush()
    finally:
        _close_mmap(output)
    manifest = {"format": "dockdack-features-v1", "completed": True, "feature_identity": feature_identity,
                "indices_sha256": hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest(),
                "batch_size": batch_size, "files": {"features.npy": _array_info(target)},
                "model_math_changed": False, "orders_started": False}
    write_new_json(stage / "completed.json", manifest)
    stage.rename(destination)
    return manifest
