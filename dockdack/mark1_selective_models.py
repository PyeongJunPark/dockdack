"""Lazy tree backends for offline mark_1 selective-signal research.

No broker/database code is imported. Classes are take-only, stop-only,
both-touch and neither. Binary heads predict class zero against all failures.
CatBoost can fit on GPU; inference and LightGBM fitting use CPU explicitly.
Only a caller-owned trial directory is written. Exact completed trials are
reused, incompatible directories rejected, and CatBoost snapshots can resume
an interrupted fit with the same data/configuration.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np


MODEL_NAMES = ("cat_binary6", "cat_binary8", "cat_joint6", "lgbm_binary")
CLASS_NAMES = ("take_only", "stop_only", "both_touch", "neither")
SCHEMA_VERSION = 1
OWNER = "dockdack.mark1_selective_models"


def _name(name: str) -> None:
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown selective model {name!r}; expected {MODEL_NAMES}")


def _integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _features(value, *, allow_empty: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.ndim != 2 or raw.shape[1] < 1 or (not allow_empty and len(raw) < 1)
            or raw.dtype.kind not in "fiu"):
        raise ValueError("features must be a finite numeric matrix [rows,features]")
    result = np.asarray(raw, dtype=np.float32, order="C")
    for start in range(0, len(result), 65536):
        if not np.isfinite(result[start:start + 65536]).all():
            raise ValueError("features must contain only finite float32 values")
    return result


def _labels(value, count: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) != count or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integer class ids [rows]")
    if np.any(raw < 0) or np.any(raw > 3):
        raise ValueError(f"{name} must contain class ids 0,1,2,3")
    return np.asarray(raw, dtype=np.int32, order="C")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    view = memoryview(array).cast("B")
    for start in range(0, len(view), 1024 * 1024):
        digest.update(view[start:start + 1024 * 1024])
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    """Atomic update of one already-resolved, owned artifact/manifest path."""
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        raise ValueError(f"refusing symlink temporary artifact {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
                         encoding="utf-8")
    os.replace(temporary, path)


def _backend(name: str):
    _name(name)
    library = "lightgbm" if name == "lgbm_binary" else "catboost"
    try:
        return importlib.import_module(library)
    except ImportError as error:
        raise ImportError(f"{name} requires optional {library}; use the isolated selective dependencies") from error


def model_path(folder, name: str) -> Path:
    _name(name)
    return Path(folder) / (name + (".txt" if name == "lgbm_binary" else ".cbm"))


def _artifact_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def _model_feature_count(model, name: str) -> int:
    if name == "lgbm_binary":
        return int(model.num_feature())
    # CatBoost 1.2.10 restores feature_names_ but reports n_features_in_ == 0
    # after a native CBM load. The names include unused input columns as well.
    names = getattr(model, "feature_names_", None)
    return len(names) if names else int(model.n_features_in_)


def _validate_model(model, name: str) -> None:
    if name == "lgbm_binary":
        if int(model.num_model_per_iteration()) != 1:
            raise ValueError("expected a binary LightGBM model")
    else:
        expected = np.arange(4) if name == "cat_joint6" else np.arange(2)
        if not np.array_equal(np.asarray(model.classes_), expected):
            raise ValueError("model class order does not match the fixed target contract")


def save_model(model, name: str, path) -> Path:
    """Export to a new path plus checksum sidecar; never replace existing files."""
    _name(name)
    _validate_model(model, name)
    artifact = Path(path).absolute()
    expected = ".txt" if name == "lgbm_binary" else ".cbm"
    if artifact.suffix != expected:
        raise ValueError(f"{name} artifact must have {expected} extension")
    sidecar = _artifact_sidecar(artifact)
    if artifact.exists() or artifact.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(f"refusing to overwrite model artifact {artifact}")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if name == "lgbm_binary":
        model.save_model(str(artifact), num_iteration=model.best_iteration or -1)
    else:
        model.save_model(str(artifact), format="cbm")
    _write_json(sidecar, {
        "owner": OWNER, "schema_version": SCHEMA_VERSION, "model_name": name,
        "model_sha256": _sha_file(artifact), "feature_count": _model_feature_count(model, name),
        "class_names": list(CLASS_NAMES),
    })
    return artifact


def load_model(name: str, path):
    """Load only a compatible, checksummed export, without pickle execution."""
    _name(name)
    artifact = Path(path).absolute()
    expected_suffix = ".txt" if name == "lgbm_binary" else ".cbm"
    if artifact.suffix != expected_suffix or not artifact.is_file() or artifact.is_symlink():
        raise ValueError("missing, symlinked or incompatible model artifact")
    sidecar = _artifact_sidecar(artifact)
    if not sidecar.is_file() or sidecar.is_symlink():
        raise ValueError("model checksum sidecar is required")
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    if (saved.get("owner") != OWNER or saved.get("schema_version") != SCHEMA_VERSION
            or saved.get("model_name") != name or saved.get("class_names") != list(CLASS_NAMES)
            or saved.get("model_sha256") != _sha_file(artifact)):
        raise ValueError("model artifact contract/checksum mismatch")
    backend = _backend(name)
    if name == "lgbm_binary":
        model = backend.Booster(model_file=str(artifact))
    else:
        model = backend.CatBoostClassifier()
        model.load_model(str(artifact), format="cbm")
    _validate_model(model, name)
    if _model_feature_count(model, name) != saved.get("feature_count"):
        raise ValueError("model feature count does not match artifact sidecar")
    return model


def joint_logits(raw) -> dict:
    """Stable joint-event log odds; no independence assumption for the barriers."""
    logits = np.asarray(raw, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 4 or not np.isfinite(logits).all():
        raise ValueError("joint logits must be finite [rows,4] in fixed class order")
    return {
        "success_logits": logits[:, 0] - np.logaddexp.reduce(logits[:, 1:], axis=1),
        "stop_logits": np.logaddexp(logits[:, 1], logits[:, 2])
        - np.logaddexp(logits[:, 0], logits[:, 3]),
    }


def predict_raw(model, name: str, features) -> dict:
    """Return CPU float64 raw log odds; binary models have no stop-risk head."""
    _name(name)
    matrix = _features(features, allow_empty=True)
    _validate_model(model, name)
    if matrix.shape[1] != _model_feature_count(model, name):
        raise ValueError("feature count does not match fitted model")
    if not len(matrix):
        return {"success_logits": np.empty(0, dtype=np.float64),
                "stop_logits": np.empty(0, dtype=np.float64) if name == "cat_joint6" else None}
    if name == "lgbm_binary":
        raw = model.predict(matrix, raw_score=True, num_iteration=model.best_iteration or -1,
                            num_threads=12)
    else:
        raw = model.predict(matrix, prediction_type="RawFormulaVal", task_type="CPU",
                            thread_count=12)
    if name == "cat_joint6":
        return joint_logits(raw)
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if len(raw) != len(matrix) or not np.isfinite(raw).all():
        raise ValueError("backend returned invalid binary raw predictions")
    return {"success_logits": raw, "stop_logits": None}


def fit_model(name, x_train, y_class4, x_tune, y_tuneclass4, folder,
              seed=42, task_type="GPU", max_iterations=3000,
              early_stopping=150, threads=12):
    """Fit/reuse one owned trial; tune data alone controls early stopping.

    ``best_iteration`` in metadata is a one-based selected tree/boosting-round
    count for both libraries. ``trained_iterations`` includes discarded rounds.
    LightGBM always uses deterministic CPU even when task_type requests GPU.
    An interrupted CatBoost fit can reuse only this exact trial's snapshot.
    """
    _name(name)
    seed = _integer(seed, "seed", 0)
    max_iterations = _integer(max_iterations, "max_iterations")
    early_stopping = _integer(early_stopping, "early_stopping")
    threads = _integer(threads, "threads")
    if task_type not in ("GPU", "CPU"):
        raise ValueError("task_type must be GPU or CPU")
    train, tune = _features(x_train), _features(x_tune)
    if train.shape[1] != tune.shape[1]:
        raise ValueError("train/tune feature counts differ")
    targets = _labels(y_class4, len(train), "y_class4")
    tune_targets = _labels(y_tuneclass4, len(tune), "y_tuneclass4")
    for target, label in ((targets, "train"), (tune_targets, "tune")):
        if not np.any(target == 0) or not np.any(target != 0):
            raise ValueError(f"{label} requires both success and failure classes")
    if name == "cat_joint6" and not np.array_equal(np.unique(targets), np.arange(4)):
        raise ValueError("joint training requires all four target classes")
    backend = _backend(name)
    root = Path(folder).absolute()
    if root.is_symlink():
        raise ValueError("trial folder must not be a symlink")
    effective_task = "CPU" if name == "lgbm_binary" else task_type
    if name == "lgbm_binary":
        params = {
            "objective": "binary", "metric": "average_precision", "num_leaves": 31,
            "min_data_in_leaf": 200, "lambda_l2": 10.0, "learning_rate": .03,
            "feature_fraction": .85, "bagging_fraction": .8, "bagging_freq": 1,
            "max_bin": 127, "deterministic": True, "force_col_wise": True,
            "device_type": "cpu", "num_threads": threads, "verbosity": -1,
            "seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed,
            "data_random_seed": seed,
        }
    else:
        params = {
            "iterations": max_iterations, "depth": 8 if name == "cat_binary8" else 6,
            "learning_rate": .03, "l2_leaf_reg": 10.0, "border_count": 128,
            "bootstrap_type": "Bernoulli", "subsample": .8, "boosting_type": "Plain",
            "random_seed": seed, "thread_count": threads, "task_type": effective_task,
            "loss_function": "MultiClass" if name == "cat_joint6" else "Logloss",
            "eval_metric": "MultiClass" if name == "cat_joint6" else "PRAUC:type=Classic",
            "allow_writing_files": True, "train_dir": str(root / "training"),
            "save_snapshot": True, "snapshot_interval": 60, "verbose": False,
        }
        if name == "cat_joint6":
            params["classes_count"] = 4
    request = {
        "owner": OWNER, "schema_version": SCHEMA_VERSION, "model_name": name,
        "seed": seed, "requested_task_type": task_type, "effective_task_type": effective_task,
        "max_iterations": max_iterations, "early_stopping": early_stopping,
        "params": params, "class_names": list(CLASS_NAMES),
        "train_shape": list(train.shape), "tune_shape": list(tune.shape),
        "data_sha256": {"x_train": _sha_array(train), "y_train": _sha_array(targets),
                        "x_tune": _sha_array(tune), "y_tune": _sha_array(tune_targets)},
        "versions": {"backend": backend.__version__, "numpy": np.__version__,
                     "python": platform.python_version()},
        "wrapper_sha256": _sha_file(Path(__file__)),
    }
    request_file, metadata_file = root / "request.json", root / "metadata.json"
    artifact = model_path(root, name)
    if root.exists() and not root.is_dir():
        raise ValueError("trial folder is not a directory")
    if root.exists() and any(root.iterdir()):
        if not request_file.is_file() or request_file.is_symlink():
            raise ValueError("nonempty trial folder has no ownership manifest")
        if json.loads(request_file.read_text(encoding="utf-8")) != request:
            raise ValueError("incompatible cached trial configuration/data; use a new trial folder")
        if metadata_file.is_file():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            if (metadata.get("request") != request or not artifact.is_file()
                    or metadata.get("model_sha256") != _sha_file(artifact)):
                raise ValueError("completed trial metadata/model checksum mismatch")
            loaded = load_model(name, artifact)
            metadata = dict(metadata, reused=True)
            return loaded, metadata
        if artifact.exists() or _artifact_sidecar(artifact).exists():
            raise ValueError("incomplete model export; preserve artifacts and use a new trial folder")
    else:
        root.mkdir(parents=True, exist_ok=True)
        _write_json(request_file, request)
    # Reject linked files/directories before backend-controlled log/snapshot writes.
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("trial artifacts must not contain symlinks")
    started = time.perf_counter()
    if name == "lgbm_binary":
        history = {}
        train_data = backend.Dataset(train, label=(targets == 0).astype(np.int8))
        tune_data = backend.Dataset(tune, label=(tune_targets == 0).astype(np.int8), reference=train_data)
        model = backend.train(
            params, train_data, num_boost_round=max_iterations,
            valid_sets=[tune_data], valid_names=["validation"],
            callbacks=[backend.early_stopping(early_stopping, first_metric_only=True, verbose=False),
                       backend.record_evaluation(history)],
        )
        best_iteration = int(model.best_iteration or model.current_iteration())
    else:
        model = backend.CatBoostClassifier(**params)
        binary = name != "cat_joint6"
        model.fit(train, (targets == 0).astype(np.int8) if binary else targets,
                  eval_set=(tune, (tune_targets == 0).astype(np.int8) if binary else tune_targets),
                  use_best_model=True, early_stopping_rounds=early_stopping)
        history = model.get_evals_result()
        best_iteration = int(model.tree_count_)
    elapsed = time.perf_counter() - started
    _validate_model(model, name)
    save_model(model, name, artifact)
    trained_iterations = max((len(values) for split in history.values() for values in split.values()), default=0)
    metadata = {
        "schema_version": SCHEMA_VERSION, "owner": OWNER, "model_name": name,
        "request": request, "params": params, "versions": request["versions"],
        "best_iteration": best_iteration, "best_iteration_convention": "one_based_round_count",
        "trained_iterations": trained_iterations, "fit_seconds": elapsed,
        "eval_history": history, "model_path": str(artifact),
        "model_sha256": _sha_file(artifact), "feature_count": train.shape[1],
        "effective_task_type": effective_task, "reused": False,
        "research_only": True, "deployment_allowed": False,
    }
    _write_json(metadata_file, metadata)
    return model, metadata
