"""Export completed 0.5/0.4 training into a NEW portable research-only bundle.

Original models, source databases and old-target bundles are never overwritten.
All six native members are checked against the frozen training run. CPU output
is compared with the source members on four 2024 histories x three prices per
market; this numerical check is not a profitability or live-readiness test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dockdack.mark1_0504_inference import (
    BUNDLE_VERSION, FEATURE_NAMES, MARKET_MODELS, OWNER, RISK_FLAGS, SCHEMA_VERSION,
    SEEDS, SEMANTICS, TARGET, TITLE, WARNINGS, HalfPercentPredictor,
    runtime_code_hashes, sha256_file, _read, _same,
)
from dockdack.mark1_metrics import calibrated_probability
from dockdack.mark1_selective_models import load_model, model_path, predict_raw


ROOT = Path(__file__).resolve().parents[1]
FOLD = "walk_2024"


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def compact_assessment(metrics):
    keys = ("count", "signal_count", "signal_days", "symbol_count", "coverage", "precision",
            "gross_mean_return", "net_mean_return", "cost_bps", "block_bootstrap")
    return {key: metrics[key] for key in keys if key in metrics}


def validation_indices(dataset):
    """Fixed date-only selection; no look at labels, returns or probabilities."""
    dates = np.asarray(dataset.target_dates)
    lower = int(np.datetime64("2024-03-01", "D").astype(np.int64))
    upper = int(np.datetime64("2024-12-31", "D").astype(np.int64))
    eligible = np.flatnonzero((dates >= lower) & (dates <= upper))
    if len(eligible) < 4:
        raise ValueError("Four eligible 2024 histories are required for CPU export equivalence")
    return eligible[np.linspace(0, len(eligible) - 1, 4, dtype=np.int64)]


def export_bundle(training_root, destination, *, cache_dir=None, old_run=None):
    training_root, destination = Path(training_root).absolute(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace existing bundle: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError("Bundle parent must be an existing, unlinked directory")
    if training_root.is_symlink() or not training_root.is_dir():
        raise ValueError("Training run must be an existing, unlinked directory")
    destination, training_root = destination.resolve(), training_root.resolve()
    # Heavy cache/data verification belongs only to export, never inference.
    from examples import train_mark1_0504 as training
    frozen = _read(training_root / "protocol.json")
    if (not _same(frozen.get("protocol"), training.canonical(training.PROTOCOL))
            or training.PROTOCOL["target"] != TARGET
            or training.PROTOCOL["feature_names"] != list(FEATURE_NAMES)
            or training.PROTOCOL["architectures"] != MARKET_MODELS
            or training.PROTOCOL["ensemble_seeds"] != list(SEEDS)):
        raise ValueError("Training protocol target, features, models or seeds differ")
    code_hashes = frozen.get("code_sha256")
    if (not isinstance(code_hashes, dict) or set(code_hashes) != set(training.CODE_FILES)
            or any(sha256_file(ROOT / name) != digest for name, digest in code_hashes.items())):
        raise ValueError("Frozen training code checksum mismatch")
    cache_dir = Path(cache_dir or frozen["raw_cache_dir"]).resolve()
    old_run = Path(old_run or frozen["old_run"]).resolve()
    if (cache_dir != Path(frozen["raw_cache_dir"]).resolve()
            or old_run != Path(frozen["old_run"]).resolve()):
        raise ValueError("Export must use the frozen raw cache and its source provenance")
    summary = _read(training_root / "summary.json")
    completion = _read(training_root / "completed.json")
    if set(summary) != set(MARKET_MODELS) or completion.get("no_orders") is not True:
        raise ValueError("Both training markets must be fully completed")
    protected = {training_root / "protocol.json", training_root / "summary.json", training_root / "completed.json",
                 Path(__file__).resolve(), ROOT / "dockdack/mark1_0504_inference.py"}
    protected.update(ROOT / name for name in code_hashes)
    for market, architecture in MARKET_MODELS.items():
        item = summary[market]
        if (not _same(item, _read(training_root / market / "summary.json"))
                or item.get("market") != market or item.get("selected") != architecture
                or item.get("target") != TARGET or item.get("completed") is not True
                or item.get("research_only") is not True or item.get("deployment_allowed") is not False
                or item.get("intraday_path_verified") is not False):
            raise ValueError("Inconsistent completed market summary")
        if (not _same(item.get("source"), _read(training_root / market / "source.json"))
                or not _same(item.get("experiment_data"), _read(training_root / market / "experiment_data.json"))
                or item["experiment_data"].get("target") != TARGET
                or item["experiment_data"].get("raw_cache_target_is_not_used_for_labels") is not True):
            raise ValueError("Source/raw-cache/new-target provenance mismatch")
        protected.update(training_root / market / name for name in ("summary.json", "source.json", "experiment_data.json"))
        source = item["source"]
        database = Path(source["database_path"])
        if sha256_file(database) != source["database_sha256"]:
            raise ValueError("Original database changed since frozen training")
        protected.add(database)
        key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
        protected.add(cache_dir / f"{market}-{key}.npz")
        if set(item.get("ensembles", {})) != set(training.PROTOCOL["folds"]):
            raise ValueError("Both chronological development ensembles are required")
        for fold, ensemble in item["ensembles"].items():
            ensemble_path = training_root / market / fold / "ensemble.json"
            if (not _same(ensemble, _read(ensemble_path)) or ensemble.get("target") != TARGET
                    or not _same(ensemble.get("policy"), {"threshold": .5, "stop_probability_cap": 1.})
                    or not _same(ensemble.get("qualification"), training.qualification(ensemble["audit"]))
                    or not _same(ensemble.get("validation_qualification"), training.qualification(ensemble["validation"]))):
                raise ValueError("Ensemble assessment/policy disagrees with training")
            protected.add(ensemble_path)
        for seed in SEEDS:
            trial = training_root / market / FOLD / f"{architecture}-{seed}"
            native = model_path(trial, architecture)
            protected.update((native, native.with_name(native.name + ".json"), trial / "metadata.json", trial / "request.json"))
    old_bundle_files = completion.get("original_prototype_sha256")
    if (not isinstance(old_bundle_files, dict) or not old_bundle_files
            or any(sha256_file(Path(path)) != digest for path, digest in old_bundle_files.items())):
        raise ValueError("Protected original prototype must remain unchanged")
    protected.update(Path(path) for path in old_bundle_files)
    before = {str(path): sha256_file(path) for path in sorted(protected)}
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-export-", dir=destination.parent)).resolve()
    if stage.parent != destination.parent or not stage.name.startswith(f".{destination.name}-export-"):
        raise ValueError("Export staging path escaped intended parent")
    references, originals = {}, {}
    for market, architecture in MARKET_MODELS.items():
        folder = stage / market
        folder.mkdir()
        ensemble = summary[market]["ensembles"][FOLD]
        members, original_models = [], []
        for seed in SEEDS:
            trial = training_root / market / FOLD / f"{architecture}-{seed}"
            native = model_path(trial, architecture)
            metadata = _read(trial / "metadata.json")
            request = _read(trial / "request.json")
            digest = sha256_file(native)
            if (metadata.get("model_sha256") != digest or ensemble["member_sha256"].get(str(seed)) != digest
                    or metadata.get("feature_count") != 184 or metadata.get("model_name") != architecture
                    or metadata.get("research_only") is not True or metadata.get("deployment_allowed") is not False
                    or not _same(metadata.get("request"), request)
                    or request.get("seed") != seed or request.get("model_name") != architecture
                    or request.get("wrapper_sha256") != code_hashes["dockdack/mark1_selective_models.py"]):
                raise ValueError("Native member metadata/provenance mismatch")
            original_models.append(load_model(architecture, native))
            target = folder / f"seed{seed}.cbm"
            shutil.copyfile(native, target)
            shutil.copyfile(native.with_name(native.name + ".json"), target.with_name(target.name + ".json"))
            members.append({"seed": seed, "path": f"{market}/{target.name}", "sha256": sha256_file(target),
                            "sidecar_path": f"{market}/{target.name}.json",
                            "sidecar_sha256": sha256_file(target.with_name(target.name + ".json")),
                            "best_iteration": metadata["best_iteration"], "training_versions": metadata["versions"]})
        originals[market] = original_models
        result = {
            "market": market, "model_name": architecture, "fold": FOLD, "seeds": list(SEEDS),
            "members": members, "calibration": ensemble["calibration"],
            "policy": ensemble["policy"], "risk_flags": RISK_FLAGS,
            "source": {"raw_cache_provenance": summary[market]["source"],
                       "new_target_experiment": summary[market]["experiment_data"]},
            "research_results": {fold: {"audit": compact_assessment(value["audit"]),
                                         "validation": compact_assessment(value["validation"]),
                                         "qualification": value["qualification"],
                                         "validation_qualification": value["validation_qualification"]}
                                 for fold, value in summary[market]["ensembles"].items()},
            "provenance": {"source_summary_sha256": sha256_file(training_root / market / "summary.json"),
                           "source_ensemble_sha256": sha256_file(training_root / market / FOLD / "ensemble.json")},
        }
        write_json(folder / "manifest.json", result)
        references[market] = {"path": f"{market}/manifest.json", "sha256": sha256_file(folder / "manifest.json")}
    manifest = {
        "owner": OWNER, "schema_version": SCHEMA_VERSION, "title": TITLE, "version": BUNDLE_VERSION,
        "completed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "markets": references, "semantics": SEMANTICS, "risk_flags": RISK_FLAGS, "warnings": WARNINGS,
        "feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
        "runtime_code_sha256": runtime_code_hashes(), "training_versions": frozen["versions"],
        "provenance": {"training_run": training_root.name, "protocol_sha256": sha256_file(training_root / "protocol.json"),
                       "summary_sha256": sha256_file(training_root / "summary.json"),
                       "frozen_training_code_sha256": code_hashes,
                       "exporter_sha256": sha256_file(Path(__file__))},
        "portability": "Requires this bundle, the matching DockDack inference code with its base package dependencies, and numpy/CatBoost; source DB/cache paths are provenance, never runtime reads. No broker clients are instantiated or APIs called.",
    }
    write_json(stage / "manifest.json", manifest)
    (stage / "manifest.sha256").write_text(sha256_file(stage / "manifest.json") + "\n", encoding="ascii")
    validations = {}
    for market, original_models in originals.items():
        portable = HalfPercentPredictor(stage, market)
        dataset, source, experiment = training.load_experiment_dataset(market, cache_dir, old_run)
        if not _same(source, summary[market]["source"]) or not _same(experiment, summary[market]["experiment_data"]):
            raise ValueError("Export validation raw windows differ from training")
        calibration = summary[market]["ensembles"][FOLD]["calibration"]
        rows = []
        for index in validation_indices(dataset):
            start = int(dataset.starts[index])
            history = dataset.bars[start:start + 30].copy()
            untouched = history.copy()
            for factor in (1., .9975, 1.0025):
                entry = float(dataset.target_ohlc[index, 0]) * factor
                features = training.features_from_history(history[None], np.array([entry]))
                raw = [predict_raw(model, MARKET_MODELS[market], features) for model in original_models]
                mean = math.fsum(float(member["success_logits"][0]) for member in raw) / len(SEEDS)
                expected = float(calibrated_probability([mean], calibration["success"])[0])
                expected_stop = None
                if calibration["stop"] is not None:
                    mean_stop = math.fsum(float(member["stop_logits"][0]) for member in raw) / len(SEEDS)
                    expected_stop = float(calibrated_probability([mean_stop], calibration["stop"])[0])
                actual = portable.predict(history, current_price=str(entry))
                difference = abs(expected - actual["probability_success"])
                stop_difference = 0. if expected_stop is None else abs(expected_stop - actual["probability_stop"])
                if (difference > 1e-12 or stop_difference > 1e-12
                        or (expected_stop is None) != (actual["probability_stop"] is None)
                        or actual["predicts_success"] != bool(expected > .5)
                        or any(actual[name] is not flag for name, flag in RISK_FLAGS.items())
                        or not np.array_equal(history, untouched)
                        or actual["candidate_take_price"] != entry * 1.005
                        or actual["candidate_stop_price"] != entry * .996):
                    raise ValueError("Portable CPU probability differs or violates new research-only contract")
                rows.append({"dataset_index": int(index), "target_date": str(np.datetime64(int(dataset.target_dates[index]), "D")),
                             "candidate_factor": factor, "probability_success": actual["probability_success"],
                             "probability_stop": actual["probability_stop"], "success_error": difference,
                             "stop_error": stop_difference, "selected_research": actual["predicts_success"]})
        validations[market] = {"histories": 4, "prediction_calls": len(rows), "passed": True,
                               "inputs_unchanged": True, "rows": rows}
        del dataset, portable
        gc.collect()
    after = {path: sha256_file(Path(path)) for path in before}
    if before != after:
        raise ValueError("A protected original artifact changed during export")
    write_json(stage / "export-validation.json", {
        "passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest_sha256": sha256_file(stage / "manifest.json"), "markets": validations,
        "research_only": True, "deployment_allowed": False, "protected_files_unchanged": True,
        "protected_sha256_before": before, "protected_sha256_after": after,
        "validation_scope": "CPU numerical equivalence on 4 deterministic 2024 histories x 3 prices per market; not a profitability or deployment test",
    })
    if stage.parent != destination.parent or destination.exists() or destination.is_symlink():
        raise ValueError("Export destination changed during validation")
    stage.rename(destination)
    return {"bundle": str(destination), "version": BUNDLE_VERSION,
            "bundle_manifest_sha256": sha256_file(destination / "manifest.json"),
            "validation": str(destination / "export-validation.json"), "prediction_comparisons": 24}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=ROOT / "outputs/mark1/half-20260920")
    parser.add_argument("--output", type=Path, default=ROOT / "models/mark1_0504")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--old-run", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.training, args.output, cache_dir=args.cache, old_run=args.old_run), indent=2))


if __name__ == "__main__":
    main()
