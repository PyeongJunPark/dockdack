"""Frozen precision-first tree research. No broker access or 2025+ selection.

Only completed historical OHLCV and the observed session OPEN form inputs.
Higher abstention cutoffs are research policies, not live-order settings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from dockdack.mark1_data import TARGET, barrier_outcomes
from dockdack.mark1_deep_data import FOLDS, class_targets, make_splits, split_manifest
from dockdack.mark1_metrics import calibrated_probability, fit_calibration
from dockdack.mark1_selective_features import FEATURE_NAMES, features_from_history
from dockdack.mark1_selective_models import MODEL_NAMES, fit_model, predict_raw, model_path
from dockdack.mark1_selective_policy import apply_policy, evaluate_signals, qualification, select_policy
from examples.train_mark1 import dataset_cache, file_hash, save_json
from examples.train_mark1_deep import atomic_json, read_sessions


ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "examples/train_mark1_selective.py", "dockdack/mark1_selective_features.py",
    "dockdack/mark1_selective_models.py", "dockdack/mark1_selective_policy.py",
    "dockdack/mark1_data.py", "dockdack/mark1_metrics.py", "dockdack/mark1_deep_data.py", "dockdack/mark1_deep_validation.py",
    "examples/train_mark1.py", "examples/train_mark1_deep.py",
)
PROTOCOL = {
    "version": "mark1-selective-v1", "target": TARGET, "folds": FOLDS,
    "architectures": list(MODEL_NAMES), "seed": 42, "ensemble_seeds": [42, 43, 44],
    "max_train": 1_500_000, "max_tune": 150_000, "price_augmentation": False,
    "calibration": "first half-year probability calibration; second half-year policy selection after 30-session purge",
    "feature_names": list(FEATURE_NAMES), "maximum_iterations": 3000, "early_stopping": 150,
    "thresholds": [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95],
    "joint_stop_caps": [1., .25], "bootstrap": {"block_sessions": 10, "replicates": 2000, "seed": 42},
    "signal_evidence": {"minimum_signals": 50, "minimum_days": 20, "minimum_symbols": 10},
    "research_gate": {"precision": .65, "precision_lower_exclusive": .579, "net_lower_exclusive": 0., "cost_bps": 20},
    "selection": "eligible in both next-year audits: maximize minimum precision lower, then mean net lower, then total signals; otherwise diagnostic mean Brier only",
    "ensemble": "after architecture lock add seeds43/44 to BOTH folds; average raw logits then recalibrate and reselect policy using that fold's own two calibration halves",
    "success_scope": "whole-session conservative daily label, NOT an arbitrary intraday entry path",
    "test_scope": "2025+ already seen historically, never used here; subsequent comparison is reused history",
    "deployment": "research only; no model promotion, no orders, no Git publication",
}


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def selective_splits(dataset, sessions, fold, *, max_train=1_500_000, max_tune=150_000):
    base = make_splits(dataset, sessions, fold, max_train=max_train, max_tune=max_tune, seed=42)
    year = FOLDS[fold]["calibration_year"]
    boundary = int(np.datetime64(f"{year}-06-30", "D").astype(np.int64))
    indices = base["calibration"]
    dates = dataset.target_dates[indices]
    ordinal = np.searchsorted(sessions, dates)
    first_context = sessions[ordinal - 30]
    result = {"train": base["train"], "tune": base["tune"],
              "probability_calibration": indices[dates <= boundary],
              "policy_calibration": indices[(dates > boundary) & (first_context > boundary)],
              "audit": base["selection"]}
    if any(not len(part) for part in result.values()):
        raise ValueError("Every selective chronological split must be nonempty")
    for part in result.values():
        part.setflags(write=False)
    return result


def describe(dataset, splits):
    return split_manifest(dataset, splits)


def feature_array(dataset, indices, *, batch_size=8192):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive feature batch size required")
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices >= len(dataset.starts)) or np.any(indices < 0):
        raise ValueError("Valid one-dimensional sample indices required")
    result = np.empty((len(indices), len(FEATURE_NAMES)), dtype=np.float32)
    offsets = np.arange(30)
    for start in range(0, len(indices), batch_size):
        picked = indices[start:start + batch_size]
        history = dataset.bars[dataset.starts[picked, None] + offsets[None, :]]
        result[start:start + len(picked)] = features_from_history(history, dataset.target_ohlc[picked, 0])
    return result


def event_data(dataset, indices):
    ohlc = dataset.target_ohlc[indices]
    outcomes = barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], ohlc[:, 0])
    return {"labels": outcomes["success"], "gross": outcomes["gross_return"],
            "classes": class_targets(ohlc)[:, 0], "dates": dataset.target_dates[indices],
            "symbols": dataset.symbol_ids[indices]}


def prepare_fold(dataset, sessions, fold, folder):
    splits = selective_splits(dataset, sessions, fold)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = describe(dataset, splits)
    index_hashes = {key: __import__("hashlib").sha256(value.tobytes()).hexdigest() for key, value in splits.items()}
    contract = {"splits": manifest, "index_sha256": index_hashes, "features": list(FEATURE_NAMES)}
    contract_path = folder / "features.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Feature cache split contract changed")
    if not contract_path.exists() and any(folder.glob("*.npy")):
        raise ValueError("Unowned feature files without a split contract")
    atomic_json(contract_path, contract)
    integrity = folder / "feature_hashes.json"
    hashes = json.loads(integrity.read_text()) if integrity.exists() else {}
    arrays, events = {}, {}
    for name, indices in splits.items():
        path = folder / f"{name}.npy"
        if path.exists():
            if name not in hashes or file_hash(path) != hashes[name]:
                raise ValueError("Existing feature cache lacks matching integrity provenance")
        else:
            print(f"  features {fold}/{name}: {len(indices):,}", flush=True)
            values = feature_array(dataset, indices)
            temporary = folder / f"{name}.building.npy"
            np.save(temporary, values, allow_pickle=False)
            temporary.replace(path)
            del values
            hashes[name] = file_hash(path)
            atomic_json(integrity, hashes)
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        if arrays[name].shape != (len(indices), len(FEATURE_NAMES)) or arrays[name].dtype != np.float32:
            raise ValueError("Invalid cached feature shape/dtype")
        events[name] = event_data(dataset, indices)
    atomic_json(contract_path, contract)
    atomic_json(integrity, hashes)
    return splits, arrays, events, manifest


def fit_calibrators(predictions, events):
    raw, event = predictions["probability_calibration"], events["probability_calibration"]
    success = fit_calibration(raw["success_logits"], event["labels"])
    stop = (fit_calibration(raw["stop_logits"], np.isin(event["classes"], [1, 2]))
            if raw["stop_logits"] is not None else None)
    return {"success": success, "stop": stop}


def calibrated(raw, calibrators):
    success = calibrated_probability(raw["success_logits"], calibrators["success"])
    stop = (calibrated_probability(raw["stop_logits"], calibrators["stop"])
            if calibrators["stop"] is not None else None)
    return success, stop


def assess(predictions, events):
    calibrators = fit_calibrators(predictions, events)
    event = events["policy_calibration"]
    p, stop = calibrated(predictions["policy_calibration"], calibrators)
    policy = select_policy(event["labels"], p, event["gross"], event["dates"], event["symbols"], stop_probabilities=stop)
    event = events["audit"]
    p, stop = calibrated(predictions["audit"], calibrators)
    selected = apply_policy(p, policy["chosen_policy"], stop_probabilities=stop)
    metrics = evaluate_signals(event["labels"], p, event["gross"], event["dates"], event["symbols"], selected)
    return {"calibration": calibrators, "policy_selection": policy, "audit": metrics,
            "qualification": qualification(metrics)}


def trial(name, seed, arrays, events, folder, args):
    started = time.monotonic()
    model, metadata = fit_model(name, arrays["train"], events["train"]["classes"],
        arrays["tune"], events["tune"]["classes"], folder, seed=seed,
        task_type=args.task_type, max_iterations=PROTOCOL["maximum_iterations"],
        early_stopping=PROTOCOL["early_stopping"], threads=args.threads)
    predictions = {part: predict_raw(model, name, arrays[part])
                   for part in ("probability_calibration", "policy_calibration", "audit")}
    result = {"architecture": name, "seed": seed, "model": metadata, **assess(predictions, events)}
    result["seconds_total"] = time.monotonic() - started
    atomic_json(folder / "result.json", result)
    np.savez(folder / "development_predictions.npz",
             **{part + "_success": raw["success_logits"] for part, raw in predictions.items()},
             **{part + "_stop": raw["stop_logits"] for part, raw in predictions.items() if raw["stop_logits"] is not None})
    print(f"  DONE {name} seed{seed}: signals={result['audit']['signal_count']} precision={result['audit']['precision']} qualified={result['qualification']['qualified']}", flush=True)
    del model
    gc.collect()
    return result, predictions


def architecture_ranking(results):
    ranked = []
    for name in MODEL_NAMES:
        metrics = [results[fold][name]["audit"] for fold in FOLDS]
        eligible = all(item["block_bootstrap"]["precision_lower"] is not None for item in metrics)
        minimum = min(item["block_bootstrap"]["precision_lower"] for item in metrics) if eligible else None
        net = float(np.mean([item["block_bootstrap"]["net_mean_lower"] for item in metrics])) if eligible else None
        ranked.append({"architecture": name, "both_audits_have_evidence": eligible,
                       "minimum_precision_lower": minimum, "mean_net_lower": net,
                       "total_signals": sum(item["signal_count"] for item in metrics),
                       "mean_brier": float(np.mean([item["overall"]["brier"] for item in metrics]))})
    ranked.sort(key=lambda row: (not row["both_audits_have_evidence"],
        -row["minimum_precision_lower"] if row["minimum_precision_lower"] is not None else 0,
        -row["mean_net_lower"] if row["mean_net_lower"] is not None else 0,
        -row["total_signals"] if row["both_audits_have_evidence"] else 0,
        row["mean_brier"], row["architecture"]))
    return ranked


def average_predictions(members):
    parts = ("probability_calibration", "policy_calibration", "audit")
    if not isinstance(members, (list, tuple)) or not members:
        raise ValueError("A nonempty ensemble is required")
    stop_schema = None
    lengths = {}
    for member in members:
        if not isinstance(member, dict) or set(member) != set(parts):
            raise ValueError("Exact ensemble prediction partitions required")
        for part in parts:
            raw = member[part]
            if not isinstance(raw, dict) or set(raw) != {"success_logits", "stop_logits"}:
                raise ValueError("Exact raw prediction schema required")
            values = np.asarray(raw["success_logits"])
            if values.ndim != 1 or values.dtype.kind not in "fiu" or not len(values) or not np.isfinite(values).all():
                raise ValueError("Finite nonempty one-dimensional logits required")
            if part in lengths and lengths[part] != len(values):
                raise ValueError("Ensemble prediction lengths differ")
            lengths[part] = len(values)
            has_stop = raw["stop_logits"] is not None
            if stop_schema is not None and stop_schema != has_stop:
                raise ValueError("Cannot mix binary and joint schemas")
            stop_schema = has_stop
            if has_stop:
                stops = np.asarray(raw["stop_logits"])
                if stops.shape != values.shape or stops.dtype.kind not in "fiu" or not np.isfinite(stops).all():
                    raise ValueError("Finite matching one-dimensional stop logits required")
    result = {}
    for part in parts:
        values = [member[part] for member in members]
        has_stop = [item["stop_logits"] is not None for item in values]
        if any(has_stop) != all(has_stop):
            raise ValueError("Cannot mix joint and binary ensemble schemas")
        result[part] = {"success_logits": np.mean([item["success_logits"] for item in values], axis=0),
                        "stop_logits": np.mean([item["stop_logits"] for item in values], axis=0) if all(has_stop) else None}
    return result


def load_trial_predictions(folder, name):
    with np.load(folder / "development_predictions.npz", allow_pickle=False) as archive:
        return {part: {"success_logits": archive[part + "_success"],
                       "stop_logits": archive[part + "_stop"] if name == "cat_joint6" else None}
                for part in ("probability_calibration", "policy_calibration", "audit")}


def run_market(market, args):
    folder = args.output_dir / market
    folder.mkdir(parents=True, exist_ok=True)
    database = args.db_dir / f"{market}_daily_clean.sqlite3"
    dataset, source = dataset_cache(database, market, args.cache_dir,
        SimpleNamespace(max_train_samples=200000, max_eval_samples=60000, seed=42))
    sessions = read_sessions(database)
    atomic_json(folder / "source.json", source)
    results = {}
    for fold in FOLDS:
        _, arrays, events, manifest = prepare_fold(dataset, sessions, fold, folder / fold / "features")
        atomic_json(folder / fold / "splits.json", manifest)
        results[fold] = {}
        for name in MODEL_NAMES:
            print(f"TRAIN {market}/{fold}/{name}/42", flush=True)
            result, _ = trial(name, 42, arrays, events, folder / fold / f"{name}-42", args)
            results[fold][name] = result
        del arrays, events
        gc.collect()
    ranking = architecture_ranking(results)
    selected = ranking[0]["architecture"]
    lock = {"market": market, "selected": selected, "ranking": ranking,
            "selection_rule": PROTOCOL["selection"], "source": source}
    locked_path = folder / "selection_locked.json"
    if locked_path.exists() and json.loads(locked_path.read_text()) != canonical(lock):
        raise ValueError("Frozen architecture selection changed")
    atomic_json(locked_path, lock)
    print(f"LOCK {market}: {selected}", flush=True)
    ensembles = {}
    for fold in FOLDS:
        _, arrays, events, _ = prepare_fold(dataset, sessions, fold, folder / fold / "features")
        members = [load_trial_predictions(folder / fold / f"{selected}-42", selected)]
        seed_results = [results[fold][selected]]
        for seed in (43, 44):
            result, predictions = trial(selected, seed, arrays, events, folder / fold / f"{selected}-{seed}", args)
            seed_results.append(result)
            members.append(predictions)
        predictions = average_predictions(members)
        assessment = assess(predictions, events)
        ensemble = {"seed_results": seed_results, **assessment,
                    "member_sha256": {str(seed): file_hash(model_path(folder / fold / f"{selected}-{seed}", selected)) for seed in (42, 43, 44)}}
        atomic_json(folder / fold / "ensemble.json", ensemble)
        np.savez(folder / fold / "ensemble_development_predictions.npz",
                 **{part + "_success": raw["success_logits"] for part, raw in predictions.items()},
                 **{part + "_stop": raw["stop_logits"] for part, raw in predictions.items() if raw["stop_logits"] is not None})
        ensembles[fold] = ensemble
        print(f"ENSEMBLE {market}/{fold}: {assessment['audit']['signal_count']} signals, precision {assessment['audit']['precision']}", flush=True)
        del arrays, events, predictions, members
        gc.collect()
    if file_hash(database) != source["database_sha256"]:
        raise RuntimeError("Cleaned DB changed during training")
    summary = {"market": market, "completed": True, "selected": selected, "ranking": ranking,
               "results": results, "ensembles": ensembles, "source": source,
               "research_qualified": all(item["qualification"]["qualified"] and item["policy_selection"]["calibration_qualified"] for item in ensembles.values()),
               "research_only": True, "deployment_allowed": False}
    atomic_json(folder / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mark1/selective-20260916"))
    parser.add_argument("--db-dir", type=Path, default=Path("C:/Users/user/Desktop/dockdack-data-collection/data/kiwoom_daily/clean-20260916-v1"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/mark1/cache"))
    parser.add_argument("--task-type", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("Positive thread count required")
    import catboost, lightgbm
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = {"protocol": PROTOCOL, "task_type": args.task_type, "threads": args.threads,
              "versions": {"numpy": np.__version__, "catboost": catboost.__version__, "lightgbm": lightgbm.__version__},
              "code_sha256": {name: file_hash(ROOT / name) for name in CODE_FILES},
              "db_dir": str(args.db_dir.resolve()), "cache_dir": str(args.cache_dir.resolve())}
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != canonical(frozen):
            raise ValueError("Frozen experiment differs; use a fresh output folder, never overwrite")
    else:
        atomic_json(protocol_path, frozen)
        atomic_json(args.output_dir / "started.json", {"utc": datetime.now(timezone.utc).isoformat()})
    protected = {str(path): file_hash(path) for path in (ROOT / "models/mark1/domestic.pt", ROOT / "models/mark1/us.pt")}
    print("PRECISION-FIRST RESEARCH ONLY; 2025+ NOT USED", flush=True)
    summaries = {market: run_market(market, args) for market in ("domestic", "us")}
    if any(file_hash(path) != digest for path, digest in protected.items()):
        raise RuntimeError("An original checkpoint changed during research")
    if any(file_hash(ROOT / name) != digest for name, digest in frozen["code_sha256"].items()):
        raise RuntimeError("Frozen research code changed during training")
    atomic_json(args.output_dir / "protected_checkpoints.json", protected)
    atomic_json(args.output_dir / "summary.json", summaries)
    print("ALL MARKETS TRAINING COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
