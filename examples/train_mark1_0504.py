"""Independent +0.5%/-0.4% research retraining; never loads broker or GUI code."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from dockdack.mark1_0504_data import TARGET, FEATURE_NAMES, features_from_history, event_data
from dockdack.mark1_data import barrier_outcomes as old_outcomes
from dockdack.mark1_deep_data import FOLDS
from dockdack.mark1_metrics import fit_calibration, calibrated_probability
from dockdack.mark1_selective_models import fit_model, predict_raw, model_path
from dockdack.mark1_selective_policy import evaluate_signals
from examples.backtest_mark1_deep import load_frozen_cache
from examples.train_mark1 import file_hash, jsonable
from examples.train_mark1_deep import read_sessions
from examples.train_mark1_selective import selective_splits, describe, average_predictions

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = {"domestic": "cat_joint6", "us": "cat_binary8"}
SEEDS = (42, 43, 44)
QUARANTINE = {"domestic": (), "us": ("FCEL", "BNED", "BBSI")}
PARTS = ("probability_calibration", "policy_calibration", "audit")
CODE_FILES = (
    "examples/train_mark1_0504.py", "dockdack/mark1_0504_data.py",
    "dockdack/mark1_selective_features.py", "dockdack/mark1_selective_models.py",
    "dockdack/mark1_selective_policy.py", "dockdack/mark1_metrics.py",
    "dockdack/mark1_data.py", "dockdack/mark1_deep_data.py", "dockdack/mark1_deep_validation.py",
    "examples/train_mark1.py", "examples/train_mark1_deep.py",
    "examples/train_mark1_selective.py", "examples/backtest_mark1_deep.py",
)
PROTOCOL = {
    "version": "mark1-0504-v1", "target": TARGET, "take_profit_pct": .5, "stop_loss_pct": .4,
    "buy_threshold": .5, "buy_comparison": "strict_greater_than", "both_touch": "stop_first_failure",
    "feature_names": list(FEATURE_NAMES), "lookback": 30, "folds": FOLDS,
    "architectures": ARCHITECTURES, "architecture_selection": "fixed prior to training; same prior selected per-market structures",
    "ensemble_seeds": list(SEEDS), "ensemble": "mean raw logits then independently fitted Platt calibration",
    "max_train": 1_500_000, "max_tune": 150_000, "max_iterations": 3000, "early_stopping": 150,
    "price_augmentation": False, "entry": "observed session OPEN; same as the previous selected tree pipeline",
    "calibration": "first half-year only; second half-year validation after 30-session purge; no threshold search",
    "quarantine": QUARANTINE, "quarantine_scope": "entire known-bad US symbols excluded before folds/sampling; raw DB unchanged",
    "data_warnings": ["known price-basis discontinuities quarantined, not corrected", "other corporate-action issues may remain; SONY flagged", "no intraday first-touch ordering", "survivorship and observed-universe limitations remain"],
    "policy": {"threshold": .5, "stop_probability_cap": 1.},
    "research_gate": {"minimum_signals": 50, "minimum_days": 20, "minimum_symbols": 10,
                      "cost_bps": 20, "precision_lower_exclusive": (0.4 + 0.2) / (0.5 + 0.4), "net_lower_exclusive": 0.},
    "evaluation": "2022/2024 development audit; 2025+ reused historical comparison, not a pristine unseen test",
    "deployment": "research only; no GUI promotion or orders",
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Refusing symlink artifacts")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def canonical(value):
    return json.loads(json.dumps(jsonable(value), allow_nan=False))


def load_experiment_dataset(market, cache_dir, old_run):
    """Read immutable raw windows, not old labels, and predeclare symbol quarantine."""
    if market not in ARCHITECTURES:
        raise ValueError("Unknown market")
    source = json.loads((Path(old_run) / market / "source.json").read_text(encoding="utf-8"))
    dataset = load_frozen_cache(source, market, Path(cache_dir))
    excluded = [row for row in dataset.manifest["symbols"] if row["symbol"] in QUARANTINE[market]]
    ids = [row["symbol_id"] for row in excluded]
    keep = ~np.isin(dataset.symbol_ids, ids)
    original_indices = np.flatnonzero(keep)
    mapping = np.full(len(keep), -1, dtype=np.int64)
    mapping[original_indices] = np.arange(len(original_indices))
    splits = {key: mapping[value[keep[value]]] for key, value in dataset.splits.items()}
    manifest = {**dataset.manifest, "target": TARGET,
                "symbols": [row for row in dataset.manifest["symbols"] if row["symbol_id"] not in ids],
                "selected_symbols": len(dataset.manifest["symbols"]) - len(excluded)}
    experiment = {"target": TARGET, "raw_cache_target_is_not_used_for_labels": True,
                  "raw_samples": len(keep), "eligible_samples": int(keep.sum()),
                  "raw_bar_count": len(dataset.bars), "eligible_symbol_count": manifest["selected_symbols"],
                  "quarantine": [{"symbol": row["symbol"], "symbol_id": row["symbol_id"],
                                  "excluded_samples": int((dataset.symbol_ids == row["symbol_id"]).sum())}
                                 for row in excluded],
                  "original_indices_sha256": hashlib.sha256(original_indices.tobytes()).hexdigest(),
                  "warnings": PROTOCOL["data_warnings"]}
    if ids:
        dataset = replace(dataset, starts=dataset.starts[keep], target_dates=dataset.target_dates[keep],
                          symbol_ids=dataset.symbol_ids[keep], target_ohlc=dataset.target_ohlc[keep],
                          splits=splits, manifest=manifest)
    else:
        dataset = replace(dataset, manifest=manifest)
    return dataset, source, experiment


def feature_array(dataset, indices, *, batch_size=8192):
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices < 0)
            or np.any(indices >= len(dataset.starts)) or type(batch_size) is not int or batch_size < 1):
        raise ValueError("Valid indices and positive batch size required")
    result = np.empty((len(indices), len(FEATURE_NAMES)), np.float32)
    offsets = np.arange(30)
    for first in range(0, len(indices), batch_size):
        batch = indices[first:first + batch_size]
        history = dataset.bars[dataset.starts[batch, None] + offsets[None, :]]
        result[first:first + len(batch)] = features_from_history(history, dataset.target_ohlc[batch, 0])
    return result


def qualification(metrics):
    reasons = []
    for key, minimum in (("signal_count", 50), ("signal_days", 20), ("symbol_count", 10)):
        if metrics.get(key, 0) < minimum:
            reasons.append(f"insufficient_{key}")
    bounds = metrics.get("block_bootstrap", {})
    lower = bounds.get("precision_lower")
    breakeven = (0.4 + 0.2) / (0.5 + 0.4)
    if lower is None or not np.isfinite(lower) or lower <= breakeven:
        reasons.append("precision_lower_not_above_two_outcome_cost_breakeven_2_over_3")
    net = bounds.get("net_mean_lower")
    if net is None or not np.isfinite(net) or net <= 0:
        reasons.append("net_return_lower_not_positive_after_20bps")
    if metrics.get("cost_bps") != 20 or bounds.get("cost_bps") != 20:
        reasons.append("incorrect_cost_assumption")
    return {"qualified": not reasons, "reasons": reasons, "two_outcome_breakeven": breakeven,
            "note": "daily timeout returns differ; net-return lower bound independently required"}


def assess(predictions, events):
    raw, event = predictions["probability_calibration"], events["probability_calibration"]
    calibration = {"success": fit_calibration(raw["success_logits"], event["labels"]),
                   "stop": fit_calibration(raw["stop_logits"], np.isin(event["classes"], [1, 2]))
                   if raw["stop_logits"] is not None else None}
    result = {"calibration": calibration, "policy": PROTOCOL["policy"]}
    for part, name in (("policy_calibration", "validation"), ("audit", "audit")):
        event = events[part]
        probabilities = calibrated_probability(predictions[part]["success_logits"], calibration["success"])
        metrics = evaluate_signals(event["labels"], probabilities, event["gross"], event["dates"],
                                   event["symbols"], probabilities > .5, cost_bps=20)
        result[name] = metrics
    result["qualification"] = qualification(result["audit"])
    result["validation_qualification"] = qualification(result["validation"])
    return result


def prepare_fold(dataset, sessions, fold, folder):
    splits = selective_splits(dataset, sessions, fold, max_train=PROTOCOL["max_train"], max_tune=PROTOCOL["max_tune"])
    contract = {"target": TARGET, "features": list(FEATURE_NAMES), "splits": describe(dataset, splits),
                "index_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in splits.items()}}
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / "contract.json"
    if saved.exists() and json.loads(saved.read_text(encoding="utf-8")) != canonical(contract):
        raise ValueError("Feature cache contract mismatch")
    if not saved.exists() and any(folder.iterdir()):
        raise ValueError("Nonempty feature folder without ownership contract")
    atomic_json(saved, contract)
    digest_file = folder / "hashes.json"
    hashes = json.loads(digest_file.read_text()) if digest_file.exists() else {}
    arrays, events, distributions = {}, {}, {}
    for part, indices in splits.items():
        path = folder / f"{part}.npy"
        if path.exists():
            if hashes.get(part) != file_hash(path):
                raise ValueError("Feature cache checksum mismatch")
        else:
            print(f"FEATURES {fold}/{part}: {len(indices):,} raw 30-day windows", flush=True)
            values = feature_array(dataset, indices)
            temporary = path.with_name(path.stem + ".building.npy")
            np.save(temporary, values, allow_pickle=False)
            temporary.replace(path)
            del values
            hashes[part] = file_hash(path)
            atomic_json(digest_file, hashes)
        arrays[part] = np.load(path, mmap_mode="r", allow_pickle=False)
        if arrays[part].shape != (len(indices), len(FEATURE_NAMES)) or arrays[part].dtype != np.float32:
            raise ValueError("Invalid cached feature dimensions")
        events[part] = event_data(dataset, indices)
        raw = dataset.target_ohlc[indices]
        old = old_outcomes(raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 0])
        classes = events[part]["classes"]
        distributions[part] = {"samples": len(indices), "new_class_counts": np.bincount(classes, minlength=4).tolist(),
                               "new_success_rate": float(events[part]["labels"].mean()),
                               "old_success_rate_same_rows": float(old["success"].mean()),
                               "old_both_touch_rate": float(old["both_touch"].mean()),
                               "new_both_touch_rate": float((classes == 2).mean())}
    atomic_json(folder / "label_audit.json", distributions)
    return arrays, events, contract


def run_market(market, args):
    folder = args.output_dir / market
    folder.mkdir(parents=True, exist_ok=True)
    dataset, source, experiment = load_experiment_dataset(market, args.cache_dir, args.old_run)
    atomic_json(folder / "source.json", source)
    atomic_json(folder / "experiment_data.json", experiment)
    sessions = read_sessions(Path(source["database_path"]))
    architecture = ARCHITECTURES[market]
    ensembles = {}
    for fold in FOLDS:
        arrays, events, contract = prepare_fold(dataset, sessions, fold, folder / fold / "features")
        atomic_json(folder / fold / "splits.json", contract["splits"])
        predictions, members = [], []
        for seed in SEEDS:
            trial = folder / fold / f"{architecture}-{seed}"
            print(f"TRAIN {market}/{fold}/{architecture}/seed{seed} GPU={args.task_type}", flush=True)
            model, metadata = fit_model(architecture, arrays["train"], events["train"]["classes"],
                arrays["tune"], events["tune"]["classes"], trial, seed=seed, task_type=args.task_type,
                max_iterations=PROTOCOL["max_iterations"], early_stopping=PROTOCOL["early_stopping"], threads=args.threads)
            prediction = {part: predict_raw(model, architecture, arrays[part]) for part in PARTS}
            assessment = assess(prediction, events)
            atomic_json(trial / "assessment.json", {"target": TARGET, **assessment})
            np.savez(trial / "development_predictions.npz", **{part + "_success": raw["success_logits"] for part, raw in prediction.items()},
                     **{part + "_stop": raw["stop_logits"] for part, raw in prediction.items() if raw["stop_logits"] is not None})
            predictions.append(prediction)
            members.append({"seed": seed, "best_iteration": metadata["best_iteration"],
                            "trained_iterations": metadata["trained_iterations"], "fit_seconds": metadata["fit_seconds"],
                            "model_sha256": metadata["model_sha256"]})
            print(f"DONE {market}/{fold}/seed{seed}: trees={metadata['best_iteration']} signals={assessment['audit']['signal_count']} precision={assessment['audit']['precision']}", flush=True)
            del model
            gc.collect()
        averaged = average_predictions(predictions)
        ensemble = {"target": TARGET, "members": members, **assess(averaged, events),
                    "member_sha256": {str(seed): file_hash(model_path(folder / fold / f"{architecture}-{seed}", architecture)) for seed in SEEDS}}
        atomic_json(folder / fold / "ensemble.json", ensemble)
        np.savez(folder / fold / "ensemble_development_predictions.npz",
                 **{part + "_success": raw["success_logits"] for part, raw in averaged.items()},
                 **{part + "_stop": raw["stop_logits"] for part, raw in averaged.items() if raw["stop_logits"] is not None})
        ensembles[fold] = ensemble
        print(f"ENSEMBLE {market}/{fold}: {ensemble['audit']['signal_count']} signals; qualified={ensemble['qualification']['qualified']}", flush=True)
        del arrays, events, predictions, averaged
        gc.collect()
    if file_hash(source["database_path"]) != source["database_sha256"]:
        raise RuntimeError("Source database changed")
    summary = {"market": market, "completed": True, "target": TARGET, "selected": architecture,
               "source": source, "source_usage": "immutable raw OHLCV windows only; old cached target labels never used",
               "experiment_data": experiment, "ensembles": ensembles,
               "research_qualified": all(value['qualification']['qualified'] and value['validation_qualification']['qualified'] for value in ensembles.values()),
               "research_only": True, "deployment_allowed": False, "intraday_path_verified": False}
    atomic_json(folder / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mark1/half-20260920")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/mark1/cache")
    parser.add_argument("--old-run", type=Path, default=ROOT / "outputs/mark1/selective-20260916")
    parser.add_argument("--task-type", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("Positive threads required")
    args.output_dir = args.output_dir.resolve()
    import catboost
    frozen = {"protocol": PROTOCOL, "task_type": args.task_type, "threads": args.threads,
              "versions": {"numpy": np.__version__, "catboost": catboost.__version__},
              "code_sha256": {name: file_hash(ROOT / name) for name in CODE_FILES},
              "raw_cache_dir": str(args.cache_dir.resolve()), "old_run": str(args.old_run.resolve())}
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != canonical(frozen):
            raise ValueError("Experiment protocol differs; use a new output folder")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Nonempty unowned experiment folder")
    atomic_json(protocol_path, frozen)
    protected = {str(path.resolve()): file_hash(path) for path in (ROOT / "models/mark1_prototype").rglob("*") if path.is_file()}
    start = time.monotonic()
    atomic_json(args.output_dir / "started.json", {"utc": datetime.now(timezone.utc).isoformat()})
    print("0504 RESEARCH: same prior architectures; new labels+features; NO TRADING", flush=True)
    summaries = {market: run_market(market, args) for market in ARCHITECTURES}
    if any(file_hash(path) != digest for path, digest in protected.items()):
        raise RuntimeError("Original prototype modified")
    if any(file_hash(ROOT / path) != digest for path, digest in frozen["code_sha256"].items()):
        raise RuntimeError("Frozen training code changed")
    atomic_json(args.output_dir / "summary.json", summaries)
    atomic_json(args.output_dir / "completed.json", {"utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-start,
                                                    "original_prototype_sha256": protected, "no_orders": True})
    print("BOTH MARKETS COMPLETED; original artifacts unchanged", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
