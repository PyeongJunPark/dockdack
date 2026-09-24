"""Matched frozen +0.5%/-0.4% versus +1%/-0.9% reused-history comparison.

This script never trains, tunes thresholds, edits the source database, contacts
brokers, or promotes a model. Both completed new-market training summaries are
required before reading evaluation prices. Entries use the actual daily OPEN;
the daily bars cannot verify an arbitrary intraday path or executable fills.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from dockdack.mark1_0504_backtest import simulate_portfolio
from dockdack.mark1_backtest_data import load_price_panel
from dockdack.mark1_selective_models import load_model, model_path, predict_raw
from dockdack.mark1_selective_policy import evaluate_signals, qualification as old_qualification
from examples.backtest_mark1 import attach_names, date_string, make_candidates
from examples.backtest_mark1_deep import _calibration_contract, common_test_indices
from examples.backtest_mark1_selective import signal_stability
from examples.backtest_mark1_selective import verify_training_artifacts as verify_old_training
from examples.train_mark1 import file_hash, save_json
from examples.train_mark1_deep import read_sessions
from examples.train_mark1_selective import calibrated
from examples.train_mark1_selective import event_data as old_event_data
from examples.train_mark1_selective import feature_array as old_feature_array


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
COSTS = (0, 10, 20, 40)
ARCHITECTURES = {"domestic": "cat_joint6", "us": "cat_binary8"}
COMPARATORS = {"old_0109": (.01, .009), "new_0504": (.005, .004)}
EVALUATION_STATUS = "reused_2025_plus_history_not_an_untouched_final_test"
EVALUATION_CODE = (
    "examples/backtest_mark1_0504.py", "dockdack/mark1_0504_backtest.py",
    "dockdack/mark1_backtest_data.py", "examples/backtest_mark1.py",
    "examples/backtest_mark1_deep.py", "examples/backtest_mark1_selective.py",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def verify_new_training(training_run):
    """Validate both markets, code, calibration and every ensemble member."""
    from examples import train_mark1_0504 as training

    training_run = Path(training_run)
    complete = read_json(training_run / "summary.json")
    frozen = read_json(training_run / "protocol.json")
    if set(complete) != set(ARCHITECTURES):
        raise ValueError("Both new-target markets must complete before evaluation")
    if frozen.get("protocol") != _canonical(training.PROTOCOL):
        raise ValueError("New-target frozen training protocol mismatch")
    code = frozen.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(training.CODE_FILES):
        raise ValueError("Missing new-target training code hashes")
    for name, digest in code.items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"New-target training code changed: {name}")
    paths = [training_run / "summary.json", training_run / "protocol.json"]
    for market, architecture in ARCHITECTURES.items():
        summary, folder = complete[market], training_run / market
        if (summary.get("completed") is not True or summary.get("market") != market
                or summary.get("target") != training.TARGET
                or summary.get("selected") != architecture
                or summary.get("research_only") is not True
                or summary.get("deployment_allowed") is not False
                or read_json(folder / "summary.json") != summary
                or summary.get("experiment_data", {}).get("target") != training.TARGET):
            raise ValueError("New-target completed summary/provenance mismatch")
        if (read_json(folder / "source.json") != summary.get("source")
                or read_json(folder / "experiment_data.json") != summary.get("experiment_data")):
            raise ValueError("New-target source and quarantine files disagree")
        if set(summary.get("ensembles", {})) != {"walk_2022", "walk_2024"}:
            raise ValueError("Both new-target chronological ensembles are required")
        paths.extend((folder / "summary.json", folder / "source.json", folder / "experiment_data.json"))
        for fold, ensemble in summary["ensembles"].items():
            ensemble_path = folder / fold / "ensemble.json"
            if read_json(ensemble_path) != ensemble or ensemble.get("target") != training.TARGET:
                raise ValueError("New-target ensemble summary mismatch")
            calibration = ensemble.get("calibration", {})
            if set(calibration) != {"success", "stop"}:
                raise ValueError("Missing new-target ensemble calibration")
            _calibration_contract(calibration["success"])
            if architecture == "cat_joint6":
                _calibration_contract(calibration["stop"])
            elif calibration["stop"] is not None:
                raise ValueError("Binary model cannot expose stop calibration")
            member_hashes = ensemble.get("member_sha256", {})
            if set(member_hashes) != {str(seed) for seed in SEEDS}:
                raise ValueError("Missing new-target ensemble hashes")
            paths.append(ensemble_path)
            for seed in SEEDS:
                trial = folder / fold / f"{architecture}-{seed}"
                artifact = model_path(trial, architecture)
                metadata_path, request_path = trial / "metadata.json", trial / "request.json"
                metadata, request = read_json(metadata_path), read_json(request_path)
                digest = file_hash(artifact)
                if (digest != member_hashes[str(seed)] or metadata.get("model_sha256") != digest
                        or metadata.get("request") != request or request.get("seed") != seed
                        or request.get("model_name") != architecture
                        or request.get("train_shape", [None, None])[1:] != [184]
                        or request.get("wrapper_sha256") != file_hash(ROOT / "dockdack/mark1_selective_models.py")):
                    raise ValueError("New-target model training/checksum mismatch")
                # Loading verifies the native artifact sidecar/schema too.
                load_model(architecture, artifact)
                paths.extend((artifact, artifact.with_name(artifact.name + ".json"),
                              metadata_path, request_path))
    hashes = {str(path.resolve()): file_hash(path) for path in paths}
    hashes.update({str((ROOT / name).resolve()): digest for name, digest in code.items()})
    return complete, hashes


def infer_ensemble(dataset, indices, market, training_run, summary, feature_function, batch_size):
    """CPU native ensemble, frozen raw-logit averaging then calibration."""
    architecture, ensemble = summary["selected"], summary["ensembles"]["walk_2024"]
    models, hashes = [], {}
    for seed in SEEDS:
        path = model_path(Path(training_run) / market / "walk_2024" / f"{architecture}-{seed}", architecture)
        digest = file_hash(path)
        if digest != ensemble["member_sha256"][str(seed)]:
            raise ValueError("Frozen model changed before inference")
        models.append(load_model(architecture, path))
        hashes[str(path.resolve())] = digest
    success = np.empty(len(indices), dtype=np.float64)
    stop = np.empty(len(indices), dtype=np.float64) if architecture == "cat_joint6" else None
    for start in range(0, len(indices), batch_size):
        picked = indices[start:start + batch_size]
        features = feature_function(dataset, picked)
        raw = [predict_raw(model, architecture, features) for model in models]
        success[start:start + len(picked)] = np.mean([item["success_logits"] for item in raw], axis=0)
        if stop is not None:
            stop[start:start + len(picked)] = np.mean([item["stop_logits"] for item in raw], axis=0)
        if start % (batch_size * 50) == 0:
            print(json.dumps({"phase": "frozen_inference", "market": market,
                              "run": str(training_run), "done": start + len(picked), "total": len(indices)}), flush=True)
    probabilities, stops = calibrated({"success_logits": success, "stop_logits": stop}, ensemble["calibration"])
    for path, digest in hashes.items():
        if file_hash(path) != digest:
            raise RuntimeError("Frozen model changed during inference")
    del models
    gc.collect()
    return probabilities, stops, hashes


def run_market(market, args, new_summary, new_hashes):
    from examples import train_mark1_0504 as training

    started = time.monotonic()
    folder = args.output_dir / market
    if folder.exists():
        raise FileExistsError("Refusing to overwrite a small-barrier backtest")
    old, old_source, old_hashes = verify_old_training(args.old_run, market)
    dataset, source, experiment_data = training.load_experiment_dataset(market, args.cache_dir, args.old_run)
    if (source != old_source or source != new_summary["source"]
            or experiment_data != new_summary["experiment_data"]):
        raise ValueError("Old/new experiment source contracts differ")
    database = Path(source["database_path"])
    indices = common_test_indices(dataset, read_sessions(database))
    start, end = int(dataset.target_dates[indices].min()), int(dataset.target_dates[indices].max())
    prices, sessions, panel = load_price_panel(database, market, dataset.manifest["symbols"], start, end)
    for index in indices:
        key = int(dataset.symbol_ids[index]), int(dataset.target_dates[index])
        if key not in prices or not np.array_equal(np.asarray(prices[key]), dataset.target_ohlc[index]):
            raise ValueError(f"Approved evaluation target differs from source: {key}")
    for symbol, day in panel["zero_volume_keys"]:
        prices.pop((symbol, day), None)
    panel["zero_volume_execution_policy"] = "Unfillable; missing held-price path, never fabricated fill"
    if file_hash(database) != source["database_sha256"]:
        raise RuntimeError("Clean source changed before comparison")
    folder.mkdir(parents=True, exist_ok=False)
    hashes = {**new_hashes, **old_hashes}
    predictions, stops, events = {}, {}, {}
    for name, run, model, features, labels in (
        ("old_0109", args.old_run, old, old_feature_array, old_event_data),
        ("new_0504", args.training_run, new_summary, training.feature_array, training.event_data),
    ):
        predictions[name], stops[name], inferred_hashes = infer_ensemble(
            dataset, indices, market, run, model, features, args.batch_size)
        for path, digest in inferred_hashes.items():
            if hashes.get(path) != digest:
                raise RuntimeError("Comparator model differs from verified training artifacts")
        events[name] = labels(dataset, indices)
    dates, symbols = dataset.target_dates[indices], dataset.symbol_ids[indices]
    for event in events.values():
        if not np.array_equal(event["dates"], dates) or not np.array_equal(event["symbols"], symbols):
            raise ValueError("Comparators do not share identical event keys")
    np.savez(folder / "predictions.npz", sample_indices=indices, dates=dates, symbol_ids=symbols,
        evaluation_status=EVALUATION_STATUS,
        **{name + "_probabilities": p for name, p in predictions.items()},
        **{name + "_labels": event["labels"] for name, event in events.items()},
        **{name + "_gross_returns": event["gross"] for name, event in events.items()},
        **{name + "_stop_probabilities": p for name, p in stops.items() if p is not None})
    initial = args.initial_krw if market == "domestic" else args.initial_usd
    summary = {"market": market, "currency": "KRW" if market == "domestic" else "USD",
        "research_evaluation": EVALUATION_STATUS, "samples": len(indices), "initial_cash": initial,
        "range": {"first": date_string(start), "last": date_string(end), "sessions": len(sessions)},
        "source": source, "experiment_data": experiment_data, "price_panel": panel,
        "artifact_sha256": hashes, "sample_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "models": {}, "cost_sensitivity": [], "research_only": True, "deployment_allowed": False,
        "candidate_contract": "Identical approved quarantined events; actual OPEN; strict calibrated probability > 0.5",
        "comparison_limitations": [
            "Each model is evaluated against its OWN barrier target; precision labels differ.",
            "New training excludes FCEL/BNED/BBSI whereas frozen old training did not; differences cannot be attributed solely to target size.",
            "2025+ was inspected in prior experiments: reused history, not untouched final-test evidence.",
            "Same daily OHLC may touch both barriers; both-hit is conservatively STOP FIRST, not observed sequence.",
            "Other undiagnosed data defects may remain; SONY split-price anomaly is not certified repaired.",
            "Inference sees 30 completed sessions plus current OPEN; arbitrary intraday entries need separate validation.",
        ]}
    for name, (take, stop) in COMPARATORS.items():
        event, probabilities = events[name], predictions[name]
        selected = probabilities > .5
        metrics = evaluate_signals(event["labels"], probabilities, event["gross"], dates, symbols, selected, cost_bps=20)
        candidates = make_candidates(dataset, indices, probabilities)
        signalled = {row["symbol_id"] for row in candidates}
        relevant = {key: value for key, value in prices.items() if key[0] in signalled}
        gate = (old_qualification if name == "old_0109" else training.qualification)(metrics)
        row = {"classification": metrics, "raw_signals": len(candidates), "qualification": gate,
            "take_profit": take, "stop_loss": stop, "portfolios": {}, "diagnostic_only": True,
            "stability": signal_stability(event["labels"], event["gross"], dates, symbols, selected)}
        save_json(folder / f"{name}-signals.json", {"research_evaluation": EVALUATION_STATUS,
                                                   "diagnostic_only": True, "signals": candidates})
        for mode in ("carry", "eod"):
            for cost in COSTS:
                simulation = simulate_portfolio(candidates, relevant, sessions, initial_cash=initial,
                    exit_mode=mode, max_positions=args.max_positions, position_fraction=args.position_fraction,
                    cost_bps=cost, volume_fraction=args.volume_fraction, take_profit=take, stop_loss=stop)
                attach_names(simulation, dataset.manifest["symbols"])
                simulation.update(market=market, model=name, exit_mode=mode, cost_bps=cost,
                    research_evaluation=EVALUATION_STATUS, diagnostic_only=True, deployment_allowed=False)
                save_json(folder / f"{name}-{mode}-cost{cost}.json", simulation)
                summary["cost_sensitivity"].append({"model": name, **simulation["summary"]})
                if cost == 20:
                    row["portfolios"][mode] = simulation["summary"]
                    print(json.dumps({"phase": "portfolio_complete", "market": market,
                                      "model": name, "mode": mode, **simulation["summary"]}), flush=True)
        summary["models"][name] = row
        save_json(folder / "summary.json", summary)
    for path, digest in hashes.items():
        if file_hash(path) != digest:
            raise RuntimeError(f"Protected artifact changed during evaluation: {path}")
    if file_hash(database) != source["database_sha256"]:
        raise RuntimeError("Clean source changed during evaluation")
    summary.update(completed=True, elapsed_seconds=round(time.monotonic() - started, 3))
    save_json(folder / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, default=Path("outputs/mark1/half-20260920"))
    parser.add_argument("--old-run", type=Path, default=Path("outputs/mark1/selective-20260916"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mark1/half-backtest-20260920"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/mark1/cache"))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--initial-krw", type=float, default=10_000_000)
    parser.add_argument("--initial-usd", type=float, default=10_000)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--position-fraction", type=float, default=.05)
    parser.add_argument("--volume-fraction", type=float, default=.001)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_positions < 1:
        parser.error("Positive batch size and maximum positions required")
    for name in ("initial_krw", "initial_usd", "position_fraction", "volume_fraction"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"Positive finite {name} required")
    if args.position_fraction > 1 or args.volume_fraction > 1:
        parser.error("Position and volume fractions must not exceed one")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output folder; overwrite is forbidden")
    # Training completion and code/model checks precede evaluation data access.
    training, hashes = verify_new_training(args.training_run)
    for market in ARCHITECTURES:
        verify_old_training(args.old_run, market)
    code = {name: file_hash(ROOT / name) for name in EVALUATION_CODE}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_json(args.output_dir / "config.json", {**vars(args), "costs_bps": list(COSTS),
        "comparators": COMPARATORS, "code_sha256": code, "research_evaluation": EVALUATION_STATUS,
        "started_utc": datetime.now(timezone.utc).isoformat(), "deployment_allowed": False})
    summaries = {market: run_market(market, args, training[market], hashes) for market in ARCHITECTURES}
    if any(file_hash(ROOT / name) != digest for name, digest in code.items()):
        raise RuntimeError("Evaluation implementation changed during run")
    save_json(args.output_dir / "summary.json", summaries)
    print("ALL 32 DIAGNOSTIC PORTFOLIOS COMPLETE; NO MODEL PROMOTION", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
