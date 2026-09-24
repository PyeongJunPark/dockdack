"""Frozen precision-first research versus two prior models on reused 2025+ data.

No fitting, threshold search, deployment or orders. Both markets must have
completed selective training before evaluation data can be opened. All four
comparators use identical eligible events. The unfiltered and selective views
retain exactly the same calibrated probabilities and differ only in abstention.
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
from types import SimpleNamespace

import numpy as np

from dockdack.mark1_backtest import simulate_portfolio
from dockdack.mark1_backtest_data import load_price_panel
from dockdack.mark1_selective_models import CLASS_NAMES, MODEL_NAMES, OWNER, SCHEMA_VERSION, load_model, model_path, predict_raw
from dockdack.mark1_selective_policy import apply_policy, evaluate_signals, qualification
from examples.backtest_mark1 import attach_names, date_string, make_candidates
from examples import backtest_mark1_deep as deep
from examples.train_mark1 import file_hash, save_json
from examples.train_mark1_selective import CODE_FILES, FEATURE_NAMES, FOLDS, PROTOCOL, ROOT, calibrated, canonical, event_data, feature_array


EVALUATION_STATUS = deep.EVALUATION_STATUS
SEEDS = (42, 43, 44)
COSTS = (0, 10, 20, 40)
COMPARATORS = ("baseline", "deep", "unfiltered", "selective")
EVALUATION_CODE = (
    "examples/backtest_mark1_selective.py", "examples/backtest_mark1_deep.py",
    "examples/backtest_mark1.py", "dockdack/mark1_backtest.py",
    "dockdack/mark1_backtest_data.py", "dockdack/mark1_deep_validation.py",
)


def read_json(path):
    return deep._read_json(path)


def _hashes(paths):
    return {str(Path(path).resolve()): file_hash(path) for path in paths}


def _calibrators(value, architecture):
    if not isinstance(value, dict) or set(value) != {"success", "stop"}:
        raise ValueError("Missing frozen ensemble calibration schema")
    deep._calibration_contract(value["success"])
    if architecture == "cat_joint6":
        deep._calibration_contract(value["stop"])
    elif value["stop"] is not None:
        raise ValueError("Binary architecture cannot have a stop calibration")


def verify_training_artifacts(training_run, market):
    """Validate both completed markets and selected model hashes before data."""
    training_run = Path(training_run)
    complete = read_json(training_run / "summary.json")
    frozen = read_json(training_run / "protocol.json")
    if (set(complete) != {"domestic", "us"}
            or any(not isinstance(row, dict) or row.get("completed") is not True
                   or row.get("market") != key for key, row in complete.items())):
        raise ValueError("Both markets must finish selective training before 2025+ evaluation")
    if frozen.get("protocol") != canonical(PROTOCOL):
        raise ValueError("Frozen selective training protocol mismatch")
    code_hashes = frozen.get("code_sha256")
    if not isinstance(code_hashes, dict) or set(code_hashes) != set(CODE_FILES):
        raise ValueError("Missing frozen selective code hashes")
    for name, digest in code_hashes.items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Frozen selective training code changed: {name}")
    protected = read_json(training_run / "protected_checkpoints.json")
    expected_originals = {str(ROOT / f"models/mark1/{key}.pt"): value
                          for key, value in deep.ORIGINAL_CHECKPOINT_SHA256.items()}
    if protected != expected_originals or any(file_hash(path) != value for path, value in protected.items()):
        raise ValueError("Original protected baseline checkpoint changed")
    paths = [training_run / name for name in ("summary.json", "protocol.json", "protected_checkpoints.json")]
    sources = {}
    # The other market's completion cannot be asserted by a bare flag: check
    # its source/selection/ensemble files and selected native artifacts too.
    for key, summary in complete.items():
        folder = training_run / key
        summary_path, source_path, lock_path = (folder / name for name in
                                               ("summary.json", "source.json", "selection_locked.json"))
        source, lock = read_json(source_path), read_json(lock_path)
        selected = summary.get("selected")
        if (read_json(summary_path) != summary or summary.get("source") != source
                or summary.get("research_only") is not True or summary.get("deployment_allowed") is not False
                or selected not in MODEL_NAMES or lock.get("market") != key
                or lock.get("source") != source or lock.get("selected") != selected
                or lock.get("ranking") != summary.get("ranking")
                or lock.get("selection_rule") != PROTOCOL["selection"]):
            raise ValueError("Frozen selective source/architecture selection mismatch")
        if (source.get("market") != key or source.get("target") != PROTOCOL["target"]
                or source.get("version") != 2 or source.get("purge_sessions") != 30
                or source.get("start") != "2010-01-01" or source.get("seed") != 42
                or source.get("max_train_samples") != 200000 or source.get("max_eval_samples") != 60000
                or not isinstance(source.get("database_path"), str)
                or not isinstance(source.get("database_sha256"), str) or len(source["database_sha256"]) != 64):
            raise ValueError("Frozen cleaned-source contract mismatch")
        if not isinstance(summary.get("ensembles"), dict) or set(summary["ensembles"]) != set(FOLDS):
            raise ValueError("Both chronological ensembles must be complete")
        paths.extend((summary_path, source_path, lock_path))
        for fold, ensemble in summary["ensembles"].items():
            ensemble_path = folder / fold / "ensemble.json"
            if read_json(ensemble_path) != ensemble:
                raise ValueError("Frozen ensemble summary mismatch")
            _calibrators(ensemble.get("calibration"), selected)
            policy = ensemble.get("policy_selection", {})
            apply_policy(np.array([.5]), policy.get("chosen_policy"),
                         np.array([.5]) if selected == "cat_joint6" else None)
            if policy.get("research_only") is not True or policy.get("deployment_allowed") is not False:
                raise ValueError("Ensemble policy must remain research-only")
            member_hashes = ensemble.get("member_sha256")
            if not isinstance(member_hashes, dict) or set(member_hashes) != {str(seed) for seed in SEEDS}:
                raise ValueError("Missing selected ensemble member checksums")
            paths.append(ensemble_path)
            for seed in SEEDS:
                trial = folder / fold / f"{selected}-{seed}"
                artifact = model_path(trial, selected)
                sidecar = artifact.with_name(artifact.name + ".json")
                metadata_path, request_path = trial / "metadata.json", trial / "request.json"
                metadata, request, side = read_json(metadata_path), read_json(request_path), read_json(sidecar)
                digest = file_hash(artifact)
                if (digest != member_hashes[str(seed)] or metadata.get("model_sha256") != digest
                        or metadata.get("request") != request or request.get("model_name") != selected
                        or request.get("seed") != seed or request.get("wrapper_sha256") != code_hashes["dockdack/mark1_selective_models.py"]
                        or request.get("class_names") != list(CLASS_NAMES)
                        or request.get("train_shape", [None, None])[1:] != [len(FEATURE_NAMES)]
                        or side.get("model_sha256") != digest or side.get("model_name") != selected
                        or side.get("owner") != OWNER or side.get("schema_version") != SCHEMA_VERSION
                        or side.get("feature_count") != len(FEATURE_NAMES) or side.get("class_names") != list(CLASS_NAMES)):
                    raise ValueError("Selected native model provenance/checksum mismatch")
                paths.extend((artifact, sidecar, metadata_path, request_path))
        sources[key] = source
    hashes = _hashes(paths)
    hashes.update({str(Path(path).resolve()): value for path, value in protected.items()})
    return complete[market], sources[market], hashes


def verify_comparators(args, market, source_contract):
    """Bind prior checkpoints to the original, completed deep backtest record."""
    summary, source, hashes = deep.verify_training_artifacts(args.deep_training_run, market)
    hashes = dict(hashes)
    old_summary_path = args.deep_backtest_run / market / "summary.json"
    old = read_json(old_summary_path)
    if (source != source_contract or old.get("source") != source_contract or old.get("completed") is not True
            or old.get("selected_architecture") != summary.get("selected")
            or old.get("ensemble_calibration") != summary.get("ensemble_calibration")):
        raise ValueError("Previous comparator source/selection contract mismatch")
    recorded = old.get("artifact_sha256", {})
    for path, digest in hashes.items():
        if recorded.get(path) != digest:
            raise ValueError("Previous comparator training artifact checksum mismatch")
    checkpoints = [args.baseline_dir / f"{market}.pt"] + [
        args.deep_training_run / market / "walk_2024" / f"{summary['selected']}-{seed}" / "model.pt"
        for seed in SEEDS]
    for path in checkpoints:
        digest = file_hash(path)
        if recorded.get(str(path.resolve())) != digest:
            raise ValueError("Previous comparator checkpoint differs from completed backtest")
        hashes[str(path.resolve())] = digest
    if file_hash(checkpoints[0]) != deep.ORIGINAL_CHECKPOINT_SHA256[market]:
        raise ValueError("Original baseline checkpoint changed")
    hashes.update(_hashes([old_summary_path]))
    return summary, hashes


def infer_selective(dataset, indices, market, args, training_summary):
    """Chunked native CPU predictions; freeze raw-logit averaging/calibration."""
    selected = training_summary["selected"]
    ensemble = training_summary["ensembles"]["walk_2024"]
    models, hashes = [], {}
    for seed in SEEDS:
        path = model_path(args.training_run / market / "walk_2024" / f"{selected}-{seed}", selected)
        digest = file_hash(path)
        if digest != ensemble["member_sha256"][str(seed)]:
            raise ValueError("Selective checkpoint changed before inference")
        models.append(load_model(selected, path))
        hashes[str(path.resolve())] = digest
    success = np.empty(len(indices), dtype=np.float64)
    stop = np.empty(len(indices), dtype=np.float64) if selected == "cat_joint6" else None
    for first in range(0, len(indices), args.batch_size):
        picked = indices[first:first + args.batch_size]
        features = feature_array(dataset, picked, batch_size=args.batch_size)
        predictions = [predict_raw(model, selected, features) for model in models]
        for raw in predictions:
            for key, expected in (("success_logits", True), ("stop_logits", stop is not None)):
                value = raw.get(key)
                if not expected:
                    if value is not None:
                        raise ValueError("Unexpected binary stop-risk predictions")
                elif (not isinstance(value, np.ndarray) or value.shape != (len(picked),)
                      or value.dtype.kind not in "fiu" or not np.isfinite(value).all()):
                    raise ValueError("Invalid selective raw prediction shape/values")
        success[first:first + len(picked)] = np.mean([raw["success_logits"] for raw in predictions], axis=0)
        if stop is not None:
            stop[first:first + len(picked)] = np.mean([raw["stop_logits"] for raw in predictions], axis=0)
    probabilities, stop_probabilities = calibrated(
        {"success_logits": success, "stop_logits": stop}, ensemble["calibration"])
    mask = apply_policy(probabilities, ensemble["policy_selection"]["chosen_policy"], stop_probabilities)
    for path, digest in hashes.items():
        if file_hash(path) != digest:
            raise RuntimeError("Selective checkpoint changed during inference")
    del models
    gc.collect()
    return probabilities, stop_probabilities, mask, hashes


def selected_candidates(dataset, indices, probabilities, selected):
    """Filter rows, never fabricate 0/1 probabilities for portfolio ranking."""
    probabilities, selected = np.asarray(probabilities), np.asarray(selected)
    if (selected.shape != (len(indices),) or selected.dtype.kind != "b"
            or probabilities.shape != selected.shape or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or np.any(selected & (probabilities <= .5))):
        raise ValueError("Invalid selective candidate mask/probabilities")
    return make_candidates(dataset, indices[selected], probabilities[selected])


def signal_stability(labels, gross_returns, dates, symbols, selected, *, cost_bps=20):
    """Descriptive frozen-policy calendar diagnostics, never threshold tuning."""
    labels, returns, dates, symbols, selected = map(np.asarray,
        (labels, gross_returns, dates, symbols, selected))
    if (labels.ndim != 1 or any(value.shape != labels.shape for value in (returns, dates, symbols, selected))
            or selected.dtype.kind != "b" or dates.dtype.kind not in "iu"
            or symbols.dtype.kind not in "iu" or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(returns).all() or not math.isfinite(cost_bps) or cost_bps < 0):
        raise ValueError("Invalid calendar signal diagnostic inputs")

    def describe(mask):
        events = mask & selected
        count = int(events.sum())
        unique, counts = np.unique(symbols[events], return_counts=True)
        return {"count": int(mask.sum()), "signal_count": count,
                "signal_days": int(len(np.unique(dates[events]))), "symbol_count": len(unique),
                "precision": float(labels[events].mean()) if count else None,
                "net_mean_return": float(returns[events].mean() - cost_bps / 10000) if count else None,
                "largest_symbol_share": float(counts.max() / count) if count else None,
                "largest_symbol_id": int(unique[np.argmax(counts)]) if count else None}

    months = dates.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
    years = months // 12 + 1970
    quarters = (months % 12) // 3 + 1
    result = describe(np.ones(len(labels), dtype=bool))
    result["cost_bps"] = cost_bps
    result["yearly"] = [{"period": str(int(year)), **describe(years == year)} for year in np.unique(years)]
    keys = years * 10 + quarters
    result["quarterly"] = [{"period": f"{int(key // 10)}Q{int(key % 10)}", **describe(keys == key)}
                           for key in np.unique(keys)]
    return result


def run_market(market, args):
    started = time.monotonic()
    training, source_contract, hashes = verify_training_artifacts(args.training_run, market)
    previous, previous_hashes = verify_comparators(args, market, source_contract)
    hashes.update(previous_hashes)
    folder = args.output_dir / market
    if folder.exists():
        raise FileExistsError("Refusing to overwrite a selective backtest market")
    dataset = deep.load_frozen_cache(source_contract, market, args.cache_dir)
    source = Path(source_contract["database_path"])
    indices = deep.common_test_indices(dataset, deep.read_sessions(source))
    start, end = int(dataset.target_dates[indices].min()), int(dataset.target_dates[indices].max())
    prices, sessions, panel = load_price_panel(source, market, dataset.manifest["symbols"], start, end)
    for index in indices:
        key = int(dataset.symbol_ids[index]), int(dataset.target_dates[index])
        if key not in prices or not np.array_equal(np.asarray(prices[key]), dataset.target_ohlc[index]):
            raise ValueError(f"Approved evaluation target differs from source: {key}")
    for symbol, day in panel["zero_volume_keys"]:
        prices.pop((symbol, day), None)
    panel["zero_volume_execution_policy"] = "Unfillable; flagged missing held-price path, never a fabricated fill"
    if file_hash(source) != source_contract["database_sha256"]:
        raise RuntimeError("Clean source changed before backtest")
    folder.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"phase": "frozen_common_inference", "market": market, "samples": len(indices)}), flush=True)
    deep_args = SimpleNamespace(training_run=args.deep_training_run, baseline_dir=args.baseline_dir,
                                device=args.device, batch_size=args.batch_size)
    probabilities, outcomes, dates, inferred_hashes = deep.infer_common(
        dataset, indices, market, deep_args, previous, source_contract)
    for path, digest in inferred_hashes.items():
        if hashes.get(path) != digest:
            raise RuntimeError("Comparator changed after provenance verification")
    p, stops, selected, new_hashes = infer_selective(dataset, indices, market, args, training)
    for path, digest in new_hashes.items():
        if hashes.get(path) != digest:
            raise RuntimeError("Selective model changed after provenance verification")
    actual = event_data(dataset, indices)
    if (not np.array_equal(dates, actual["dates"]) or not np.array_equal(outcomes["success"], actual["labels"])
            or not np.array_equal(outcomes["gross_return"], actual["gross"])):
        raise ValueError("Comparators do not share identical events/outcomes")
    probabilities.update(unfiltered=p, selective=p)
    masks = {name: probabilities[name] > .5 for name in COMPARATORS}
    masks["selective"] = selected
    np.savez(folder / "predictions.npz", sample_indices=indices, dates=dates,
             symbol_ids=actual["symbols"], labels=actual["labels"], gross_returns=actual["gross"],
             both_touch=outcomes["both_touch"], evaluation_status=EVALUATION_STATUS,
             **{name + "_probabilities": value for name, value in probabilities.items()},
             **{name + "_selected": value for name, value in masks.items()},
             **({"selective_stop_probabilities": stops} if stops is not None else {}))
    initial = args.initial_krw if market == "domestic" else args.initial_usd
    ensemble = training["ensembles"]["walk_2024"]
    summary = {
        "market": market, "currency": "KRW" if market == "domestic" else "USD",
        "research_evaluation": EVALUATION_STATUS, "samples": len(indices), "initial_cash": initial,
        "range": {"first": date_string(start), "last": date_string(end), "sessions": len(sessions)},
        "selected_architecture": training["selected"], "ensemble_seeds": list(SEEDS),
        "source": source_contract, "price_panel": panel, "artifact_sha256": hashes,
        "sample_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "ensemble_calibration": ensemble["calibration"], "frozen_policy": ensemble["policy_selection"]["chosen_policy"],
        "training_research_qualified": training["research_qualified"], "models": {}, "cost_sensitivity": [],
        "research_only": True, "deployment_allowed": False,
        "deployment": "NOT PROMOTED; diagnostic simulated trades even when unqualified",
        "candidate_contract": "Identical approved events and actual OPEN; original calibrated probabilities retained; selective abstention filters only candidate rows",
        "universe": "Uncapped approved 2014-2021 train symbols; original 30-session 2025 boundary purge",
        "comparator_prediction_source": "Recomputed from checksummed native/deep checkpoints; no unverified prediction cache",
    }
    for name in COMPARATORS:
        metrics = evaluate_signals(actual["labels"], probabilities[name], actual["gross"], dates,
                                   actual["symbols"], masks[name], cost_bps=20)
        candidates = selected_candidates(dataset, indices, probabilities[name], masks[name])
        symbols = {row["symbol_id"] for row in candidates}
        relevant = {key: value for key, value in prices.items() if key[0] in symbols}
        row = {"classification": metrics, "raw_signals": len(candidates),
               "qualification": qualification(metrics), "portfolios": {}, "diagnostic_only": True,
               "stability": signal_stability(actual["labels"], actual["gross"], dates,
                                               actual["symbols"], masks[name])}
        save_json(folder / f"{name}-signals.json", {"research_evaluation": EVALUATION_STATUS,
                                                   "diagnostic_only": True, "signals": candidates})
        for mode in ("carry", "eod"):
            for cost in COSTS:
                simulation = simulate_portfolio(candidates, relevant, sessions, initial_cash=initial,
                    exit_mode=mode, max_positions=args.max_positions, position_fraction=args.position_fraction,
                    cost_bps=cost, volume_fraction=args.volume_fraction)
                attach_names(simulation, dataset.manifest["symbols"])
                simulation.update(market=market, model=name, exit_mode=mode, cost_bps=cost,
                                  research_evaluation=EVALUATION_STATUS, diagnostic_only=True,
                                  deployment="NOT PROMOTED", deployment_allowed=False)
                suffix = "" if cost == 20 else f"-cost{cost}"
                save_json(folder / f"{name}-{mode}{suffix}.json", simulation)
                summary["cost_sensitivity"].append({"model": name, "exit_mode": mode,
                                                     "cost_bps": cost, **simulation["summary"]})
                if cost == 20:
                    row["portfolios"][mode] = simulation["summary"]
                    print(json.dumps({"market": market, "model": name, "mode": mode,
                                      **simulation["summary"]}), flush=True)
        summary["models"][name] = row
        save_json(folder / "summary.json", summary)
    for path, digest in hashes.items():
        if file_hash(path) != digest:
            raise RuntimeError(f"Frozen research artifact changed during evaluation: {path}")
    if file_hash(source) != source_contract["database_sha256"]:
        raise RuntimeError("Clean source changed during backtest")
    summary.update(completed=True, elapsed_seconds=round(time.monotonic() - started, 3))
    save_json(folder / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, default=Path("outputs/mark1/selective-20260916"))
    parser.add_argument("--deep-training-run", type=Path, default=Path("outputs/mark1/deep-20260916"))
    parser.add_argument("--deep-backtest-run", type=Path, default=Path("outputs/mark1/deep-backtest-20260916"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mark1/selective-backtest-20260916"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/mark1/cache"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("models/mark1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
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
        raise FileExistsError("Use a fresh selective backtest output folder; overwrite is forbidden")
    # These checks precede output creation and all evaluation-cache access.
    for market in ("domestic", "us"):
        _, source, _ = verify_training_artifacts(args.training_run, market)
        verify_comparators(args, market, source)
    code = {name: file_hash(ROOT / name) for name in EVALUATION_CODE}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_json(args.output_dir / "config.json", {**vars(args), "costs_bps": list(COSTS),
        "comparators": list(COMPARATORS), "research_evaluation": EVALUATION_STATUS,
        "code_sha256": code, "started_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_allowed": False})
    summaries = {market: run_market(market, args) for market in ("domestic", "us")}
    if any(file_hash(ROOT / name) != digest for name, digest in code.items()):
        raise RuntimeError("Evaluation implementation changed during this run")
    save_json(args.output_dir / "summary.json", summaries)
    print("ALL 64 DIAGNOSTIC PORTFOLIOS COMPLETE; NO MODEL PROMOTION", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
