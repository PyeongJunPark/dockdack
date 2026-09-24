"""Frozen deep ensemble versus original Mark_1 on identical reused history.

Research only: the already-inspected 2025+ history is NOT an untouched test.
No calibration fitting, architecture selection, model promotion or orders.
Requires the complete training-run summary before reading evaluation data.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from dockdack.mark1_backtest import simulate_portfolio
from dockdack.mark1_backtest_data import load_price_panel
from dockdack.mark1_deep_data import FOLDS
from dockdack.mark1_deep_models import CLASS_NAMES, FEATURE_NAMES, TARGET, build_model
from dockdack.mark1_deep_validation import evaluate
from dockdack.mark1_inference import Predictor
from dockdack.mark1_metrics import calibrated_probability
from examples.backtest_mark1 import attach_names, date_string, full_test_indices, make_candidates
from examples.train_mark1 import BatchBank, dataset_cache, file_hash, save_json
from examples.train_mark1_deep import DeepBank, MODEL_CONFIG, PROTOCOL, read_sessions


EVALUATION_STATUS = "reused_historical_evaluation_not_untouched_test"
SEEDS = (42, 43, 44)
COSTS = (0, 10, 20, 40)
ORIGINAL_CHECKPOINT_SHA256 = {
    "domestic": "482462c064e77f8dc91ba201f5fa463e7d145d0fa184af222644b79ec3b58818",
    "us": "b6b28534c055dcab03e032106bb39296f5c4251bd0516593d87488d1a78265a7",
}


def _canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _calibration_contract(value):
    if (not isinstance(value, dict) or type(value.get("fit_samples")) is not int
            or value["fit_samples"] < 1 or value.get("weighted") is not False):
        raise ValueError("Frozen independent unweighted calibration is required")
    calibrated_probability([0.0], value)


def verify_training_artifacts(training_run: Path, market: str):
    """Reject partial training or mismatched locks before loading 2025 data."""
    folder = training_run / market
    paths = [training_run / "summary.json", training_run / "protocol.json",
             folder / "summary.json", folder / "selection_locked.json", folder / "source.json"]
    values = [_read_json(path) for path in paths]
    complete, frozen, summary, locked, source = values
    if set(complete) != {"domestic", "us"} or any(not isinstance(value, dict) for value in complete.values()):
        raise ValueError("Both markets must finish training before reused historical evaluation")
    if complete[market] != summary:
        raise ValueError("Market summary differs from completed training run")
    if (frozen.get("protocol") != _canonical(PROTOCOL)
            or summary.get("protocol") != _canonical(PROTOCOL)):
        raise ValueError("Frozen training protocol mismatch")
    if (summary.get("market") != market or locked.get("market") != market
            or summary.get("selected") != locked.get("selected")
            or locked.get("selected") not in PROTOCOL["architectures"]
            or locked.get("ranking") != summary.get("ranking")
            or locked.get("selection_rule") != PROTOCOL["architecture_selection"]):
        raise ValueError("Frozen architecture selection mismatch")
    _calibration_contract(summary.get("ensemble_calibration"))
    if (source.get("market") != market or source.get("target") != TARGET
            or source.get("version") != 2 or source.get("purge_sessions") != 30
            or source.get("start") != "2010-01-01"
            or source.get("max_train_samples") != 200000
            or source.get("max_eval_samples") != 60000 or source.get("seed") != 42
            or not isinstance(source.get("database_path"), str)
            or not isinstance(source.get("database_sha256"), str)
            or len(source["database_sha256"]) != 64):
        raise ValueError("Frozen cleaned-source contract mismatch")
    # Training code is part of the frozen experiment. Different inference
    # feature/model implementations cannot silently reuse these checkpoints.
    root = Path(__file__).resolve().parents[1]
    expected_files = ("examples/train_mark1_deep.py", "dockdack/mark1_deep_data.py",
                      "dockdack/mark1_deep_models.py", "dockdack/mark1_deep_validation.py",
                      "dockdack/mark1_data.py", "dockdack/mark1_metrics.py", "examples/train_mark1.py")
    hashes = frozen.get("code_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {Path(name).name for name in expected_files}:
        raise ValueError("Missing frozen training-code hashes")
    for name in expected_files:
        if file_hash(root / name) != hashes[Path(name).name]:
            raise ValueError(f"Frozen training code changed: {name}")
    return summary, source, {str(path.resolve()): file_hash(path) for path in paths}


def load_frozen_cache(source_contract, market, cache_dir):
    """Require an existing cache: evaluation never builds or rewrites it."""
    key = hashlib.sha256(json.dumps(source_contract, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"{market}-{key}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Required read-only training cache missing: {path}")
    source = Path(source_contract["database_path"])
    dataset, actual = dataset_cache(
        source, market, Path(cache_dir),
        SimpleNamespace(max_train_samples=200000, max_eval_samples=60000, seed=42))
    if actual != source_contract:
        raise ValueError("Evaluation must use the exact frozen training source contract")
    return dataset


def common_test_indices(dataset, sessions):
    """All original-purged 2025+ events in the uncapped final train universe."""
    original = full_test_indices(dataset)
    dates = np.asarray(dataset.target_dates)
    symbols = np.asarray(dataset.symbol_ids)
    calendar = np.asarray(sessions)
    if (dates.ndim != 1 or symbols.shape != dates.shape or dates.dtype.kind not in "iu"
            or symbols.dtype.kind not in "iu" or calendar.ndim != 1
            or calendar.dtype.kind not in "iu" or not len(calendar)
            or np.any(calendar[1:] <= calendar[:-1])):
        raise ValueError("Invalid integer sample dates/symbols/session calendar")
    fold = FOLDS["walk_2024"]
    lower = int(np.datetime64(fold["train_start"], "D").astype(np.int64))
    upper = int(np.datetime64(fold["train_end"], "D").astype(np.int64))
    train = (dates >= lower) & (dates <= upper)
    if not train.any():
        raise ValueError("No uncapped final-fold training universe")
    eligible = np.unique(symbols[train])
    indices = original[np.isin(symbols[original], eligible)]
    if not len(indices):
        raise ValueError("No common approved reused evaluation samples")
    ordinals = np.searchsorted(calendar, dates[indices])
    if (np.any(ordinals >= len(calendar)) or np.any(ordinals < 30)
            or np.any(calendar[ordinals] != dates[indices])):
        raise ValueError("Evaluation target missing 30-session calendar context")
    cutoff = int(np.datetime64("2024-12-31", "D").astype(np.int64))
    if np.any(calendar[ordinals - 30] <= cutoff):
        raise ValueError("Original 30-session test boundary purge was not preserved")
    keys = np.stack([symbols[indices], dates[indices]], axis=1)
    if len(np.unique(keys, axis=0)) != len(indices):
        raise ValueError("Duplicate approved evaluation event")
    indices = indices.astype(np.int64, copy=True)
    indices.setflags(write=False)
    return indices


def load_deep_checkpoint(checkpoint, *, market, architecture, seed, source_contract, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Deep checkpoint must contain metadata and weights")
    context = payload.get("context")
    if (payload.get("target") != TARGET
            or _canonical(payload.get("feature_names")) != list(FEATURE_NAMES)
            or _canonical(payload.get("class_names")) != list(CLASS_NAMES)
            or payload.get("model_config") != MODEL_CONFIG
            or payload.get("architecture") != architecture
            or type(payload.get("seed")) is not int or payload["seed"] != seed
            or payload.get("research_only") is not True
            or payload.get("intraday_path_verified") is not False
            or _canonical(payload.get("protocol")) != _canonical(PROTOCOL)
            or not isinstance(context, dict) or context.get("market") != market
            or context.get("fold") != "walk_2024" or context.get("source") != source_contract):
        raise ValueError("Incompatible deep checkpoint target/features/classes/provenance")
    for key, expected in (("threshold", .5), ("take_profit_pct", 1.), ("stop_loss_pct", .9)):
        if type(payload.get(key)) not in (int, float) or payload[key] != expected:
            raise ValueError(f"Frozen deep checkpoint requires {key}={expected}")
    _calibration_contract(payload.get("calibration"))
    state = payload.get("state_dict")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
                   or not bool(torch.isfinite(value).all()) for value in state.values())):
        raise ValueError("Deep checkpoint contains missing or invalid weights")
    model = build_model(architecture, **MODEL_CONFIG)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("Deep checkpoint weights do not match architecture") from exc
    return model.to(device).eval(), payload


def infer_common(dataset, indices, market, args, training_summary, source_contract):
    """Frozen mean logits then frozen Platt calibration; never fit anything."""
    bank = DeepBank(dataset, {"test": indices}, args.device)
    selected = training_summary["selected"]
    logits, hashes = [], {}
    for seed in SEEDS:
        path = args.training_run / market / "walk_2024" / f"{selected}-{seed}" / "model.pt"
        before = file_hash(path)
        model, metadata = load_deep_checkpoint(
            path, market=market, architecture=selected, seed=seed,
            source_contract=source_contract, device=args.device)
        logits.append(bank.logits(model, "test", args.batch_size).astype(np.float64))
        if file_hash(path) != before:
            raise RuntimeError("Deep checkpoint changed during inference")
        hashes[str(path.resolve())] = before
        del model
    ensemble_logits = np.mean(np.stack(logits), axis=0)
    deep_probabilities = calibrated_probability(ensemble_logits, training_summary["ensemble_calibration"])
    part = bank.parts["test"]
    outcomes, dates = part["outcomes"], part["dates"]
    del bank
    if args.device == "cuda":
        torch.cuda.empty_cache()
    # A dataclass view replaces only the split dictionary, never cached arrays.
    old_bank = BatchBank(replace(dataset, splits={**dataset.splits, "test": indices}), args.device)
    path = args.baseline_dir / f"{market}.pt"
    before = file_hash(path)
    if before != ORIGINAL_CHECKPOINT_SHA256[market]:
        raise ValueError("Original Mark_1 baseline checkpoint changed")
    baseline = Predictor(path, device=args.device)
    if (baseline.market != market or baseline.metadata.get("dataset") != source_contract
            or baseline.metadata.get("variant") != "mlp_no_price_aug"):
        raise ValueError("Original Mark_1 baseline source/selection mismatch")
    baseline_logits = old_bank.logits(baseline.model, "test", args.batch_size)
    baseline_probabilities = calibrated_probability(baseline_logits, baseline.metadata["calibration"])
    if file_hash(path) != before:
        raise RuntimeError("Original baseline changed during inference")
    hashes[str(path.resolve())] = before
    del old_bank, baseline
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return {"baseline": baseline_probabilities, "deep": deep_probabilities}, outcomes, dates, hashes


def run_market(market, args):
    started = time.monotonic()
    training_summary, source_contract, artifact_hashes = verify_training_artifacts(args.training_run, market)
    dataset = load_frozen_cache(source_contract, market, args.cache_dir)
    source = Path(source_contract["database_path"])
    indices = common_test_indices(dataset, read_sessions(source))
    start, end = int(dataset.target_dates[indices].min()), int(dataset.target_dates[indices].max())
    prices, sessions, panel = load_price_panel(source, market, dataset.manifest["symbols"], start, end)
    for index in indices:
        key = int(dataset.symbol_ids[index]), int(dataset.target_dates[index])
        if key not in prices or not np.array_equal(np.asarray(prices[key]), dataset.target_ohlc[index]):
            raise ValueError(f"Approved target differs from frozen source DB: {key}")
    for symbol_id, day in panel["zero_volume_keys"]:
        prices.pop((symbol_id, day), None)
    panel["zero_volume_execution_policy"] = "Unfillable; treated as missing held-price path, never a fabricated fill"
    if file_hash(source) != source_contract["database_sha256"]:
        raise RuntimeError("Clean source changed before backtest")
    folder = args.output_dir / market
    folder.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"market": market, "phase": "frozen_inference", "samples": len(indices),
                      "evaluation_status": EVALUATION_STATUS}), flush=True)
    probabilities, outcomes, dates, checkpoint_hashes = infer_common(
        dataset, indices, market, args, training_summary, source_contract)
    artifact_hashes.update(checkpoint_hashes)
    np.savez(folder / "predictions.npz", sample_indices=indices, dates=dates,
             symbol_ids=dataset.symbol_ids[indices], labels=outcomes["success"],
             gross_returns=outcomes["gross_return"], both_touch=outcomes["both_touch"],
             baseline_probabilities=probabilities["baseline"], deep_probabilities=probabilities["deep"],
             evaluation_status=EVALUATION_STATUS)
    initial = args.initial_krw if market == "domestic" else args.initial_usd
    summary = {"market": market, "currency": "KRW" if market == "domestic" else "USD",
               "research_evaluation": EVALUATION_STATUS, "samples": len(indices),
               "selected_architecture": training_summary["selected"], "ensemble_seeds": list(SEEDS),
               "range": {"first": date_string(start), "last": date_string(end), "sessions": len(sessions)},
               "initial_cash": initial, "models": {}, "cost_sensitivity": [],
               "source": source_contract, "price_panel": panel, "artifact_sha256": artifact_hashes,
               "sample_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
               "ensemble_calibration": training_summary["ensemble_calibration"],
               "training_research_qualified": training_summary.get("research_qualified"),
               "deployment": "NOT PROMOTED; original model files and order settings unchanged",
               "candidate_contract": "Identical approved indices and actual-open entries; strict p>.5; last20 completed volumes only",
               "universe": "Symbols with uncapped approved 2014-2021 targets; original 30-session 2025 boundary purge preserved"}
    for name in ("baseline", "deep"):
        metrics = evaluate(outcomes["success"], probabilities[name], outcomes["gross_return"], dates, cost_bps=20)
        candidates = make_candidates(dataset, indices, probabilities[name])
        signal_symbols = {row["symbol_id"] for row in candidates}
        relevant = {key: value for key, value in prices.items() if key[0] in signal_symbols}
        row = {"classification": metrics, "raw_signals": len(candidates), "portfolios": {}}
        save_json(folder / f"{name}-signals.json", {
            "research_evaluation": EVALUATION_STATUS, "signals": candidates})
        for mode in ("carry", "eod"):
            for cost in COSTS:
                simulation = simulate_portfolio(
                    candidates, relevant, sessions, initial_cash=initial, exit_mode=mode,
                    max_positions=args.max_positions, position_fraction=args.position_fraction,
                    cost_bps=cost, volume_fraction=args.volume_fraction)
                attach_names(simulation, dataset.manifest["symbols"])
                simulation.update(market=market, model=name, exit_mode=mode, cost_bps=cost,
                                  research_evaluation=EVALUATION_STATUS, deployment="NOT PROMOTED")
                suffix = "" if cost == 20 else f"-cost{cost}"
                save_json(folder / f"{name}-{mode}{suffix}.json", simulation)
                summary["cost_sensitivity"].append({"model": name, "exit_mode": mode,
                                                     "cost_bps": cost, **simulation["summary"]})
                if cost == 20:
                    row["portfolios"][mode] = simulation["summary"]
                    print(json.dumps({"market": market, "model": name, "mode": mode,
                                      "phase": "portfolio_complete", **simulation["summary"]}), flush=True)
        summary["models"][name] = row
        save_json(folder / "summary.json", summary)
    for path, digest in artifact_hashes.items():
        if file_hash(Path(path)) != digest:
            raise RuntimeError(f"Frozen artifact changed during backtest: {path}")
    if file_hash(source) != source_contract["database_sha256"]:
        raise RuntimeError("Clean source changed during backtest")
    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    summary["completed"] = True
    save_json(folder / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, default=Path("outputs/mark1/deep-20260916"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mark1/deep-backtest-20260916"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/mark1/cache"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("models/mark1"))
    parser.add_argument("--market", choices=("all", "domestic", "us"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--initial-krw", type=float, default=10_000_000)
    parser.add_argument("--initial-usd", type=float, default=10_000)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--position-fraction", type=float, default=.05)
    parser.add_argument("--volume-fraction", type=float, default=.001)
    args = parser.parse_args(argv)
    if (args.batch_size < 1 or args.max_positions < 1
            or any(not math.isfinite(value) or value <= 0 for value in (args.initial_krw, args.initial_usd))
            or not 0 < args.position_fraction <= 1 or not 0 < args.volume_fraction <= 1):
        parser.error("Invalid backtest portfolio limits")
    markets = ("domestic", "us") if args.market == "all" else (args.market,)
    for market in markets:
        verify_training_artifacts(args.training_run, market)
    if args.device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        parser.error("Requested CUDA BF16 is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    save_json(args.output_dir / "config.json", {
        **vars(args), "created_at": datetime.now(timezone.utc).isoformat(),
        "research_evaluation": EVALUATION_STATUS, "costs_bps": COSTS,
        "threshold": "strict probability > 0.5", "take_profit": .01, "stop_loss": .009,
        "both_touch": "stop first; known opening gaps handled first",
        "entry": "Observed actual session open; optimistic immediate-fill assumption",
        "cost": "Hypothetical symmetric rate, not actual broker fee/tax schedule",
        "missing_price": "Uncertain held path and stale marks; next observed OPEN liquidation",
        "comparison": "Same uncapped approved 2025+ indices, frozen original MLP and frozen three-seed deep ensemble",
        "limitation": "Daily OHLC proxy, no bid/ask, impact, auction queue or guaranteed fills; historical data already inspected",
        "deployment": "No model promotion or live orders", "torch": str(torch.__version__)})
    results = {market: run_market(market, args) for market in markets}
    save_json(args.output_dir / "summary.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
