"""Frozen Mark_1 models: full approved 2025+ test-period portfolio simulation.

Research only. No broker, network, retraining, threshold tuning or live orders.
Entry assumes an immediate fill at the observed open, a disclosed optimistic
latency assumption, and never treats synthetic training queries as real trades.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from dockdack.mark1_backtest import simulate_portfolio
from dockdack.mark1_backtest_data import load_price_panel
from dockdack.mark1_inference import Predictor
from dockdack.mark1_metrics import binary_metrics, calibrated_probability
from examples.train_mark1 import BatchBank, dataset_cache, file_hash, save_json


def date_string(day):
    return str(np.datetime64(int(day), "D"))


def full_test_indices(dataset):
    """Stored starts already passed the original 30-session boundary purge."""
    cut = int(np.datetime64("2024-12-31", "D").astype(np.int64))
    indices = np.flatnonzero(dataset.target_dates > cut)
    if not len(indices) or not np.isin(dataset.splits["test"], indices).all():
        raise ValueError("Missing original held-out test samples")
    return indices


def make_candidates(dataset, indices, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(indices),) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probabilities for approved candidates")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities outside [0,1]")
    picked = np.flatnonzero(probabilities > .5)
    candidates = []
    for local in picked:
        sample = int(indices[local]); start = int(dataset.starts[sample])
        # Known before entry: only the LAST 20 of the previous 30 completed bars.
        volume = float(np.median(dataset.bars[start + 10:start + 30, 4].astype(np.float64)))
        candidates.append({"symbol_id": int(dataset.symbol_ids[sample]),
                           "date": int(dataset.target_dates[sample]),
                           "probability": float(probabilities[local]), "liquidity_shares": volume})
    return candidates


def attach_names(simulation, symbols):
    lookup = {row["symbol_id"]: row for row in symbols}
    for trade in simulation["trades"]:
        row = lookup[trade["symbol_id"]]
        trade.update(symbol=row["symbol"], exchange=row["exchange"])
        for key in ("entry_date", "exit_date"):
            if key in trade:
                trade[key + "_iso"] = date_string(trade[key])
    return simulation


def run_market(market, args):
    started = time.monotonic()
    folder = args.output_dir / market
    folder.mkdir(parents=True, exist_ok=False)
    original = args.training_run / market
    locked_path = original / "selection_locked.json"
    locked_hash = file_hash(locked_path)
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    source = args.db_dir / f"{market}_daily_clean.sqlite3"
    dataset, cache_contract = dataset_cache(source, market, args.cache_dir, args)
    original_contract = json.loads((original / "cache_contract.json").read_text(encoding="utf-8"))
    if cache_contract != original_contract:
        raise ValueError("Backtest must reuse the exact frozen training data contract")
    original_test = dataset.splits["test"].copy()
    indices = full_test_indices(dataset)
    dataset.splits["test"] = indices
    start, end = int(dataset.target_dates[indices].min()), int(dataset.target_dates[indices].max())
    print(json.dumps({"market": market, "phase": "load_full_test", "samples": len(indices),
                      "first": date_string(start), "last": date_string(end)}), flush=True)
    prices, sessions, panel_metadata = load_price_panel(source, market, dataset.manifest["symbols"], start, end)
    # Every approved target must still be exactly the source DB OHLC row.
    for index in indices:
        key = int(dataset.symbol_ids[index]), int(dataset.target_dates[index])
        if key not in prices or not np.array_equal(np.asarray(prices[key]), dataset.target_ohlc[index]):
            raise ValueError(f"Approved cache target differs from source DB at {key}")
    # A zero-volume row supplies no evidence of an executable fill. Treat it
    # as unavailable to this OHLC execution engine; held paths become flagged.
    for symbol_id, day in panel_metadata["zero_volume_keys"]:
        prices.pop((symbol_id, day), None)
    panel_metadata["zero_volume_execution_policy"] = "Unfillable; treated as missing held-price path, not a fabricated trade"
    if file_hash(source) != cache_contract["database_sha256"]:
        raise RuntimeError("Cleaned source changed while building backtest")
    bank = BatchBank(dataset, args.device)
    initial_cash = args.initial_krw if market == "domestic" else args.initial_usd
    summary = {"market": market, "currency": "KRW" if market == "domestic" else "USD",
               "winner_frozen": locked["winner"], "selection_file_sha256": locked_hash,
               "source": cache_contract, "price_panel": panel_metadata,
               "test_samples": len(indices), "previous_test_sample_count": len(original_test),
               "range": {"first": date_string(start), "last": date_string(end), "sessions": len(sessions)},
               "initial_cash": initial_cash, "results": [], "cost_sensitivity": []}
    part = bank.parts["test"]
    for model_row in locked["models"]:
        variant = model_row["variant"]
        checkpoint = original / variant / "model.pt"
        checkpoint_hash = file_hash(checkpoint)
        predictor = Predictor(checkpoint, device=args.device)
        if predictor.market != market or predictor.metadata["dataset"] != cache_contract:
            raise ValueError("Frozen checkpoint provenance mismatch")
        logits = bank.logits(predictor.model, "test", args.batch_size)
        probabilities = calibrated_probability(logits, predictor.metadata["calibration"])
        # Expanded test results must agree with already published 60k evaluation.
        published = np.load(original / variant / "test_predictions.npz", allow_pickle=False)
        positions = np.searchsorted(indices, original_test)
        difference = float(np.max(np.abs(probabilities[positions] - published["probabilities"])))
        if difference > 1e-5:
            raise ValueError("Frozen prediction reproducibility check failed")
        metrics = binary_metrics(part["outcomes"]["success"], probabilities, dates=part["dates"])
        candidates = make_candidates(dataset, indices, probabilities)
        np.savez(folder / f"{variant}-predictions.npz", probabilities=probabilities,
                 sample_indices=indices, dates=part["dates"], symbol_ids=dataset.symbol_ids[indices])
        # Computing on only symbols with a signal changes neither selection nor
        # time ordering; all their available dates are kept for held positions.
        signalled_symbols = {row["symbol_id"] for row in candidates}
        relevant_prices = {key: value for key, value in prices.items() if key[0] in signalled_symbols}
        model_result = {"variant": variant, "checkpoint_sha256": checkpoint_hash,
                        "original_prediction_max_difference": difference, "classification": metrics,
                        "raw_signals": len(candidates), "portfolios": {}}
        save_json(folder / f"{variant}-signals.json", candidates)
        for exit_mode in ("carry", "eod"):
            simulation = simulate_portfolio(candidates, relevant_prices, sessions,
                                            initial_cash=initial_cash, exit_mode=exit_mode,
                                            max_positions=args.max_positions, position_fraction=args.position_fraction,
                                            cost_bps=args.cost_bps, volume_fraction=args.volume_fraction)
            attach_names(simulation, dataset.manifest["symbols"])
            simulation.update(market=market, variant=variant, cost_bps=args.cost_bps,
                              exit_mode=exit_mode, frozen_winner=variant == locked["winner"])
            save_json(folder / f"{variant}-{exit_mode}.json", simulation)
            model_result["portfolios"][exit_mode] = simulation["summary"]
            print(json.dumps({"market": market, "variant": variant, "mode": exit_mode,
                              "phase": "portfolio_complete", **simulation["summary"]}, ensure_ascii=False), flush=True)
            if variant == locked["winner"]:
                for cost in (0, 10, 20, 40):
                    sensitivity = (simulation if cost == args.cost_bps else
                                   simulate_portfolio(candidates, relevant_prices, sessions,
                                                      initial_cash=initial_cash, exit_mode=exit_mode,
                                                      max_positions=args.max_positions, position_fraction=args.position_fraction,
                                                      cost_bps=cost, volume_fraction=args.volume_fraction))
                    summary["cost_sensitivity"].append({"exit_mode": exit_mode, "cost_bps": cost,
                                                         **sensitivity["summary"]})
        summary["results"].append(model_result)
        save_json(folder / "summary.json", summary)
        if file_hash(checkpoint) != checkpoint_hash:
            raise RuntimeError("Checkpoint changed during backtest")
        del predictor
        if args.device == "cuda":
            torch.cuda.empty_cache()
    if file_hash(source) != cache_contract["database_sha256"] or file_hash(locked_path) != locked_hash:
        raise RuntimeError("Source or frozen model selection changed")
    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    summary["completed"] = True
    save_json(folder / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, default=Path("outputs/mark1/experiment-20260916"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/mark1/cache"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--market", choices=("all", "domestic", "us"), default="all")
    parser.add_argument("--initial-krw", type=float, default=10_000_000)
    parser.add_argument("--initial-usd", type=float, default=10_000)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--position-fraction", type=float, default=.05)
    parser.add_argument("--volume-fraction", type=float, default=.001)
    parser.add_argument("--cost-bps", type=float, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args(argv)
    training = json.loads((args.training_run / "run_config.json").read_text(encoding="utf-8"))
    args.db_dir = Path(training["db_dir"])
    for key in ("seed", "max_train_samples", "max_eval_samples"):
        setattr(args, key, training[key])
    if (any(not math.isfinite(x) or x <= 0 for x in (args.initial_krw, args.initial_usd))
            or args.max_positions < 1 or args.batch_size < 1 or not 0 < args.position_fraction <= 1
            or not 0 < args.volume_fraction <= 1 or not math.isfinite(args.cost_bps) or not 0 <= args.cost_bps < 10000):
        parser.error("Invalid portfolio limits")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.65)
    config = {**vars(args), "created_at": datetime.now(timezone.utc).isoformat(),
              "entry": "Observed actual session open; assumes immediate fill after probability evaluation",
              "input": "30 completed OHLCV bars plus today's candidate entry price",
              "both_touch": "stop first for intraday ambiguous touches; known opening gap evaluated first",
              "threshold": "strict probability > 0.5", "take_profit": .01, "stop_loss": .009,
              "default_primary_exit": "carry until barrier; eod timeout reported separately",
              "selected_model": "Previously locked 2024 winner; no reselection using backtest",
              "cost": "Hypothetical symmetric entry/exit friction; not Kiwoom actual fee/tax schedule",
              "capital": "No leverage/shorting; whole shares; max20 by default; 5%startdayequity perposition",
              "sampling": "ALL approved 2025+ samples from frozen training-availability universe; no60kcap",
              "missing_price": "Freeze last mark and cash; uncertain path; exit on next observed open, not retroactively",
              "limitation": "Daily OHLC research simulation; no bid/ask/latency/auction queue/price-limit fill modeling",
              "torch": torch.__version__}
    save_json(args.output_dir / "config.json", config)
    for market in (("domestic", "us") if args.market == "all" else (args.market,)):
        run_market(market, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
