"""Run Mark1.5--1.7 research on both markets; never send broker orders.

The three variants share one causal 30-bar candidate universe, frozen time
splits, calibration grid, and exact cash-only portfolio proxy.  A completed
run has manifest.json; partial output is intentionally retained for audit.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import platform
import time

import numpy as np
import torch

from dockdack.clean_daily_dataset import load_sessions, source_fingerprints
from dockdack.mark1_4_data import load_mark14_candidates, mark14_chronological_splits
from dockdack.mark1_4_evolution import simulate_portfolio
from dockdack.mark1_4_followup_selection import select_calibrated_policy
from dockdack.mark1_series_models import SCORE_UNITS, VARIANTS, fit_series_variant
from dockdack.research_artifacts import sha256_file, write_new_json


ROOT = Path(__file__).resolve().parents[1]


def _periods(samples, sessions):
    base = mark14_chronological_splits(
        samples, sessions, train_start="2018-01-01", train_end="2021-12-31",
        validation_end="2022-12-31", test_end="2024-12-31",
        embargo_sessions=30,
    )
    targets = np.asarray(samples.target_dates)
    ordinals = np.asarray(samples.target_ordinals)
    train = np.flatnonzero((targets >= "2018-01-01") & (targets <= "2020-12-31"))
    if not len(train):
        raise ValueError("No 2018--2020 training candidates")
    last_train = int(ordinals[train].max())
    calibration = np.flatnonzero((targets >= "2021-01-01") &
                                 (targets <= "2021-12-31") &
                                 (ordinals > last_train + 30))
    periods = {
        "train_2018_2020": train,
        "calibration_2021": calibration,
        "development_2022": base["validation"],
        "reused_history_2023_2024": base["test"],
    }
    if any(not len(rows) for rows in periods.values()):
        raise ValueError("Empty chronological period after 30-session embargo")
    all_rows = [set(rows.tolist()) for rows in periods.values()]
    if any(all_rows[i] & all_rows[j] for i in range(4) for j in range(i + 1, 4)):
        raise AssertionError("Period rows overlap")
    return periods


def _model_run(destination: Path, samples, periods, *, variant: str,
               seeds: tuple[int, ...], device: str, epochs: int,
               batch_size: int, day_batch_size: int, hidden: int,
               recurrent_cell: str, initial_equity: float) -> dict:
    scores_by_seed = {}
    models = {}
    for seed in seeds:
        started = time.perf_counter()
        fit = fit_series_variant(
            samples, periods["train_2018_2020"], variant, seed=seed,
            device=device, epochs=epochs, batch_size=batch_size,
            day_batch_size=day_batch_size, hidden=hidden,
            recurrent_cell=recurrent_cell, cost_bps=20.0,
        )
        scores_by_seed[seed] = fit.scores
        stem = f"{samples.source['market']}-{variant}-seed{seed}"
        checkpoint = destination / f"{stem}-weights.npz"
        with checkpoint.open("xb") as stream:
            np.savez_compressed(stream, **fit.state)
        score_file = destination / f"{stem}-scores.npz"
        with score_file.open("xb") as stream:
            np.savez_compressed(stream, scores=fit.scores)
        metadata = destination / f"{stem}-model.json"
        fit.artifact["checkpoint_file"] = checkpoint.name
        fit.artifact["checkpoint_sha256"] = sha256_file(checkpoint)
        fit.artifact["score_file"] = score_file.name
        fit.artifact["score_sha256"] = sha256_file(score_file)
        fit.artifact["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_new_json(metadata, fit.artifact)
        models[str(seed)] = {
            "model": metadata.name, "checkpoint": checkpoint.name,
            "scores": score_file.name,
            "fit_seconds": fit.artifact["elapsed_seconds"],
        }
        print(f"{variant} {samples.source['market']} seed {seed}: "
              f"{len(fit.scores):,} candidate scores in "
              f"{fit.artifact['elapsed_seconds']:.1f}s", flush=True)
    selection = select_calibrated_policy(
        samples, scores_by_seed, periods["train_2018_2020"],
        periods["calibration_2021"], cost_bps=20.0, allocation=.1,
        max_positions=10, initial_equity=initial_equity,
        min_executed_trades=20,
    )
    chosen = selection["selected"]
    if chosen is None:
        scores = np.zeros(len(samples.windows), dtype=np.float32)
        threshold = 1.0
    else:
        scores = scores_by_seed[int(chosen["seed"])]
        threshold = float(chosen["train_numeric_score_threshold"])
    evaluations = {}
    for name, rows in periods.items():
        evaluations[name] = {
            str(cost): simulate_portfolio(
                samples, rows, scores, threshold=threshold,
                cost_bps=cost, allocation=.1, max_positions=10,
                initial_equity=initial_equity,
            )
            for cost in (0.0, 20.0, 40.0)
        }
    return {
        "variant": variant, "score_unit": SCORE_UNITS[variant],
        "research_only": True, "deployment_allowed": False,
        "not_paper_reproduction": True,
        "model_artifacts": models,
        "calibration_selection": selection,
        "frozen_policy": chosen,
        "cash_only_if_no_eligible_calibration_policy": chosen is None,
        "periods": {
            name: {"candidates": len(rows),
                   "first_target": str(min(samples.target_dates[rows])),
                   "last_target": str(max(samples.target_dates[rows]))}
            for name, rows in periods.items()
        },
        "evaluations": evaluations,
        "cost_grid_bps": [0.0, 20.0, 40.0],
        "source": samples.source,
        "limitations": [
            "Current instrument catalog is not point-in-time and has survivorship bias.",
            "2022--2024 was inspected previously; this is not an independent holdout.",
            "Opening/closing auction fills and 20bp roundtrip costs are hypothetical.",
            "A selected missing target invalidates the exact account path.",
            "No GUI, demo, or live order integration is permitted by this research run.",
        ],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", nargs="+", choices=("domestic", "us"),
                        default=("domestic", "us"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda",
                        help="Research training requires the local CUDA GPU")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--day-batch-size", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--recurrent-cell", choices=("lstm", "gru"), default="lstm")
    parser.add_argument("--max-symbols", type=int, default=100)
    parser.add_argument("--database", type=Path,
                        help="Optional raw DB override, allowed only for one market")
    args = parser.parse_args(argv)
    if len(set(args.markets)) != len(args.markets) or len(set(args.variants)) != len(args.variants):
        parser.error("Markets and variants must not repeat")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be unique nonnegative integers")
    if args.database is not None and len(args.markets) != 1:
        parser.error("--database override requires exactly one market")
    if not torch.cuda.is_available():
        parser.error("Mark1.5--1.7 training requires CUDA; no CPU fallback")
    destination = args.output_dir.resolve()
    allowed = (ROOT / "outputs" / "mark1").resolve()
    if (not destination.is_relative_to(allowed) or destination == allowed
            or destination.exists()):
        parser.error("Choose a new, unused child directory under outputs/mark1")
    databases = {}
    for market in args.markets:
        database = (args.database or ROOT / "data" / "kiwoom_daily" /
                    f"{market}_daily.sqlite3").resolve()
        if not database.is_file():
            parser.error(f"Raw database not found: {database}")
        databases[market] = database
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "series": "Mark1.5--1.7", "research_only": True,
        "deployment_allowed": False, "not_paper_reproduction": True,
        "output_directory": str(destination),
        "configuration": {"markets": list(args.markets),
                          "variants": list(args.variants),
                          "seeds": list(args.seeds),
                          "epochs": args.epochs, "batch_size": args.batch_size,
                          "day_batch_size": args.day_batch_size,
                          "hidden": args.hidden,
                          "recurrent_cell": args.recurrent_cell,
                          "max_symbols": args.max_symbols,
                          "device": args.device},
        "runtime": {"python": platform.python_version(),
                    "numpy": np.__version__, "torch": torch.__version__,
                    "cuda_device": torch.cuda.get_device_name()},
        "reports": {},
    }
    for market in args.markets:
        database = databases[market]
        before = source_fingerprints(database)
        sessions, calendar_info = load_sessions(
            market, date(2017, 1, 1), date(2025, 1, 1))
        samples = load_mark14_candidates(
            database, market, start="2017-01-01",
            calibration_end="2017-12-31", train_end="2021-12-31",
            test_end="2024-12-31", max_symbols=args.max_symbols,
            session_dates=sessions,
        )
        periods = _periods(samples, sessions)
        for variant in args.variants:
            run = _model_run(
                destination, samples, periods, variant=variant,
                seeds=tuple(args.seeds), device=args.device,
                epochs=args.epochs, batch_size=args.batch_size,
                day_batch_size=args.day_batch_size, hidden=args.hidden,
                recurrent_cell=args.recurrent_cell,
                initial_equity=(10_000_000.0 if market == "domestic" else 10_000.0),
            )
            after = source_fingerprints(database)
            if before != after:
                raise ValueError("Raw DB/WAL changed while running; report not published")
            run["calendar"] = calendar_info
            run["session_dates"] = list(sessions)
            run["raw_source_fingerprints"] = before
            filename = f"{variant}-{market}-report.json"
            write_new_json(destination / filename, run)
            manifest["reports"][f"{variant}-{market}"] = filename
    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_new_json(destination / "manifest.json", manifest)
    print(f"Research-only series: {destination / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
