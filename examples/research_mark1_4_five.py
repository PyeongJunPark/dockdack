"""Run preregistered Mark1.4 E1/E2/E3/E5 research; never place orders.

Each invocation works on one market and one experiment. The program writes a
new ignored report directory and refuses to overwrite previous experiments.
E4 uses the frozen v2-artifact horizon runner instead of retraining a model.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import platform
import time

import numpy as np
import torch

from dockdack.clean_daily_dataset import load_sessions, source_fingerprints
from dockdack.mark1_4_data import load_mark14_candidates, mark14_chronological_splits
from dockdack.mark1_4_evolution import simulate_portfolio
from dockdack.mark1_4_followup_models import fit_score_variant
from dockdack.mark1_4_followup_selection import select_calibrated_policy
from dockdack.mark1_4_robust_ga import evolve_robust
from dockdack.research_artifacts import write_new_json


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (41, 42, 43)
MODEL_VARIANTS = {
    "e1": "e1_rank", "e2": "e2_uncertainty", "e5": "e5_features",
}


def _summary(result: dict) -> dict:
    """Keep daily equity evidence but omit only redundant proxy returns."""
    return {key: value for key, value in result.items()
            if key != "observed_day_proxy_returns"}


def _model_splits(samples, sessions, base_splits) -> tuple[np.ndarray, np.ndarray]:
    target_days = np.asarray(samples.target_dates)
    ordinals = np.asarray(samples.target_ordinals)
    train = np.flatnonzero((target_days >= "2018-01-01") &
                           (target_days <= "2020-12-31"))
    if not len(train):
        raise ValueError("No 2018-2020 neural training rows")
    last_train_day = int(ordinals[train].max())
    calibration = np.flatnonzero((target_days >= "2021-01-01") &
                                     (target_days <= "2021-12-31") &
                                     (ordinals > last_train_day + 30))
    if not len(calibration) or len(sessions) <= last_train_day + 30:
        raise ValueError("No 2021 calibration rows after 30-session embargo")
    if (not set(train).isdisjoint(calibration) or
            not set(calibration).isdisjoint(base_splits["validation"]) or
            not set(calibration).isdisjoint(base_splits["test"])):
        raise AssertionError("Research splits overlap")
    return train, calibration


def _period(samples, indices) -> dict:
    rows = np.asarray(indices)
    return {"candidates": len(rows),
            "first_target": str(min(samples.target_dates[rows])),
            "last_target": str(max(samples.target_dates[rows]))}


def _write_model_report(destination: Path, samples, splits, sessions, *,
                        experiment: str, initial_equity: float,
                        device: str, epochs: int, batch_size: int) -> dict:
    train, calibration = _model_splits(samples, sessions, splits)
    variant = MODEL_VARIANTS[experiment]
    scores_by_seed = {}
    artifacts = {}
    timings = {}
    for seed in SEEDS:
        started = time.perf_counter()
        scores, artifact = fit_score_variant(
            samples, train, variant, seed=seed, device=device,
            epochs=epochs, batch_size=batch_size, cost_bps=20.0)
        scores_by_seed[seed] = scores
        # E2's complete per-candidate mean/sigma are saved as compressed arrays
        # beside the artifact instead of inflating the comparison report.
        components = artifact.pop("prediction_components", None)
        if components is not None:
            with (destination / f"seed{seed}-components.npz").open("xb") as stream:
                np.savez_compressed(stream, **{
                    key: np.asarray(value, dtype=np.float32)
                    for key, value in components.items()
                })
            artifact["prediction_components_file"] = f"seed{seed}-components.npz"
        with (destination / f"seed{seed}-scores.npz").open("xb") as stream:
            np.savez_compressed(stream, scores=scores)
        write_new_json(destination / f"seed{seed}-model.json", artifact)
        artifacts[str(seed)] = {"model": f"seed{seed}-model.json",
                                "scores": f"seed{seed}-scores.npz"}
        timings[str(seed)] = round(time.perf_counter() - started, 3)
        print(f"{experiment} {samples.source['market']} seed {seed} scored "
              f"{len(scores):,} candidates in {timings[str(seed)]:.1f}s", flush=True)

    selection = select_calibrated_policy(
        samples, scores_by_seed, train, calibration,
        cost_bps=20.0, allocation=.1, max_positions=10,
        initial_equity=initial_equity, min_executed_trades=20)
    selected = selection.get("selected")
    if selected is None:
        scores = np.zeros(len(samples.windows), dtype=np.float32)
        threshold = 1.0  # exact cash-only comparator
    else:
        scores = scores_by_seed[int(selected["seed"])]
        threshold = float(selected["train_numeric_score_threshold"])
    periods = {"train": train, "calibration": calibration,
               "development_2022": splits["validation"],
               "reused_history_2023_2024": splits["test"]}
    evaluations = {}
    for name, rows in periods.items():
        evaluations[name] = {
            str(cost): _summary(simulate_portfolio(
                samples, rows, scores, threshold=threshold, cost_bps=cost,
                allocation=.1, max_positions=10, initial_equity=initial_equity))
            for cost in (0.0, 20.0, 40.0)
        }
    return {
        "experiment": experiment,
        "variant": variant,
        "research_only": True,
        "deployment_allowed": False,
        "winner_selection_uses_2022_or_later": False,
        "neural_fit_years": "2018-2020",
        "calibration_year": "2021 (after 30 market sessions)",
        "calibration_selection": selection,
        "frozen_policy": selected,
        "cash_only_if_no_eligible_calibration_policy": selected is None,
        "model_artifacts": artifacts,
        "fit_seconds_by_seed": timings,
        "periods": {name: _period(samples, rows) for name, rows in periods.items()},
        "cost_grid_bps": [0.0, 20.0, 40.0],
        "evaluations": evaluations,
        "source": samples.source,
        "limitations": [
            "Current catalog is not point-in-time and can cause survivorship bias.",
            "2022 and 2023-2024 have been inspected in prior research and are not pristine holdouts.",
            "OPEN/CLOSE are hypothetical fills; 20bp is an assumed roundtrip cost.",
            "Missing selected outcomes invalidate exact portfolio return rather than count as zero.",
            "No signal is connected to the GUI or any broker.",
        ],
    }


def _write_e3_report(destination: Path, samples, splits, *,
                     initial_equity: float, device: str,
                     population_size: int, generations: int) -> dict:
    reports = {}
    for seed in SEEDS:
        started = time.perf_counter()
        report, champion = evolve_robust(
            samples, splits, seed=seed, population_size=population_size,
            generations=generations, device=device, cost_bps_train=40.0,
            cost_bps_eval=20.0, allocation=.1, max_positions=10,
            initial_equity=initial_equity)
        if champion is not None:
            with (destination / f"seed{seed}-champion.npz").open("xb") as stream:
                np.savez_compressed(
                    stream, genome=champion,
                    frozen_threshold=np.asarray(
                        report["selected_strategy"]["frozen_numeric_score_threshold"],
                        dtype=np.float64))
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_new_json(destination / f"seed{seed}-report.json", report)
        reports[str(seed)] = {
            "report": f"seed{seed}-report.json",
            "train_qualified": not report["no_train_qualified_strategy"],
            "fitness": (report["selected_strategy"]["train_robust_fitness"]
                        if report["selected_strategy"] else None),
            "elapsed_seconds": report["elapsed_seconds"],
        }
        print(f"e3 {samples.source['market']} seed {seed} completed in "
              f"{report['elapsed_seconds']:.1f}s; train qualified="
              f"{reports[str(seed)]['train_qualified']}", flush=True)
    qualified = [(int(seed), info["fitness"]) for seed, info in reports.items()
                 if info["train_qualified"]]
    chosen_seed = (max(qualified, key=lambda item: (item[1], -item[0]))[0]
                   if qualified else None)
    return {
        "experiment": "e3", "research_only": True,
        "deployment_allowed": False,
        "winner_selection_uses_2022_or_later": False,
        "train_years": "2018-2021", "train_cost_bps": 40.0,
        "evaluation_cost_bps": 20.0,
        "seeds": reports, "train_selected_seed": chosen_seed,
        "cash_only_if_no_train_qualified_strategy": chosen_seed is None,
        "source": samples.source,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=(*MODEL_VARIANTS, "e3"), required=True)
    parser.add_argument("--market", choices=("domestic", "us"), required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=50)
    args = parser.parse_args(argv)
    destination = args.output_dir.resolve()
    allowed = (ROOT / "outputs" / "mark1").resolve()
    if not destination.is_relative_to(allowed) or destination == allowed or destination.exists():
        parser.error("Choose a new, unused child folder under outputs/mark1")
    database = (args.database or ROOT / "data" / "kiwoom_daily" /
                f"{args.market}_daily.sqlite3").resolve()
    if not database.is_file():
        parser.error(f"Raw database not found: {database}")
    started = time.perf_counter()
    before = source_fingerprints(database)
    sessions, calendar_info = load_sessions(
        args.market, date(2017, 1, 1), date(2025, 1, 1))
    samples = load_mark14_candidates(
        database, args.market, start="2017-01-01",
        calibration_end="2017-12-31", train_end="2021-12-31",
        test_end="2024-12-31", max_symbols=100, session_dates=sessions)
    splits = mark14_chronological_splits(
        samples, sessions, train_start="2018-01-01",
        train_end="2021-12-31", validation_end="2022-12-31",
        test_end="2024-12-31")
    loaded = time.perf_counter()
    initial_equity = 10_000_000.0 if args.market == "domestic" else 10_000.0
    destination.mkdir(parents=True, exist_ok=False)
    try:
        if args.experiment == "e3":
            report = _write_e3_report(
                destination, samples, splits, initial_equity=initial_equity,
                device=args.device, population_size=args.population_size,
                generations=args.generations)
        else:
            report = _write_model_report(
                destination, samples, splits, sessions,
                experiment=args.experiment, initial_equity=initial_equity,
                device=args.device, epochs=args.epochs, batch_size=args.batch_size)
        after = source_fingerprints(database)
        if before != after:
            raise ValueError("Raw DB/WAL changed while running; report not published")
        report["calendar"] = calendar_info
        report["session_dates"] = list(sessions)
        report["raw_source_fingerprints"] = before
        report["runtime"] = {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda_device": (
                torch.cuda.get_device_name() if torch.cuda.is_available() and
                args.device != "cpu" else None),
            "candidate_loading_seconds": round(loaded - started, 3),
            "experiment_seconds": round(time.perf_counter() - loaded, 3),
        }
        write_new_json(destination / "report.json", report)
    except Exception:
        # The created directory and any partial artifacts stay for inspection;
        # it never becomes a successful report without the final report.json.
        raise
    print(f"Research-only report: {destination / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
