"""Run research-only Mark1.8 direct-allocation study on the existing 100-name set.

CUDA is mandatory. Neither this runner nor its model can place an order or
modify the source database. 2022 and 2023-24 are already-seen diagnostics.
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
from dockdack.mark1_8_allocation import fit_allocation, simulate_allocations
from dockdack.mark1_8_bundle import write_bundle_manifest
from dockdack.research_artifacts import write_new_json


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (41, 42, 43)
ENTRY_QUANTILES = (.005, .01, .02, .05, .1)


def periods(samples, sessions):
    target = np.asarray(samples.target_dates)
    ordinals = np.asarray(samples.target_ordinals)
    train = np.flatnonzero((target >= "2018-01-01") & (target <= "2020-12-31"))
    if not len(train):
        raise ValueError("No 2018-2020 training candidates")
    last_train = int(ordinals[train].max())
    calibration = np.flatnonzero((target >= "2021-01-01") &
                                    (target <= "2021-12-31") &
                                    (ordinals > last_train + 30))
    standard = mark14_chronological_splits(
        samples, sessions, train_start="2018-01-01", train_end="2021-12-31",
        validation_end="2022-12-31", test_end="2024-12-31")
    if not len(calibration) or not set(train).isdisjoint(calibration):
        raise ValueError("Missing embargoed 2021 calibration")
    if not set(calibration).isdisjoint(standard["validation"]):
        raise AssertionError("Calibration/development overlap")
    return {"train_2018_2020": train, "calibration_2021": calibration,
            "development_2022": standard["validation"],
            "seen_history_2023_2024": standard["test"]}


def calibrate(samples, fits, train, calibration, *, initial_equity: float) -> dict:
    """Freeze a qualified 2021 signal policy, even when cash performs better.

    Financial suitability is reported separately. A user-selectable research
    signal must not be mislabeled as a profitable or validated strategy.
    """
    candidates = []
    for seed in sorted(fits):
        fit = fits[seed]
        for fraction in ENTRY_QUANTILES:
            threshold = float(np.quantile(fit.scores[train], 1 - fraction))
            result = simulate_allocations(
                samples, calibration, fit.scores, fit.sizes,
                threshold=threshold, cost_bps=20.0,
                initial_equity=initial_equity)
            valid = (not result["incomplete_data"] and
                     result["executed_trades"] >= 20)
            utility = (result["compound_net_return"] +
                       .5 * result["max_drawdown"] if valid else None)
            candidates.append({"seed": seed, "train_target_fraction": fraction,
                               "threshold": threshold, "qualified": valid,
                               "calibration_trades": result["executed_trades"],
                               "calibration_signals": result["signals"],
                               "calibration_unresolved":
                                   result["unresolved_selected_outcomes"],
                               "calibration_return": result["compound_net_return"],
                               "calibration_drawdown": result["max_drawdown"],
                               "calibration_utility": utility})
    eligible = [item for item in candidates if item["qualified"] and
                item["calibration_utility"] is not None]
    winner = max(eligible, key=lambda item: (
        item["calibration_utility"], -item["seed"],
        -item["train_target_fraction"])) if eligible else None
    return {"selection_period": "embargoed 2021 only",
            "threshold_source": "2018-2020 training score quantile",
            "target_fractions": list(ENTRY_QUANTILES),
            "minimum_integer_trades": 20,
            "rule": "max 2021 net return + 0.5 * drawdown among exact, >=20-trade policies, even if below cash",
            "cash_outperformed_selected": (winner is not None and
                                          winner["calibration_utility"] <= 0),
            "candidates": candidates, "winner": winner}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=("domestic", "us"), required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-days", type=int, default=16)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required; Mark1.8 never falls back to CPU training")
    destination = args.output_dir.resolve()
    allowed = (ROOT / "outputs" / "mark1").resolve()
    if not destination.is_relative_to(allowed) or destination == allowed or destination.exists():
        parser.error("Choose a new unused child of outputs/mark1")
    database = (args.database or ROOT / "data" / "kiwoom_daily" /
                f"{args.market}_daily.sqlite3").resolve()
    if not database.is_file():
        parser.error(f"Source database not found: {database}")
    initial_equity = 10_000_000.0 if args.market == "domestic" else 10_000.0
    started = time.perf_counter()
    before = source_fingerprints(database)
    sessions, calendar_info = load_sessions(
        args.market, date(2017, 1, 1), date(2025, 1, 1))
    samples = load_mark14_candidates(
        database, args.market, start="2017-01-01",
        calibration_end="2017-12-31", train_end="2021-12-31",
        test_end="2024-12-31", max_symbols=100, session_dates=sessions)
    split = periods(samples, sessions)
    loaded = time.perf_counter()
    destination.mkdir(parents=True, exist_ok=False)
    fits = {}
    fitting = {}
    for seed in SEEDS:
        t0 = time.perf_counter()
        fit = fit_allocation(samples, split["train_2018_2020"], seed=seed,
                             epochs=args.epochs, batch_days=args.batch_days)
        fits[seed] = fit
        with (destination / f"seed{seed}-model.pt").open("xb") as stream:
            torch.save({"state_dict": fit.model.state_dict(), "seed": seed,
                        "architecture": "150-64-32, score+size",
                        "research_only": True, "deployment_allowed": False}, stream)
        with (destination / f"seed{seed}-scores.npz").open("xb") as stream:
            np.savez_compressed(stream, scores=fit.scores, sizes=fit.sizes)
        fitting[str(seed)] = {"seconds": round(time.perf_counter() - t0, 3),
                              "training_rows": fit.training_rows,
                              "cuda_name": fit.cuda_name,
                              "epochs": list(fit.history),
                              "weights": f"seed{seed}-model.pt",
                              "outputs": f"seed{seed}-scores.npz"}
        print(f"Mark1.8 {args.market} CUDA seed {seed} finished in "
              f"{fitting[str(seed)]['seconds']:.1f}s", flush=True)
    calibration = calibrate(samples, fits, split["train_2018_2020"],
                            split["calibration_2021"],
                            initial_equity=initial_equity)
    winner = calibration["winner"]
    evaluations = {}
    if winner is not None:
        fit = fits[winner["seed"]]
        for name, rows in split.items():
            evaluations[name] = {str(cost): simulate_allocations(
                samples, rows, fit.scores, fit.sizes,
                threshold=winner["threshold"], cost_bps=cost,
                initial_equity=initial_equity) for cost in (0.0, 20.0, 40.0)}
    after = source_fingerprints(database)
    if before != after:
        raise ValueError("Source DB or WAL changed during research; report not finalized")
    report = {"version": "Mark1.8", "market": args.market,
              "research_only": True, "deployment_allowed": False,
              "objective": "differentiable net open-to-close daily return minus downside/turnover penalties; small direction auxiliary",
              "surrogate_vs_exact": "training uses soft top-10; evaluation uses independent hard top-10 integer-share cash ledger",
              "model": "30x5 normalized OHLCV -> 64/32 GELU -> rank score and 0..10% allocation",
              "split": {key: {"rows": len(rows),
                              "first_target": str(min(samples.target_dates[rows])),
                              "last_target": str(max(samples.target_dates[rows]))}
                        for key, rows in split.items()},
              "fitting": fitting, "calibration": calibration,
              "evaluations": evaluations,
              "cash_only": winner is None,
              "signal_policy_is_research_only": True,
              "source": samples.source, "source_fingerprints": before,
              "calendar": calendar_info,
              "runtime": {"python": platform.python_version(),
                          "numpy": np.__version__, "torch": torch.__version__,
                          "cuda": torch.cuda.get_device_name(0),
                          "load_seconds": round(loaded - started, 3),
                          "total_seconds": round(time.perf_counter() - started, 3)},
              "limitations": [
                  "Current catalog is not point-in-time; universe survivorship bias remains.",
                  "Hypothetical exact open/close fills are not executable fill evidence.",
                  "2022 and 2023-24 were seen in previous research, not fresh holdouts.",
                  "20bp is an assumption; market impact and tax may differ.",
                  "Never connected to GUI, demo orders, or live orders.",
              ]}
    write_new_json(destination / "report.json", report)
    write_bundle_manifest(destination)
    print(f"Research-only Mark1.8 report: {destination / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
