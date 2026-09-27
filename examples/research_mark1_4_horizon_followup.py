"""Frozen Mark1.4 v2 H1-vs-H3/H5 research-only follow-up.

This reads the six already-frozen sparse champions and original Kiwoom daily
SQLite databases without training, GUI access or order submission.  Test years
have been inspected in earlier Mark1.4 research and are not a fresh holdout.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from dockdack.clean_daily_dataset import load_sessions, source_fingerprints
from dockdack.mark1_4_data import load_mark14_candidates, mark14_chronological_splits
from dockdack.mark1_4_evolution import GENOME_SIZE
from dockdack.mark1_4_horizon import compare_fixed_horizon, load_horizon_closes
from dockdack.mark1_4_sparse import score_sparse_genome
from dockdack.research_artifacts import write_new_json


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (41, 42, 43)
COSTS_BPS = (0.0, 20.0, 40.0)
HORIZONS = (3, 5)
ARTIFACT_SUFFIX = "20260927"
DATES = {
    "start": "2017-01-01",
    "calibration_end": "2017-12-31",
    "train_start": "2018-01-01",
    "train_end": "2021-12-31",
    "validation_end": "2022-12-31",
    "test_end": "2024-12-31",
}


def _champion(path: Path, market: str, seed: int, raw_fingerprint: dict):
    if not path.is_file():
        raise FileNotFoundError(f"Frozen v2 champion missing: {path}")
    report_path = path.parent / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Source report missing: {report_path}")
    artifact_bytes = path.read_bytes()
    artifact = json.loads(artifact_bytes)
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (artifact.get("model") != "mark1-4-v2-sparse-random-neural-evolution"
            or artifact.get("research_only") is not True
            or artifact.get("deployment_allowed") is not False
            or artifact.get("genome_size") != GENOME_SIZE
            or source_report.get("source", {}).get("market") != market
            or source_report.get("raw_source_fingerprints") != raw_fingerprint):
        raise ValueError(f"Frozen champion/source identity mismatch: {path}")
    genome = np.asarray(artifact.get("genome"), dtype=np.float32)
    if genome.shape != (GENOME_SIZE,) or not np.isfinite(genome).all():
        raise ValueError(f"Invalid frozen genome: {path}")
    threshold = float(artifact["frozen_train_numeric_threshold"])
    provenance = artifact.get("threshold_provenance", {})
    if (not np.isfinite(threshold)
            or provenance.get("validation_or_test_used") is not False
            or provenance.get("source") != "train_scores_only"
            or threshold != float(provenance.get("numeric_score_threshold"))):
        raise ValueError(f"Threshold was not train-frozen: {path}")
    if (source_report.get("splits", {}).get("validation", {}).get("first_target") != "2022-02-17"
            and market == "domestic"):
        raise ValueError(f"Unexpected domestic split provenance: {path}")
    source_symbols = source_report.get("source", {}).get("selected_symbols")
    return genome, threshold, source_symbols, {
        "path": str(path),
        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "genome_sha256": hashlib.sha256(genome.tobytes()).hexdigest(),
        "threshold": threshold,
        "threshold_provenance": provenance,
        "source_report_path": str(report_path),
        "seed": seed,
    }


def _compact(arm: dict, *, one_day: bool) -> dict:
    keys = (
        "signals", "observed_signals", "unresolved_signals", "executed_trades",
        "executed_trades_on_exact_prefix", "final_equity", "compound_net_return",
        "incomplete_data", "initial_equity", "cost_bps", "allocation", "max_positions",
    )
    result = {key: arm[key] for key in keys}
    if one_day:
        result["max_drawdown"] = arm["max_drawdown"]
        result["active_entry_sessions"] = arm["active_sessions"]
        for key in ("daily_dates", "daily_ordinals", "daily_returns",
                    "daily_signals", "daily_selected_symbol_ids", "daily_drawdowns"):
            result[key] = arm[key]
    else:
        result["max_drawdown_at_exits_only"] = arm["max_drawdown_at_exits_only"]
        result["active_entry_sessions"] = arm["active_entry_sessions"]
        result["entry_sessions"] = arm["entry_sessions"]
        result["ineligible_trailing_entry_sessions"] = arm["ineligible_trailing_entry_sessions"]
        for key in ("entry_dates", "entry_ordinals", "exit_ordinals",
                    "entry_signal_counts", "entry_selected_symbol_ids",
                    "realized_block_returns", "drawdowns_at_exit_points"):
            result[key] = arm[key]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New, unused child directory under ignored outputs/mark1")
    args = parser.parse_args(argv)
    output_root = (ROOT / "outputs" / "mark1").resolve()
    destination = args.output_dir.resolve()
    if (not destination.is_relative_to(output_root) or destination == output_root
            or destination.exists()):
        parser.error("Choose a new, unused child directory under outputs/mark1")
    started = time.perf_counter()
    result = {
        "experiment": "mark1-4-e4-frozen-v2-horizon-followup",
        "research_only": True,
        "deployment_allowed": False,
        "artifact_suffix": ARTIFACT_SUFFIX,
        "frozen_seeds": list(SEEDS),
        "horizons": list(HORIZONS),
        "roundtrip_costs_bps": list(COSTS_BPS),
        "protocol": {
            "lookback": "30 completed OHLCV bars through t; signal frozen before t+1",
            "entry": "t+1 session open, hypothetical fill",
            "exit": "Hth scheduled session close, ordinal t+1+(H-1)",
            "cadence": "new entries every H sessions; no overlapping positions or leverage",
            "baseline": "same frozen score/threshold/ranking and same entry dates, sold at entry-session close",
            "sizing": "up to 10 names, 10% start-equity budget per selected name, integer shares",
            "missing": "selected missing entry or exit makes exact total return null, never zero/backfilled",
            "split": "2018-21 train; 2022 validation; 2023-24 reused historical test; 30-session embargo",
            "training": "none; six 2026-09-27 Mark1.4 v2 champions are loaded unchanged",
            "candidate_count": 100,
        },
        "markets": {},
        "limitations": [
            "2023-24 history has been inspected in earlier Mark1.4 experiments; not independent OOS",
            "current Kiwoom catalog is not a historical point-in-time security master",
            "open/close execution, slippage, corporate actions, FX and taxes are not independently verified",
            "H3/H5 interim mark-to-market drawdowns are unavailable; only exit-equity drawdowns are reported",
            "different H3 and H5 entry cadences mean their cohorts are not identical to each other",
        ],
    }
    for market in ("domestic", "us"):
        market_start = time.perf_counter()
        database = ROOT / "data" / "kiwoom_daily" / f"{market}_daily.sqlite3"
        before = source_fingerprints(database)
        calendar, calendar_meta = load_sessions(
            market, date.fromisoformat(DATES["start"]),
            date.fromisoformat(DATES["test_end"]) + timedelta(days=1))
        samples = load_mark14_candidates(
            database, market, start=DATES["start"],
            calibration_end=DATES["calibration_end"],
            train_end=DATES["train_end"], test_end=DATES["test_end"],
            max_symbols=100, session_dates=calendar)
        splits = mark14_chronological_splits(
            samples, calendar, train_start=DATES["train_start"],
            train_end=DATES["train_end"], validation_end=DATES["validation_end"],
            test_end=DATES["test_end"], embargo_sessions=30)
        loaded = load_horizon_closes(database, samples, calendar, HORIZONS)
        after_loading = source_fingerprints(database)
        if before != after_loading:
            raise RuntimeError(f"{market} raw DB/WAL changed during read-only loading")
        initial_equity = 10_000_000.0 if market == "domestic" else 10_000.0
        market_result = {
            "candidate_rows": len(samples.windows),
            "calendar": calendar_meta,
            "session_dates": list(calendar),
            "source": {key: value for key, value in samples.source.items()
                       if key != "selected_symbols"},
            "raw_source_fingerprints": before,
            "horizon_exit_coverage": {
                str(horizon): int(np.isfinite(loaded[horizon]).sum())
                for horizon in HORIZONS},
            "splits": {
                split: {"candidates": len(indices),
                        "first_target": min(samples.target_dates[indices].tolist()),
                        "last_target": max(samples.target_dates[indices].tolist())}
                for split, indices in splits.items()
            },
            "champions": {},
        }
        for seed in SEEDS:
            path = (ROOT / "outputs" / "mark1"
                    / f"mark1-4-v2-{market}-seed{seed}-{ARTIFACT_SUFFIX}"
                    / "champion.json")
            genome, threshold, source_symbols, metadata = _champion(
                path, market, seed, before)
            if source_symbols != samples.source["selected_symbols"]:
                raise ValueError(f"Universe changed relative to champion: {path}")
            scores = score_sparse_genome(samples.windows, genome, device=args.device)
            champion_results = {"frozen_artifact": metadata, "splits": {}}
            for split in ("validation", "test"):
                indices = splits[split]
                start = int(samples.target_ordinals[indices].min())
                end = int(samples.target_ordinals[indices].max())
                cost_results = {}
                for cost in COSTS_BPS:
                    horizon_results = {}
                    for horizon in HORIZONS:
                        compared = compare_fixed_horizon(
                            samples, indices, scores, loaded[horizon],
                            horizon=horizon, threshold=threshold, cost_bps=cost,
                            allocation=.1, max_positions=10,
                            initial_equity=initial_equity,
                            split_start_ordinal=start, split_end_ordinal=end)
                        horizon_results[str(horizon)] = {
                            "one_day_same_entries": _compact(compared["one_day"], one_day=True),
                            "held_horizon": _compact(compared["multi_day"], one_day=False),
                            "entry_sessions": len(compared["entry_ordinals"]),
                            "ineligible_trailing_entry_sessions":
                                compared["ineligible_trailing_entry_sessions"],
                            "same_entry_signals": compared["same_entry_signals"],
                        }
                    cost_results[str(int(cost))] = horizon_results
                champion_results["splits"][split] = cost_results
            market_result["champions"][str(seed)] = champion_results
            print(f"{market} seed{seed}: frozen H1/H3/H5 comparisons complete", flush=True)
        end_fingerprint = source_fingerprints(database)
        if before != end_fingerprint:
            raise RuntimeError(f"{market} raw DB/WAL changed during experiment")
        market_result["raw_source_unchanged"] = True
        market_result["seconds"] = round(time.perf_counter() - market_start, 3)
        result["markets"][market] = market_result
    result["runtime"] = {
        "seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device_requested": args.device,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    destination.mkdir(parents=True, exist_ok=False)
    write_new_json(destination / "report.json", result)
    print(f"Research report: {destination / 'report.json'}", flush=True)
    return result


if __name__ == "__main__":
    main()
