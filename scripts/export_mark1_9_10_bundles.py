"""Seal already-completed Mark1.4 E2/E3 research as Mark1.9/1.10 models.

No retraining, broker calls, operational database writes, or overwrite path.
The source candidate DB is read-only and checked for changes before/after.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import shutil

import numpy as np

from dockdack.clean_daily_dataset import load_sessions, source_fingerprints
from dockdack.mark1_4_data import load_mark14_candidates, mark14_chronological_splits
from dockdack.mark1_4_evolution import simulate_portfolio
from dockdack.mark1_4_followup_models import _SmallScorer
from dockdack.mark1_4_sparse import score_sparse_genome
from dockdack.research_artifacts import read_json, sha256_file, write_new_json


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "1.9": {"experiment": "e2", "suffix": "contractfix", "seeds":
            {"domestic": 42, "us": 43}, "thresholds":
            {"domestic": -1.2470741236209881, "us": -0.8052668985724445}},
    "1.10": {"experiment": "e3", "suffix": "auditfix", "seeds":
             {"domestic": 41, "us": 41}, "thresholds":
             {"domestic": 1.9375118432461458, "us": 2.0943155987308897}},
}


def _e2_scores(windows, artifact):
    import torch
    from dockdack.mark1_4_evolution import normalize_windows

    scaler = artifact["feature_scaler"]
    raw = normalize_windows(windows).reshape(len(windows), 150)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    scaled = np.clip((raw - mean) / std, -6, 6).astype(np.float32)
    model = _SmallScorer(150, "e2_uncertainty")
    state = {key: torch.as_tensor(value, dtype=torch.float32)
             for key, value in artifact["model_state"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(scaled)).numpy()
    sigma = np.exp(np.clip(output[:, 1], -3.0, 3.0)).astype(np.float32)
    return (output[:, 0].astype(np.float32) - sigma).astype(np.float32)


def main() -> None:
    from dockdack.mark1_4_evolution import normalize_windows
    from dockdack import mark1_4_evolution, mark1_4_followup_models, mark1_4_sparse

    code_hashes = {"mark1_4_evolution.py": sha256_file(Path(mark1_4_evolution.__file__)),
                   "mark1_4_followup_models.py": sha256_file(Path(mark1_4_followup_models.__file__)),
                   "mark1_4_sparse.py": sha256_file(Path(mark1_4_sparse.__file__))}
    for version, specification in SOURCES.items():
        folder = ROOT / "models" / ("mark1_9" if version == "1.9" else "mark1_10")
        if folder.exists():
            raise FileExistsError(f"Refusing overwrite of immutable bundle: {folder}")
        folder.mkdir(parents=True, exist_ok=False)
        markets = {}
        for market in ("domestic", "us"):
            source = ROOT / "outputs" / "mark1" / (
                f"mark1-4-five-{specification['experiment']}-{market}-"
                f"{specification['suffix']}-20260927")
            if not source.is_dir():
                raise FileNotFoundError(source)
            report = read_json(source / "report.json")
            seed = specification["seeds"][market]
            threshold = specification["thresholds"][market]
            database = ROOT / "data" / "kiwoom_daily" / f"{market}_daily.sqlite3"
            before = source_fingerprints(database)
            sessions, _ = load_sessions(market, date(2017, 1, 1), date(2025, 1, 1))
            samples = load_mark14_candidates(
                database, market, start="2017-01-01", calibration_end="2017-12-31",
                train_end="2021-12-31", test_end="2024-12-31",
                max_symbols=100, session_dates=sessions)
            indices = np.unique(np.linspace(0, len(samples.windows) - 1, 16,
                                            dtype=np.int64))
            if version == "1.9":
                frozen = report["frozen_policy"]
                if (frozen["seed"] != seed or
                        frozen["train_numeric_score_threshold"] != threshold):
                    raise ValueError("E2 selected policy differs from frozen contract")
                source_name = f"seed{seed}-model.json"
                artifact = read_json(source / source_name)
                if artifact["variant"] != "e2_uncertainty":
                    raise ValueError("Expected frozen E2 model")
                destination_name = f"{market}-{source_name}"
                shutil.copyfile(source / source_name, folder / destination_name)
                audit_scores = _e2_scores(samples.windows[indices], artifact)
                saved_scores = np.load(source / f"seed{seed}-scores.npz",
                                       allow_pickle=False)["scores"][indices]
                if not np.allclose(audit_scores, saved_scores, atol=3e-6, rtol=1e-6):
                    raise ValueError("E2 exported scorer diverges from saved CUDA vector")
                metric = "predicted_net_percent_minus_one_sigma_percent"
                score_unit = "percent"
                source_selection = "2021 embargoed calibration"
                source_detail_sha = sha256_file(source / source_name)
            else:
                if report["train_selected_seed"] != seed:
                    raise ValueError("E3 selected seed differs from frozen contract")
                detail_file = source / f"seed{seed}-report.json"
                detail = read_json(detail_file)
                frozen = detail["selected_strategy"]
                if (frozen is None or
                        frozen["frozen_numeric_score_threshold"] != threshold):
                    raise ValueError("E3 selected genome differs from frozen contract")
                source_name = f"seed{seed}-champion.npz"
                saved = np.load(source / source_name, allow_pickle=False)
                genome = np.asarray(saved["genome"], dtype=np.float32)
                if float(saved["frozen_threshold"]) != threshold or len(genome) != 1826:
                    raise ValueError("E3 genotype or threshold mismatches report")
                destination_name = f"{market}-{source_name}"
                shutil.copyfile(source / source_name, folder / destination_name)
                audit_scores = score_sparse_genome(samples.windows[indices], genome,
                                                   device="cuda")
                # Verify the entire frozen strategy reproduces its saved 2022
                # integer-share path, not just a few scorer outputs.
                all_scores = score_sparse_genome(samples.windows, genome,
                                                 device="cuda")
                splits = mark14_chronological_splits(
                    samples, sessions, train_start="2018-01-01",
                    train_end="2021-12-31", validation_end="2022-12-31",
                    test_end="2024-12-31")
                observed = simulate_portfolio(
                    samples, splits["validation"], all_scores,
                    threshold=threshold, cost_bps=20.0, allocation=.1,
                    max_positions=10,
                    initial_equity=10_000_000.0 if market == "domestic" else 10_000.0)
                reference = frozen["results_at_20bp"]["validation"]
                if (observed["executed_trades"] != reference["executed_trades"] or
                        not np.isclose(observed["compound_net_return"],
                                       reference["compound_net_return"], atol=1e-10)):
                    raise ValueError("E3 frozen strategy diverges from saved CUDA report")
                metric = "unscaled_sparse_linear_rank_score"
                score_unit = "arbitrary_rank_score"
                source_selection = "2018-2021 worst-year 40bp genetic fitness"
                source_detail_sha = sha256_file(detail_file)
            audit_name = f"{market}-audit.npz"
            with (folder / audit_name).open("xb") as stream:
                np.savez_compressed(stream, windows=samples.windows[indices],
                                    indices=indices, scores=audit_scores)
            selected_symbols = [{"symbol": row["symbol"],
                                 "exchange": row["exchange"]}
                                for row in samples.source["selected_symbols"]]
            if source_fingerprints(database) != before:
                raise RuntimeError("Raw database changed while exporting bundle")
            markets[market] = {
                "seed": seed, "threshold": threshold, "score_metric": metric,
                "score_unit": score_unit, "model_file": destination_name,
                "model_sha256": sha256_file(folder / destination_name),
                "audit_file": audit_name,
                "audit_sha256": sha256_file(folder / audit_name),
                "selected_symbols": selected_symbols,
                "source_report_sha256": sha256_file(source / "report.json"),
                "source_detail_sha256": source_detail_sha,
                "source_selection": source_selection,
                "source_raw_db_sha256": before[database.name]["sha256"],
                "catalog_point_in_time": False,
                "research_only": True, "deployment_allowed": False,
            }
            print(f"Exported Mark{version} {market}: seed {seed}, threshold {threshold}",
                  flush=True)
        manifest = {
            "schema_version": 1, "title": f"mark{version}",
            "research_only": True, "deployment_allowed": False,
            "lookback": 30,
            "bar_columns": ["open", "high", "low", "close", "volume"],
            "entry": "next_session_open", "exit": "same_session_close",
            "cost_bps": 20.0, "max_positions": 10,
            "threshold_comparison": "strict_greater_than",
            "research_code_sha256": code_hashes, "markets": markets,
            "warning": "Previously seen history, survivorship bias and hypothetical fills; not a profit guarantee",
        }
        write_new_json(folder / "manifest.json", manifest)
        with (folder / "manifest.sha256").open("x", encoding="ascii") as stream:
            stream.write(sha256_file(folder / "manifest.json") + "\n")
        print(f"Sealed {folder}", flush=True)


if __name__ == "__main__":
    main()
