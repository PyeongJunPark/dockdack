"""Package frozen E4 v2 champions as Mark1.11/H3 and Mark1.12/H5.

No fitting takes place. The default seed is chosen by each market's *original
2018--2021 train fitness* only; the 2022/2023--24 horizon outcomes never choose
it. This is a new post-hoc research model, not the E4 report's selected winner.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from dockdack.clean_daily_dataset import load_sessions, source_fingerprints
from dockdack.mark1_4_data import load_mark14_candidates
from dockdack.mark1_4_evolution import GENOME_SIZE
from dockdack.mark1_4_sparse import score_sparse_genome


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "mark1" / "mark1-4-horizon-followup-auditfix-20260927" / "report.json"
DESTINATION = ROOT / "models" / "mark1_horizons"
MARKETS = ("domestic", "us")
SEEDS = (41, 42, 43)
VARIANTS = {"mark1.11": 3, "mark1.12": 5}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def main() -> None:
    if DESTINATION.exists():
        raise ValueError("Frozen horizon bundle already exists; never overwrite it")
    source = _read(REPORT)
    if (source.get("experiment") != "mark1-4-e4-frozen-v2-horizon-followup"
            or source.get("artifact_suffix") != "20260927"
            or source.get("frozen_seeds") != list(SEEDS)
            or source.get("horizons") != [3, 5]
            or source.get("research_only") is not True
            or source.get("deployment_allowed") is not False
            or source.get("runtime", {}).get("device_requested") != "cuda"
            or set(source.get("markets", {})) != set(MARKETS)):
        raise ValueError("Not the completed CUDA E4 auditfix report")
    selected: dict[str, dict] = {}
    references: dict[str, dict] = {}
    for market in MARKETS:
        item = source["markets"][market]
        db = ROOT / "data" / "kiwoom_daily" / f"{market}_daily.sqlite3"
        if (item.get("raw_source_unchanged") is not True
                or source_fingerprints(db) != item.get("raw_source_fingerprints")
                or set(item.get("champions", {})) != {str(seed) for seed in SEEDS}):
            raise ValueError(f"Raw DB or six champions differ from E4 report: {market}")
        seeds: dict[str, dict] = {}
        source_universe = None
        for seed in SEEDS:
            path = (ROOT / "outputs" / "mark1"
                    / f"mark1-4-v2-{market}-seed{seed}-20260927" / "champion.json")
            report_path = path.parent / "report.json"
            model = _read(path)
            original = _read(report_path)
            e4 = item["champions"][str(seed)]["frozen_artifact"]
            genome = np.asarray(model.get("genome"), dtype=np.float32)
            fitness = float(original["selected_strategy"]["train_fitness"])
            threshold = float(model.get("frozen_train_numeric_threshold"))
            provenance = model.get("threshold_provenance", {})
            if (model.get("model") != "mark1-4-v2-sparse-random-neural-evolution"
                    or model.get("research_only") is not True
                    or model.get("deployment_allowed") is not False
                    or model.get("genome_size") != GENOME_SIZE
                    or genome.shape != (GENOME_SIZE,) or not np.isfinite(genome).all()
                    or not np.isfinite(threshold) or not np.isfinite(fitness)
                    or threshold != float(provenance.get("numeric_score_threshold"))
                    or provenance.get("source") != "train_scores_only"
                    or provenance.get("validation_or_test_used") is not False
                    or original.get("source", {}).get("market") != market
                    or original.get("raw_source_fingerprints") != item["raw_source_fingerprints"]
                    or e4.get("sha256") != _sha256(path)
                    or e4.get("genome_sha256") != hashlib.sha256(genome.tobytes()).hexdigest()
                    or e4.get("threshold") != threshold):
                raise ValueError(f"Frozen E4 champion identity mismatch: {market} seed{seed}")
            universe = original["source"]["selected_symbols"]
            if source_universe is None:
                source_universe = universe
            elif universe != source_universe:
                raise ValueError("E4 champion seeds do not share the frozen 100-name universe")
            seeds[str(seed)] = {
                "seed": seed,
                "train_fitness": fitness,
                "frozen_numeric_score_threshold": threshold,
                "target_train_candidate_coverage": provenance["target_train_candidate_coverage"],
                "champion_file": f"{market}-seed{seed}-champion.json",
                "champion_sha256": _sha256(path),
                "genome_sha256": e4["genome_sha256"],
                "source_report_sha256": _sha256(report_path),
                "source_path": path,
                "genome": genome,
            }
        if len(source_universe) != 100:
            raise ValueError("E4 frozen universe is not 100 symbols")
        chosen = max(SEEDS, key=lambda seed: (seeds[str(seed)]["train_fitness"], -seed))
        if chosen != 43:
            raise ValueError("Train-only champion selection drifted from reviewed seed43")
        calendar, _ = load_sessions(market, date(2017, 1, 1), date(2025, 1, 1))
        samples = load_mark14_candidates(
            db, market, start="2017-01-01", calibration_end="2017-12-31",
            train_end="2021-12-31", test_end="2024-12-31",
            max_symbols=100, session_dates=calendar,
        )
        if (len(samples.windows) != item["candidate_rows"]
                or samples.source["selected_symbols"] != source_universe):
            raise ValueError("Candidate rows or universe changed from E4 report")
        period = np.flatnonzero((samples.target_dates >= "2022-01-01") &
                                (samples.target_dates <= "2022-12-31"))
        days, counts = np.unique(samples.target_ordinals[period], return_counts=True)
        if not len(days):
            raise ValueError("No 2022 reference cross-section")
        day = int(days[np.flatnonzero(counts == counts.max())[0]])
        rows = np.flatnonzero(samples.target_ordinals == day)
        identities = [source_universe[int(symbol_id)] for symbol_id in samples.symbol_ids[rows]]
        if not torch.cuda.is_available():
            raise ValueError("CUDA required to reproduce frozen source scores for export")
        source_scores = score_sparse_genome(
            samples.windows[rows], seeds[str(chosen)]["genome"], device="cuda")
        references[market] = {
            "windows": samples.windows[rows],
            "symbols": np.asarray([row["symbol"] for row in identities], dtype="U32"),
            "exchanges": np.asarray([row["exchange"] for row in identities], dtype="U8"),
            "target_date": np.asarray([samples.target_dates[rows[0]]], dtype="U10"),
            "seed43_cuda_scores": source_scores,
        }
        selected[market] = {"seed": chosen, "seeds": seeds,
                            "selected_symbols": source_universe}
    DESTINATION.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "title": "mark1-horizons",
        "variants": list(VARIANTS),
        "lookback": 30,
        "bar_columns": ["open", "high", "low", "close", "volume"],
        "training": "none_for_horizons_reuse_frozen_mark1_4_v2_neuroevolution",
        "entry": "next_session_open",
        "backtest_exit": "Hth_scheduled_session_close_entry_session_is_day_1",
        "backtest_entry_stride_sessions": "equals_horizon_no_overlap",
        "runtime_exit": "Hth_scheduled_session_intraday_not_backtest_equivalent",
        "cost_bps": 20.0,
        "research_only": True,
        "research_qualified": False,
        "deployment_allowed": False,
        "new_posthoc_model_not_e4_selected_winner": True,
        "seed_selection": "highest_original_2018_2021_train_fitness_only_among_41_42_43",
        "source_e4_report_sha256": _sha256(REPORT),
        "scoring_code_sha256": {
            "mark1_4_sparse.py": _sha256(ROOT / "dockdack" / "mark1_4_sparse.py"),
            "mark1_4_evolution.py": _sha256(ROOT / "dockdack" / "mark1_4_evolution.py"),
        },
        "markets": {},
    }
    for market in MARKETS:
        chosen = selected[market]["seed"]
        reference_file = f"{market}-reference.npz"
        with (DESTINATION / reference_file).open("xb") as stream:
            np.savez_compressed(stream, **references[market])
        model_specs = {}
        for seed in SEEDS:
            item = selected[market]["seeds"][str(seed)]
            shutil.copyfile(item["source_path"], DESTINATION / item["champion_file"])
            del item["source_path"]
            del item["genome"]
        for variant, horizon in VARIANTS.items():
            outcome = source["markets"][market]["champions"][str(chosen)]["splits"]
            val = outcome["validation"]["20"][str(horizon)]["held_horizon"]
            test = outcome["test"]["20"][str(horizon)]["held_horizon"]
            model_specs[variant] = {
                "strategy_id": "mark1-11-prototype" if horizon == 3 else "mark1-12-prototype",
                "horizon_sessions": horizon,
                "backtest_entry_stride_sessions": horizon,
                "backtest_exit_ordinal_offset_from_entry": horizon - 1,
                "runtime_exit_timing": "Hth_scheduled_session_intraday",
                "runtime_not_backtest_equivalent": True,
                "selected_seed": chosen,
                "score_metric": "frozen_v2_sparse_neural_rank_score",
                "frozen_numeric_score_threshold": selected[market]["seeds"][str(chosen)]["frozen_numeric_score_threshold"],
                "validation_2022_net_return_after_20bps": val["compound_net_return"],
                "validation_2022_exact_fills": val["executed_trades"],
                "validation_2022_signals": val["signals"],
                "reused_history_2023_2024_net_return_after_20bps": test["compound_net_return"],
                "reused_history_2023_2024_exact_fills": test["executed_trades"],
            }
        manifest["markets"][market] = {
            "selected_seed": chosen,
            "selected_symbols": [
                {"symbol": row["symbol"], "exchange": row["exchange"]}
                for row in selected[market]["selected_symbols"]
            ],
            "catalog_point_in_time": False,
            "champions": selected[market]["seeds"],
            "reference_file": reference_file,
            "reference_sha256": _sha256(DESTINATION / reference_file),
            "reference_target_date": str(references[market]["target_date"][0]),
            "reference_cross_section_size": len(references[market]["windows"]),
            "models": model_specs,
        }
    _write(DESTINATION / "manifest.json", manifest)
    (DESTINATION / "manifest.sha256").write_text(
        _sha256(DESTINATION / "manifest.json") + "\n", encoding="ascii")
    print(DESTINATION / "manifest.json")


if __name__ == "__main__":
    main()
