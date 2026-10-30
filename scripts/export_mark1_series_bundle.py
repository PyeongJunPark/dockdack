"""Seal the completed CUDA Mark1.5--1.7 research selections for offline inference.

This copies existing weights; it never fits models or connects to a broker.
The small reference arrays contain one *complete same-date cross-section* from
the exact source database, with previously saved GPU scores for parity tests.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from dockdack.clean_daily_dataset import load_sessions, source_fingerprints
from dockdack.mark1_4_data import load_mark14_candidates
from dockdack.mark1_series_models import SCORE_UNITS, VARIANTS


ROOT = Path(__file__).resolve().parents[1]
MARKETS = ("domestic", "us")
SOURCE = ROOT / "outputs" / "mark1" / "mark1-series-100x2-contractfix-20260927"
DESTINATION = ROOT / "models" / "mark1_series"
# Full-history CPU rescoring against the saved RTX 5080 CUDA score arrays.
# The margins round *up* beyond each observed maximum. They are not a proof
# about unseen inputs or a replacement for future runtime calibration.
CPU_GPU_AUDIT = {
    "domestic": {
        "mark1.5": (154216, 0.004840373992919922, 4, 0.005),
        "mark1.6": (154216, 0.000012516975402832031, 1, 0.00002),
        "mark1.7": (154216, 0.000000476837158203125, 0, 0.000001),
    },
    "us": {
        "mark1.5": (172680, 0.0026769042015075684, 3, 0.003),
        "mark1.6": (172680, 0.000014036893844604492, 3, 0.00002),
        "mark1.7": (172680, 0.0000005960464477539062, 0, 0.000001),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _selected_models(source: Path) -> tuple[dict, dict]:
    run = _read(source / "manifest.json")
    expected = {f"{version}-{market}" for market in MARKETS for version in VARIANTS}
    if (set(run.get("reports", {})) != expected
            or run.get("configuration", {}).get("device") != "cuda"
            or run["configuration"].get("max_symbols") != 100
            or run.get("runtime", {}).get("cuda_device") is None
            or run.get("deployment_allowed") is not False):
        raise ValueError("Source run is not the completed 100-symbol CUDA series")
    models: dict[str, dict] = {}
    reports: dict[str, dict] = {}
    for market in MARKETS:
        models[market] = {}
        for version in VARIANTS:
            name = f"{version}-{market}"
            report_path = source / run["reports"][name]
            report = _read(report_path)
            policy = report.get("frozen_policy")
            if (report.get("variant") != version or report.get("source", {}).get("market") != market
                    or report.get("deployment_allowed") is not False
                    or not isinstance(policy, dict)
                    or report.get("calibration_selection", {}).get("decision") != "selected"
                    or report.get("calibration_selection", {}).get("selected") != policy):
                raise ValueError(f"Missing frozen 2021 selection: {name}")
            seed = int(policy["seed"])
            item = report["model_artifacts"][str(seed)]
            model_path = source / item["model"]
            weight_path = source / item["checkpoint"]
            score_path = source / item["scores"]
            model = _read(model_path)
            if (model.get("device") != "cuda" or model.get("variant") != version
                    or model.get("seed") != seed or model.get("score_unit") != SCORE_UNITS[version]
                    or model.get("research_only") is not True
                    or model.get("deployment_allowed") is not False
                    or model.get("checkpoint_file") != weight_path.name
                    or model.get("score_file") != score_path.name
                    or _sha256(weight_path) != model.get("checkpoint_sha256")
                    or _sha256(score_path) != model.get("score_sha256")):
                raise ValueError(f"Selected CUDA model/artifact mismatch: {name}")
            scores = np.load(score_path, allow_pickle=False)["scores"]
            if scores.ndim != 1 or not np.isfinite(scores).all():
                raise ValueError(f"Invalid saved scores: {name}")
            evaluation = report["evaluations"]["development_2022"]["20.0"]
            if evaluation["incomplete_data"] or evaluation["compound_net_return"] is None:
                raise ValueError(f"Missing exact 2022 development result: {name}")
            threshold = float(policy["train_numeric_score_threshold"])
            coverage = float(policy["target_train_candidate_coverage"])
            if not np.isfinite(threshold) or not 0 < coverage < 1:
                raise ValueError(f"Invalid frozen threshold or coverage: {name}")
            models[market][version] = {
                "strategy_id": f"mark1-{version.split('.')[1]}-prototype",
                "signal_phase": "preopen",
                "exit_after_sessions": 0,
                "exit_timing": "preclose",
                "cpu_gpu_audit_candidate_rows": CPU_GPU_AUDIT[market][version][0],
                "cpu_gpu_observed_max_abs_score_delta": CPU_GPU_AUDIT[market][version][1],
                "cpu_gpu_observed_threshold_flips": CPU_GPU_AUDIT[market][version][2],
                "cpu_numeric_guard_margin": CPU_GPU_AUDIT[market][version][3],
                "seed": seed,
                "score_unit": SCORE_UNITS[version],
                "frozen_numeric_score_threshold": threshold,
                "target_train_candidate_coverage": coverage,
                "model_file": model_path.name,
                "model_sha256": _sha256(model_path),
                "weights_file": weight_path.name,
                "weights_sha256": _sha256(weight_path),
                "source_report_sha256": _sha256(report_path),
                "development_2022_net_return_after_20bps": float(evaluation["compound_net_return"]),
                "development_2022_exact_fills": int(evaluation["executed_trades"]),
            }
            reports[name] = {"report": report, "scores": scores,
                             "model_path": model_path, "weight_path": weight_path}
    return models, reports


def _reference(market: str, reports: dict) -> dict:
    report = reports[f"mark1.5-{market}"]["report"]
    database = ROOT / "data" / "kiwoom_daily" / f"{market}_daily.sqlite3"
    if source_fingerprints(database) != report["raw_source_fingerprints"]:
        raise ValueError(f"Current {market} database differs from frozen CUDA run")
    sessions, _ = load_sessions(market, date(2017, 1, 1), date(2025, 1, 1))
    samples = load_mark14_candidates(
        database, market, start="2017-01-01", calibration_end="2017-12-31",
        train_end="2021-12-31", test_end="2024-12-31", max_symbols=100,
        session_dates=sessions,
    )
    if samples.source != report["source"]:
        raise ValueError(f"Candidate source differs from frozen CUDA run: {market}")
    for version in VARIANTS:
        if len(reports[f"{version}-{market}"]["scores"]) != len(samples.windows):
            raise ValueError(f"Saved score count differs from candidates: {version}-{market}")
    # Prefer a complete 100-name 2022 cross-section; fall back to the largest
    # available group only if the historical quality gates made that impossible.
    period = np.flatnonzero((samples.target_dates >= "2022-01-01") &
                            (samples.target_dates <= "2022-12-31"))
    days, counts = np.unique(samples.target_ordinals[period], return_counts=True)
    if not len(days):
        raise ValueError(f"No 2022 reference sessions: {market}")
    day = int(days[np.flatnonzero(counts == counts.max())[0]])
    rows = np.flatnonzero(samples.target_ordinals == day)
    selected = samples.source["selected_symbols"]
    pairs = [selected[int(symbol_id)] for symbol_id in samples.symbol_ids[rows]]
    result = {
        "windows": samples.windows[rows],
        "symbols": np.asarray([pair["symbol"] for pair in pairs], dtype="U32"),
        "exchanges": np.asarray([pair["exchange"] for pair in pairs], dtype="U8"),
        "target_date": np.asarray([samples.target_dates[rows[0]]], dtype="U10"),
    }
    for version in VARIANTS:
        result[f"{version}_scores"] = reports[f"{version}-{market}"]["scores"][rows]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    parser.add_argument("--refresh-manifest", action="store_true",
                        help="Reseal metadata of an existing, hash-matching generated bundle")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    destination = args.output_dir.resolve()
    if source != SOURCE.resolve() or destination != DESTINATION.resolve():
        parser.error("This export seals only the reviewed contractfix run and models/mark1_series")
    models, reports = _selected_models(source)
    if destination.exists():
        if not args.refresh_manifest:
            parser.error("Destination already exists; use --refresh-manifest only for metadata")
        current = _read(destination / "manifest.json")
        previous_seal = (destination / "manifest.sha256").read_text(encoding="ascii").strip()
        if (previous_seal != _sha256(destination / "manifest.json")
                or current.get("source_run_manifest_sha256") != _sha256(source / "manifest.json")
                or set(current.get("markets", {})) != set(MARKETS)):
            raise ValueError("Existing bundle seal/source mismatch")
        for market in MARKETS:
            item = current["markets"][market]
            if _sha256(destination / item["reference_file"]) != item["reference_sha256"]:
                raise ValueError("Existing reference checksum mismatch")
            for version in VARIANTS:
                old = item["models"][version]
                new = models[market][version]
                for key in ("seed", "model_file", "model_sha256", "weights_file",
                            "weights_sha256", "source_report_sha256",
                            "frozen_numeric_score_threshold", "score_unit"):
                    if old.get(key) != new[key]:
                        raise ValueError("Existing model and frozen policy differ from source")
                if (_sha256(destination / old["model_file"]) != old["model_sha256"]
                        or _sha256(destination / old["weights_file"]) != old["weights_sha256"]):
                    raise ValueError("Existing copied model checksum mismatch")
                item["models"][version] = new
        _write_json(destination / "manifest.json", current)
        (destination / "manifest.sha256").write_text(
            _sha256(destination / "manifest.json") + "\n", encoding="ascii")
        print(destination / "manifest.json")
        return
    references = {market: _reference(market, reports) for market in MARKETS}
    manifest = {
        "schema_version": 1,
        "title": "mark1-series",
        "versions": list(VARIANTS),
        "markets": {},
        "lookback": 30,
        "bar_columns": ["open", "high", "low", "close", "volume"],
        "entry": "next_session_open",
        "exit": "same_session_close",
        "cost_bps": 20.0,
        "threshold_comparison": "strict_greater_than",
        "training_device": "cuda",
        "research_only": True,
        "research_qualified": False,
        "deployment_allowed": False,
        "source_run_manifest_sha256": _sha256(source / "manifest.json"),
        "scoring_code_sha256": {
            "mark1_series_models.py": _sha256(ROOT / "dockdack" / "mark1_series_models.py"),
            "mark1_4_evolution.py": _sha256(ROOT / "dockdack" / "mark1_4_evolution.py"),
        },
    }
    destination.mkdir(parents=True, exist_ok=False)
    for market in MARKETS:
        reference_file = f"{market}-reference.npz"
        with (destination / reference_file).open("xb") as stream:
            np.savez_compressed(stream, **references[market])
        first_report = reports[f"mark1.5-{market}"]["report"]
        selected_symbols = [
            {"symbol": row["symbol"], "exchange": row["exchange"]}
            for row in first_report["source"]["selected_symbols"]
        ]
        manifest["markets"][market] = {
            "catalog_point_in_time": False,
            "selected_universe_size": 100,
            "selected_symbols": selected_symbols,
            "reference_file": reference_file,
            "reference_sha256": _sha256(destination / reference_file),
            "reference_target_date": str(references[market]["target_date"][0]),
            "reference_cross_section_size": len(references[market]["symbols"]),
            "models": models[market],
        }
        for version in VARIANTS:
            selected = reports[f"{version}-{market}"]
            for key in ("model_path", "weight_path"):
                file = selected[key]
                shutil.copyfile(file, destination / file.name)
    _write_json(destination / "manifest.json", manifest)
    (destination / "manifest.sha256").write_text(
        _sha256(destination / "manifest.json") + "\n", encoding="ascii")
    print(destination / "manifest.json")


if __name__ == "__main__":
    main()
