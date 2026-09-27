"""Freeze the two selected Mark1.4 research artifacts for offline inference.

This is an explicit, one-time export from the final contract-fixed reports.
It does not train, score, inspect newer outcomes, or write to a broker store.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "models" / "mark1_4"
SOURCES = {
    "domestic": {
        "directory": "mark1-4-five-e5-domestic-contractfix-20260927",
        "experiment": "e5", "variant": "e5_features", "seed": 43,
        "coverage": 0.02, "threshold": 0.451765888929366,
        "score_metric": "predicted_net_return_percent",
        "report_sha256": "082bbac668eaafd13b5c17060b81e9805902d9f5a28f472d76621365e442019d",
        "model_sha256": "439205fcd1262539c5449754f40fc5446f1a45689217f78481a49c8b57cbf36f",
    },
    "us": {
        "directory": "mark1-4-five-e1-us-contractfix-20260927",
        "experiment": "e1", "variant": "e1_rank", "seed": 41,
        "coverage": 0.005, "threshold": 0.6831178557872781,
        "score_metric": "unscaled_pairwise_ranking_score",
        "report_sha256": "bbbd8d92d4698b0a2f7d7715ee1bbf3c978f8047a74fac9ab993bde2cc505d9a",
        "model_sha256": "d90143d017e7c6c7d2b159364462e1280c8e337e77b1dde4d0c4dd6329c107b9",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to replace existing bundle: {DESTINATION}")
    markets = {}
    models = {}
    for market, expected in SOURCES.items():
        origin = ROOT / "outputs" / "mark1" / expected["directory"]
        report_path = origin / "report.json"
        model_path = origin / f"seed{expected['seed']}-model.json"
        if (digest(report_path) != expected["report_sha256"]
                or digest(model_path) != expected["model_sha256"]):
            raise ValueError(f"{market}: final report or model bytes changed")
        report, model = read_json(report_path), read_json(model_path)
        selected = report.get("frozen_policy")
        if (report.get("experiment") != expected["experiment"]
                or report.get("variant") != expected["variant"]
                or report.get("research_only") is not True
                or report.get("deployment_allowed") is not False
                or not isinstance(selected, dict)
                or selected.get("seed") != expected["seed"]
                or selected.get("target_train_candidate_coverage") != expected["coverage"]
                or selected.get("train_numeric_score_threshold") != expected["threshold"]
                or report.get("source", {}).get("market") != market
                or report.get("source", {}).get("catalog_point_in_time") is not False
                or model.get("variant") != expected["variant"]
                or model.get("seed") != expected["seed"]
                or model.get("score_formula") != expected["score_metric"]
                or model.get("research_only") is not True
                or model.get("deployment_allowed") is not False
                or model.get("score_uses_only_completed_30_bars") is not True):
            raise ValueError(f"{market}: model and frozen policy provenance mismatch")
        source_symbols = report["source"]["selected_symbols"]
        symbols = [{"symbol": row["symbol"], "exchange": row["exchange"]}
                   for row in source_symbols]
        if (len(symbols) != 100 or len({(row["symbol"], row["exchange"])
                                        for row in symbols}) != 100):
            raise ValueError(f"{market}: expected 100 unique source-universe symbols")
        filename = f"{market}-seed{expected['seed']}-model.json"
        models[filename] = model_path
        markets[market] = {
            "experiment": expected["experiment"],
            "variant": expected["variant"],
            "seed": expected["seed"],
            "score_metric": expected["score_metric"],
            "target_train_candidate_coverage": expected["coverage"],
            "frozen_numeric_score_threshold": expected["threshold"],
            "model_file": filename,
            "model_sha256": expected["model_sha256"],
            "source_report": f"outputs/mark1/{expected['directory']}/report.json",
            "source_report_sha256": expected["report_sha256"],
            "calibration_cutoff_session": report["source"]["calibration_cutoff_session"],
            "catalog_point_in_time": False,
            "selected_symbols": symbols,
        }
    manifest = {
        "schema_version": 1,
        "title": "mark1.4",
        "research_only": True,
        "deployment_allowed": False,
        "lookback": 30,
        "bar_columns": ["open", "high", "low", "close", "volume"],
        "entry": "next_session_open",
        "exit": "same_session_close",
        "cost_bps": 20.0,
        "threshold_comparison": "strict_greater_than",
        "markets": markets,
        "research_code_sha256": {
            "mark1_4_followup_models.py": "4ae7a7746e7616e5b835418f47d404c9944f2d8f9917252c324202fa374e094c",
            "mark1_4_evolution.py": "81b17fc137cf011d03a117efe6392c4bfb424cfb38d5ab37b56116ad565cd834",
        },
    }
    for filename, source in models.items():
        if not source.is_file():
            raise FileNotFoundError(source)
    DESTINATION.mkdir(parents=True, exist_ok=False)
    for filename, source in models.items():
        shutil.copyfile(source, DESTINATION / filename)
    payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                          indent=2, allow_nan=False) + "\n").encode("utf-8")
    (DESTINATION / "manifest.json").write_bytes(payload)
    (DESTINATION / "manifest.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii")
    print(DESTINATION)


if __name__ == "__main__":
    main()
