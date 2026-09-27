"""Copy selected, already CUDA-trained Mark1.8 research heads into a compact seal.

The ignored training output remains untouched. No fitting or operational DB
write occurs. The fixed 100-name source is rebuilt only to audit predictions.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import shutil

import numpy as np

from dockdack.clean_daily_dataset import load_sessions, source_fingerprints
from dockdack.mark1_4_data import load_mark14_candidates
from dockdack.mark1_8_bundle import load_bundle
from dockdack.research_artifacts import read_json, sha256_file, write_new_json


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"domestic": (42, 0.1103103756904602),
            "us": (41, 0.07466766238212585)}


def main() -> None:
    target = ROOT / "models" / "mark1_8"
    if target.exists():
        raise FileExistsError(f"Refusing to replace sealed Mark1.8 bundle: {target}")
    target.mkdir(parents=True, exist_ok=False)
    markets = {}
    for market, (seed, threshold) in EXPECTED.items():
        source = ROOT / "outputs" / "mark1" / (
            f"mark1-8-{market}-signalpolicy-20260927")
        prior = load_bundle(source, market=market)
        if prior.model is None or prior.threshold != threshold:
            raise ValueError("Prior CUDA result is not the selected signal policy")
        report = read_json(source / "report.json")
        winner = report["calibration"]["winner"]
        if (winner["seed"] != seed or
                report["calibration"]["cash_outperformed_selected"] is not True):
            raise ValueError("Research loss warning or selected seed changed")
        database = ROOT / "data" / "kiwoom_daily" / f"{market}_daily.sqlite3"
        before = source_fingerprints(database)
        sessions, _ = load_sessions(market, date(2017, 1, 1), date(2025, 1, 1))
        samples = load_mark14_candidates(
            database, market, start="2017-01-01", calibration_end="2017-12-31",
            train_end="2021-12-31", test_end="2024-12-31",
            max_symbols=100, session_dates=sessions)
        indices = np.unique(np.linspace(0, len(samples.windows) - 1, 16,
                                        dtype=np.int64))
        score, size = prior.predict(samples.windows[indices])
        with np.load(source / f"seed{seed}-scores.npz",
                     allow_pickle=False) as saved:
            if (not np.allclose(score, saved["scores"][indices], atol=1e-6) or
                    not np.allclose(size, saved["sizes"][indices], atol=1e-6)):
                raise ValueError("Tracked inference diverges from saved CUDA outputs")
        model_name = f"{market}-seed{seed}-model.pt"
        shutil.copyfile(source / f"seed{seed}-model.pt", target / model_name)
        audit_name = f"{market}-audit.npz"
        with (target / audit_name).open("xb") as stream:
            np.savez_compressed(stream, windows=samples.windows[indices],
                                indices=indices, scores=score, sizes=size)
        if source_fingerprints(database) != before:
            raise RuntimeError("Source DB changed during Mark1.8 export")
        markets[market] = {
            "seed": seed, "threshold": threshold,
            "score_metric": "uncalibrated_rank_score",
            "score_unit": "arbitrary_rank_score",
            "allocation_unit": "fraction_of_start_day_equity",
            "model_file": model_name, "model_sha256": sha256_file(target / model_name),
            "audit_file": audit_name, "audit_sha256": sha256_file(target / audit_name),
            "source_manifest_sha256": sha256_file(source / "manifest.json"),
            "source_report_sha256": sha256_file(source / "report.json"),
            "source_raw_db_sha256": before[database.name]["sha256"],
            "selected_symbols": [{"symbol": item["symbol"],
                                  "exchange": item["exchange"]}
                                 for item in samples.source["selected_symbols"]],
            "cash_outperformed_selected_in_2021": True,
            "catalog_point_in_time": False,
            "research_only": True, "deployment_allowed": False,
        }
        print(f"Audited Mark1.8 {market}: seed {seed}, threshold {threshold}",
              flush=True)
    manifest = {
        "schema_version": 1, "title": "mark1.8",
        "research_only": True, "deployment_allowed": False,
        "lookback": 30,
        "bar_columns": ["open", "high", "low", "close", "volume"],
        "entry": "next_session_open", "exit": "same_session_close",
        "cost_bps": 20.0, "max_positions": 10,
        "max_fraction_per_position": 0.1,
        "threshold_comparison": "strict_greater_than",
        "markets": markets,
        "warning": "2021 cash beat both selected signal policies; 2022/2023-24 are seen history",
    }
    write_new_json(target / "manifest.json", manifest)
    with (target / "manifest.sha256").open("x", encoding="ascii") as stream:
        stream.write(sha256_file(target / "manifest.json") + "\n")
    print(f"Sealed {target}", flush=True)


if __name__ == "__main__":
    main()
