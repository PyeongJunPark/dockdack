"""Export verified selected ensembles into a new portable research-only bundle.

The source is never modified. An interrupted export retains its newly-created
staging directory for inspection; no existing bundle or model is overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dockdack.mark1_prototype_inference import (
    BUNDLE_VERSION, FEATURE_NAMES, MARKET_MODELS, OWNER, RISK_FLAGS, SCHEMA_VERSION,
    SEEDS, SEMANTICS, TITLE, WARNINGS, PrototypePredictor, runtime_code_hashes, sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def compact_assessment(metrics):
    keys = ("count", "signal_count", "signal_days", "symbol_count", "coverage", "precision",
            "gross_mean_return", "net_mean_return", "cost_bps", "block_bootstrap")
    return {key: metrics[key] for key in keys if key in metrics}


def export_bundle(training_root, destination, *, cache_dir=None, backtest_root=None):
    # Only the exporter imports the complete frozen research verifier / Torch.
    from dockdack.mark1_selective_inference import SelectivePredictor
    from dockdack.mark1_selective_models import model_path

    training_root, destination = Path(training_root).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace existing bundle: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError("Bundle parent must be an existing, unlinked directory")
    destination = destination.resolve()
    cache_dir = Path(cache_dir or ROOT / "outputs/mark1/cache").resolve()
    backtest_root = Path(backtest_root or ROOT / "outputs/mark1/selective-backtest-20260916").resolve()
    protocol = read_json(training_root / "protocol.json")
    summary = read_json(training_root / "summary.json")
    backtest_global = read_json(backtest_root / "summary.json")
    # Instantiation cross-checks global+market completion, frozen sources,
    # selected seeds, positive calibration, policies and all native checksums.
    originals = {market: SelectivePredictor(training_root / market) for market in MARKET_MODELS}
    for market, original in originals.items():
        if original.model_name != MARKET_MODELS[market] or original.research_qualified:
            raise ValueError("This prototype version requires the diagnosed final selected research models")
    protected = {training_root / "protocol.json", training_root / "summary.json", Path(__file__).resolve()}
    protected.update(ROOT / path for path in protocol["code_sha256"])
    protected.add(ROOT / "dockdack/mark1_selective_inference.py")
    protected.add(ROOT / "dockdack/mark1_prototype_inference.py")
    cache_paths = {}
    for market in MARKET_MODELS:
        source = summary[market]["source"]
        key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
        cache_paths[market] = cache_dir / f"{market}-{key}.npz"
        protected.update((cache_paths[market], Path(source["database_path"]),
                          ROOT / "models/mark1" / f"{market}.pt", training_root / market / "summary.json",
                          training_root / market / "source.json", training_root / market / "selection_locked.json",
                          training_root / market / "cpu-inference-audit.json", backtest_root / market / "summary.json"))
        for fold in ("walk_2022", "walk_2024"):
            protected.add(training_root / market / fold / "ensemble.json")
        for seed in SEEDS:
            trial = training_root / market / "walk_2024" / f"{MARKET_MODELS[market]}-{seed}"
            artifact = model_path(trial, MARKET_MODELS[market])
            protected.update((artifact, artifact.with_name(artifact.name + ".json"), trial / "metadata.json"))
    before = {str(path): sha256_file(path) for path in sorted(protected)}
    # Both source database checksums are rechecked, not merely recorded.
    for market in MARKET_MODELS:
        source = summary[market]["source"]
        if before[str(Path(source["database_path"]))] != source["database_sha256"]:
            raise ValueError("Source database has changed since training")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-export-", dir=destination.parent)).resolve()
    if stage.parent != destination.parent or not stage.name.startswith(f".{destination.name}-export-"):
        raise ValueError("Export staging path escaped the intended parent")
    references = {}
    for market, original in originals.items():
        folder = stage / market
        folder.mkdir()
        selected = MARKET_MODELS[market]
        ensemble = summary[market]["ensembles"]["walk_2024"]
        members = []
        for seed in SEEDS:
            trial = training_root / market / "walk_2024" / f"{selected}-{seed}"
            native = model_path(trial, selected)
            target = folder / f"seed{seed}.cbm"
            shutil.copyfile(native, target)
            shutil.copyfile(native.with_name(native.name + ".json"), target.with_name(target.name + ".json"))
            metadata = read_json(trial / "metadata.json")
            members.append({"seed": seed, "path": f"{market}/{target.name}", "sha256": sha256_file(target),
                            "sidecar_path": f"{market}/{target.name}.json",
                            "sidecar_sha256": sha256_file(target.with_name(target.name + ".json")),
                            "best_iteration": metadata["best_iteration"], "training_versions": metadata["versions"]})
        backtest = read_json(backtest_root / market / "summary.json")
        if (backtest.get("completed") is not True or backtest.get("selected_architecture") != selected
                or backtest.get("source") != summary[market]["source"]
                or backtest_global.get(market) != backtest):
            raise ValueError("Completed backtest summary disagrees with selected training artifacts")
        evaluation = backtest["models"]["selective"]
        result = {
            "market": market, "model_name": selected, "fold": "walk_2024", "seeds": list(SEEDS),
            "members": members, "calibration": ensemble["calibration"],
            "policy": ensemble["policy_selection"]["chosen_policy"], "risk_flags": RISK_FLAGS,
            "source": summary[market]["source"],
            "research_results": {
                "development": {fold: {"audit": compact_assessment(value["audit"]),
                                       "qualification": value["qualification"]}
                                for fold, value in summary[market]["ensembles"].items()},
                "reused_evaluation": {"description": backtest["research_evaluation"], "range": backtest["range"],
                                      "samples": backtest["samples"],
                                      "classification": compact_assessment(evaluation["classification"]),
                                      "qualification": evaluation["qualification"],
                                      "portfolios": evaluation["portfolios"]},
            },
            "provenance": {"source_summary_sha256": sha256_file(training_root / market / "summary.json"),
                           "source_ensemble_sha256": sha256_file(training_root / market / "walk_2024/ensemble.json"),
                           "backtest_summary_sha256": sha256_file(backtest_root / market / "summary.json")},
        }
        write_json(folder / "manifest.json", result)
        references[market] = {"path": f"{market}/manifest.json", "sha256": sha256_file(folder / "manifest.json")}
    manifest = {
        "owner": OWNER, "schema_version": SCHEMA_VERSION, "title": TITLE, "version": BUNDLE_VERSION,
        "completed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "markets": references, "semantics": SEMANTICS, "risk_flags": RISK_FLAGS, "warnings": WARNINGS,
        "feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
        "runtime_code_sha256": runtime_code_hashes(), "training_versions": protocol["versions"],
        "provenance": {"training_run": training_root.name, "protocol_sha256": sha256_file(training_root / "protocol.json"),
                       "summary_sha256": sha256_file(training_root / "summary.json"),
                       "frozen_training_code_sha256": protocol["code_sha256"],
                       "exporter_sha256": sha256_file(Path(__file__)),
                       "original_inference_sha256": sha256_file(ROOT / "dockdack/mark1_selective_inference.py")},
        "portability": "Only this bundle and the local numpy/CatBoost inference modules are needed; source paths are provenance, never runtime reads.",
    }
    write_json(stage / "manifest.json", manifest)
    (stage / "manifest.sha256").write_text(sha256_file(stage / "manifest.json") + "\n", encoding="ascii")
    validations = {}
    for market, original in originals.items():
        portable = PrototypePredictor(stage, market)
        audited = read_json(training_root / market / "cpu-inference-audit.json")
        if audited.get("success") is not True:
            raise ValueError("Original CPU audit must have passed")
        chosen = [row for row in audited["rows"] if row["candidate_factor"] == 1.][:4]
        if len(chosen) != 4 or len({row["dataset_index"] for row in chosen}) != 4:
            raise ValueError("Four unique approved 2024 histories are required")
        with np.load(cache_paths[market], allow_pickle=False) as archive:
            bars, starts = archive["bars"], archive["starts"]
            histories = [(row, bars[int(starts[row["dataset_index"]]):int(starts[row["dataset_index"]]) + 30].copy())
                         for row in chosen]
        rows = []
        for row, history in histories:
            if not str(row["target_date"]).startswith("2024-"):
                raise ValueError("Export equivalence must use approved pre-2025 histories only")
            untouched = history.copy()
            for factor in (1., .9975, 1.0025):
                entry = float(row["actual_open"]) * factor
                expected = original.predict(history, entry)
                actual = portable.predict(history, current_price=str(entry))
                difference = abs(expected["probability_success"] - actual["probability_success"])
                stop_difference = (0. if expected["probability_stop"] is None else
                                   abs(expected["probability_stop"] - actual["probability_stop"]))
                if (difference > 1e-12 or stop_difference > 1e-12
                        or actual["predicts_success"] is not expected["selected_research"]
                        or any(actual[name] is not flag for name, flag in RISK_FLAGS.items())
                        or not np.array_equal(history, untouched)
                        or actual["candidate_take_price"] != entry * 1.01
                        or actual["candidate_stop_price"] != entry * .991):
                    raise ValueError("Portable CPU prediction differs from original or violates research-only contract")
                rows.append({"dataset_index": row["dataset_index"], "target_date": row["target_date"],
                             "candidate_factor": factor, "probability_success": actual["probability_success"],
                             "probability_stop": actual["probability_stop"], "success_error": difference,
                             "stop_error": stop_difference, "selected_research": actual["predicts_success"]})
        validations[market] = {"histories": 4, "prediction_calls": len(rows), "passed": True,
                               "inputs_unchanged": True, "rows": rows}
    after = {path: sha256_file(Path(path)) for path in before}
    if before != after:
        raise ValueError("A protected original artifact changed during export")
    write_json(stage / "export-validation.json", {
        "passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest_sha256": sha256_file(stage / "manifest.json"), "markets": validations,
        "research_only": True, "deployment_allowed": False, "protected_files_unchanged": True,
        "protected_sha256_before": before, "protected_sha256_after": after,
        "validation_scope": "CPU numerical equivalence on 4 approved 2024 histories x 3 entry prices per market; not a profitability or deployment test",
    })
    # No arbitrary directory move/delete: both fully-resolved siblings are
    # validated again and the destination must still not exist.
    if stage.parent != destination.parent or destination.exists() or destination.is_symlink():
        raise ValueError("Export destination changed during validation")
    stage.rename(destination)
    return {"bundle": str(destination), "version": BUNDLE_VERSION,
            "bundle_manifest_sha256": sha256_file(destination / "manifest.json"),
            "validation": str(destination / "export-validation.json"), "prediction_comparisons": 24}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=ROOT / "outputs/mark1/selective-20260916")
    parser.add_argument("--output", type=Path, default=ROOT / "models/mark1_prototype")
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/mark1/cache")
    parser.add_argument("--backtest", type=Path, default=ROOT / "outputs/mark1/selective-backtest-20260916")
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.training, args.output, cache_dir=args.cache, backtest_root=args.backtest), indent=2))


if __name__ == "__main__":
    main()
