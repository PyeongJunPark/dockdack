"""Export a completed two-market MK1.2 run without altering any training artifact."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from dockdack.mark1_2_inference import (
    FLAGS, MARKETS, OWNER, SEEDS, SEMANTICS, TITLE, WARNINGS, Predictor,
    checked_file, load_weights, read_json, runtime_code_hashes, sha256_file,
    unlinked_path, validate_calibration, validate_config,
)
from dockdack.mark1_2_models import MODEL_NAMES, build_model
from dockdack.mark1_deep_models import CLASS_NAMES, FEATURE_NAMES, TARGET, features_from_history, success_logit
from dockdack.mark1_metrics import calibrated_probability

ROOT = Path(__file__).resolve().parents[1]
TRAINING_CODE_FILES = (
    "examples/train_mark1_2.py", "dockdack/mark1_2_data.py", "dockdack/mark1_2_models.py",
    "dockdack/mark1_deep_models.py", "dockdack/mark1_deep_data.py", "dockdack/mark1_data.py",
    "dockdack/mark1_metrics.py", "dockdack/mark1_selective_policy.py",
    "dockdack/research_artifacts.py", "dockdack/research_compat.py", "dockdack/research_arrays.py",
)


def inspect_training_run(run, workspace=ROOT):
    """Validate completed source evidence and return records; no writes or DB I/O."""
    root, workspace = unlinked_path(run), unlinked_path(workspace)
    if not root.is_dir():
        raise ValueError("Training run is missing")
    hashes = {}
    def document(relative):
        path = unlinked_path(root / relative)
        before = sha256_file(path)
        result = read_json(path)
        if sha256_file(path) != before:
            raise ValueError("Training source changed while reading")
        hashes[relative] = before
        return result
    protocol, summary, status = document("protocol.json"), document("summary.json"), document("status.json")
    code = protocol.get("code_sha256")
    recipe = protocol.get("protocol")
    if (protocol.get("phase") != "train" or sorted(protocol.get("markets", [])) != sorted(MARKETS)
            or set(summary) != set(MARKETS) or status.get("status") != "completed"
            or status.get("phase") != "train" or status.get("training_only") is not True
            or not isinstance(code, dict) or set(code) != set(TRAINING_CODE_FILES)
            or not isinstance(recipe, dict) or recipe.get("version") != "mark1-2-neural-price-augmentation-v1"
            or recipe.get("target") != TARGET or recipe.get("take_profit_pct") != 1.
            or recipe.get("stop_loss_pct") != .9 or recipe.get("threshold") != .5
            or recipe.get("both_touch") != "stop_first_failure" or recipe.get("ensemble_seeds") != list(SEEDS)
            or recipe.get("price_factors") != [1., .99, .995, 1.005, 1.01]
            or recipe.get("architectures") != list(MODEL_NAMES)
            or recipe.get("deployment_allowed") is not False or recipe.get("intraday_path_verified") is not False):
        raise ValueError("Completed two-market non-pilot training contract required")
    for relative, digest in code.items():
        checked_file(workspace, relative, digest)
    validate_config(recipe.get("model_config"))
    markets = {}
    for market in MARKETS:
        item = document(f"{market}/summary.json")
        lock = document(f"{market}/selection_locked.json")
        architecture = item.get("selected")
        if (item != summary[market] or item.get("completed") is not True or item.get("pilot") is not False
                or item.get("market") != market or architecture not in MODEL_NAMES
                or item.get("protocol") != recipe or item.get("deployment_allowed") is not False
                or item.get("intraday_path_verified") is not False or type(item.get("research_qualified")) is not bool
                or lock.get("selected") != architecture or lock.get("ranking") != item.get("ranking")
                or not isinstance(item.get("ranking"), list) or not item["ranking"]
                or item["ranking"][0].get("architecture") != architecture
                or lock.get("code_sha256") != code or lock.get("source") != item.get("source")
                or lock.get("rule") != recipe.get("selection")
                or not isinstance(item.get("ensemble"), dict) or set(item["ensemble"]) != {"walk_2022", "walk_2024"}):
            raise ValueError("Market summary/selection lock mismatch")
        for fold in ("walk_2022", "walk_2024"):
            ensemble = document(f"{market}/{fold}/ensemble.json")
            if (ensemble != item["ensemble"][fold] or not isinstance(ensemble.get("qualification"), dict)
                    or type(ensemble["qualification"].get("qualified")) is not bool):
                raise ValueError("Ensemble evidence differs from summary")
            validate_calibration(ensemble.get("calibration"))
        if item["research_qualified"] != all(value["qualification"]["qualified"] for value in item["ensemble"].values()):
            raise ValueError("Research qualification summary mismatch")
        members = []
        for seed in SEEDS:
            prefix = f"{market}/walk_2024/{architecture}-{seed}"
            result = document(prefix + "/result.json")
            contract = document(prefix + "/contract.json")
            seal = document(prefix + "/model.sha256.json")
            artifacts = result.get("artifact_sha256")
            context = contract.get("context", {})
            if (result.get("completed") is not True or result.get("contract") != contract
                    or result.get("architecture") != architecture or type(result.get("seed")) is not int or result["seed"] != seed
                    or contract.get("architecture") != architecture or contract.get("seed") != seed
                    or contract.get("pilot") is not False or contract.get("protocol") != recipe
                    or context.get("market") != market or context.get("fold") != "walk_2024"
                    or context.get("code_sha256") != code or context.get("source") != item.get("source")
                    or context.get("data_receipt") != item.get("data_receipt")
                    or not isinstance(artifacts, dict) or set(artifacts) != {"model.pt", "predictions.npz", "history.json"}
                    or seal != {"sha256": artifacts.get("model.pt")}):
                raise ValueError("Trial artifact/identity contract mismatch")
            for filename, digest in artifacts.items():
                checked_file(root, prefix + "/" + filename, digest)
                hashes[prefix + "/" + filename] = digest
            checkpoint = load_weights(root / prefix / "model.pt")
            if (checkpoint.get("contract") != contract or checkpoint.get("context") != context
                    or checkpoint.get("architecture") != architecture or checkpoint.get("seed") != seed
                    or checkpoint.get("model_config") != recipe["model_config"]
                    or checkpoint.get("feature_names") != list(FEATURE_NAMES)
                    or checkpoint.get("class_names") != list(CLASS_NAMES) or checkpoint.get("target") != TARGET
                    or checkpoint.get("threshold") != .5 or checkpoint.get("take_profit_pct") != 1.
                    or checkpoint.get("stop_loss_pct") != .9
                    or any(checkpoint.get(key) is not value for key, value in FLAGS.items())
                    or checkpoint.get("calibration") != result.get("calibration")):
                raise ValueError("Checkpoint training identity/risk contract mismatch")
            validate_calibration(checkpoint.get("calibration"))
            model = build_model(architecture, **recipe["model_config"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            members.append({"seed": seed, "checkpoint": checkpoint, "path": root / prefix / "model.pt",
                            "sha256": artifacts["model.pt"], "result_sha256": hashes[prefix + "/result.json"]})
        markets[market] = {"summary": item, "selected": architecture, "model_config": recipe["model_config"],
                           "calibration": item["ensemble"]["walk_2024"]["calibration"], "members": members}
    for relative, digest in hashes.items():
        checked_file(root, relative, digest)
    return {"protocol": protocol, "summary": summary, "markets": markets, "source_files_sha256": hashes}


def synthetic_cases():
    """Four positive histories x three price queries, not real observations."""
    histories, entries = [], []
    for pattern in range(4):
        t = np.arange(30, dtype=np.float32)
        close = (100 + pattern * 20 + .2 * t + np.sin(t / (pattern + 2))).astype(np.float32)
        bars = np.stack((close - .15, close + 1, close - 1, close,
                         10000 + 30 * t + 100 * pattern), axis=-1)
        for factor in (.99, 1., 1.01):
            histories.append(bars)
            entries.append(np.float32(close[-1] * factor))
    return np.asarray(histories), np.asarray(entries)


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _publish_manifest(stage, manifest, validation):
    _write_json(stage / "export-validation.json", validation)
    manifest["validation_sha256"] = sha256_file(stage / "export-validation.json")
    _write_json(stage / "manifest.json", manifest)
    (stage / "manifest.sha256").write_text(sha256_file(stage / "manifest.json") + "\n", encoding="ascii")


def export_bundle(run, destination=None, *, workspace=ROOT):
    """Create a NEW bundle through sibling staging; never overwrite a destination."""
    workspace = unlinked_path(workspace)
    target = unlinked_path(destination if destination is not None else workspace / "models/mark1_2_prototype")
    if target.exists():
        raise ValueError("Destination exists; existing models are never overwritten")
    source = inspect_training_run(run, workspace)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mark1-2-building-", dir=target.parent))
    try:
        manifest = {"owner": OWNER, "title": TITLE, "schema_version": 1, "completed": True,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(), "semantics": SEMANTICS,
                    "risk_flags": FLAGS, "warnings": WARNINGS, "feature_names": list(FEATURE_NAMES),
                    "runtime_code_sha256": runtime_code_hashes(), "markets": {},
                    "source_run": {"source_files_sha256": source["source_files_sha256"],
                                   "protocol_sha256": source["source_files_sha256"]["protocol.json"],
                                   "summary_sha256": source["source_files_sha256"]["summary.json"]},
                    "exporter_sha256": sha256_file(Path(__file__)),
                    "versions": {"torch": str(torch.__version__), "numpy": np.__version__}}
        for market, item in source["markets"].items():
            (stage / market).mkdir()
            members = []
            for member in item["members"]:
                relative = f"{market}/seed{member['seed']}.pt"
                payload = {"state_dict": member["checkpoint"]["state_dict"], "architecture": item["selected"],
                           "model_config": item["model_config"], "market": market, "seed": member["seed"],
                           "target": TARGET, "risk_flags": FLAGS}
                torch.save(payload, stage / relative)
                members.append({"seed": member["seed"], "path": relative, "sha256": sha256_file(stage / relative),
                                "source_checkpoint_sha256": member["sha256"]})
            manifest["markets"][market] = {"market": market, "architecture": item["selected"],
                "model_config": item["model_config"], "calibration": item["calibration"], "fold": "walk_2024",
                "members": members, "research_qualified": item["summary"]["research_qualified"], "risk_flags": FLAGS}
        # Staging is never published until the same public reader has checked
        # every copied model and numerical equivalence below has succeeded.
        validation = {"completed": True, "cases": 24, "scope": "synthetic_cpu_equivalence_not_profitability"}
        _publish_manifest(stage, manifest, validation)
        histories, entries = synthetic_cases()
        differences = {}
        with torch.inference_mode(), torch.autocast("cpu", enabled=False):
            features = features_from_history(torch.from_numpy(histories), torch.from_numpy(entries), validate=True)
            for market, item in source["markets"].items():
                raw = []
                for member in item["members"]:
                    model = build_model(item["selected"], **item["model_config"]).cpu().eval()
                    model.load_state_dict(member["checkpoint"]["state_dict"], strict=True)
                    raw.append(success_logit(model(features)).numpy())
                expected = calibrated_probability(np.mean(raw, axis=0), item["calibration"])
                observed = Predictor(stage, market).predict_proba(histories, entries)
                np.testing.assert_allclose(observed, expected, atol=1e-7, rtol=0)
                differences[market] = float(np.max(np.abs(observed - expected)))
        validation["maximum_absolute_difference"] = differences
        validation["cases_per_market"] = {market: 12 for market in MARKETS}
        _publish_manifest(stage, manifest, validation)
        for market in MARKETS:
            Predictor(stage, market)
        for relative, digest in source["source_files_sha256"].items():
            checked_file(run, relative, digest)
        if target.exists():
            raise ValueError("Destination appeared during export; refusing overwrite")
        stage.rename(target)
        return {"bundle": str(target), "manifest_sha256": sha256_file(target / "manifest.json"), "validation": validation}
    finally:
        if stage.exists():
            # Only our fresh, exact temporary sibling directory is discarded.
            unlinked_path(stage)
            if stage.parent != target.parent or not stage.name.startswith(".mark1-2-building-"):
                raise ValueError("Unexpected temporary export directory")
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/mark1_2_prototype")
    args = parser.parse_args(argv)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        print(json.dumps(export_bundle(args.training_run, args.output_dir), ensure_ascii=False))
    finally:
        torch.set_num_threads(previous)


if __name__ == "__main__":
    main()
