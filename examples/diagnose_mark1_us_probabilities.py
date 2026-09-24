"""Read-only US probability-flow audit; never fit, tune, promote, or order.

Existing frozen development arrays are preferred. Tiny native CPU checks only
verify label/logit interfaces. Reused 2025+ history is descriptive, never used
to select a new model or policy. Writes one new diagnostic JSON, no originals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dockdack.mark1_deep_data import class_targets
from dockdack.mark1_metrics import binary_metrics, calibrated_probability
from dockdack.mark1_selective_models import load_model, model_path, predict_raw
from dockdack.mark1_selective_policy import apply_policy
from examples.train_mark1 import file_hash
from examples.train_mark1_deep import read_sessions
from examples.train_mark1_selective import CODE_FILES, MODEL_NAMES, PROTOCOL, selective_splits


ROOT = Path(__file__).resolve().parents[1]
PARTS = ("probability_calibration", "policy_calibration", "audit")
QUANTILES = (0., .01, .1, .5, .9, .95, .99, .999, 1.)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1. / (1. + np.exp(-values[positive]))
    exp = np.exp(values[~positive])
    result[~positive] = exp / (1. + exp)
    return result


def distribution(values):
    values = np.asarray(values)
    return {"count": len(values), "nonfinite_count": int((~np.isfinite(values)).sum()),
            "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))},
            "mean": float(values.mean()), "standard_deviation": float(values.std()),
            "count_exactly_0_5": int((values == .5).sum()),
            "count_above_0_5": int((values > .5).sum()),
            "count_above_0_6": int((values > .6).sum())}


def top_quantiles(labels, probabilities, dates, symbols):
    result = []
    for fraction in (.1, .05, .01, .005, .001):
        threshold = float(np.quantile(probabilities, 1. - fraction))
        mask = probabilities >= threshold
        result.append({"requested_top_fraction": fraction, "inclusive_cutoff": threshold,
                       "count": int(mask.sum()), "actual_fraction": float(mask.mean()),
                       "precision": float(labels[mask].mean()), "mean_probability": float(probabilities[mask].mean()),
                       "signal_days": int(np.unique(dates[mask]).size),
                       "symbols": int(np.unique(symbols[mask]).size),
                       "ties": "all cutoff ties included; descriptive only, not a selected policy"})
    return result


def stage(raw_logits, calibration, classes, dates, symbols, *, probabilities=None, raw_origin="stored_native_logits"):
    labels = classes == 0
    raw_logits = np.asarray(raw_logits, dtype=np.float64)
    assert raw_logits.shape == labels.shape and np.isfinite(raw_logits).all()
    before = sigmoid(raw_logits)
    after = calibrated_probability(raw_logits, calibration) if probabilities is None else np.asarray(probabilities)
    assert np.isfinite(after).all()
    metrics_before, metrics_after = binary_metrics(labels, before), binary_metrics(labels, after)
    raw_mask, calibrated_mask = before > .5, after > .5
    return {
        "count": len(labels), "first": str(np.datetime64(int(dates.min()), "D")),
        "last": str(np.datetime64(int(dates.max()), "D")),
        "class_counts": {name: int((classes == index).sum()) for index, name in enumerate(("take_only", "stop_only", "both_touch", "neither"))},
        "success_prevalence": float(labels.mean()), "raw_origin": raw_origin,
        "raw_logits": distribution(raw_logits), "before_calibration": distribution(before),
        "after_calibration": distribution(after), "calibration": calibration,
        "raw_logit_needed_for_calibrated_0_5": -calibration["bias"] / calibration["slope"],
        "crossed_up_through_0_5": int((~raw_mask & calibrated_mask).sum()),
        "crossed_down_through_0_5": int((raw_mask & ~calibrated_mask).sum()),
        "metrics_before": metrics_before, "metrics_after": metrics_after,
        "top_quantiles_after": top_quantiles(labels, after, dates, symbols),
    }


def inverse_platt(probabilities, calibration):
    values = np.asarray(probabilities, dtype=np.float64)
    if np.any((values <= 0) | (values >= 1)):
        raise ValueError("Cannot reconstruct finite raw logits from saturated stored probabilities")
    return (np.log(values) - np.log1p(-values) - calibration["bias"]) / calibration["slope"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/mark1/us-diagnostic-20260916/probability-audit.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a fresh diagnostic output; overwriting is forbidden")
    selective = ROOT / "outputs/mark1/selective-20260916/us"
    deep = ROOT / "outputs/mark1/deep-20260916/us"
    prior = ROOT / "outputs/mark1/experiment-20260916/us/mlp_no_price_aug"
    summary, source = read(selective / "summary.json"), read(selective / "source.json")
    selected, deep_summary = summary["selected"], read(deep / "summary.json")
    source_key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
    cache = ROOT / f"outputs/mark1/cache/us-{source_key}.npz"
    protected = {ROOT / name for name in CODE_FILES}
    protected.update((Path(source["database_path"]), cache, ROOT / "models/mark1/us.pt",
                      ROOT / "models/mark1/domestic.pt", Path(__file__).resolve()))
    for folder in (selective, deep, prior):
        protected.update(path for path in folder.rglob("*") if path.is_file() and path.suffix in (".pt", ".cbm", ".txt", ".npz", ".json"))
    reused = ROOT / "outputs/mark1/selective-backtest-20260916/us/predictions.npz"
    protected.add(reused)
    before = {str(path.resolve()): file_hash(path) for path in sorted(protected)}
    with np.load(cache, allow_pickle=False) as archive:
        assert json.loads(str(archive["cache_config"].item())) == source
        dataset = SimpleNamespace(target_dates=archive["target_dates"], symbol_ids=archive["symbol_ids"], target_ohlc=archive["target_ohlc"])
        baseline_selection_indices = archive["split_selection"]
    sessions = read_sessions(Path(source["database_path"]))
    report = {"version": 1, "market": "us", "mode": "diagnosis_only_no_change",
              "selected": selected, "selection_ranking": summary["ranking"],
              "frozen_protocol": PROTOCOL, "source": source, "folds": {}, "native_interface_checks": [],
              "no_retraining": True, "no_threshold_selection": True, "deployment_allowed": False}
    for fold in PROTOCOL["folds"]:
        print("Audit probability flow", fold, flush=True)
        splits = selective_splits(dataset, sessions, fold)
        contract = read(selective / fold / "features/features.json")
        for name, indices in splits.items():
            assert hashlib.sha256(indices.tobytes()).hexdigest() == contract["index_sha256"][name]
        events = {part: {"classes": class_targets(dataset.target_ohlc[indices])[:, 0],
                         "dates": dataset.target_dates[indices], "symbols": dataset.symbol_ids[indices]}
                  for part, indices in splits.items()}
        fold_report = {"target_prevalence_by_split": {
            part: {"count": len(event["classes"]), "success_prevalence": float((event["classes"] == 0).mean()),
                   "class_counts": {name: int((event["classes"] == index).sum()) for index, name in enumerate(("take_only", "stop_only", "both_touch", "neither"))}}
            for part, event in events.items()}, "single_candidates": {}, "selected_members": {}, "ensemble": {}}
        for architecture in MODEL_NAMES:
            trial = selective / fold / f"{architecture}-42"
            result = read(trial / "result.json")
            with np.load(trial / "development_predictions.npz", allow_pickle=False) as archive:
                analyzed = {part: stage(archive[part + "_success"], result["calibration"]["success"], **events[part]) for part in PARTS}
            fold_report["single_candidates"][architecture] = {"best_iteration": result["model"]["best_iteration"],
                "trained_iterations": result["model"]["trained_iterations"], "stages": analyzed,
                "policy": result["policy_selection"]["chosen_policy"], "qualification": result["qualification"]}
        for seed in (42, 43, 44):
            trial = selective / fold / f"{selected}-{seed}"
            result = read(trial / "result.json")
            with np.load(trial / "development_predictions.npz", allow_pickle=False) as archive:
                analyzed = {part: stage(archive[part + "_success"], result["calibration"]["success"], **events[part]) for part in PARTS}
            fold_report["selected_members"][str(seed)] = {"best_iteration": result["model"]["best_iteration"], "stages": analyzed}
        ensemble = read(selective / fold / "ensemble.json")
        policy = ensemble["policy_selection"]["chosen_policy"]
        with np.load(selective / fold / "ensemble_development_predictions.npz", allow_pickle=False) as archive:
            for part in PARTS:
                logits = archive[part + "_success"]
                analyzed = stage(logits, ensemble["calibration"]["success"], **events[part])
                probabilities = calibrated_probability(logits, ensemble["calibration"]["success"])
                mask = apply_policy(probabilities, policy)
                analyzed["frozen_policy"] = policy
                analyzed["after_policy_count"] = int(mask.sum())
                analyzed["additional_filter_rejections_after_p_gt_0_5"] = int(((probabilities > .5) & ~mask).sum())
                fold_report["ensemble"][part] = analyzed
        # Verify the interface against native class probabilities and saved
        # logits on fixed CPU-only rows, independently of the wrapper math.
        for architecture in (selected, "cat_joint6"):
            trial = selective / fold / f"{architecture}-42"
            model = load_model(architecture, model_path(trial, architecture))
            features = np.load(selective / fold / "features/audit.npy", mmap_mode="r", allow_pickle=False)
            picked = np.linspace(0, len(features) - 1, 32, dtype=int)
            raw = predict_raw(model, architecture, features[picked])
            probabilities = model.predict_proba(features[picked], task_type="CPU", thread_count=2)
            success = probabilities[:, 0] if architecture == "cat_joint6" else probabilities[:, 1]
            max_probability_error = float(np.abs(sigmoid(raw["success_logits"]) - success).max())
            with np.load(trial / "development_predictions.npz", allow_pickle=False) as archive:
                max_saved_logit_error = float(np.abs(raw["success_logits"] - archive["audit_success"][picked]).max())
            stop_error = float(np.abs(sigmoid(raw["stop_logits"]) - probabilities[:, 1:3].sum(axis=1)).max()) if architecture == "cat_joint6" else None
            assert max_probability_error < 1e-12 and max_saved_logit_error < 1e-12
            assert stop_error is None or stop_error < 1e-12
            report["native_interface_checks"].append({"fold": fold, "architecture": architecture,
                "rows": 32, "classes": np.asarray(model.classes_).tolist(), "native_positive_class": 0 if architecture == "cat_joint6" else 1,
                "target_success_definition": "original class0 take_only; binarytraining label=1 iff original class0",
                "sigmoid_logit_vs_native_probability_maxerror": max_probability_error,
                "new_vs_saved_native_logit_maxerror": max_saved_logit_error,
                "joint_stop_sum_maxerror": stop_error, "device": "CPU", "passed": True})
            del model, features
        # Prior deep model: 2022 has only seed42; 2024 has frozen3seedensemble.
        old_seeds = (42,) if fold == "walk_2022" else (42, 43, 44)
        all_calibration, all_audit, calibration_indices, audit_indices = [], [], None, None
        for seed in old_seeds:
            path = deep / fold / f"{deep_summary['selected']}-{seed}/predictions.npz"
            with np.load(path, allow_pickle=False) as archive:
                if calibration_indices is not None:
                    assert np.array_equal(calibration_indices, archive["calibration_indices"])
                    assert np.array_equal(audit_indices, archive["selection_indices"])
                calibration_indices, audit_indices = archive["calibration_indices"], archive["selection_indices"]
                all_calibration.append(archive["calibration_logits"].astype(np.float64))
                all_audit.append(archive["selection_logits"].astype(np.float64))
        old_calibration = (deep_summary["ensemble_calibration"] if len(old_seeds) == 3 else deep_summary["results"][fold][deep_summary["selected"]]["calibration"])
        old_calibration_logits, old_audit_logits = np.mean(all_calibration, axis=0), np.mean(all_audit, axis=0)
        old_stages = {}
        for part in PARTS:
            original_indices, original_logits = ((audit_indices, old_audit_logits) if part == "audit" else (calibration_indices, old_calibration_logits))
            positions = np.searchsorted(original_indices, splits[part])
            assert np.array_equal(original_indices[positions], splits[part])
            old_stages[part] = stage(original_logits[positions], old_calibration, **events[part])
        fold_report["prior_deep"] = {"architecture": deep_summary["selected"], "seeds": list(old_seeds),
            "calibration_scope": "original full calendar calibration year; H1/H2 rows overlap original calibration, not independent audit",
            "stages": old_stages}
        report["folds"][fold] = fold_report
    # Baseline only saved 60k approved2024 calibrated predictions; no archived
    # H1/H2 raw arrays. Inverse Platt is algebraic reconstruction, not rerunning.
    baseline_result = read(prior / "result.json")
    baseline_calibration = baseline_result["calibration"]
    with np.load(prior / "selection_predictions.npz", allow_pickle=False) as archive:
        probabilities, saved_labels, dates = archive["probabilities"], archive["labels"], archive["dates"]
    classes = class_targets(dataset.target_ohlc[baseline_selection_indices])[:, 0]
    assert np.array_equal(saved_labels, classes == 0) and np.array_equal(dates, dataset.target_dates[baseline_selection_indices])
    report["prior_baseline_stored_2024_subset"] = stage(inverse_platt(probabilities, baseline_calibration), baseline_calibration,
        classes, dates, dataset.symbol_ids[baseline_selection_indices], probabilities=probabilities,
        raw_origin="algebraically reconstructed from stored calibrated probabilities; not an independent raw archive")
    report["prior_baseline_comparison_limit"] = "60k original random approved2024 subset, not whole selectivefoldaudit; H1/H2 raw predictions not archived; no baseline2022 audit claimed because2022 used for tuning"
    with np.load(reused, allow_pickle=False) as archive:
        indices, dates, symbols, labels = archive["sample_indices"], archive["dates"], archive["symbol_ids"], archive["labels"]
        classes = class_targets(dataset.target_ohlc[indices])[:, 0]
        assert np.array_equal(classes == 0, labels)
        comparisons = {}
        calibrations = {"baseline": baseline_calibration, "deep": deep_summary["ensemble_calibration"],
                        "selective": read(selective / "walk_2024/ensemble.json")["calibration"]["success"]}
        for model, calibration in calibrations.items():
            probabilities = archive[model + "_probabilities"]
            comparisons[model] = stage(inverse_platt(probabilities, calibration), calibration, classes, dates, symbols,
                probabilities=probabilities, raw_origin="algebraically reconstructed from frozen calibrated probabilities, descriptive reused2025+ only")
        report["reused_historical_2025_plus"] = {"status": "descriptive only; already-seen data, no model or threshold selection",
            "comparators": comparisons, "selective_unfiltered_probabilities_identical": bool(np.array_equal(archive["selective_probabilities"], archive["unfiltered_probabilities"])),
            "selective_unfiltered_masks_identical": bool(np.array_equal(archive["selective_selected"], archive["unfiltered_selected"]))}
    after = {path: file_hash(path) for path in before}
    assert before == after
    report.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), protected_files_unchanged=True,
                  protected_sha256_before=before, protected_sha256_after=after,
                  finding="SelectedUS raw model probabilities are already below0.5 in all fixeddevelopmentstages; monotonePlatt slightly raises high tail, frozenpolicy adds no further rejections. Otherjointcandidate yields few signals but fails evidence gate. Weak ranking rather than a hardcodedzero signal or label/logit reversal.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("DIAGNOSTIC_COMPLETE", args.output, "sha256", file_hash(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
