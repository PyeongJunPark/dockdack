"""Read-only diagnostic of saved US training/selection and feature health.

Does not fit models, select a new policy, query a broker, or change source data.
Writes only the requested diagnostic JSON. Sparse metric logs are not treated
as one record per iteration; only the full stopping-metric curve is indexed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_health(path, names):
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != len(names) or len(values) == 0:
        raise ValueError("Unexpected feature schema")
    lower = np.full(len(names), np.inf)
    upper = np.full(len(names), -np.inf)
    clips = np.zeros(len(names), dtype=np.int64)
    nonfinite = 0
    for start in range(0, len(values), 65536):
        block = values[start:start + 65536]
        nonfinite += int((~np.isfinite(block)).sum())
        lower = np.minimum(lower, block.min(axis=0))
        upper = np.maximum(upper, block.max(axis=0))
        clips += (np.abs(block) >= 20).sum(axis=0)
    chosen = ["query_log_gap", "query_gap_vol_scaled", "historical_return_std",
              "w30_past_take_only_rate", "w30_past_both_touch_rate"]
    return {
        "shape": list(values.shape), "nonfinite_values": nonfinite,
        "constant_features": [name for name, lo, hi in zip(names, lower, upper) if lo == hi],
        "clipped_value_count": int(clips.sum()),
        "clipped_value_fraction": float(clips.sum() / values.size),
        "most_clipped_features": [
            {"name": names[i], "count": int(clips[i]), "fraction": float(clips[i] / len(values))}
            for i in np.argsort(-clips)[:8] if clips[i]
        ],
        "selected_feature_quantiles": {
            name: np.quantile(values[:, names.index(name)], [0, .01, .5, .99, 1]).tolist()
            for name in chosen
        },
    }


def audit(run, output):
    run, output = Path(run), Path(output)
    if output.suffix != ".json" or output.resolve().is_relative_to(run.resolve()):
        raise ValueError("Write a diagnostic JSON outside the frozen training run")
    summary_path, protocol_path = run / "us/summary.json", run / "protocol.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not summary["completed"]:
        raise ValueError("Requires complete frozen research artifacts")
    inputs = [summary_path, protocol_path]
    inputs.extend(sorted((run / "us").glob("walk_*/*/metadata.json")))
    inputs.extend(run / "us" / fold / "features" / f"{part}.npy"
                  for fold in ("walk_2022", "walk_2024") for part in ("train", "audit"))
    before = {str(path): sha(path) for path in inputs}
    trials = []
    for path in sorted((run / "us").glob("walk_*/*/metadata.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        params = data["params"]
        history = data["eval_history"]
        validation = history["validation"]
        metric = params.get("eval_metric", params.get("metric"))
        curve = np.asarray(validation[metric], dtype=np.float64)
        minimize = metric == "MultiClass"
        best = int(np.argmin(curve) if minimize else np.argmax(curve))
        dense = len(curve) == data["trained_iterations"]
        trials.append({
            "fold": path.parent.parent.name, "trial": path.parent.name,
            "train_samples": data["request"]["train_shape"][0],
            "best_iteration": data["best_iteration"],
            "trained_iterations": data["trained_iterations"],
            "stopping_metric": metric, "metric_records": len(curve),
            "best_record": best + 1, "curve_dense": dense,
            "first_metric": float(curve[0]), "best_metric": float(curve[best]),
            "last_metric": float(curve[-1]),
            "metadata_best_matches_dense_curve":
                bool(best + 1 == data["best_iteration"]) if dense else None,
            "curve_lengths": {part: {key: len(value) for key, value in metrics.items()}
                              for part, metrics in history.items()},
            "no_class_reweighting": not any(key in params for key in
                ("class_weights", "auto_class_weights", "scale_pos_weight", "is_unbalance")),
        })
    health = {}
    names = protocol["protocol"]["feature_names"]
    for fold in ("walk_2022", "walk_2024"):
        health[fold] = {}
        for part in ("train", "audit"):
            print(f"Feature scan {fold}/{part}", flush=True)
            health[fold][part] = feature_health(run / "us" / fold / "features" / f"{part}.npy", names)
    after = {str(path): sha(path) for path in inputs}
    if before != after:
        raise RuntimeError("A protected diagnostic input changed")
    result = {
        "scope": "US frozen pre2025 training and feature diagnosis only; no retraining or new policy",
        "selected": summary["selected"], "research_qualified": summary["research_qualified"],
        "ranking": summary["ranking"],
        "selection_fell_back_to_brier": not any(row["both_audits_have_evidence"] for row in summary["ranking"]),
        "trials": trials, "feature_health": health,
        "inputs_unchanged": before == after, "input_sha256": before,
        "limitations": [
            "Low ranking signal or early stopping does not prove data corruption.",
            "A fallback Brier winner is a diagnostic artifact, not a qualified BUY model.",
            "Sparse auxiliary metric histories are not indexed as consecutive iterations.",
            "This scan does not independently validate broker source prices or future profitability.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "selected": result["selected"],
                      "brier_fallback": result["selection_fell_back_to_brier"],
                      "trials": len(trials)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/mark1/selective-20260916"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mark1/us-diagnostic-20260916/training-audit.json"))
    args = parser.parse_args()
    audit(args.run, args.output)
