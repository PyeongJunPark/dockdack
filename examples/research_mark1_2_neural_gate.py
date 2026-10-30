"""Research-only second-stage neural gate for the frozen MK1.2 ensembles.

The gate sees the selected three-member raw success logit, 30 completed bars,
and the observed target OPEN. It cannot see target HIGH/LOW/CLOSE. Within each
walk-forward fold, the first 70% of calibration-year sessions fit the gate and
the later 30% stop training and set a label-blind signal-frequency threshold.
The next selection year is evaluated once without refitting. Those selection
years helped select the original MK1.2 architecture, so these results remain
development evidence rather than an independent final test.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dockdack.mark1_2_data import load_dataset, read_sessions
from dockdack.mark1_deep_data import FOLDS, make_splits
from dockdack.mark1_deep_models import features_from_history
from dockdack.mark1_data import barrier_outcomes
from dockdack.mark1_metrics import calibrated_probability
from dockdack.research_artifacts import read_json, sha256_file, write_new_json


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "mark1" / "mark1-2-training-20260924-v1"
MARKETS = ("domestic", "us")
SEEDS = (42, 43, 44)
FEATURE_NAMES = (
    "selected_ensemble_raw_success_logit", "open_gap_over_history_volatility",
    "log_open_level_div10", "past_log_return_scale", "past_mean_return_over_scale",
    "oldest_close_relative_to_last_close", "mean_5_return", "mean_20_return",
    "std_20_return", "mean_5_volume_z", "mean_20_volume_z",
    "mean_5_range", "mean_20_range", "mean_5_candle_body",
    "mean_5_open_gap", "last_close_range_position",
)
COST_BPS = 20
TRAIN_CAP = 120_000
VALID_CAP = 60_000
SIGNAL_MULTIPLIER = 1.3
MIN_TARGET_SIGNALS = 30


class GateNet(nn.Module):
    def __init__(self, input_size: int = len(FEATURE_NAMES)) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 32), nn.GELU(),
                                    nn.Dropout(.1), nn.Linear(32, 16), nn.GELU(),
                                    nn.Linear(16, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def chronological_partition(dates: np.ndarray, fraction: float = .7) -> tuple[np.ndarray, np.ndarray]:
    """Split by distinct sessions even when original rows are symbol ordered."""
    dates = np.asarray(dates)
    if dates.ndim != 1 or dates.dtype.kind not in "iu" or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Integer dates and a proper chronological fraction are required")
    unique = np.unique(dates)
    if len(unique) < 2:
        raise ValueError("At least two calibration sessions are required")
    boundary = unique[min(len(unique) - 1, max(1, int(len(unique) * fraction)))]
    train, validation = np.flatnonzero(dates < boundary), np.flatnonzero(dates >= boundary)
    if not len(train) or not len(validation) or dates[train].max() >= dates[validation].min():
        raise ValueError("Chronological calibration split failed")
    return train, validation


def sample_positions(positions: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Label-blind uniform cap; preserves the source array's row order."""
    positions = np.asarray(positions)
    if positions.ndim != 1 or positions.dtype.kind not in "iu" or cap < 1 or seed < 0:
        raise ValueError("Valid positions, cap and seed required")
    if len(positions) <= cap:
        return positions.copy()
    picked = np.sort(np.random.default_rng(seed).choice(len(positions), cap, replace=False))
    return positions[picked].copy()


def causal_summary(history: torch.Tensor, opens: torch.Tensor,
                   raw_logits: torch.Tensor) -> torch.Tensor:
    """Small fixed summary of the existing causal feature contract."""
    if raw_logits.ndim != 1 or raw_logits.shape != opens.shape or history.shape != (len(opens), 30, 5):
        raise ValueError("History, OPEN and logit batch shapes differ")
    if not history.is_floating_point() or history.dtype != opens.dtype or history.dtype != raw_logits.dtype:
        raise ValueError("All causal inputs must use the same floating type")
    x = features_from_history(history, opens, validate=True)
    past = x[:, :30]
    values = torch.stack((
        raw_logits.clamp(-12, 12), x[:, 30, 0], x[:, 30, 11],
        past[:, -1, 13], past[:, -1, 14], past[:, 0, 3],
        past[:, -5:, 5].mean(1), past[:, -20:, 5].mean(1),
        past[:, -20:, 5].std(1, unbiased=False),
        past[:, -5:, 4].mean(1), past[:, -20:, 4].mean(1),
        past[:, -5:, 6].mean(1), past[:, -20:, 6].mean(1),
        past[:, -5:, 7].mean(1), past[:, -5:, 15].mean(1),
        past[:, -1, 8],
    ), dim=1)
    if values.shape[1] != len(FEATURE_NAMES) or not bool(torch.isfinite(values).all()):
        raise ValueError("Invalid neural gate features")
    return values


def gate_features(dataset, indices: np.ndarray, logits: np.ndarray, *,
                  batch_size: int = 2048, device: str = "cpu") -> np.ndarray:
    indices, logits = np.asarray(indices), np.asarray(logits)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != len(logits)
            or logits.ndim != 1 or not np.isfinite(logits).all() or batch_size < 1
            or np.any(indices >= len(dataset.starts))):
        raise ValueError("Aligned finite logits and dataset indices required")
    device = torch.device(device)
    result = np.empty((len(indices), len(FEATURE_NAMES)), dtype=np.float32)
    offsets = np.arange(30)
    for first in range(0, len(indices), batch_size):
        selected = indices[first:first + batch_size]
        past = np.asarray(dataset.bars[dataset.starts[selected, None] + offsets[None, :]], dtype=np.float32)
        # The target-day OPEN is available at entry. Target HIGH/LOW/CLOSE are
        # only read by the outcome evaluator, never by feature construction.
        opens = np.asarray(dataset.target_ohlc[selected, 0], dtype=np.float32)
        with torch.inference_mode():
            features = causal_summary(torch.as_tensor(past, device=device),
                                      torch.as_tensor(opens, device=device),
                                      torch.as_tensor(logits[first:first + len(selected)],
                                                      dtype=torch.float32, device=device))
        result[first:first + len(selected)] = features.cpu().numpy()
    return result


def signal_metrics(success: np.ndarray, gross: np.ndarray, dates: np.ndarray,
                   symbols: np.ndarray, mask: np.ndarray) -> dict:
    success, gross, dates, symbols, mask = map(np.asarray, (success, gross, dates, symbols, mask))
    if (any(row.ndim != 1 or len(row) != len(mask) for row in (success, gross, dates, symbols))
            or mask.ndim != 1 or mask.dtype.kind != "b" or not np.isfinite(gross).all()):
        raise ValueError("Aligned one-dimensional outcome rows and boolean mask required")
    count = int(mask.sum())
    return {"events": len(mask), "signals": count,
            "signal_rate": float(count / len(mask)) if len(mask) else None,
            "precision_success": float(success[mask].mean()) if count else None,
            "net_mean_20bp": float(gross[mask].mean() - COST_BPS / 10_000) if count else None,
            "signal_days": int(np.unique(dates[mask]).size),
            "symbols": int(np.unique(symbols[mask]).size)}


def frequency_threshold(scores: np.ndarray, baseline_count: int, *,
                        multiplier: float = SIGNAL_MULTIPLIER) -> tuple[float, int]:
    """Set frequency from validation scores only, without selection labels."""
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1)) or baseline_count < 0
            or not np.isfinite(multiplier) or multiplier <= 1):
        raise ValueError("Finite validation scores and a positive signal target required")
    target = min(len(scores), max(MIN_TARGET_SIGNALS, math.ceil(baseline_count * multiplier)))
    kth = float(np.partition(scores, -target)[-target])
    return float(np.nextafter(kth, -np.inf)), target


def fit_gate(train_x: np.ndarray, train_gross: np.ndarray,
             valid_x: np.ndarray, valid_gross: np.ndarray, *,
             seed: int = 42, max_epochs: int = 25) -> tuple[GateNet, np.ndarray, np.ndarray, dict]:
    """Weighted BCE estimates whether a 20bp-adjusted entry has positive value."""
    train_x, valid_x = np.asarray(train_x), np.asarray(valid_x)
    train_gross, valid_gross = np.asarray(train_gross), np.asarray(valid_gross)
    if (train_x.ndim != 2 or valid_x.ndim != 2 or train_x.shape[1] != len(FEATURE_NAMES)
            or valid_x.shape[1] != len(FEATURE_NAMES) or not len(train_x) or not len(valid_x)
            or train_gross.shape != (len(train_x),) or valid_gross.shape != (len(valid_x),)
            or not all(np.isfinite(x).all() for x in (train_x, valid_x, train_gross, valid_gross))
            or max_epochs < 1):
        raise ValueError("Finite training and validation features/returns required")
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(train_x.std(axis=0, dtype=np.float64), .001).astype(np.float32)
    x_train = torch.from_numpy(np.clip((train_x - mean) / std, -8, 8).astype(np.float32))
    x_valid = torch.from_numpy(np.clip((valid_x - mean) / std, -8, 8).astype(np.float32))
    y_train = torch.from_numpy(((train_gross - .002) > 0).astype(np.float32))
    y_valid = torch.from_numpy(((valid_gross - .002) > 0).astype(np.float32))
    weight_train = torch.from_numpy(np.clip(np.abs(train_gross - .002) / .01, .1, 3).astype(np.float32))
    weight_valid = torch.from_numpy(np.clip(np.abs(valid_gross - .002) / .01, .1, 3).astype(np.float32))
    torch.manual_seed(seed)
    model = GateNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for first in range(0, len(order), 2048):
            batch = order[first:first + 2048]
            prediction = model(x_train[batch])
            loss = (F.binary_cross_entropy_with_logits(prediction, y_train[batch], reduction="none")
                    * weight_train[batch]).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite gate loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            pieces = []
            for first in range(0, len(x_valid), 8192):
                prediction = model(x_valid[first:first + 8192])
                pieces.append(F.binary_cross_entropy_with_logits(
                    prediction, y_valid[first:first + 8192], reduction="none")
                    * weight_valid[first:first + 8192])
            val_loss = float(torch.cat(pieces).mean())
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch >= 5 and stale >= 4:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, mean, std, {"best_epoch": best_epoch, "epochs": epoch,
                              "validation_weighted_bce": best_loss,
                              "train_events": len(train_x), "validation_events": len(valid_x)}


def predict_gate(model: GateNet, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Finite gate features required")
    result = np.empty(len(x), dtype=np.float64)
    with torch.inference_mode():
        for first in range(0, len(x), 8192):
            values = torch.from_numpy(np.clip((x[first:first + 8192] - mean) / std, -8, 8).astype(np.float32))
            result[first:first + len(values)] = model(values).sigmoid().numpy()
    return result


def verified_predictions(folder: Path, selected: str, fold: str,
                         splits: dict, summary: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    calibration, selection = [], []
    checked = {}
    for seed in SEEDS:
        trial = folder / f"{selected}-{seed}"
        result_path, prediction_path = trial / "result.json", trial / "predictions.npz"
        row = read_json(result_path)
        context = row.get("contract", {}).get("context", {})
        if (row.get("completed") is not True or row.get("architecture") != selected
                or row.get("seed") != seed or context.get("market") != summary["market"]
                or context.get("fold") != fold or context.get("source") != summary["source"]
                or context.get("data_receipt") != summary["data_receipt"]
                or row.get("artifact_sha256", {}).get("predictions.npz") != sha256_file(prediction_path)):
            raise ValueError("Selected model prediction provenance mismatch")
        checked[str(result_path)] = sha256_file(result_path)
        checked[str(prediction_path)] = sha256_file(prediction_path)
        with np.load(prediction_path, allow_pickle=False) as saved:
            for name in ("calibration", "selection"):
                if not np.array_equal(saved[f"{name}_indices"], splits[name]):
                    raise ValueError("Cached predictions and frozen folds are misaligned")
            calibration.append(saved["calibration_logits"].copy())
            selection.append(saved["selection_logits"].copy())
    cal_logits = np.mean(calibration, axis=0)
    sel_logits = np.mean(selection, axis=0)
    if not np.isfinite(cal_logits).all() or not np.isfinite(sel_logits).all():
        raise ValueError("Nonfinite source model logits")
    ensemble_predictions = folder / "ensemble_predictions.npz"
    ensemble_result = folder / "ensemble.json"
    with np.load(ensemble_predictions, allow_pickle=False) as saved:
        if (not np.array_equal(saved["sample_indices"], splits["selection"])
                or not np.array_equal(saved["raw_logits"], sel_logits)):
            raise ValueError("Selected ensemble differs from verified seed logits")
    checked[str(ensemble_predictions)] = sha256_file(ensemble_predictions)
    checked[str(ensemble_result)] = sha256_file(ensemble_result)
    return cal_logits, sel_logits, checked


def run_market(market: str, output_dir: Path, *, feature_device: str = "cpu") -> dict:
    dataset, source, receipt = load_dataset(market, ROOT)
    summary_path = RUN / market / "summary.json"
    summary = read_json(summary_path)
    if (summary.get("completed") is not True or summary.get("pilot") is not False
            or summary.get("market") != market or summary.get("research_qualified") is not False
            or summary.get("deployment_allowed") is not False
            or summary.get("source") != source or summary.get("data_receipt") != receipt):
        raise ValueError("Completed research-only MK1.2 training does not match frozen data")
    selected = summary["selected"]
    sessions = read_sessions(receipt["source"]["physical_path"])
    output = {"market": market, "selected_architecture": selected, "folds": {},
              "source_sha256": sha256_file(summary_path),
              "data_receipt": receipt, "research_only": True, "deployment_allowed": False}
    for fold in FOLDS:
        indices = make_splits(dataset, sessions, fold,
                              max_train=summary["protocol"]["max_train"],
                              max_tune=summary["protocol"]["max_tune"], seed=42)
        source_folder = RUN / market / fold
        cal_logits, sel_logits, checked = verified_predictions(source_folder, selected, fold, indices, summary)
        ensemble = read_json(source_folder / "ensemble.json")
        cal_idx, sel_idx = indices["calibration"], indices["selection"]
        cal_dates = dataset.target_dates[cal_idx]
        train_pos, valid_pos = chronological_partition(cal_dates)
        train_pos = sample_positions(train_pos, TRAIN_CAP, 42)
        valid_pos = sample_positions(valid_pos, VALID_CAP, 43)
        cal_open = dataset.target_ohlc[cal_idx, 0]
        cal_target = dataset.target_ohlc[cal_idx]
        cal_outcomes = barrier_outcomes(cal_target[:, 1], cal_target[:, 2], cal_target[:, 3], cal_open)
        train_x = gate_features(dataset, cal_idx[train_pos], cal_logits[train_pos], device=feature_device)
        valid_x = gate_features(dataset, cal_idx[valid_pos], cal_logits[valid_pos], device=feature_device)
        model, mean, std, training = fit_gate(train_x, cal_outcomes["gross_return"][train_pos],
                                              valid_x, cal_outcomes["gross_return"][valid_pos])
        valid_scores = predict_gate(model, valid_x, mean, std)
        baseline_val = calibrated_probability(cal_logits[valid_pos], ensemble["calibration"]) > .5
        threshold, target_count = frequency_threshold(valid_scores, int(baseline_val.sum()))
        valid_gate = valid_scores > threshold
        valid_baseline_metrics = signal_metrics(cal_outcomes["success"][valid_pos],
            cal_outcomes["gross_return"][valid_pos], cal_dates[valid_pos],
            dataset.symbol_ids[cal_idx[valid_pos]], baseline_val)
        valid_gate_metrics = signal_metrics(cal_outcomes["success"][valid_pos],
            cal_outcomes["gross_return"][valid_pos], cal_dates[valid_pos],
            dataset.symbol_ids[cal_idx[valid_pos]], valid_gate)
        sel_x = gate_features(dataset, sel_idx, sel_logits, device=feature_device)
        sel_scores = predict_gate(model, sel_x, mean, std)
        baseline_prob = calibrated_probability(sel_logits, ensemble["calibration"])
        sel_ohlc = dataset.target_ohlc[sel_idx]
        sel_outcomes = barrier_outcomes(sel_ohlc[:, 1], sel_ohlc[:, 2], sel_ohlc[:, 3], sel_ohlc[:, 0])
        baseline = signal_metrics(sel_outcomes["success"], sel_outcomes["gross_return"],
                                  dataset.target_dates[sel_idx], dataset.symbol_ids[sel_idx], baseline_prob > .5)
        gate = signal_metrics(sel_outcomes["success"], sel_outcomes["gross_return"],
                              dataset.target_dates[sel_idx], dataset.symbol_ids[sel_idx], sel_scores > threshold)
        published_signals = ensemble["selection"]["signal_count"]
        if baseline["signals"] != published_signals:
            raise ValueError("Recomputed baseline differs from frozen selection metrics")
        target = output_dir / market / fold
        target.mkdir(parents=True, exist_ok=False)
        checkpoint = {"state_dict": model.state_dict(), "normalization_mean": mean,
                      "normalization_std": std, "threshold": threshold,
                      "feature_names": FEATURE_NAMES, "source_prediction_sha256": checked,
                      "research_only": True, "deployment_allowed": False}
        torch.save(checkpoint, target / "gate.pt")
        output["folds"][fold] = {
            "calibration_train_last": str(np.datetime64(int(cal_dates[train_pos].max()), "D")),
            "calibration_validation_first": str(np.datetime64(int(cal_dates[valid_pos].min()), "D")),
            "training": training, "threshold": threshold,
            "validation_frequency_target": target_count,
            "validation_baseline": valid_baseline_metrics, "validation_gate": valid_gate_metrics,
            "selection_baseline": baseline, "selection_gate": gate,
            "selection_net_positive": gate["net_mean_20bp"] is not None and gate["net_mean_20bp"] > 0,
            "checkpoint_sha256": sha256_file(target / "gate.pt"), "source_prediction_sha256": checked,
        }
        for path, digest in checked.items():
            if sha256_file(Path(path)) != digest:
                raise RuntimeError("Source prediction changed during gate research")
        print(json.dumps({"market": market, "fold": fold, "baseline": baseline,
                          "gate": gate}, ensure_ascii=False, allow_nan=False), flush=True)
    if sha256_file(summary_path) != output["source_sha256"]:
        raise RuntimeError("Source model summary changed during gate research")
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--markets", nargs="+", choices=MARKETS, default=list(MARKETS))
    args = parser.parse_args(argv)
    destination = args.output_dir.resolve()
    allowed = (ROOT / "outputs" / "mark1").resolve()
    if not destination.is_relative_to(allowed) or destination == allowed or destination.exists():
        parser.error("Choose a new dedicated child directory under outputs/mark1")
    if args.feature_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(8)
    destination.mkdir(parents=True, exist_ok=False)
    results = {"experiment": "mark1-2-cost-aware-neural-gate-v1",
               "source_training": str(RUN), "cost_bps": COST_BPS,
               "frequency_rule": "validation signal count >= 1.3 times frozen p>0.5 count, minimum 30; threshold chosen without outcome labels",
               "outcome": "+1% before -0.9%; both touch stop first; same-day CLOSE timeout; 20bp hypothetical roundtrip cost",
               "validation_years": [2021, 2023], "selection_years": [2022, 2024],
               "selection_caveat": "original architecture already chosen using selection years; not independent final evidence",
               "training_cap": TRAIN_CAP, "validation_cap": VALID_CAP,
               "feature_names": FEATURE_NAMES, "research_only": True,
               "deployment_allowed": False, "markets": {}}
    for market in args.markets:
        results["markets"][market] = run_market(market, destination, feature_device=args.feature_device)
    write_new_json(destination / "result.json", results)


if __name__ == "__main__":
    main()
