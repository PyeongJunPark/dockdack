"""Read-only mark_1 development-split audit; never evaluate 2025+ outcomes.

Uses historical train and the previously declared 2024 selection split only.
The full cache's packed bars are merely storage; only 30 bars immediately
before each permitted target are indexed. No test prediction file is opened.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dockdack.mark1_data import barrier_outcomes, features_from_history
from dockdack.mark1_metrics import binary_metrics, calibrated_probability
from dockdack.mark1_models import build_model


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs/mark1/experiment-20260916"
CUTOFF = np.datetime64("2024-12-31", "D").astype(np.int64)
N = 50_000


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in (0, .01, .1, .25, .5, .75, .9, .99, 1)}


def overview(labels, probabilities):
    metrics = binary_metrics(labels, probabilities)
    keys = ("count", "base_rate", "brier", "log_loss", "pr_auc", "roc_auc", "signal_count", "coverage", "precision")
    return {**{key: metrics[key] for key in keys}, "mean_probability": float(np.mean(probabilities)) if len(probabilities) else None}


def bucketed(values, boundaries, labels, probabilities):
    buckets = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (values >= lower) & (values < upper)
        buckets.append({"lower": None if np.isneginf(lower) else float(lower),
                        "upper": None if np.isposinf(upper) else float(upper),
                        **overview(labels[mask], probabilities[mask])})
    return buckets


@torch.inference_mode()
def infer(model, calibration, history, entries):
    parts = []
    for first in range(0, len(history), 2048):
        features = features_from_history(torch.from_numpy(history[first:first + 2048]),
                                         torch.as_tensor(entries[first:first + 2048], dtype=torch.float32))
        parts.append(model(features).numpy())
    return calibrated_probability(np.concatenate(parts), calibration)


def audit_market(market):
    cache_path, = (ROOT / "outputs/mark1/cache").glob(market + "-*.npz")
    payload = torch.load(EXPERIMENT / market / "mlp_no_price_aug/model.pt", map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    model = build_model(metadata["model_name"], **metadata["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    output = {"market": market, "frozen_model": "mlp_no_price_aug", "scope": "TRAIN 2010-2021 and SELECTION 2024 ONLY", "sample_cap": N}
    with np.load(cache_path, allow_pickle=False) as cache:
        output["cache_config"] = json.loads(str(cache["cache_config"].item()))
        dates = cache["target_dates"]
        all_starts = cache["starts"]
        all_ohlc = cache["target_ohlc"]
        bars = cache["bars"]
        data = {}
        for ordinal, split in enumerate(("train", "selection")):
            original_indices = cache["split_" + split]
            rng = np.random.default_rng(np.random.SeedSequence([314159, ordinal]))
            local_indices = np.sort(rng.choice(len(original_indices), min(N, len(original_indices)), replace=False))
            indices = original_indices[local_indices]
            assert np.all(dates[indices] <= CUTOFF), "Audit may not inspect 2025+ outcomes"
            if split == "selection":
                assert np.all(dates[indices] >= np.datetime64("2024-01-01", "D").astype(np.int64))
            history = bars[all_starts[indices, None] + np.arange(30)[None, :]]
            ohlc = all_ohlc[indices]
            del indices
            entries = ohlc[:, 0]
            returns = np.diff(np.log(history[:, :, 3].astype(np.float64)), axis=1)
            vol = returns.std(axis=1, ddof=1)
            query_gap = np.log(entries / history[:, -1, 3])
            labels = barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], entries)["success"]
            probabilities = infer(model, metadata["calibration"], history, entries)
            features = features_from_history(torch.from_numpy(history), torch.as_tensor(entries, dtype=torch.float32)).numpy()
            data[split] = {"history": history, "ohlc": ohlc, "labels": labels, "probabilities": probabilities,
                           "volatility": vol, "query_gap": query_gap}
            output[split] = {"actual_entry_metrics": overview(labels, probabilities), "volatility_quantiles": quantiles(vol),
                             "query_gap_quantiles": quantiles(query_gap), "historical_feature_std": features[:, :30, :7].std(axis=(0, 1)).tolist(),
                             "probability_quantiles": quantiles(probabilities),
                             "probability_buckets": bucketed(probabilities, [0, .2, .3, .4, .45, .5, .55, .6, 1.0000001], labels, probabilities)}
            factors = []
            for factor in (.99, .995, 1., 1.005, 1.01):
                query = entries * factor
                outcome = barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], query)
                factors.append({"factor": factor, "success_rate": float(outcome["success"].mean()),
                                "both_touch_rate": float(outcome["both_touch"].mean()),
                                "entry_outside_observed_daily_range_rate": float(((query < ohlc[:, 2]) | (query > ohlc[:, 1])).mean()),
                                "gap_quantiles": quantiles(np.log(query / history[:, -1, 3]))})
            output[split]["hypothetical_price_augmentation"] = factors
        vol_edges = [-np.inf, *np.quantile(data["train"]["volatility"], [.25, .5, .75]), np.inf]
        gap_edges = [-np.inf, -.02, -.01, -.005, 0, .005, .01, .02, np.inf]
        for split, part in data.items():
            output[split]["volatility_buckets_train_quartiles"] = bucketed(part["volatility"], vol_edges, part["labels"], part["probabilities"])
            output[split]["query_gap_buckets"] = bucketed(part["query_gap"], gap_edges, part["labels"], part["probabilities"])
            low, high = np.quantile(data["train"]["query_gap"], [.01, .99])
            outside = (part["query_gap"] < low) | (part["query_gap"] > high)
            output[split]["outside_train_gap_p01_p99"] = overview(part["labels"][outside], part["probabilities"][outside])
            output[split]["outside_train_gap_p01_p99"]["fraction"] = float(outside.mean())
        del bars, all_ohlc, all_starts, dates, data
    output["prior_architectures"] = []
    for variant in ("mlp", "lstm", "gru", "tcn", "transformer", "mlp_no_price_aug"):
        folder = EXPERIMENT / market / variant
        history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(folder / "model.pt", weights_only=True, map_location="cpu")
        md = checkpoint["metadata"]
        output["prior_architectures"].append({"variant": variant, "history": history,
                                               "calibration": md["calibration"], "selection": md["selection_metrics"]})
    return output


def main():
    torch.set_num_threads(4)
    output = {"method": "Uniform deterministic samples from prior train/2024 selection only; no test file opened or 2025+ label computed",
              "features_are_causal": "30 historical bars + target OPEN entry query; no target high/low/close/volume features",
              "markets": {}}
    for market in ("domestic", "us"):
        output["markets"][market] = audit_market(market)
        print(json.dumps({"market": market, "done": True}), flush=True)
    path = ROOT / "outputs/mark1/deep-audit.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
