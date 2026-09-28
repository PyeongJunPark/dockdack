"""Train five independent DEMO intraday-query models on verified daily proxies.

This offline command reads the sealed, cleaned daily cache. It NEVER contacts a
broker, reads an account, changes an operating database or starts trading.
Target-day OPEN/HIGH/LOW/CLOSE are used only to make the conservative label;
the models see the previous 30 completed daily bars and that OPEN. This is not
minute-bar training, and its query-price use later in the day is unverified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from dockdack.mark1_2_data import load_dataset
from dockdack.mark1_data import barrier_outcomes
from dockdack.mark1_intraday_models import ARCHITECTURES, VARIANTS, build_model, feature_matrix
from dockdack.research_artifacts import sha256_file


ROOT = Path(__file__).resolve().parents[1]
MARKETS = ("domestic", "us")
SEED = 20260928
EPOCHS = 8
BATCH_SIZE = 8192
RISK_FLAGS = {"research_only": True, "research_qualified": False,
              "deployment_allowed": False, "intraday_path_verified": False}
TARGET = "daily_open_to_whole_session_take_only_1pct_without_0.9pct_stop"
EFFECTIVE_LOOKBACK = {"mark1.13": 21, "mark1.14": 30, "mark1.15": 30,
                      "mark1.16": 21, "mark1.17": 30}


def _selected(dataset, name):
    indices = dataset.splits[name]
    starts = dataset.starts[indices]
    history = dataset.bars[starts[:, None] + np.arange(30)[None, :]]
    ohlc = dataset.target_ohlc[indices]
    if not np.isfinite(ohlc).all() or np.any(ohlc <= 0):
        raise ValueError("Nonfinite or nonpositive target OHLC")
    labels = barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], ohlc[:, 0])["success"]
    return history, ohlc[:, 0], labels.astype(np.float32)


def _fit_model(variant, train, *, seed=SEED, epochs=EPOCHS,
               feature_fn=feature_matrix, model_builder=build_model):
    history, entries, labels = train
    x = feature_fn(history, entries, variant)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-5))
    x = np.clip((x - mean) / scale, -8., 8.)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite normalized training features")
    torch.manual_seed(seed)
    model = model_builder(variant).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    features = torch.from_numpy(x)
    target = torch.from_numpy(labels)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(len(features), generator=generator)
        model.train()
        for first in range(0, len(order), BATCH_SIZE):
            positions = order[first:first + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            score = model(features[positions]).flatten()
            loss = F.binary_cross_entropy_with_logits(score, target[positions])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        print(json.dumps({"phase": "fit", "variant": variant,
                          "epoch": epoch + 1, "epochs": epochs}), flush=True)
    model.eval()
    with torch.inference_mode():
        # Functional integrity only; no historical-profit or test-set claim.
        probes = model(features[:min(len(features), 32)]).flatten()
    if not bool(torch.isfinite(probes).all()):
        raise ValueError("Trained model returned nonfinite scores")
    return model, mean, scale, {"train_examples": len(labels),
                                "train_positive": int(labels.sum()),
                                "epochs": epochs, "seed": seed}


def _write_bundle(workspace: Path, output: Path, *, variants=VARIANTS,
                  architectures=ARCHITECTURES, lookbacks=EFFECTIVE_LOOKBACK,
                  feature_fn=feature_matrix, model_builder=build_model,
                  feature_code="dockdack/mark1_intraday_models.py",
                  owner="dockdack.mark1_intraday"):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite model bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    stage = Path(tempfile.mkdtemp(prefix="mark1-intraday-training-", dir=output.parent))
    try:
        manifest = {
            "schema_version": 1, "owner": owner,
            "created_utc_date": "2026-09-28", "source_kind": "verified_clean_daily_cache",
            "target": TARGET, "history_bars": 30,
            "entry_training_basis": "target_session_observed_open_only",
            "query_inference_basis": "current_price_relative_to_last_completed_daily_close",
            "threshold": 0.5, "take_profit_pct": 1.0, "stop_loss_pct": 0.9,
            "risk_flags": RISK_FLAGS,
            "warnings": [
                "No minute/tick training data: this is not an intraday-path model.",
                "Whole-session daily OHLC cannot establish barrier order after an arbitrary intraday entry.",
                "The model score is not a demonstrated live win probability or profitability result.",
                "Connected DEMO source still requires separate user monitoring and order consent.",
            ],
            "runtime_feature_code_sha256": sha256_file(
                workspace / feature_code),
            "variants": {}, "markets": {},
        }
        for market in MARKETS:
            dataset, source, receipt = load_dataset(market, workspace)
            train = _selected(dataset, "train")
            market_info = {
                "market": market, "source_database_sha256": source["database_sha256"],
                "source_cache_sha256": receipt["cache_sha256"],
                "source_json_sha256": receipt["source_json_sha256"],
                "train_target_period": ["2010-01-01", "2021-12-31"],
                "sample_count": len(train[2]),
            }
            manifest["markets"][market] = market_info
            for offset, variant in enumerate(variants):
                model, mean, scale, diagnostic = _fit_model(
                    variant, train, seed=SEED + offset,
                    feature_fn=feature_fn, model_builder=model_builder)
                payload = {
                    "variant": variant, "market": market, "target": TARGET,
                    "feature_names": variants[variant],
                    "architecture": architectures[variant],
                    "mean": torch.from_numpy(mean.copy()),
                    "scale": torch.from_numpy(scale.copy()),
                    "state_dict": {name: value.detach().cpu().clone()
                                   for name, value in model.state_dict().items()},
                }
                filename = f"{market}-{variant.replace('.', '_')}.pt"
                path = stage / filename
                torch.save(payload, path)
                manifest["variants"].setdefault(variant, {
                    "model_id": variant.replace("mark1.", "mark1-") + "-prototype",
                    "feature_names": variants[variant],
                    "architecture": architectures[variant],
                    "effective_lookback": lookbacks[variant],
                    "models": {},
                })["models"][market] = {
                    "path": filename, "sha256": sha256_file(path), **diagnostic,
                }
            # The source cache is explicitly verified again after this market.
            if sha256_file(Path(receipt["cache_path"])) != receipt["cache_sha256"]:
                raise RuntimeError("Frozen daily cache changed during training")
            del dataset, train
        manifest_bytes = json.dumps(manifest, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")
        (stage / "manifest.json").write_bytes(manifest_bytes)
        (stage / "manifest.sha256").write_text(hashlib.sha256(manifest_bytes).hexdigest(),
                                               encoding="ascii")
        # No partial bundle becomes loadable while fitting; output is new-only.
        stage.rename(output)
        return manifest
    except Exception:
        # Retain a failed new stage for inspection; never remove existing data.
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "mark1_intraday")
    args = parser.parse_args(argv)
    result = _write_bundle(args.workspace.resolve(), args.output.resolve())
    print(json.dumps({"completed": True, "output": str(args.output.resolve()),
                      "variants": list(result["variants"]),
                      "markets": list(result["markets"]),
                      "intraday_path_verified": False,
                      "profitability_validated": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
