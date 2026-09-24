"""Preregistered deeper Mark_1 research; never arms trading or replaces models.

Two chronological development folds, four architectures, then three fixed-seed
ensemble of the architecture chosen by average relative Brier skill. The 2025+
period is already seen historical data and is NOT used in this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from dockdack.mark1_data import barrier_outcomes
from dockdack.mark1_deep_data import FOLDS, make_splits, class_targets
from dockdack.mark1_deep_models import (
    FEATURE_NAMES, CLASS_NAMES, MODEL_NAMES, TARGET, build_model,
    features_from_history, parameter_count, success_logit,
)
from dockdack.mark1_deep_validation import evaluate, qualification
from dockdack.mark1_metrics import fit_calibration, calibrated_probability
from examples.train_mark1 import dataset_cache, file_hash, save_json, seed_all


FACTORS = (1.0, .995, .9975, 1.0025, 1.005)
MODEL_CONFIG = dict(input_size=18, sequence_length=31, width=32, dropout=.2)
PROTOCOL = {
    "version": "mark1-deep-v1", "folds": FOLDS,
    "architectures": list(MODEL_NAMES), "model_config": MODEL_CONFIG,
    "screen_seed": 42, "ensemble_seeds": [42, 43, 44],
    "max_train": 750000, "max_tune": 100000,
    "max_epochs": 40, "minimum_epochs": 10, "patience": 8,
    "batch_size": 1024, "lr": .0003, "weight_decay": .001,
    "gradient_clip": 1., "warmup_epochs": 3,
    "loss": "actual-open binary BCE + 0.25 four-class CE + 0.10 counterfactual BCE",
    "class_weights": None, "synthetic_factors": FACTORS,
    "augmentation": "One uniform non-unit factor per base event per epoch; not independent real trades",
    "early_stop_metric": "actual-open tune binary BCE, uncalibrated",
    "architecture_selection": "mean over two folds of 1 - selection Brier / calibration-prevalence constant Brier; tie name",
    "ensemble": "mean three raw success logits, then independent calibration-year positive-slope Platt",
    "calibration": "unweighted monotone Platt using actual opens only",
    "threshold": .5, "threshold_rule": "strictly_greater", "take": .01, "stop": .009,
    "both_touch": "stop_first", "cost_bps": 20,
    "qualification": "Both single-seed development folds and final ensemble must pass fixed 200-signal/50-day/10-day-block CI precision>.5 and net proxy>0 gate; not live authorization",
    "test_policy": "2025+ excluded from training/selection; later comparison is REUSED historical evaluation, never fresh OOS",
    "deployment": "No automatic promotion; original checkpoints and trading settings unchanged",
}


def read_sessions(database):
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return np.asarray([int(np.datetime64(row[0], "D").astype(np.int64))
                           for row in connection.execute("SELECT session_date FROM sessions ORDER BY ordinal")], dtype=np.int64)
    finally:
        connection.close()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_json(temporary, value)
    temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class DeepBank:
    def __init__(self, dataset, splits, device):
        self.dataset, self.device = dataset, torch.device(device)
        self.bars = torch.tensor(dataset.bars, device=self.device)
        self.offsets = torch.arange(30, device=self.device)
        self.parts = {}
        for name, indices in splits.items():
            indices = np.asarray(indices)
            ohlc = dataset.target_ohlc[indices]
            outcomes = barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], ohlc[:, 0])
            self.parts[name] = {
                "indices": indices, "ohlc": ohlc, "dates": dataset.target_dates[indices],
                "outcomes": outcomes,
                "starts": torch.tensor(dataset.starts[indices], device=self.device),
                "entries": torch.tensor(ohlc[:, 0], dtype=torch.float32, device=self.device),
                "classes": torch.tensor(class_targets(ohlc, FACTORS), device=self.device),
            }
        self.factors = torch.tensor(FACTORS, dtype=torch.float32, device=self.device)

    def features(self, part, index, factor_indices=None):
        history = self.bars[part["starts"][index, None] + self.offsets[None, :]]
        entries = part["entries"][index]
        if factor_indices is not None:
            entries = entries * self.factors[factor_indices]
        return features_from_history(history, entries)

    @torch.inference_mode()
    def logits(self, model, split, batch_size=2048):
        model.eval()
        part, pieces = self.parts[split], []
        for first in range(0, len(part["indices"]), batch_size):
            index = torch.arange(first, min(first + batch_size, len(part["indices"])), device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logits = success_logit(model(self.features(part, index)).float())
            pieces.append(logits.cpu().numpy())
        result = np.concatenate(pieces)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite inference")
        return result


def binary_loss(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    return float((np.logaddexp(0., logits) - labels * logits).mean())


def result_metrics(part, probabilities):
    return evaluate(part["outcomes"]["success"], probabilities,
                    part["outcomes"]["gross_return"], part["dates"], cost_bps=20)


def train_trial(bank, architecture, seed, folder, context, *, epochs=40, batch_size=1024):
    folder.mkdir(parents=True, exist_ok=True)
    contract = dict(context, architecture=architecture, seed=seed, epochs=epochs,
                    batch_size=batch_size, model_config=MODEL_CONFIG, protocol=PROTOCOL)
    if (folder / "contract.json").exists():
        if json.loads((folder / "contract.json").read_text("utf-8")) != json.loads(json.dumps(contract)):
            raise ValueError("Existing trial contract mismatch")
    else:
        atomic_json(folder / "contract.json", contract)
    if (folder / "result.json").exists():
        if not all((folder / name).is_file() for name in ("model.pt", "predictions.npz", "history.json")):
            raise ValueError("Completed trial is missing required artifacts")
        print(f"Reuse completed trial {folder}", flush=True)
        return json.loads((folder / "result.json").read_text("utf-8"))
    seed_all(seed)
    model = build_model(architecture, **MODEL_CONFIG).to(bank.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL["lr"], weight_decay=PROTOCOL["weight_decay"])
    history, best, best_epoch, stale, epoch_start = [], float("inf"), 0, 0, 1
    if (folder / "resume.pt").exists():
        resume = torch.load(folder / "resume.pt", map_location=bank.device, weights_only=False)
        model.load_state_dict(resume["model"]); optimizer.load_state_dict(resume["optimizer"])
        history, best, best_epoch, stale = resume["history"], resume["best"], resume["best_epoch"], resume["stale"]
        epoch_start = len(history) + 1
        torch.set_rng_state(resume["cpu_rng"].cpu())
        if bank.device.type == "cuda":
            torch.cuda.set_rng_state(resume["cuda_rng"].cpu())
    train = bank.parts["train"]
    for epoch in range(epoch_start, epochs + 1):
        if epoch > PROTOCOL["minimum_epochs"] and stale >= PROTOCOL["patience"]:
            break
        started = time.monotonic()
        model.train()
        multiplier = (epoch / 3 if epoch <= 3 else
                      .1 + .9 * (1 + math.cos(math.pi * (epoch - 3) / max(1, epochs - 3))) / 2)
        for group in optimizer.param_groups:
            group["lr"] = PROTOCOL["lr"] * multiplier
        order = torch.randperm(len(train["indices"]), device=bank.device)
        loss_sum, seen = 0., 0
        for first in range(0, len(order), batch_size):
            index = order[first:first + batch_size]
            factors = torch.randint(1, len(FACTORS), (len(index),), device=bank.device)
            actual = bank.features(train, index)
            augmented = bank.features(train, index, factors)
            classes = train["classes"][index, 0]
            augmented_labels = (train["classes"][index, factors] == 0).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bank.device.type == "cuda"):
                output = model(torch.cat((actual, augmented)))
                actual_logits, augmented_logits = output[:len(index)].float(), output[len(index):].float()
                loss = (F.binary_cross_entropy_with_logits(success_logit(actual_logits), (classes == 0).float())
                        + .25 * F.cross_entropy(actual_logits, classes)
                        + .1 * F.binary_cross_entropy_with_logits(success_logit(augmented_logits), augmented_labels))
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            seen += len(index)
        tune_logits = bank.logits(model, "tune", batch_size * 2)
        tune_loss = binary_loss(tune_logits, bank.parts["tune"]["outcomes"]["success"])
        if tune_loss < best - 1e-5:
            best, best_epoch, stale = tune_loss, epoch, 0
            atomic_torch(folder / "best.pt", {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                                              "architecture": architecture, "model_config": MODEL_CONFIG,
                                              "feature_names": FEATURE_NAMES, "class_names": CLASS_NAMES,
                                              "target": TARGET, "context": context, "seed": seed,
                                              "best_epoch": best_epoch, "research_only": True})
        else:
            stale += 1
        row = dict(epoch=epoch, training_loss=loss_sum / seen, tune_binary_loss=tune_loss,
                   seconds=time.monotonic() - started, lr=optimizer.param_groups[0]["lr"],
                   best_epoch=best_epoch, train_base_samples=seen)
        history.append(row)
        atomic_json(folder / "history.json", history)
        atomic_torch(folder / "resume.pt", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
            "best": best, "best_epoch": best_epoch, "stale": stale,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if bank.device.type == "cuda" else None})
        print(json.dumps(dict(market=context["market"], fold=context["fold"], model=architecture, seed=seed, **row)), flush=True)
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    calibration_logits = bank.logits(model, "calibration", batch_size * 2)
    selection_logits = bank.logits(model, "selection", batch_size * 2)
    calibration = fit_calibration(calibration_logits, bank.parts["calibration"]["outcomes"]["success"])
    probabilities = calibrated_probability(selection_logits, calibration)
    metrics = result_metrics(bank.parts["selection"], probabilities)
    prevalence = float(bank.parts["calibration"]["outcomes"]["success"].mean())
    constant_brier = float(np.square(bank.parts["selection"]["outcomes"]["success"] - prevalence).mean())
    result = dict(architecture=architecture, seed=seed, parameters=parameter_count(model),
                  epochs=len(history), best_epoch=best_epoch, best_tune_loss=best,
                  seconds=sum(row["seconds"] for row in history), calibration=calibration,
                  selection=metrics, constant_brier=constant_brier,
                  brier_skill=1 - metrics["brier"] / constant_brier,
                  qualification=qualification(metrics))
    checkpoint.update(calibration=calibration, threshold=.5, take_profit_pct=1., stop_loss_pct=.9,
                      intraday_path_verified=False, protocol=PROTOCOL)
    atomic_torch(folder / "model.pt", checkpoint)
    np.savez(folder / "predictions.npz", calibration_logits=calibration_logits,
             selection_logits=selection_logits, probabilities=probabilities,
             calibration_indices=bank.parts["calibration"]["indices"], selection_indices=bank.parts["selection"]["indices"])
    atomic_json(folder / "result.json", result)
    print(json.dumps(dict(phase="trial_completed", market=context["market"], fold=context["fold"], **result)), flush=True)
    del model, optimizer
    if bank.device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_market(market, args):
    source = args.db_dir / f"{market}_daily_clean.sqlite3"
    # Reuse byte-verified full sample arrays; original split caps are not reused.
    dataset, source_contract = dataset_cache(source, market, args.cache_dir,
                                              SimpleNamespace(max_train_samples=200000, max_eval_samples=60000, seed=42))
    sessions = read_sessions(source)
    folder = args.output_dir / market
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "source.json", source_contract)
    results = {}
    for fold in FOLDS:
        splits = make_splits(dataset, sessions, fold, max_train=PROTOCOL["max_train"],
                             max_tune=PROTOCOL["max_tune"], seed=42)
        manifest = {name: {"samples": len(indices), "first": str(np.datetime64(int(dataset.target_dates[indices].min()), "D")),
                           "last": str(np.datetime64(int(dataset.target_dates[indices].max()), "D")),
                           "symbols": int(len(np.unique(dataset.symbol_ids[indices])))} for name, indices in splits.items()}
        fold_folder = folder / fold
        fold_folder.mkdir(parents=True, exist_ok=True)
        atomic_json(fold_folder / "split_manifest.json", manifest)
        bank = DeepBank(dataset, splits, args.device)
        results[fold] = {}
        context = dict(market=market, fold=fold, source=source_contract, splits=manifest)
        for architecture in MODEL_NAMES:
            results[fold][architecture] = train_trial(bank, architecture, 42, fold_folder / f"{architecture}-42", context)
        del bank
        torch.cuda.empty_cache()
    ranking = sorted((dict(architecture=name, score=float(np.mean([results[fold][name]["brier_skill"] for fold in FOLDS])))
                      for name in MODEL_NAMES), key=lambda row: (-row["score"], row["architecture"]))
    selected = ranking[0]["architecture"]
    lock = dict(market=market, selected=selected, ranking=ranking, selection_rule=PROTOCOL["architecture_selection"],
                note="Locked before seed ensemble and any reused 2025+ evaluation")
    locked_file = folder / "selection_locked.json"
    if locked_file.exists() and json.loads(locked_file.read_text("utf-8")) != lock:
        raise ValueError("Selection lock mismatch")
    atomic_json(locked_file, lock)
    final_fold = "walk_2024"
    splits = make_splits(dataset, sessions, final_fold, max_train=PROTOCOL["max_train"], max_tune=PROTOCOL["max_tune"], seed=42)
    bank = DeepBank(dataset, splits, args.device)
    manifest = json.loads((folder / final_fold / "split_manifest.json").read_text("utf-8"))
    context = dict(market=market, fold=final_fold, source=source_contract, splits=manifest)
    seed_results = [results[final_fold][selected]]
    for seed in (43, 44):
        seed_results.append(train_trial(bank, selected, seed, folder / final_fold / f"{selected}-{seed}", context))
    predictions = [np.load(folder / final_fold / f"{selected}-{seed}" / "predictions.npz", allow_pickle=False) for seed in (42, 43, 44)]
    for item in predictions:
        if not (np.array_equal(item["calibration_indices"], splits["calibration"])
                and np.array_equal(item["selection_indices"], splits["selection"])):
            raise ValueError("Ensemble prediction membership mismatch")
    cal_logits = np.mean([item["calibration_logits"].astype(np.float64) for item in predictions], axis=0)
    sel_logits = np.mean([item["selection_logits"].astype(np.float64) for item in predictions], axis=0)
    calibration = fit_calibration(cal_logits, bank.parts["calibration"]["outcomes"]["success"])
    probabilities = calibrated_probability(sel_logits, calibration)
    metrics = result_metrics(bank.parts["selection"], probabilities)
    np.savez(folder / "ensemble-selection.npz", probabilities=probabilities, indices=splits["selection"])
    for item in predictions:
        item.close()
    gate = qualification(metrics)
    fold_gates = {fold: results[fold][selected]["qualification"] for fold in FOLDS}
    summary = dict(market=market, selected=selected, ranking=ranking, results=results,
                   seed_results=seed_results, ensemble_calibration=calibration, ensemble_selection=metrics,
                   ensemble_qualification=gate, fold_qualification=fold_gates,
                   research_qualified=gate["qualified"] and all(item["qualified"] for item in fold_gates.values()),
                   deployment="NOT PROMOTED; no order settings changed", protocol=PROTOCOL)
    atomic_json(folder / "summary.json", summary)
    if file_hash(source) != source_contract["database_sha256"]:
        raise RuntimeError("Source DB changed during experiment")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=Path("C:/Users/user/Desktop/dockdack-data-collection/data/kiwoom_daily/clean-20260916-v1"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/mark1/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mark1/deep-20260916"))
    parser.add_argument("--markets", nargs="+", choices=("domestic", "us"), default=["domestic", "us"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable")
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This frozen protocol requires BF16 support")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_files = [Path(__file__), Path("dockdack/mark1_deep_data.py"), Path("dockdack/mark1_deep_models.py"),
                  Path("dockdack/mark1_deep_validation.py"), Path("dockdack/mark1_data.py"),
                  Path("dockdack/mark1_metrics.py"), Path("examples/train_mark1.py")]
    contract = dict(protocol=PROTOCOL, torch=torch.__version__, device=args.device,
                    gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                    code_sha256={str(path.name): file_hash(path) for path in code_files})
    path = args.output_dir / "protocol.json"
    if path.exists() and json.loads(path.read_text("utf-8")) != json.loads(json.dumps(contract)):
        raise ValueError("Frozen experiment configuration/code changed: use a fresh output directory")
    atomic_json(path, contract)
    results = {market: run_market(market, args) for market in args.markets}
    atomic_json(args.output_dir / "summary.json", results)


if __name__ == "__main__":
    main()
