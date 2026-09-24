"""Compare Mark_1 networks using immutable cleaned daily DBs and a held-out test.

No broker, monitoring, automatic-order or GUI code is executed here. Augmented
entry prices are hypothetical training queries, not independent real trades.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from dockdack.mark1_data import (
    FEATURE_NAMES, TARGET, Mark1Dataset, barrier_outcomes, features_from_history, load_dataset,
)
from dockdack.mark1_metrics import binary_metrics, calibrated_probability, fit_calibration
from dockdack.mark1_models import MODEL_NAMES, build_model, parameter_count


FACTORS = (0.99, 0.995, 1.0, 1.005, 1.01)
MODEL_CONFIG = {"input_size": 9, "sequence_length": 31, "hidden_size": 64, "dropout": 0.15}
SPLITS = ("train", "tune", "calibration", "selection", "test")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_cache(database, market, folder, args):
    database = Path(database).resolve()
    wal = Path(str(database) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Clean source must be an immutable completed database without a nonempty WAL")
    source_hash = file_hash(database)
    config = {"version": 2, "database_path": str(database), "database_sha256": source_hash, "market": market, "start": "2010-01-01",
              "max_train_samples": args.max_train_samples, "max_eval_samples": args.max_eval_samples,
              "seed": args.seed, "purge_sessions": 30, "target": TARGET}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{market}-{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if json.loads(str(cached["cache_config"].item())) != config:
                raise ValueError("Cache contract mismatch")
            dataset = Mark1Dataset(
                bars=cached["bars"], starts=cached["starts"], target_dates=cached["target_dates"],
                symbol_ids=cached["symbol_ids"], target_ohlc=cached["target_ohlc"],
                splits={name: cached["split_" + name] for name in SPLITS},
                manifest=json.loads(str(cached["manifest"].item())))
        if (wal.exists() and wal.stat().st_size) or file_hash(database) != source_hash:
            raise RuntimeError("Clean source database changed while reading cache")
        print(f"Loaded immutable cache {path}", flush=True)
        return dataset, config
    dataset = load_dataset(database, market, max_train_samples=args.max_train_samples,
                           max_eval_samples=args.max_eval_samples, seed=args.seed, purge_sessions=30)
    if (wal.exists() and wal.stat().st_size) or file_hash(database) != source_hash:
        raise RuntimeError("Clean source database changed while loading")
    temporary = path.with_suffix(".building.npz")
    if temporary.exists():
        raise FileExistsError("Unfinished cache exists; select a fresh cache directory")
    np.savez(temporary, bars=dataset.bars, starts=dataset.starts, target_dates=dataset.target_dates,
             symbol_ids=dataset.symbol_ids, target_ohlc=dataset.target_ohlc,
             **{"split_" + k: v for k, v in dataset.splits.items()},
             manifest=json.dumps(jsonable(dataset.manifest)), cache_config=json.dumps(config))
    temporary.rename(path)
    return dataset, config


class BatchBank:
    """Small gathered windows over one shared GPU-resident historical bar array."""

    def __init__(self, dataset, device):
        self.dataset, self.device = dataset, torch.device(device)
        if any(not len(dataset.splits[name]) for name in SPLITS):
            raise ValueError("Every chronological split must contain approved samples")
        self.bars = torch.as_tensor(dataset.bars, device=self.device)
        self.offsets = torch.arange(30, device=self.device)
        self.parts = {}
        for name in SPLITS:
            indices = dataset.splits[name]
            ohlc = dataset.target_ohlc[indices]
            outcomes = barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], ohlc[:, 0])
            self.parts[name] = {"indices": indices, "ohlc": ohlc, "outcomes": outcomes,
                                "dates": dataset.target_dates[indices],
                                "starts": torch.as_tensor(dataset.starts[indices], device=self.device),
                                "entries": torch.as_tensor(ohlc[:, 0], dtype=torch.float32, device=self.device)}

    def features(self, part, local_indices, entries=None):
        starts = part["starts"][local_indices]
        history = self.bars[starts[:, None] + self.offsets[None, :]]
        return features_from_history(history, part["entries"][local_indices] if entries is None else entries)

    @torch.inference_mode()
    def logits(self, model, split, batch_size):
        model.eval()
        part = self.parts[split]
        pieces = []
        for start in range(0, len(part["indices"]), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(part["indices"])), device=self.device)
            pieces.append(model(self.features(part, indices)).float().cpu().numpy())
        result = np.concatenate(pieces)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite model outputs")
        return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))


def checkpoint_metadata(market, architecture, variant, calibration, dataset_config, args):
    return {"schema_version": 1, "strategy_version": "mark_1", "market": market,
            "model_name": architecture, "architecture": architecture, "variant": variant,
            "model_config": MODEL_CONFIG, "lookback": 30, "sequence_length": 31,
            "feature_names": list(FEATURE_NAMES), "target": TARGET,
            "buy_threshold": 0.5, "take_profit_pct": 1.0, "stop_loss_pct": 0.9,
            "calibration": calibration, "dataset": dataset_config, "seed": args.seed,
            "entry_context": "actual session open for evaluation; hypothetical entry-price augmentation in training",
            "intraday_path_verified": False,
            "limitations": ["Whole-session daily OHLC target, not the remaining intraday first-touch probability",
                            "Both barriers touched is a failure; ordering and actual fills are unknown",
                            "Synthetic prices are counterfactual and may never have traded",
                            "Current catalog survivorship and target observability selection remain",
                            "A >50% model estimate is not a promise of profit"],
            "created_at": datetime.now(timezone.utc).isoformat()}


def train_variant(bank, market, variant, output, dataset_config, args):
    architecture = "mlp" if variant == "mlp_no_price_aug" else variant
    factors = (1.0,) * len(FACTORS) if variant == "mlp_no_price_aug" else FACTORS
    output.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    model = build_model(architecture, **MODEL_CONFIG).to(bank.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.BCEWithLogitsLoss()
    train_part = bank.parts["train"]
    entries = train_part["ohlc"][:, :1] * np.asarray(factors)[None, :]
    train_outcomes = barrier_outcomes(train_part["ohlc"][:, 1:2], train_part["ohlc"][:, 2:3],
                                     train_part["ohlc"][:, 3:4], entries)
    labels = torch.as_tensor(np.asarray(train_outcomes["success"], dtype=np.float32).reshape(-1), device=bank.device)
    entries = torch.as_tensor(entries.astype(np.float32).reshape(-1), device=bank.device)
    n_train, best_loss, bad_epochs = len(labels), float("inf"), 0
    best_state, best_epoch, history = None, None, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        order = torch.randperm(n_train, device=bank.device)
        total_loss = torch.zeros((), device=bank.device)
        for start in range(0, n_train, args.batch_size):
            slots = order[start:start + args.batch_size]
            base = torch.div(slots, len(factors), rounding_mode="floor")
            features = bank.features(train_part, base, entries[slots])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels[slots])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.detach() * len(slots)
        train_loss = float((total_loss / n_train).item())
        tune_logits = bank.logits(model, "tune", args.batch_size)
        tune = binary_metrics(bank.parts["tune"]["outcomes"]["success"], sigmoid(tune_logits))
        if not math.isfinite(train_loss) or not math.isfinite(tune["log_loss"]):
            raise ValueError("Training produced nonfinite loss")
        row = {"epoch": epoch, "train_bce": train_loss, "tune_bce": tune["log_loss"],
               "tune_brier": tune["brier"], "seconds": round(time.monotonic() - epoch_start, 3)}
        history.append(row)
        if tune["log_loss"] < best_loss - 1e-6:
            best_loss, best_epoch, bad_epochs = tune["log_loss"], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "epoch": epoch, "variant": variant}, output / "training_best.pt")
        else:
            bad_epochs += 1
        save_json(output / "history.json", history)
        print(json.dumps({"market": market, "variant": variant, **row}, ensure_ascii=False), flush=True)
        if bad_epochs >= args.patience:
            break
    model.load_state_dict(best_state)
    calibration_logits = bank.logits(model, "calibration", args.batch_size)
    calibration = fit_calibration(calibration_logits, bank.parts["calibration"]["outcomes"]["success"])
    metadata = checkpoint_metadata(market, architecture, variant, calibration, dataset_config, args)
    selection_logits = bank.logits(model, "selection", args.batch_size)
    probabilities = calibrated_probability(selection_logits, calibration)
    selection = binary_metrics(bank.parts["selection"]["outcomes"]["success"], probabilities,
                               gross_returns=bank.parts["selection"]["outcomes"]["gross_return"], cost_bps=args.cost_bps)
    result = {"variant": variant, "architecture": architecture, "parameters": parameter_count(model),
              "augmentation_factors": list(factors), "unique_training_base_events": len(train_part["indices"]),
              "training_presentations_per_epoch": n_train, "best_epoch": best_epoch,
              "epochs_completed": len(history), "history": history, "calibration": calibration,
              "selection": selection, "training_seconds": round(time.monotonic() - started, 3)}
    metadata["selection_metrics"] = selection
    torch.save({"metadata": jsonable(metadata), "model_state_dict": best_state}, output / "model.pt")
    save_json(output / "result.json", result)
    np.savez(output / "selection_predictions.npz", probabilities=probabilities,
             labels=bank.parts["selection"]["outcomes"]["success"], dates=bank.parts["selection"]["dates"])
    del model, optimizer, labels, entries
    if bank.device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_market(market, args):
    folder = args.output_dir / market
    folder.mkdir(parents=True, exist_ok=False)
    dataset, data_config = dataset_cache(args.db_dir / f"{market}_daily_clean.sqlite3", market, args.cache_dir, args)
    save_json(folder / "dataset_manifest.json", dataset.manifest)
    save_json(folder / "cache_contract.json", data_config)
    bank = BatchBank(dataset, args.device)
    baseline_probability = float(np.mean(bank.parts["train"]["outcomes"]["success"]))
    variants = list(args.models)
    results = [train_variant(bank, market, variant, folder / variant, data_config, args) for variant in variants]
    # Selection is permanently recorded BEFORE the final test is scored.
    winner = min(results, key=lambda item: (item["selection"]["brier"], item["selection"]["log_loss"], item["variant"]))
    selected = {"market": market, "winner": winner["variant"], "criterion": "2024 selection Brier, then log loss",
                "selection_locked_before_test": True, "seed": args.seed,
                "models": [{"variant": item["variant"], "brier": item["selection"]["brier"],
                            "log_loss": item["selection"]["log_loss"]} for item in results]}
    save_json(folder / "selection_locked.json", selected)
    part = bank.parts["test"]
    for result in results:
        variant_folder = folder / result["variant"]
        payload = torch.load(variant_folder / "model.pt", weights_only=True, map_location="cpu")
        model = build_model(result["architecture"], **MODEL_CONFIG).to(bank.device)
        model.load_state_dict(payload["model_state_dict"])
        logits = bank.logits(model, "test", args.batch_size)
        probabilities = calibrated_probability(logits, result["calibration"])
        result["test"] = binary_metrics(part["outcomes"]["success"], probabilities,
                                       gross_returns=part["outcomes"]["gross_return"], dates=part["dates"],
                                       cost_bps=args.cost_bps)
        result["test"]["cost_sensitivity"] = {
            str(cost): binary_metrics(part["outcomes"]["success"], probabilities,
                                     gross_returns=part["outcomes"]["gross_return"], cost_bps=cost)["net_mean_return"]
            for cost in (0, 10, 20, 40)}
        np.savez(variant_folder / "test_predictions.npz", probabilities=probabilities,
                 labels=part["outcomes"]["success"], gross_returns=part["outcomes"]["gross_return"],
                 both_touch=part["outcomes"]["both_touch"], dates=part["dates"],
                 symbol_ids=dataset.symbol_ids[part["indices"]])
        save_json(variant_folder / "result.json", result)
        del model
    baseline = {"probability": baseline_probability,
                "selection": binary_metrics(bank.parts["selection"]["outcomes"]["success"],
                                            np.full(len(bank.parts["selection"]["indices"]), baseline_probability)),
                "test": binary_metrics(part["outcomes"]["success"], np.full(len(part["indices"]), baseline_probability),
                                       gross_returns=part["outcomes"]["gross_return"], cost_bps=args.cost_bps)}
    summary = {"market": market, "winner": selected["winner"], "selection": selected, "baseline": baseline,
               "results": results, "dataset": dataset.manifest,
               "test_entry": "actual target-session open; no synthetic evaluation prices",
               "return_metric": "idealized one-session stop-first OHLC proxy, neither-hit exits at close; not executed P&L",
               "warning": "No intraday timing/fills/spreads; current intraday probability remains unverified"}
    save_json(folder / "summary.json", summary)
    print(json.dumps({"market": market, "winner": selected["winner"], "completed": True}), flush=True)
    del bank, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/mark1/cache"))
    parser.add_argument("--market", choices=("domestic", "us", "all"), default="all")
    parser.add_argument("--models", nargs="+", choices=(*MODEL_NAMES, "mlp_no_price_aug"),
                        default=[*MODEL_NAMES, "mlp_no_price_aug"])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-train-samples", type=int, default=200000)
    parser.add_argument("--max-eval-samples", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-bps", type=float, default=20)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.65)
    args = parser.parse_args(argv)
    if (min(args.epochs, args.patience, args.batch_size, args.max_train_samples, args.max_eval_samples) < 1
            or len(set(args.models)) != len(args.models)
            or not math.isfinite(args.cost_bps) or not 0 <= args.cost_bps <= 10000
            or args.seed < 0
            or not 0 < args.gpu_memory_fraction <= 1):
        parser.error("Invalid experiment limits")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA explicitly requested but unavailable; refusing silent CPU fallback")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    runtime = {"torch": torch.__version__, "cuda": torch.version.cuda,
               "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"}
    save_json(args.output_dir / "run_config.json", {**vars(args), **runtime, "model_config": MODEL_CONFIG,
              "augmentation_factors": FACTORS, "selection_policy": "2024 Brier then log loss; never use test to pick",
              "splits": {"train": "2010-2021", "tune": "2022", "calibration": "2023", "selection": "2024", "test": "2025+"},
              "purge": "input start after previous split's last target date", "single_seed_screen": True})
    markets = ("domestic", "us") if args.market == "all" else (args.market,)
    for market in markets:
        run_market(market, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
