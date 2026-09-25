"""Independent GPU price-augmented neural comparison; never imports a trading UI.

Frozen MK1/MK1.1 data and models are read-only. All new artifacts live in an
explicit new run. 2025+ history is excluded from fitting and model selection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dockdack.mark1_2_data import BuildAugmentedBank, load_dataset, read_sessions
from dockdack.mark1_2_models import MODEL_NAMES, build_model, parameter_count
from dockdack.mark1_deep_data import FOLDS, make_splits, split_manifest
from dockdack.mark1_deep_models import FEATURE_NAMES, CLASS_NAMES, TARGET, success_logit
from dockdack.mark1_metrics import fit_calibration, calibrated_probability, binary_metrics
from dockdack.mark1_selective_policy import evaluate_signals, qualification
from dockdack.research_artifacts import sha256_file, read_json

ROOT = Path(__file__).resolve().parents[1]
FACTORS = (1., .99, .995, 1.005, 1.01)
CONFIG = dict(input_size=18, sequence_length=31, width=48, dropout=.2)
SEEDS = (42, 43, 44)
CODE_FILES = (
    "examples/train_mark1_2.py", "dockdack/mark1_2_data.py", "dockdack/mark1_2_models.py",
    "dockdack/mark1_deep_models.py", "dockdack/mark1_deep_data.py", "dockdack/mark1_data.py",
    "dockdack/mark1_metrics.py", "dockdack/mark1_selective_policy.py",
    "dockdack/research_artifacts.py", "dockdack/research_compat.py", "dockdack/research_arrays.py",
)
PROTOCOL = {
    "version": "mark1-2-neural-price-augmentation-v1", "target": TARGET,
    "take_profit_pct": 1., "stop_loss_pct": .9, "threshold": .5,
    "both_touch": "stop_first_failure", "folds": FOLDS,
    "architectures": list(MODEL_NAMES), "model_config": CONFIG,
    "screen_seed": 42, "ensemble_seeds": SEEDS,
    "max_train": 1_500_000, "max_tune": 150_000,
    "max_epochs": 60, "minimum_epochs": 12, "patience": 12,
    "batch_size": 1024, "learning_rate": .0003, "weight_decay": .001,
    "gradient_clip": 1., "warmup_epochs": 3,
    "price_factors": FACTORS, "augmentation": "actual open plus one uniform nonunit factor per event per epoch",
    "sampling": "dedicated per-epoch generator independent of architecture/dropout RNG",
    "loss": "actual BCE + .25 actual four-class CE + .5 augmented BCE + .125 augmented four-class CE",
    "loss_actual_to_augmented_weight": "2:1", "class_weights": None,
    "tuning": "observed-open FP32 tune BCE only; no calibration or model selection on augmented siblings",
    "calibration": "unweighted monotone Platt on independent-year observed opens, CPU FP32 inference",
    "selection": "qualified both folds first, supported both folds second: worst precision CI lower, mean net CI lower, mean Brier skill; otherwise Brier diagnostic fallback",
    "ensemble": "mean three raw success logits then refit independent CPU Platt; both folds checked",
    "qualification": "existing selective gate: >=50 signals/20 dates/10 symbols, precision>=.65, CI lower>.579, 20bp net CI lower>0",
    "evaluation": "2022/2024 development; 2025+ reserved for later REUSED historical comparison, not pristine OOS",
    "quarantine_us": ["FCEL", "BNED", "BBSI", "SONY"],
    "training_precision": "BF16 autocast; recurrent internals FP32 with Windows CUDA cuDNN RNN disabled",
    "inference_precision": "CPU FP32 for calibration/selection/export; CUDA FP32 for epoch tuning",
    "deployment_allowed": False, "intraday_path_verified": False,
}


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Linked artifacts are forbidden")
    temporary.write_text(json.dumps(canonical(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Linked checkpoints are forbidden")
    torch.save(value, temporary)
    temporary.replace(path)
    atomic_json(path.with_suffix(".sha256.json"), {"sha256": sha256_file(path)})


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    seal = read_json(path.with_suffix(".sha256.json"))
    if path.is_symlink() or sha256_file(path) != seal.get("sha256"):
        raise ValueError("Checkpoint checksum mismatch; preserve artifacts and inspect")
    return torch.load(path, map_location=device, weights_only=True)


def save_resume(folder, value):
    """Two generations: publish the pointer last; keep the preceding valid slot."""
    folder = Path(folder)
    slot = len(value["history"]) % 2
    path = folder / f"resume-{slot}.pt"
    save_checkpoint(path, value)
    atomic_json(folder / "resume.json", {"slot": slot, "sha256": sha256_file(path)})


def load_resume(folder, device):
    folder = Path(folder)
    pointer = read_json(folder / "resume.json")
    if type(pointer.get("slot")) is not int or pointer["slot"] not in (0, 1):
        raise ValueError("Invalid resume generation pointer")
    path = folder / f"resume-{pointer['slot']}.pt"
    if sha256_file(path) != pointer.get("sha256"):
        raise ValueError("Published resume generation changed")
    return load_checkpoint(path, device)


@contextmanager
def run_lock(folder):
    """Local OS lock, released on process death; never a broker/account lock."""
    path = Path(folder) / "run.lock"
    if path.is_symlink():
        raise ValueError("Linked run lock")
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def verify_completed_trial(folder, result, contract):
    required = {"model.pt", "predictions.npz", "history.json"}
    hashes = result.get("artifact_sha256")
    if (result.get("contract") != contract or result.get("completed") is not True
            or not isinstance(hashes, dict) or set(hashes) != required):
        raise ValueError("Completed trial contract/artifact set mismatch")
    for name, digest in hashes.items():
        path = Path(folder) / name
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or path.is_symlink() or not path.is_file() or sha256_file(path) != digest):
            raise ValueError("Completed trial artifact changed")


def lock_json(path, value):
    value = canonical(value)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Run contract changed: {path}; use a new output directory")
    else:
        atomic_json(path, value)


def binary_loss(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    return float(np.mean(np.logaddexp(0., logits) - np.asarray(labels) * logits))


def augmented_loss(actual_logits, augmented_logits, classes, augmented_classes):
    """Equal base-event weighting; synthetic events are not independent trades."""
    return (F.binary_cross_entropy_with_logits(success_logit(actual_logits.float()), (classes == 0).float())
            + .25 * F.cross_entropy(actual_logits.float(), classes)
            + .5 * F.binary_cross_entropy_with_logits(success_logit(augmented_logits.float()), (augmented_classes == 0).float())
            + .125 * F.cross_entropy(augmented_logits.float(), augmented_classes))


def epoch_generator(seed, epoch, device):
    # Model initialization and different dropout implementations cannot perturb
    # the base-event ordering or price sampling across architecture comparisons.
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1_000_003 * epoch)
    return generator


def score_part(part, probabilities):
    metrics = evaluate_signals(part["outcomes"]["success"], probabilities,
        part["outcomes"]["gross_return"], part["dates"], part["symbols"], probabilities > .5, cost_bps=20)
    return metrics, qualification(metrics)


def training_state(model, optimizer, history, best, best_epoch, stale, best_state, contract):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
        best=best, best_epoch=best_epoch, stale=stale, best_state=best_state, contract=contract,
        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())


def train_trial(bank, architecture, seed, folder, context, *, epochs=60, pilot=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    contract = canonical(dict(context=context, architecture=architecture, seed=seed, epochs=epochs,
                              pilot=pilot, protocol=PROTOCOL))
    lock_json(folder / "contract.json", contract)
    if (folder / "result.json").exists():
        result = read_json(folder / "result.json")
        verify_completed_trial(folder, result, contract)
        return result
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(architecture, **CONFIG).to(bank.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
    history, best, best_epoch, stale, best_state = [], float("inf"), 0, 0, None
    if (folder / "resume.json").exists():
        resume = load_resume(folder, bank.device)
        if resume["contract"] != contract:
            raise ValueError("Resume contract mismatch")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        history, best = resume["history"], resume["best"]
        best_epoch, stale = resume["best_epoch"], resume["stale"]
        best_state = {key: value.cpu() for key, value in resume["best_state"].items()}
        torch.set_rng_state(resume["cpu_rng"].cpu())
        torch.cuda.set_rng_state(resume["cuda_rng"].cpu())
        del resume
    train = bank.parts["train"]
    batch_size = PROTOCOL["batch_size"]
    for epoch in range(len(history) + 1, epochs + 1):
        if not pilot and epoch > PROTOCOL["minimum_epochs"] and stale >= PROTOCOL["patience"]:
            break
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        multiplier = epoch / 3 if epoch <= 3 else .1 + .9 * (1 + math.cos(math.pi * (epoch - 3) / max(1, epochs - 3))) / 2
        for group in optimizer.param_groups:
            group["lr"] = PROTOCOL["learning_rate"] * multiplier
        generator = epoch_generator(seed, epoch, bank.device)
        order = torch.randperm(len(train["indices"]), generator=generator, device=bank.device)
        loss_sum = torch.zeros((), device=bank.device)
        seen = 0
        for first in range(0, len(order), batch_size):
            index = order[first:first + batch_size]
            factor = torch.randint(1, len(FACTORS), (len(index),), generator=generator, device=bank.device)
            actual, augmented = bank.features(train, index), bank.features(train, index, factor)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(torch.cat((actual, augmented)))
                loss = augmented_loss(output[:len(index)], output[len(index):],
                                      train["classes"][index, 0], train["classes"][index, factor])
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.detach() * len(index)
            seen += len(index)
        tune_logits = bank.logits(model, "tune", batch_size=batch_size * 2)
        tune_loss = binary_loss(tune_logits, bank.parts["tune"]["outcomes"]["success"])
        if not math.isfinite(tune_loss):
            raise ValueError("Non-finite tune loss")
        if tune_loss < best - 1e-5:
            best, best_epoch, stale = tune_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        row = dict(epoch=epoch, training_loss=float(loss_sum) / seen, tune_binary_loss=tune_loss,
            seconds=time.monotonic() - started, lr=optimizer.param_groups[0]["lr"], best_epoch=best_epoch,
            base_events=seen, augmented_presentations=seen, total_presentations=seen * 2,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        history.append(row)
        save_resume(folder, training_state(model, optimizer, history, best, best_epoch, stale, best_state, contract))
        atomic_json(folder / "history.json", history)
        print(json.dumps(dict(phase="epoch", market=context["market"], fold=context["fold"],
                              architecture=architecture, seed=seed, **row)), flush=True)
    model.load_state_dict(best_state)
    calibration_logits = bank.logits(model, "calibration", batch_size=batch_size, device="cpu")
    selection_logits = bank.logits(model, "selection", batch_size=batch_size, device="cpu")
    calibration = fit_calibration(calibration_logits, bank.parts["calibration"]["outcomes"]["success"])
    probabilities = calibrated_probability(selection_logits, calibration)
    metrics, gate = score_part(bank.parts["selection"], probabilities)
    prevalence = float(bank.parts["calibration"]["outcomes"]["success"].mean())
    constant_brier = float(np.mean((bank.parts["selection"]["outcomes"]["success"] - prevalence) ** 2))
    checkpoint = dict(state_dict=best_state, architecture=architecture, model_config=CONFIG,
        feature_names=list(FEATURE_NAMES), class_names=list(CLASS_NAMES), target=TARGET,
        calibration=calibration, threshold=.5, take_profit_pct=1., stop_loss_pct=.9,
        seed=seed, context=context, contract=contract, best_epoch=best_epoch,
        research_only=True, deployment_allowed=False, intraday_path_verified=False)
    save_checkpoint(folder / "model.pt", checkpoint)
    np.savez(folder / "predictions.npz", calibration_logits=calibration_logits,
        selection_logits=selection_logits, probabilities=probabilities,
        calibration_indices=bank.parts["calibration"]["indices"], selection_indices=bank.parts["selection"]["indices"])
    result = dict(completed=True, contract=contract, architecture=architecture, seed=seed,
        parameters=parameter_count(model), epochs=len(history), best_epoch=best_epoch, best_tune_loss=best,
        training_seconds=sum(row["seconds"] for row in history), calibration=calibration,
        selection=metrics, qualification=gate, constant_brier=constant_brier,
        brier_skill=1 - metrics["overall"]["brier"] / constant_brier,
        artifact_sha256={name: sha256_file(folder / name) for name in ("model.pt", "predictions.npz", "history.json")})
    atomic_json(folder / "result.json", result)
    print(json.dumps(dict(phase="trial_completed", market=context["market"], fold=context["fold"],
        architecture=architecture, seed=seed, epochs=len(history), brier_skill=result["brier_skill"],
        signals=metrics["signal_count"], precision=metrics["precision"], qualification=gate)), flush=True)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def rank_architectures(results):
    ranking = []
    for name in MODEL_NAMES:
        rows = [results[fold][name] for fold in FOLDS]
        blocks = [row["selection"]["block_bootstrap"] for row in rows]
        supported = all(block.get("reason") is None and block.get("precision_lower") is not None
                        and block.get("net_mean_lower") is not None for block in blocks)
        ranking.append(dict(architecture=name, supported_both_folds=supported,
            qualified_both_folds=all(row["qualification"]["qualified"] for row in rows),
            worst_precision_lower=min(block["precision_lower"] for block in blocks) if supported else None,
            mean_net_lower=float(np.mean([block["net_mean_lower"] for block in blocks])) if supported else None,
            mean_brier_skill=float(np.mean([row["brier_skill"] for row in rows]))))
    supported = any(row["supported_both_folds"] for row in ranking)
    ranking.sort(key=lambda row: (-int(row["qualified_both_folds"]), -(int(row["supported_both_folds"]) if supported else 0),
        -(row["worst_precision_lower"] if supported and row["supported_both_folds"] else -1),
        -(row["mean_net_lower"] if supported and row["supported_both_folds"] else -1),
        -row["mean_brier_skill"], row["architecture"]))
    return ranking


def run_market(market, args, code_hashes):
    dataset, source, receipt = load_dataset(market, ROOT)
    sessions = read_sessions(receipt["source"]["physical_path"])
    folder = args.output_dir / market
    folder.mkdir(parents=True, exist_ok=True)
    lock_json(folder / "source.json", dict(source=source, receipt=receipt))
    results, manifests = {}, {}
    folds = ("walk_2024",) if args.phase == "pilot" else tuple(FOLDS)
    for fold in folds:
        splits = make_splits(dataset, sessions, fold,
            max_train=20_000 if args.phase == "pilot" else PROTOCOL["max_train"],
            max_tune=5_000 if args.phase == "pilot" else PROTOCOL["max_tune"], seed=42)
        if args.phase == "pilot":
            splits = {name: indices[:5000] if name in ("calibration", "selection") else indices
                      for name, indices in splits.items()}
        manifests[fold] = split_manifest(dataset, splits)
        lock_json(folder / fold / "splits.json", manifests[fold])
        bank = BuildAugmentedBank(dataset, splits, "cuda", factors=FACTORS)
        context = canonical(dict(market=market, fold=fold, source=source, data_receipt=receipt,
                                 splits=manifests[fold], code_sha256=code_hashes))
        results[fold] = {}
        for architecture in MODEL_NAMES:
            results[fold][architecture] = train_trial(bank, architecture, 42,
                folder / fold / f"{architecture}-42", context,
                epochs=2 if args.phase == "pilot" else PROTOCOL["max_epochs"], pilot=args.phase == "pilot")
        del bank
        gc.collect()
        torch.cuda.empty_cache()
    if args.phase == "pilot":
        summary = dict(completed=True, pilot=True, market=market, results=results,
                       note="Throughput/safety only; not a selected model or full-sample performance")
        atomic_json(folder / "summary.json", summary)
        return summary
    ranking = rank_architectures(results)
    selected = ranking[0]["architecture"]
    lock_json(folder / "selection_locked.json", dict(selected=selected, ranking=ranking,
        rule=PROTOCOL["selection"], source=source, code_sha256=code_hashes))
    ensembles = {}
    for fold in FOLDS:
        splits = make_splits(dataset, sessions, fold, max_train=PROTOCOL["max_train"],
                             max_tune=PROTOCOL["max_tune"], seed=42)
        bank = BuildAugmentedBank(dataset, splits, "cuda", factors=FACTORS)
        context = canonical(dict(market=market, fold=fold, source=source, data_receipt=receipt,
                                 splits=manifests[fold], code_sha256=code_hashes))
        for seed in SEEDS[1:]:
            train_trial(bank, selected, seed, folder / fold / f"{selected}-{seed}", context)
        predictions = []
        for seed in SEEDS:
            with np.load(folder / fold / f"{selected}-{seed}" / "predictions.npz", allow_pickle=False) as saved:
                for name in ("calibration", "selection"):
                    if not np.array_equal(saved[f"{name}_indices"], bank.parts[name]["indices"]):
                        raise ValueError("Ensemble prediction rows differ")
                predictions.append((saved["calibration_logits"].copy(), saved["selection_logits"].copy()))
        calibration_logits, selection_logits = (np.mean([row[index] for row in predictions], axis=0) for index in (0, 1))
        calibration = fit_calibration(calibration_logits, bank.parts["calibration"]["outcomes"]["success"])
        probabilities = calibrated_probability(selection_logits, calibration)
        metrics, gate = score_part(bank.parts["selection"], probabilities)
        ensembles[fold] = dict(calibration=calibration, selection=metrics, qualification=gate)
        atomic_json(folder / fold / "ensemble.json", ensembles[fold])
        np.savez(folder / fold / "ensemble_predictions.npz", probabilities=probabilities,
                 sample_indices=bank.parts["selection"]["indices"], raw_logits=selection_logits)
        del bank
        gc.collect()
        torch.cuda.empty_cache()
    summary = dict(completed=True, pilot=False, market=market, selected=selected, ranking=ranking,
        ensemble=ensembles, source=source, data_receipt=receipt, protocol=canonical(PROTOCOL),
        research_qualified=all(item["qualification"]["qualified"] for item in ensembles.values()),
        deployment_allowed=False, intraday_path_verified=False,
        next_required="Frozen-bundle export, price sensitivity and reused 2025+ portfolio backtest; not yet performed")
    atomic_json(folder / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "train"), default="train")
    parser.add_argument("--markets", nargs="+", choices=("domestic", "us"), default=["domestic", "us"])
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    allowed = (ROOT / "outputs" / "mark1").resolve()
    if not args.output_dir.is_relative_to(allowed) or args.output_dir == allowed:
        parser.error("Output must be a dedicated child of outputs/mark1; historical runs must be preserved")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; refusing silent CPU training")
    if "5080" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This requested experiment requires the RTX 5080")
    torch.set_num_threads(12)
    torch.cuda.set_per_process_memory_fraction(.65)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    code_hashes = {name: sha256_file(ROOT / name) for name in CODE_FILES}
    contract = dict(protocol=PROTOCOL, code_sha256=code_hashes, phase=args.phase, markets=args.markets,
        runtime={"torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda,
                 "gpu": torch.cuda.get_device_name(0)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(args.output_dir):
        if (any(path.name != "run.lock" for path in args.output_dir.iterdir())
                and not (args.output_dir / "protocol.json").exists()):
            raise ValueError("Nonempty unowned output directory")
        lock_json(args.output_dir / "protocol.json", contract)
        started = time.monotonic()
        try:
            atomic_json(args.output_dir / "status.json", {"status": "running", "phase": args.phase})
            summaries = {}
            for market in args.markets:
                summaries[market] = run_market(market, args, code_hashes)
            atomic_json(args.output_dir / "summary.json", summaries)
            atomic_json(args.output_dir / "status.json", {"status": "completed", "phase": args.phase,
                        "elapsed_seconds": time.monotonic() - started, "training_only": args.phase == "train"})
        except BaseException as error:
            atomic_json(args.output_dir / "status.json", {"status": "failed", "phase": args.phase,
                        "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
