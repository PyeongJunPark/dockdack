"""Train the Mark1.3 pre-open research candidate on CUDA and seal a DEMO bundle.

Historical OPEN/CLOSE are evaluation proxies, never evidence of broker fills.
No operational store, GUI, account, broker, or order path is opened here.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dockdack.clean_daily_dataset import load_sessions
from dockdack import mark1_3_research
from dockdack.mark1_3_research import (
    DailyNetRegressor, FEATURE_NAMES, chronological_splits,
    fixed_open_close_returns, load_raw_daily_candidates,
)
from dockdack.mark1_3_preopen import NUMERIC_GUARD_PERCENT


ROOT = Path(__file__).resolve().parents[1]
FIRST = "2017-01-01"
TRAIN_END = "2021-12-31"
VALIDATION_END = "2022-12-31"
TEST_END = "2024-12-31"
COST_BPS = 20


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _save_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return _sha(path)


def _metrics(prediction: np.ndarray, net: np.ndarray, indices: np.ndarray) -> dict:
    selected = indices[prediction[indices] * 100 > NUMERIC_GUARD_PERCENT]
    observed = selected[np.isfinite(net[selected])]
    return {
        "eligible_candidates": int(len(indices)),
        "positive_score_signals": int(len(selected)),
        "observed_proxy_outcomes": int(len(observed)),
        "missing_proxy_outcomes": int(len(selected) - len(observed)),
        "mean_observed_net_return_percent":
            float(net[observed].mean() * 100) if len(observed) else None,
        "positive_observed_net_fraction":
            float(np.mean(net[observed] > 0)) if len(observed) else None,
    }


def _fit_cuda(samples, splits, *, market: str, epochs: int, train_cap: int, seed: int) -> tuple[dict, dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("Mark1.3 training requires a CUDA GPU; no CPU fallback")
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _, net = fixed_open_close_returns(samples.entry_open, samples.exit_close, cost_bps=COST_BPS)
    train = splits["train"][np.isfinite(net[splits["train"]])]
    validation = splits["validation"][np.isfinite(net[splits["validation"]])]
    if len(train) > train_cap:
        train = np.sort(np.random.default_rng(seed).choice(train, train_cap, replace=False))
    if len(train) < 20 or len(validation) < 5:
        raise ValueError("Insufficient observed Mark1.3 train/validation outcomes")
    mean = samples.features[train].mean(axis=0).astype(np.float32)
    scale = samples.features[train].std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1
    scaled = np.clip((samples.features - mean) / scale, -8, 8).astype(np.float32)
    x = torch.from_numpy(scaled).to(device)
    y = torch.from_numpy((np.nan_to_num(net, nan=0.0) * 100).astype(np.float32)).to(device)
    train_tensor = torch.as_tensor(train, dtype=torch.long, device=device)
    validation_tensor = torch.as_tensor(validation, dtype=torch.long, device=device)
    model = DailyNetRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.SmoothL1Loss(beta=1.0)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        for order in torch.randperm(len(train), generator=generator).split(2048):
            optimizer.zero_grad(set_to_none=True)
            index = train_tensor[order.to(device)]
            loss = criterion(model(x[index]), y[index])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            value = float(criterion(model(x[validation_tensor]), y[validation_tensor]))
        if value < best_loss:
            best_loss, best_epoch = value, epoch
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Mark1.3 CUDA training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        gpu_prediction_percent = model(x).cpu().numpy().astype(np.float64)
    model.cpu().eval()
    with torch.inference_mode():
        cpu_prediction_percent = model(torch.from_numpy(scaled)).numpy().astype(np.float64)
    observed_delta_percent = float(np.max(np.abs(gpu_prediction_percent - cpu_prediction_percent)))
    if observed_delta_percent > NUMERIC_GUARD_PERCENT:
        raise ValueError("Observed CPU/CUDA score delta exceeds Mark1.3 numeric guard")
    prediction = cpu_prediction_percent / 100
    if not np.isfinite(prediction).all():
        raise ValueError("Mark1.3 CUDA checkpoint has nonfinite predictions")
    training = {
        "training_device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "seed": seed, "best_epoch": best_epoch,
        "observed_cpu_gpu_max_abs_score_delta_percent": observed_delta_percent,
        "numeric_guard_percent": NUMERIC_GUARD_PERCENT,
        "validation_huber_percentage_points": best_loss,
        "observed_train_used": int(len(train)), "observed_validation": int(len(validation)),
        "score_policy": "predicted_net_return_percent_gt_0.0001_numeric_guard_without_test_tuning",
        "splits": {name: _metrics(prediction, net, splits[name])
                   for name in ("train", "validation", "test")},
        "test_is_previously_reused_historical_period": True,
    }
    artifact = {
        "model_id": "mark1-3-prototype", "market": market,
        "features": list(FEATURE_NAMES), "mean": mean.tolist(), "scale": scale.tolist(),
        "state_dict": {name: value.numpy().tolist() for name, value in best_state.items()},
        "threshold": NUMERIC_GUARD_PERCENT, "trained_on": "cuda",
        "training_symbols": [{"symbol": item["symbol"], "exchange": item["exchange"]}
                             for item in samples.source["selected_symbols"]],
        "training": training,
    }
    return artifact, training


def train_bundle(destination: Path, *, max_symbols: int = 100, epochs: int = 20,
                 train_cap: int = 100_000, seed: int = 42,
                 replace_seal: str | None = None) -> dict:
    if (not 1 <= max_symbols <= 500 or not 1 <= epochs <= 100
            or not 20 <= train_cap <= 1_000_000 or seed < 0):
        raise ValueError("Invalid Mark1.3 GPU training configuration")
    if destination.exists():
        files = {item.name for item in destination.iterdir()}
        existing_seal = destination / "manifest.sha256"
        if (replace_seal is None
                or files != {"manifest.json", "manifest.sha256", "domestic.json", "us.json"}
                or not existing_seal.is_file()
                or existing_seal.read_text(encoding="ascii").strip() != replace_seal
                or _sha(destination / "manifest.json") != replace_seal):
            raise FileExistsError(f"Refusing to replace an unrecognized model bundle: {destination}")
    if not torch.cuda.is_available():
        raise RuntimeError("Mark1.3 training requires CUDA; no CPU fallback")
    prepared = {}
    for market in ("domestic", "us"):
        database = ROOT / "data" / "kiwoom_daily" / (market + "_daily.sqlite3")
        if not database.is_file():
            raise FileNotFoundError(database)
        sessions, _ = load_sessions(market, date.fromisoformat(FIRST),
                                    date.fromisoformat(TEST_END) + timedelta(days=1))
        print(f"{market}: loading up to {max_symbols} raw read-only daily histories", flush=True)
        samples = load_raw_daily_candidates(
            database, market, start=FIRST, train_end=TRAIN_END, test_end=TEST_END,
            max_symbols=max_symbols, seed=seed, session_dates=sessions)
        splits = chronological_splits(samples, train_end=TRAIN_END,
                                      validation_end=VALIDATION_END, test_end=TEST_END,
                                      sessions=tuple(sessions))
        artifact, training = _fit_cuda(samples, splits, market=market,
                                       epochs=epochs, train_cap=train_cap, seed=seed)
        prepared[market] = (artifact, training)
        print(f"{market}: CUDA fit complete; historical test proxy {training['splits']['test']}", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    markets = {}
    for market, (artifact, training) in prepared.items():
        filename = market + ".json"
        markets[market] = {"file": filename, "sha256": _save_json(destination / filename, artifact),
                           "training": training}
    manifest = {
        "schema_version": 1, "model_id": "mark1-3-prototype", "version": "1.3",
        "features": list(FEATURE_NAMES), "lookback": 30,
        "entry": "next_session_open_proxy", "exit": "same_session_close_proxy",
        "cost_bps": COST_BPS, "training_device": "cuda", "research_only": True,
        "research_qualified": False, "deployment_allowed": False,
        "feature_and_architecture_sha256": _sha(Path(mark1_3_research.__file__)),
        "selection_policy": "score_predicted_net_return_percent_strictly_above_0.0001_numeric_guard_then_top_ten",
        "limitations": "Historical proxy; catalog survivorship and unverified open/close fills; not real performance",
        "markets": markets,
    }
    seal = _save_json(destination / "manifest.json", manifest)
    (destination / "manifest.sha256").write_text(seal + "\n", encoding="ascii")
    print(f"Mark1.3 bundle: {destination}; manifest SHA256: {seal}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "mark1_3")
    parser.add_argument("--max-symbols", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-cap", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace-seal", help="Replace only a previously generated bundle with this exact seal")
    args = parser.parse_args(argv)
    train_bundle(args.output.resolve(), max_symbols=args.max_symbols, epochs=args.epochs,
                 train_cap=args.train_cap, seed=args.seed, replace_seal=args.replace_seal)


if __name__ == "__main__":
    main()
