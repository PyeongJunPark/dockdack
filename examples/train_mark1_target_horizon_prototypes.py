"""Train six DEMO-only target/horizon prototype bundles from completed daily bars.

This intentionally uses the train and calibration splits only.  It does not
read selection or final-test outcomes, submit orders, or modify account data.
The observed next-session OPEN is a retrospective training query/entry proxy;
an intraday current-price query has an unvalidated time/price distribution.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dockdack.mark1_2_data import load_dataset
from dockdack.mark1_target_horizon_data import load_horizon_bank
from dockdack.research_artifacts import sha256_file, write_new_json
from examples.research_mark1_target_horizon import (
    FeatureCache, _eligible_features, _fit_torch, _temperature,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / "mark1_target_horizon_v1"
SPECS = (
    ("mark1-23-prototype", 20, 10, 3.0, "gru"),
    ("mark1-24-prototype", 30, 20, 3.0, "mlp"),
    ("mark1-25-prototype", 20, 20, 4.0, "gru"),
    ("mark1-26-prototype", 20, 20, 2.0, "linear"),
    ("mark1-27-prototype", 20, 10, 3.0, "cnn"),
    ("mark1-28-prototype", 10, 10, 4.0, "mlp"),
)


def _sample(dataset, name: str, cap: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(dataset.splits[name], dtype=np.int64)
    if not len(values):
        raise ValueError(f"empty approved {name} split")
    return np.sort(rng.choice(values, size=min(cap, len(values)), replace=False))


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def train_bundle(output: Path, *, train_cap: int = 20_000,
                 calibration_cap: int = 10_000, epochs: int = 5,
                 seed: int = 53) -> Path:
    output = Path(output).resolve()
    if (output.exists() or output == ROOT or not output.is_relative_to(ROOT / "models")
            or train_cap < 1 or calibration_cap < 1 or epochs < 1 or seed < 0):
        raise ValueError("select a new models/ child and positive training parameters")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = {
        "schema_version": 1,
        "owner": "mark1_target_horizon_v1",
        "research_only": True,
        "deployment_allowed": False,
        "intraday_path_verified": False,
        "trading_mode": "demo",
        "feature_contract": "completed_ohlcv_plus_query_log_gap_v2_6_channels",
        "training_query": "observed_next_session_open",
        "proposed_runtime_query": "current_intraday_price",
        "entry_proxy": "observed_next_session_open_not_executable_at_same_open",
        "exit_proxy": "first_daily_high_target_touch_else_Hth_close_no_stop",
        "warnings": [
            "The observed next OPEN cannot both inform the decision and guarantee that same OPEN fill.",
            "The runtime intraday quote differs from the training query; target-day HIGH may predate query.",
            "Daily HIGH touch does not guarantee limit execution; H-day CLOSE is not a live fill.",
            "No profit, intraday path, brokerage execution, or live deployment is validated.",
        ],
        "feature_module_sha256": sha256_file(ROOT / "dockdack" / "mark1_target_horizon_models.py"),
        "training_script_sha256": sha256_file(Path(__file__)),
        "train_cap_per_market": train_cap,
        "calibration_cap_per_market": calibration_cap,
        "epochs": epochs,
        "seed": seed,
        "specs": {},
        "markets": {},
    }
    for market in ("domestic", "us"):
        dataset, source, _ = load_dataset(market, ROOT)
        rng = np.random.default_rng(seed)
        chosen = np.sort(np.concatenate((
            _sample(dataset, "train", train_cap, rng),
            _sample(dataset, "calibration", calibration_cap, rng),
        )))
        bank = load_horizon_bank(dataset, source, ROOT, market,
                                 sample_indices=chosen, max_horizon=20)
        cache = FeatureCache(bank, dataset)
        manifest["markets"][market] = {
            "source_sha256": bank.source_sha256,
            "sample_indices_sha256": _array_hash(bank.sample_indices),
            "train_count": int(np.count_nonzero(bank.split_names == "train")),
            "calibration_count": int(np.count_nonzero(bank.split_names == "calibration")),
        }
        for index, (model_id, lookback, horizon, target_pct, family) in enumerate(SPECS):
            train = bank.outcomes(target_pct=target_pct, horizon=horizon,
                                  cost_bps=20., slippage_bps=5.,
                                  split_name="train", lookback=lookback)
            calibration = bank.outcomes(target_pct=target_pct, horizon=horizon,
                                        cost_bps=20., slippage_bps=5.,
                                        split_name="calibration", lookback=lookback)
            train_positions, train_x = _eligible_features(cache, lookback, "train", train)
            calibration_positions, calibration_x = _eligible_features(
                cache, lookback, "calibration", calibration)
            if len(train_positions) < 30 or len(calibration_positions) < 30:
                raise ValueError(f"insufficient valid paths for {market} {model_id}")
            fitted = _fit_torch(
                family, lookback, train_x, train.hit[train_positions],
                train.net_return[train_positions], seed=seed + index,
                epochs=epochs, batch_size=512, device="cpu")
            logits, _ = fitted.predict(calibration_x)
            temperature, brier = _temperature(logits, calibration.hit[calibration_positions])
            name = f"{market}-{model_id}.pt"
            path = output / name
            with path.open("xb") as stream:
                torch.save({"state_dict": fitted.model.state_dict(),
                            "family": family, "lookback": lookback}, stream)
            info = manifest["specs"].setdefault(model_id, {
                "lookback": lookback, "horizon": horizon,
                "target_pct": target_pct, "family": family,
                "markets": {},
            })
            info["markets"][market] = {
                "weights": name,
                "weights_sha256": sha256_file(path),
                "normalization_center": fitted.center.astype(float).tolist(),
                "normalization_scale": fitted.scale.astype(float).tolist(),
                "temperature": temperature,
                "calibration_brier": brier,
                "eligible_train": int(len(train_positions)),
                "eligible_calibration": int(len(calibration_positions)),
                "train_target_hits": int(train.hit[train_positions].sum()),
            }
            print(f"{market} {model_id}: {len(train_positions)} train, "
                  f"{len(calibration_positions)} calibration", flush=True)
    write_new_json(output / "manifest.json", manifest)
    with (output / "manifest.sha256").open("x", encoding="ascii") as stream:
        stream.write(sha256_file(output / "manifest.json") + "\n")
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-cap", type=int, default=20_000)
    parser.add_argument("--calibration-cap", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args(argv)
    print(train_bundle(args.output_dir, train_cap=args.train_cap,
                       calibration_cap=args.calibration_cap,
                       epochs=args.epochs, seed=args.seed), flush=True)


if __name__ == "__main__":
    main()
