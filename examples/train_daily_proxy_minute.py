"""Train daily-only generic-bar models for experimental 5m DEMO inference.

No minute observations enter fitting, scaling, threshold selection or holdout
evaluation. Three architectures are fitted per invocation. Use separate
lookback/horizon invocations for additional prototype identities. A DEMO
eligibility flag describes complete artifact/schedule metadata, not proven
intraday performance. No brokerage, account or order API is called here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import torch

from dockdack.research.daily_proxy_minute import (
    ARCHITECTURES, FORMAT, load_daily_proxy_artifact, sha256_file,
    train_daily_proxy,
)
from dockdack.research.minute_transfer import load_bars


def run(args) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing model bundle: {output}")
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(timezone.utc)
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must include a UTC offset")
    daily_stats: dict = {}
    daily = load_bars(args.daily, kind="daily", as_of=as_of, stats=daily_stats)
    if {bar.market for bar in daily} != {args.market}:
        raise ValueError("market does not match daily input")
    daily_hash = sha256_file(args.daily)
    receipt_path = Path(args.daily).with_suffix(Path(args.daily).suffix + ".receipt.json")
    receipt_verified = False
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("source") != "clean_daily_sqlite_read_only"
                or receipt.get("bar_kind") != "observed_daily_not_minute"
                or receipt.get("market") != args.market
                or receipt.get("sha256") != daily_hash
                or receipt.get("accepted_bars") != daily_stats["accepted_bars"]):
            raise ValueError("daily clean DB export receipt does not match input")
        receipt_verified = True
    artifacts, summary = train_daily_proxy(
        daily, lookback=args.lookback, horizon=args.horizon,
        fee_bps_per_side=args.fee_bps,
        slippage_bps_per_side=args.slippage_bps,
        purge_sessions=args.purge_sessions, epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        min_samples_per_split=args.min_samples_per_split,
        min_validation_candidates=args.min_validation_candidates,
        min_candidate_days=args.min_candidate_days,
        min_training_symbols=args.min_training_symbols,
        seed=args.seed,
    )
    model_records = {}
    for index, architecture in enumerate(ARCHITECTURES):
        art = artifacts[architecture]
        model_records[architecture] = {
            "model_id": f"mark1-{args.first_model_number + index}-prototype",
            "state_file": f"{architecture}.pt",
            "validation_threshold": art["threshold"],
            "validation": art["validation"],
            "daily_test": art["daily_test"],
            "demo_experimental_eligible": bool(receipt_verified and art["threshold"] is not None),
            "minute_backtest_completed": False,
        }
    manifest = {
        "format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of.isoformat(), "market": args.market,
        "source": {"daily_sha256": daily_hash,
                   "daily_export_receipt_verified": receipt_verified,
                   "latest_completed_daily_label": max(bar.timestamp for bar in daily).isoformat(),
                   "daily": daily_stats},
        "training": {"epochs": args.epochs, "seed": args.seed,
                     "max_train_samples": args.max_train_samples,
                     "min_validation_candidates": args.min_validation_candidates,
                     "min_candidate_days": args.min_candidate_days},
        "study": summary, "models": model_records,
        "safety": {
            "real_money_allowed": False,
            "demo_experimental_only": True,
            "minute_profitability_claim": False,
            "model_does_not_place_orders": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="daily-proxy-", dir=output.parent) as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        for architecture in ARCHITECTURES:
            art = artifacts[architecture]
            state_path = staging / f"{architecture}.pt"
            torch.save({
                "architecture": architecture,
                "lookback": args.lookback, "horizon": args.horizon,
                "state_dict": art["model"].state_dict(),
                "mean": torch.from_numpy(art["mean"].copy()),
                "scale": torch.from_numpy(art["scale"].copy()),
            }, state_path)
            model_records[architecture]["state_sha256"] = sha256_file(state_path)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(output)
    # Self-check the exact saved files, not only the in-memory model objects.
    for architecture in ARCHITECTURES:
        load_daily_proxy_artifact(output, architecture)
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=("domestic", "us"), required=True)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", help="aware ISO-8601 cutoff; defaults to now")
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--first-model-number", type=int, required=True)
    parser.add_argument("--fee-bps", type=float, default=2.)
    parser.add_argument("--slippage-bps", type=float, default=8.)
    parser.add_argument("--purge-sessions", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=50_000)
    parser.add_argument("--min-samples-per-split", type=int, default=100)
    parser.add_argument("--min-validation-candidates", type=int, default=20)
    parser.add_argument("--min-candidate-days", type=int, default=10)
    parser.add_argument("--min-training-symbols", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args(argv)
    manifest = run(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
