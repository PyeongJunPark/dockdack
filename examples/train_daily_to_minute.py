"""Offline daily-to-5-minute transfer research; never connects to an account.

Example (real exchange data required, never synthetic data for a published run):

  uv run --extra research python -m examples.train_daily_to_minute \
      --market domestic --daily path/to/daily.jsonl \
      --minute path/to/complete_5m.jsonl --output path/to/new_research_bundle

Both inputs use one JSON object per line:
  {"market":"domestic","symbol":"005930",
   "timestamp":"2026-09-28T15:30:00+09:00","open":70000,
   "high":71000,"low":69000,"close":70500,"volume":123456,
   "bar_minutes":1440}

Five-minute rows use bar_minutes=5 and the broker's original bar time label.
Their label may denote bar start or end; a full additional five minutes must
elapse before the row is accepted. Daily timestamps are exchange close times.
Use separate invocations for domestic and US to avoid pooled market leakage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import torch

from dockdack.research.minute_transfer import (ARCHITECTURES, load_bars,
                                               sha256_file, train_transfer)


def run(args) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace an existing research bundle: {output}")
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(timezone.utc)
    daily_stats, minute_stats = {}, {}
    daily = load_bars(args.daily, kind="daily", as_of=as_of, stats=daily_stats)
    minute = load_bars(args.minute, kind="minute", as_of=as_of, stats=minute_stats)
    minute_hash = sha256_file(args.minute)
    receipt_path = Path(args.minute).with_suffix(Path(args.minute).suffix + ".receipt.json")
    receipt_verified = False
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("source") != "kiwoom_rest_demo_minute_chart"
                or receipt.get("sha256") != minute_hash
                or receipt.get("market") != args.market
                or receipt.get("interval_minutes") != 5
                or receipt.get("regular_session_filter") is not True
                or receipt.get("accepted_bars") != minute_stats["jsonl_rows"]):
            raise ValueError("Kiwoom minute collector receipt does not match this input")
        receipt_verified = True
    if {bar.market for bar in daily} != {args.market} or {bar.market for bar in minute} != {args.market}:
        raise ValueError("both files must contain only the selected market")
    artifacts, summary = train_transfer(
        daily, minute, lookback=args.lookback, horizon=args.horizon,
        fee_bps_per_side=args.fee_bps, slippage_bps_per_side=args.slippage_bps,
        purge_sessions=args.purge_sessions, pretrain_epochs=args.pretrain_epochs,
        adapt_epochs=args.adapt_epochs, min_samples_per_split=args.min_samples_per_split,
        min_validation_trades=args.min_validation_trades,
        max_train_samples=args.max_train_samples, seed=args.seed)
    summary["minute_collector_receipt_verified"] = receipt_verified
    manifest = {
        "format": "dockdack-minute-transfer-research-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of.isoformat(), "market": args.market,
        "sources": {"daily_sha256": sha256_file(args.daily),
                    "minute_sha256": minute_hash,
                    "minute_collector_receipt_verified": receipt_verified,
                    "daily_latest_bar_label": max(bar.timestamp for bar in daily).isoformat(),
                    "minute_latest_bar_label": max(bar.timestamp for bar in minute).isoformat(),
                    "daily": daily_stats, "minute": minute_stats},
        "training": {"seed": args.seed, "pretrain_epochs": args.pretrain_epochs,
                     "adapt_epochs": args.adapt_epochs,
                     "max_train_samples": args.max_train_samples,
                     "min_validation_trades": args.min_validation_trades},
        "study": summary,
        "models": {name: {"validation": artifacts[name]["validation"],
                          "test": artifacts[name]["test"],
                          "validation_threshold": artifacts[name]["threshold"],
                          "state_file": f"{name}.pt"}
                   for name in ARCHITECTURES},
        "safety": {"research_only": True, "deployment_allowed": False,
                   "order_routing_connected": False,
                   "historical_profitability_claim": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="minute-transfer-", dir=output.parent) as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        for name in ARCHITECTURES:
            art = artifacts[name]
            torch.save({"state_dict": art["model"].state_dict(),
                        "mean": torch.from_numpy(art["minute_mean"].copy()),
                        "scale": torch.from_numpy(art["minute_scale"].copy()),
                        "lookback": args.lookback, "architecture": name},
                       staging / f"{name}.pt")
            manifest["models"][name]["state_sha256"] = sha256_file(staging / f"{name}.pt")
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True, choices=("domestic", "us"))
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--minute", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", help="aware ISO-8601 cutoff; defaults to now UTC")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--fee-bps", type=float, default=2.)
    parser.add_argument("--slippage-bps", type=float, default=8.)
    parser.add_argument("--purge-sessions", type=int, default=3)
    parser.add_argument("--pretrain-epochs", type=int, default=4)
    parser.add_argument("--adapt-epochs", type=int, default=4)
    parser.add_argument("--min-samples-per-split", type=int, default=30)
    parser.add_argument("--min-validation-trades", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)
    manifest = run(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                     allow_nan=False))


if __name__ == "__main__":
    main()
