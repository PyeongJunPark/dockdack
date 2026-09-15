"""Adapter example for an external model. Running this file publishes HOLD only.

Replace the calling model's decisions, not the broker's safety checks. Reuse the same
decision_id AND generated_at when retrying a decision; never renew IDs to retry orders.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dockdack.signal_bridge import atomic_json, identifier, timestamp


def build_signals(charts, decisions, *, source_id, decision_id, generated_at):
    """decisions maps watch_id to {action, quantity, max_notional}; absent => hold.

    This does not choose a strategy or place an order. Quantities are integer shares;
    max_notional is a string in each stock's original KRW/USD currency.
    """
    identifier(source_id, "source_id")
    identifier(decision_id, "decision_id")
    if generated_at.tzinfo is None:
        raise ValueError("generated_at requires a timezone")
    signals = []
    for stock in charts["stocks"]:
        if stock.get("status") != "ok" or not stock.get("complete"):
            continue
        decision = decisions.get(stock["watch_id"], {"action": "hold"})
        if decision.get("action") not in {"buy", "sell", "hold"}:
            raise ValueError("The model must return buy, sell or hold")
        signal = {"signal_id": uuid5(NAMESPACE_URL, f"{source_id}:{decision_id}:{stock['watch_id']}").hex,
                  "export_id": charts["export_id"],
                  **{field: stock[field] for field in ("market", "symbol", "exchange")},
                  "action": decision["action"], "generated_at": generated_at.isoformat(),
                  "expires_at": (generated_at + timedelta(minutes=2)).isoformat()}
        if decision["action"] != "hold":
            signal.update(quantity=decision["quantity"], max_notional=decision["max_notional"])
        signals.append(signal)
    return {"schema_version": 1, "source_id": source_id, "signals": signals}


def main():
    parser = argparse.ArgumentParser(description="Publish HOLD-only example signals; no broker/API calls")
    parser.add_argument("--charts", type=Path, default=Path(".dockdack/exchange/charts.json"))
    parser.add_argument("--output", type=Path, default=Path(".dockdack/exchange/signals.example.json"))
    args = parser.parse_args()
    if args.charts.resolve() == args.output.resolve() or args.output.suffix.lower() != ".json":
        parser.error("output must be a different .json file")
    with args.charts.open(encoding="utf-8") as stream:
        charts = json.load(stream)
    # Fixed to the export for this HOLD-only example: repeat runs cannot extend TTL.
    created = timestamp(charts["created_at"], "created_at")
    if not 0 <= (datetime.now(timezone.utc) - created).total_seconds() <= 300:
        parser.error("refresh/export the charts first (export must be under 5 minutes old)")
    payload = build_signals(charts, {}, source_id="external-model", decision_id=charts["export_id"], generated_at=created)
    atomic_json(args.output, payload)
    print(f"Published {len(payload['signals'])} HOLD signals to {args.output.resolve()}; no orders")


if __name__ == "__main__":
    main()
