"""One-shot demo-only LSTM adapter; publishes preview JSON, never an order.

Run as ``python -m examples.publish_lstm30 --help`` from the repository root.
Quantities and KRW/USD caps have no default: these are user trading decisions.
Retain the sibling .state.json file between runs to preserve idempotency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dockdack.lstm30_adapter import (
    LSTM30SignalProducer, atomic_json, demo_position_provider, read_json,
)


def fixture_position_provider(payload):
    """Offline snapshots must explicitly include each flat/held instrument."""
    if not isinstance(payload, dict) or payload.get("trading_mode") != "demo":
        raise ValueError("Position fixture must explicitly declare trading_mode='demo'")
    rows = payload.get("positions")
    if not isinstance(rows, list):
        raise ValueError("Position fixture requires a positions list")
    positions = {}
    for row in rows:
        key = tuple(row.get(field) for field in ("market", "exchange", "symbol"))
        if key in positions:
            raise ValueError("Duplicate position fixture instrument")
        positions[key] = row

    def provide(stock):
        return positions.get(tuple(stock[field] for field in ("market", "exchange", "symbol")))

    return provide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charts", type=Path, default=Path(".dockdack/exchange/charts.json"))
    parser.add_argument("--output", type=Path, default=Path(".dockdack/exchange/signals.preview.json"))
    parser.add_argument("--state", type=Path, help="Immutable decision history; defaults beside output")
    parser.add_argument("--domestic-checkpoint", type=Path)
    parser.add_argument("--us-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu", help="Model inference device, e.g. cpu or cuda")
    parser.add_argument("--buy-threshold", type=float, default=0.4,
                        help="Inclusive runtime BUY probability threshold (default 0.4); checkpoint weights stay unchanged")
    parser.add_argument("--quantity", type=int, required=True, help="BUY shares and maximum shares per SELL")
    parser.add_argument("--max-krw", required=True, help="Per-order KRW cap; 0 blocks domestic signals")
    parser.add_argument("--max-usd", required=True, help="Per-order USD cap; 0 blocks US signals")
    positions = parser.add_mutually_exclusive_group(required=True)
    positions.add_argument("--positions", type=Path, help="Offline demo position snapshots JSON")
    positions.add_argument("--kiwoom-demo", action="store_true", help="Read current Kiwoom DEMO holdings only")
    args = parser.parse_args()
    if not 0 < args.buy_threshold < 1:
        parser.error("--buy-threshold must be a finite probability strictly between 0 and 1")
    state_path = args.state or args.output.with_name(args.output.stem + ".state.json")
    diagnostic_path = args.output.with_name(args.output.stem + ".diagnostics.json")
    output_paths = [path.resolve() for path in (args.output, state_path, diagnostic_path)]
    protected = {path.resolve() for path in (args.charts, args.positions, args.domestic_checkpoint,
                                             args.us_checkpoint) if path is not None}
    if (len(set(output_paths)) != 3 or protected.intersection(output_paths)
            or any(path.suffix.lower() != ".json" for path in output_paths)):
        parser.error("Use three distinct .json output/state/diagnostic files, separate from all inputs")
    try:
        charts = read_json(args.charts)
        predictors = {}
        for market, path in (("domestic", args.domestic_checkpoint), ("us", args.us_checkpoint)):
            if path is not None:
                try:
                    from dockdack.ml30 import Predictor
                    predictors[market] = Predictor(path, device=args.device, buy_threshold=args.buy_threshold)
                except (ValueError, RuntimeError, OSError, ImportError) as exc:
                    # Missing/broken models must not disable cost-based held exits.
                    print(f"{market} model unavailable: {exc}; flat positions will HOLD", file=sys.stderr)
        if args.positions:
            provider = fixture_position_provider(read_json(args.positions))
        else:
            from dockdack import KiwoomBroker, KiwoomConfig, Market, TradingMode
            brokers = {}

            def provider(stock):
                market = stock["market"]
                if market not in brokers:
                    config = KiwoomConfig.from_env(TradingMode.DEMO, market=Market(market))
                    brokers[market] = KiwoomBroker(config)
                return demo_position_provider(brokers)(stock)

        producer = LSTM30SignalProducer(predictors, position_provider=provider,
                                       quantity=args.quantity, max_krw=args.max_krw,
                                       max_usd=args.max_usd, state_path=state_path)
        payload, diagnostics = producer(charts)
        atomic_json(args.output, payload)
        atomic_json(diagnostic_path, {"signal_only": True, "diagnostics": diagnostics})
        counts = {action: sum(row["action"] == action for row in payload["signals"])
                  for action in ("buy", "sell", "hold")}
        print(f"Published demo signals {counts} to {args.output.resolve()}; no orders or background monitoring")
        print(f"Diagnostics: {diagnostic_path.resolve()}")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
