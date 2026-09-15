"""Verify an exported main source snapshot against the local adapter, with fakes only.

No keys, network, live DB or actual orders are used. main is not merged/checked out.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-source", type=Path, required=True)
    parser.add_argument("--domestic-checkpoint", type=Path)
    parser.add_argument("--us-checkpoint", type=Path)
    args = parser.parse_args()
    main_source = args.main_source.resolve()
    workspace = Path(__file__).resolve().parents[1]
    if not (main_source / "tests/test_autotrade.py").is_file():
        parser.error("main-source must be a main snapshot with the fake-broker tests")
    sys.path.insert(0, str(main_source))
    sys.path.insert(0, str(main_source / "tests"))
    import dockdack
    dockdack.__path__.append(str(workspace / "dockdack"))
    from dockdack.autotrade import AutoTrader
    from dockdack.lstm30_adapter import LSTM30SignalProducer, SOURCE_ID
    from dockdack.signal_bridge import ExternalPolicy, export_charts, ingest_signals
    from dockdack.watchlist import WatchItem, WatchStore
    from test_autotrade import FakeTradingService, NOW, position

    results = []
    cases = [
        ("buy", "100", False, 0.8, "buy", None),
        ("hold", "100", False, 0.2, "hold", None),
        ("take_profit", "101", True, 0.2, "sell", "cost_profit_pct"),
        ("stop_loss", "99.2", True, 0.8, "sell", "cost_loss_pct"),
    ]
    for name, price, held, probability, expected, exit_field in cases:
        with tempfile.TemporaryDirectory(prefix="lstm30-bridge-") as folder:
            root = Path(folder)
            store = WatchStore(root / "watch.sqlite3")
            service = FakeTradingService()
            service.prices = [Decimal(price)]
            service.positions = (position(),) if held else ()
            item = WatchItem(service.resolve("005930"), "Samsung")
            store.save_item(item)
            engine = AutoTrader(service, store, clock=lambda: NOW)
            engine.snapshot(item)
            chart = export_charts(store, root / "charts.json", now=NOW)
            model = SimpleNamespace(
                metadata={"market": "domestic"},
                predict=lambda bars: {"probability_ge_1pct": probability,
                                      "buy_threshold": 0.5, "predicts_gain": probability >= 0.5},
            )

            def position_provider(stock):
                return {**{k: stock[k] for k in ("market", "symbol", "exchange", "currency")},
                        "quantity": "1" if held else "0", "sellable_quantity": "1" if held else "0",
                        "average_price": "100" if held else None, "fetched_at": NOW.isoformat()}

            producer = LSTM30SignalProducer(
                {"domestic": model}, position_provider=position_provider,
                quantity=1, max_krw="1000", max_usd="0", clock=lambda: NOW,
            )
            payload, diagnostics = producer(chart)
            signal = payload["signals"][0]
            assert signal["action"] == expected, (name, payload, diagnostics)
            if exit_field:
                assert signal[exit_field] == ("1" if exit_field == "cost_profit_pct" else "0.8")
            policy = ExternalPolicy(SOURCE_ID, 1, Decimal("1000"), Decimal("0"))
            engine.external_policy, engine.external_only = policy, True
            received = ingest_signals(store, payload, policy, now=NOW)
            assert received["hold" if expected == "hold" else "queued"] == 1
            assert ingest_signals(store, payload, policy, now=NOW)["duplicates"] == 1
            engine.poll()
            assert not engine.orders_enabled and not service.submitted
            # Activate only this in-memory FAKE broker to exercise the real main gates.
            if expected != "hold":
                engine.enable_orders("DEMO_AUTOTRADE")
                engine.poll()
                assert len(service.submitted) == 1, (name, store.events())
                assert service.submitted[0].side.value == expected
            results.append({"case": name, "action": expected, "parser": "passed",
                            "idempotency": "passed", "orders_off_blocks": True,
                            "fake_submissions": len(service.submitted)})
    for market, checkpoint, symbol in (("domestic", args.domestic_checkpoint, "005930"),
                                       ("us", args.us_checkpoint, "AAPL")):
        if checkpoint is None:
            continue
        from dockdack.ml30 import Predictor
        model = Predictor(checkpoint)
        with tempfile.TemporaryDirectory(prefix="lstm30-trained-bridge-") as folder:
            root = Path(folder)
            store = WatchStore(root / "watch.sqlite3")
            service = FakeTradingService()
            item = WatchItem(service.resolve(symbol), symbol)
            store.save_item(item)
            engine = AutoTrader(service, store, clock=lambda: NOW)
            engine.snapshot(item)
            chart = export_charts(store, root / "chart.json", now=NOW)

            def flat_position(stock):
                return {**{k: stock[k] for k in ("market", "symbol", "exchange", "currency")},
                        "quantity": "0", "sellable_quantity": "0", "average_price": None,
                        "fetched_at": NOW.isoformat()}

            producer = LSTM30SignalProducer(
                {market: model}, position_provider=flat_position, quantity=1,
                max_krw="1000", max_usd="1000", clock=lambda: NOW,
            )
            payload, diagnostics = producer(chart)
            assert "prediction" in diagnostics[0], diagnostics
            action = payload["signals"][0]["action"]
            expected = "buy" if diagnostics[0]["prediction"]["predicts_gain"] else "hold"
            assert action == expected
            policy = ExternalPolicy(SOURCE_ID, 1, Decimal("1000"), Decimal("1000"))
            received = ingest_signals(store, payload, policy, now=NOW)
            assert received["hold" if action == "hold" else "queued"] == 1
            assert not engine.orders_enabled and not service.submitted
            results.append({"case": market + "_trained_checkpoint", "action": action,
                            "checkpoint": str(checkpoint.resolve()), "parser": "passed",
                            "prediction": diagnostics[0]["prediction"], "fake_submissions": 0})
    print(json.dumps({"main_source": str(main_source), "checks": results,
                      "network_calls": 0, "real_orders": 0}, indent=2))


if __name__ == "__main__":
    main()
