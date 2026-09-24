"""Packaged external mark1/mark1.1 worker; no account or order authority."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import socket
import sys
import time


def deny_network(*args, **kwargs):
    raise RuntimeError("External prototype workers have no network/order authority")


def main(argv=None):
    from dockdack.prototype_external import MODEL_IDS, PrototypeWorker, serve
    from dockdack.lstm30_adapter import atomic_json, read_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--stdio", action="store_true", help="JSON request/reply protocol for the GUI")
    parser.add_argument("--chart", type=Path)
    parser.add_argument("--positions", type=Path, help="Verified timestamped positions by watch_id, including explicit zero holdings")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-krw", default="0")
    parser.add_argument("--max-usd", default="0")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=5.)
    args = parser.parse_args(argv)
    if args.interval < .1 or args.interval > 3600:
        parser.error("--interval must be between 0.1 and 3600 seconds")
    if args.stdio and any((args.chart, args.positions, args.output, args.once)):
        parser.error("--stdio cannot be combined with file mode")
    if not args.stdio and not all((args.chart, args.positions, args.output, args.state)):
        parser.error("File mode requires --chart, --positions, --output and --state")
    # Defense in depth, not an operating-system sandbox. No broker/config is
    # instantiated; even accidental Python outbound connection attempts fail.
    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    worker = PrototypeWorker(args.model, bundle_root=args.bundle, state_path=args.state)
    if args.stdio:
        serve(worker)
        return 0
    while True:
        try:
            result = worker.dispatch({"schema_version": 1, "model_id": args.model, "operation": "produce",
                                      "chart": read_json(args.chart), "positions": read_json(args.positions),
                                      "now": datetime.now(timezone.utc).isoformat(),
                                      "max_krw": args.max_krw, "max_usd": args.max_usd})
            atomic_json(args.output, result["payload"])
        except Exception:
            # A stopped standalone writer must not leave yesterday's BUY file.
            atomic_json(args.output, {"schema_version": 1, "source_id": worker.bridge_type.source_id,
                                      "trading_mode": "demo", "signals": []})
            raise
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
