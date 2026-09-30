"""Read-only, same-session paper inference for three daily-to-minute models.

This command reads a previously created transfer-research bundle and a fresh
Kiwoom DEMO minute JSONL snapshot with its matching collector receipt. It
neither loads credentials nor imports a broker, account, order, GUI, or live
trading component. It NEVER places an order.

Example (during the selected exchange's current regular session):

  uv run --extra research python -m examples.paper_minute_signal \
      --bundle path/to/research_bundle --minute path/to/new_5m.jsonl \
      --market domestic --exchange KRX --symbol 005930 \
      --session 2026-09-29 --output path/to/new_paper_report.json
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from dockdack.market_schedule import session_on
from dockdack.models import Market
from dockdack.research.minute_transfer import (
    ARCHITECTURES, MARKET_ZONE, load_bars, load_paper_artifact,
    paper_probability, sha256_file,
)


MAX_SNAPSHOT_AGE = timedelta(minutes=15)
MAX_LAST_BAR_AGE = timedelta(minutes=15)


def _receipt(path: Path, *, data_sha256: str, market: str,
             exchange: str, symbol: str, as_of: datetime,
             raw_rows: int) -> dict:
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    symbols = receipt.get("symbols")
    if (receipt.get("source") != "kiwoom_rest_demo_minute_chart"
            or receipt.get("sha256") != data_sha256
            or receipt.get("market") != market
            or receipt.get("exchange") != exchange
            or receipt.get("interval_minutes") != 5
            or receipt.get("regular_session_filter") is not True
            or type(receipt.get("accepted_bars")) is not int
            or receipt["accepted_bars"] != raw_rows
            or not isinstance(symbols, list) or symbol not in symbols):
        raise ValueError("minute collector receipt does not match the complete regular-session JSONL")
    try:
        collected_at = datetime.fromisoformat(receipt["collected_at_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("minute collector receipt needs a valid collected_at_utc") from exc
    if collected_at.tzinfo is None or collected_at.utcoffset() != timedelta():
        raise ValueError("minute collector receipt collected_at_utc must use UTC")
    if not timedelta() <= as_of - collected_at <= MAX_SNAPSHOT_AGE:
        raise ValueError("minute collector receipt is stale or dated in the future")
    return {"content": receipt, "path": receipt_path,
            "collected_at": collected_at}


def run(args, *, now: datetime | None = None) -> dict:
    as_of = now or datetime.now(timezone.utc)
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("paper clock must be timezone-aware")
    market = Market(args.market)
    if (not isinstance(args.exchange, str) or not args.exchange.strip()
            or args.exchange != args.exchange.strip()
            or not isinstance(args.symbol, str) or not args.symbol.strip()
            or args.symbol != args.symbol.strip()):
        raise ValueError("exchange and symbol must be nonempty trimmed strings")
    if args.exchange == "INDEX":
        raise ValueError("stock transfer model is not an index model")
    try:
        selected_day = date.fromisoformat(args.session)
    except (TypeError, ValueError) as exc:
        raise ValueError("session must be YYYY-MM-DD") from exc
    local_now = as_of.astimezone(ZoneInfo(MARKET_ZONE[market.value]))
    if selected_day != local_now.date():
        raise ValueError("paper signal accepts only the currently ongoing local session")
    session = session_on(market, selected_day)
    if session is None or not session.opened <= as_of < session.closed:
        raise ValueError("selected exchange regular session is not in progress")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing paper report: {output}")

    data = Path(args.minute).resolve(strict=True)
    data_sha256 = sha256_file(data)
    data_stats = {}
    bars = load_bars(data, kind="minute", as_of=as_of, stats=data_stats)
    receipt = _receipt(data, data_sha256=data_sha256, market=market.value,
                       exchange=args.exchange, symbol=args.symbol,
                       as_of=as_of, raw_rows=data_stats["jsonl_rows"])
    matching = tuple(sorted((bar for bar in bars
                             if (bar.market == market.value and bar.exchange == args.exchange
                                 and bar.symbol == args.symbol
                                 and bar.session_date == selected_day)),
                            key=lambda bar: bar.timestamp))
    if not matching:
        raise ValueError("no completed current-session bars for the selected stock")
    last = matching[-1]
    if not timedelta(minutes=5) < as_of - last.timestamp <= MAX_LAST_BAR_AGE:
        raise ValueError("most recent selected bar is incomplete or too stale")

    artifacts = {}
    manifest = None
    for name in ARCHITECTURES:
        artifact, loaded = load_paper_artifact(args.bundle, name)
        if manifest is None:
            manifest = loaded
        elif loaded != manifest:
            raise ValueError("research manifest changed during paper inference")
        artifacts[name] = artifact
    if (manifest["market"] != market.value
            or manifest["study"]["market"] != market.value):
        raise ValueError("paper market differs from the trained research bundle")
    if {"exchange": args.exchange, "symbol": args.symbol} not in manifest["study"]["minute_adaptation_identities"]:
        raise ValueError("paper stock/exchange was not in minute adaptation training")
    if manifest["sources"].get("minute_collector_receipt_verified") is not True:
        raise ValueError("paper inference needs a receipt-verified minute training source")
    if data_sha256 == manifest["sources"]["minute_sha256"]:
        raise ValueError("paper snapshot is identical to the minute training input")
    try:
        training_latest = datetime.fromisoformat(manifest["sources"]["minute_latest_bar_label"])
        bundle_created = datetime.fromisoformat(manifest["created_at_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("research bundle lacks forward-paper timing metadata") from exc
    if (training_latest.tzinfo is None or bundle_created.tzinfo is None
            or training_latest.date() >= selected_day
            or receipt["collected_at"] <= bundle_created):
        raise ValueError("paper snapshot must be from a later session and collection than training")
    lookback = manifest["study"]["lookback"]
    if len(matching) < lookback:
        raise ValueError(f"current session needs {lookback} completed selected bars")
    history = matching[-lookback:]
    predictions = {}
    for name in ARCHITECTURES:
        output_row = paper_probability(artifacts[name], history,
                                       architecture=name, as_of=as_of)
        if output_row["validation_threshold"] is None and output_row["paper_candidate"]:
            raise ValueError("threshold-less model cannot emit a paper candidate")
        predictions[name] = output_row

    report = {
        "format": "dockdack-minute-transfer-paper-v1",
        "generated_at_utc": as_of.astimezone(timezone.utc).isoformat(),
        "market": market.value, "exchange": args.exchange,
        "symbol": args.symbol, "session": selected_day.isoformat(),
        "last_broker_bar_label": last.timestamp.isoformat(),
        "completed_contiguous_history_count": len(history),
        "broker_bar_label_meaning": "start_or_end_unverified",
        "source": {"minute_sha256": data_sha256,
                   "collector_receipt_sha256": sha256_file(receipt["path"]),
                   "collector_receipt_verified": True,
                   "regular_session_excluded": data_stats["regular_session_excluded"]},
        "model_bundle": {"manifest_sha256": sha256_file(Path(args.bundle) / "manifest.json"),
                         "market": manifest["market"],
                         "training_latest_bar_label": training_latest.isoformat()},
        "models": predictions,
        "safety": {"paper_only": True, "deployment_allowed": False,
                   "order_routing_connected": False,
                   "actual_execution_or_profitability_claim": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--minute", type=Path, required=True)
    parser.add_argument("--market", choices=("domestic", "us"), required=True)
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--session", required=True, help="YYYY-MM-DD, current local session only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({"report": str(args.output.resolve()), "market": report["market"],
                      "symbol": report["symbol"], "session": report["session"],
                      "paper_only": True, "models": report["models"]},
                     ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
