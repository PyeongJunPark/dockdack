"""Build a separate, audited, liquidity-filtered training database.

Sources are opened read-only. The approved sample index, not adjacency after
deleting bad rows, defines the 30 completed bars and next-session target.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

from dockdack.dataset_identity import classify_instrument, resolve_catalog_key
from dockdack.dataset_quality import QualityPolicy, assess_series


SCHEMA_VERSION = "clean-daily-v1"
CATALOG_COLUMNS = ("symbol", "exchange", "name", "english_name", "listing_market",
                   "catalog_market_code", "is_etf", "raw_json", "discovered_at")
BAR_COLUMNS = ("symbol", "exchange", "trade_date", "open", "high", "low", "close",
               "volume", "trade_value", "change", "change_rate", "adjustment_type",
               "adjustment_rate", "currency", "collected_at")


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def open_source(path):
    resolved = Path(path).resolve(strict=True)
    connection = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def source_fingerprints(path):
    """Include durable WAL contents; SHM read marks are not database content."""
    result = {}
    for file in (Path(path), Path(str(path) + "-wal")):
        if not file.exists():
            continue
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[file.name] = {"bytes": file.stat().st_size, "sha256": digest.hexdigest()}
    return result


def load_sessions(market, earliest, as_of):
    import exchange_calendars
    from dockdack.market_schedule import EXTRA_CLOSURES
    from dockdack.models import Market

    name = "XKRX" if market == "domestic" else "XNYS"
    calendar = exchange_calendars.get_calendar(name, start=earliest,
                                              end=(as_of - timedelta(days=1)).isoformat())
    removed = {day.isoformat() for selected, day in EXTRA_CLOSURES if selected == Market(market)}
    days = tuple(day.date().isoformat() for day in calendar.sessions if day.date().isoformat() not in removed)
    return days, {"library": "exchange_calendars", "version": exchange_calendars.__version__,
                  "calendar": name, "first": days[0], "last": days[-1],
                  "extra_closures": sorted(removed),
                  "note": "Historical exchange calendar, including historical KRX Saturdays; not weekday filling."}


def normalized_turnover(value, market):
    try:
        amount = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    amount *= 1_000_000 if market == "domestic" else 1
    return amount if math.isfinite(amount) and amount >= 0 else None


def create_schema(db):
    db.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE instruments(
            symbol TEXT NOT NULL,exchange TEXT NOT NULL,name TEXT NOT NULL,
            english_name TEXT,listing_market TEXT,catalog_market_code TEXT,is_etf INTEGER,
            raw_json TEXT NOT NULL,discovered_at TEXT NOT NULL,PRIMARY KEY(symbol,exchange));
        CREATE TABLE source_instruments AS SELECT * FROM instruments WHERE 0;
        CREATE TABLE instrument_audit(
            symbol TEXT NOT NULL,exchange TEXT NOT NULL,status TEXT NOT NULL,reasons TEXT NOT NULL,
            source_rows INTEGER NOT NULL,valid_rows INTEGER NOT NULL,stored_rows INTEGER NOT NULL,
            input_eligible_endpoints INTEGER NOT NULL,samples INTEGER NOT NULL,
            first_source_date TEXT,last_source_date TEXT,candidate_keys TEXT NOT NULL,
            PRIMARY KEY(symbol,exchange));
        CREATE TABLE daily_bars(
            symbol TEXT NOT NULL,exchange TEXT NOT NULL,trade_date TEXT NOT NULL,
            open TEXT NOT NULL,high TEXT NOT NULL,low TEXT NOT NULL,close TEXT NOT NULL,
            volume INTEGER NOT NULL,trade_value TEXT,change TEXT,change_rate TEXT,
            adjustment_type TEXT,adjustment_rate TEXT,currency TEXT NOT NULL,collected_at TEXT NOT NULL,
            segment_id INTEGER NOT NULL,source_rowid INTEGER NOT NULL,quality_flags TEXT NOT NULL,
            PRIMARY KEY(symbol,exchange,trade_date),
            FOREIGN KEY(symbol,exchange) REFERENCES instruments(symbol,exchange));
        CREATE TABLE training_samples(
            symbol TEXT NOT NULL,exchange TEXT NOT NULL,input_start_date TEXT NOT NULL,
            input_end_date TEXT NOT NULL,target_date TEXT NOT NULL,target_up INTEGER NOT NULL CHECK(target_up IN (0,1)),
            segment_id INTEGER NOT NULL,
            PRIMARY KEY(symbol,exchange,input_end_date),
            FOREIGN KEY(symbol,exchange,input_start_date) REFERENCES daily_bars(symbol,exchange,trade_date),
            FOREIGN KEY(symbol,exchange,input_end_date) REFERENCES daily_bars(symbol,exchange,trade_date),
            FOREIGN KEY(symbol,exchange,target_date) REFERENCES daily_bars(symbol,exchange,trade_date));
        CREATE TABLE bar_audit(
            symbol TEXT NOT NULL,exchange TEXT NOT NULL,trade_date TEXT,
            source_rowid INTEGER NOT NULL,decision TEXT NOT NULL,reasons TEXT NOT NULL,
            PRIMARY KEY(symbol,exchange,source_rowid));
        CREATE TABLE sessions(session_date TEXT PRIMARY KEY,ordinal INTEGER NOT NULL UNIQUE);
    """)


def write_metadata(db, values):
    db.executemany("INSERT OR REPLACE INTO metadata VALUES(?,?)",
                   [(key, json_text(value)) for key, value in values.items()])


def build_database(source_path, output_path, *, market, as_of, policy, session_dates=None, progress_every=100):
    """Refuse overwrite, record every source key, and expose only approved windows."""
    if market not in {"domestic", "us"}:
        raise ValueError("market must be domestic or us")
    if type(as_of) is not date:
        raise ValueError("as_of must be a date")
    if not isinstance(policy, QualityPolicy):
        raise TypeError("policy must be a QualityPolicy")
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError("progress_every must be a positive integer")
    source_path = Path(source_path).resolve(strict=True)
    output_path = Path(output_path).resolve()
    if source_path == output_path or output_path.exists() or output_path.with_suffix(".summary.json").exists():
        raise ValueError("Output must be a new database, never the source or an existing file")
    building = output_path.with_name(output_path.name + ".building")
    if building.exists():
        raise ValueError("An unfinished output exists; choose a fresh output path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with building.open("xb"):
        pass
    started = time.monotonic()
    before = source_fingerprints(source_path)
    source, destination = None, None
    try:
        source = open_source(source_path)
        source.execute("BEGIN")
        source_market = source.execute("SELECT value FROM metadata WHERE key='market'").fetchone()
        if source_market is None or source_market[0] != market:
            raise ValueError("Source database market metadata does not match the requested market")
        catalog_rows = [dict(row) for row in source.execute("SELECT * FROM instruments ORDER BY exchange,symbol")]
        catalog = {(row["exchange"], row["symbol"]): row for row in catalog_rows}
        keys = [tuple(row) for row in source.execute(
            "SELECT symbol,exchange,COUNT(*),MIN(trade_date),MAX(trade_date) FROM daily_bars GROUP BY symbol,exchange")]
        key_set = {(exchange, symbol) for symbol, exchange, *_ in keys}
        if session_dates is None:
            dates = [row[0] for row in source.execute("SELECT DISTINCT trade_date FROM daily_bars ORDER BY trade_date")]
            canonical = []
            for day in dates:
                try:
                    parsed = date.fromisoformat(day)
                    if parsed.isoformat() == day and parsed < as_of:
                        canonical.append(day)
                except (ValueError, TypeError):
                    pass
            if not canonical:
                raise ValueError("No completed canonical dates in source")
            session_dates, calendar_info = load_sessions(market, min(canonical), as_of)
        else:
            session_dates = tuple(session_dates)
            if not session_dates:
                raise ValueError("session_dates must not be empty")
            calendar_info = {"calendar": "provided", "first": session_dates[0], "last": session_dates[-1]}
        # Validate even when there are no classifiable source instruments.
        assess_series((), market=market, as_of=as_of, session_dates=session_dates, policy=policy)
        ordinal = {day: index for index, day in enumerate(session_dates)}
        destination = sqlite3.connect(building)
        destination.execute("PRAGMA journal_mode=DELETE")
        create_schema(destination)
        destination.executemany("INSERT INTO sessions VALUES(?,?)", [(day, i) for day, i in ordinal.items()])
        insert_catalog = "INSERT INTO {} VALUES(" + ",".join("?" for _ in CATALOG_COLUMNS) + ")"
        destination.executemany(insert_catalog.format("source_instruments"),
                                [tuple(row.get(key) for key in CATALOG_COLUMNS) for row in catalog_rows])
        metadata = {
            "schema_version": SCHEMA_VERSION, "build_status": "building", "market": market,
            "source_database": str(source_path), "source_fingerprints": before,
            "as_of_exclusive": as_of.isoformat(), "policy": asdict(policy), "calendar": calendar_info,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target": "next scheduled trading-session close >= input endpoint close * 1.01",
            "requires_training_samples": True,
            "turnover_multiplier": 1_000_000 if market == "domestic" else 1,
            "turnover_basis": "reported daily trade_value; KR million KRW converted to KRW; US USD empirical unit verification",
            "selection_information": "Liquidity thresholds use input-endpoint history only; target requires an observed valid positive-volume next session, but no target turnover threshold or return trimming",
            "limitations": ["Current catalog survivorship and security subtype limitations remain",
                            "Positive-volume observed targets are required; target availability/activity selection remains",
                            "Adjusted-history consistency and all corporate actions are not independently verified",
                            "This is an audited training dataset, not proof of profitability or point-in-time corporate-action availability"],
        }
        write_metadata(destination, metadata)
        totals = Counter(source_catalog_keys=len(catalog), source_bar_keys=len(keys),
                         source_rows=sum(row[2] for row in keys))
        counts, flags, reasons = Counter(), Counter(), Counter()
        for number, (symbol, exchange, source_count, first, last) in enumerate(keys, 1):
            identity_status, candidates, identity_reasons = resolve_catalog_key(symbol, exchange, catalog)
            status, classification_reasons = ("review", identity_reasons)
            item = catalog.get((exchange, symbol))
            if identity_status == "exact":
                status, classification_reasons = classify_instrument(item, market)
            valid_count = kept_count = endpoint_count = sample_count = 0
            if status == "eligible":
                rows = [dict(row) for row in source.execute(
                    "SELECT rowid AS source_rowid,* FROM daily_bars WHERE symbol=? AND exchange=? ORDER BY trade_date",
                    (symbol, exchange))]
                for row in rows:
                    row["date"] = row["trade_date"]
                    row["liquidity_turnover"] = normalized_turnover(row["trade_value"], market)
                assessment = assess_series(rows, market=market, as_of=as_of,
                                           session_dates=session_dates, policy=policy)
                bars = {bar.source_index: bar for bar in assessment.bars}
                valid_positions = {bar.date: bar.source_index for bar in assessment.bars if bar.valid}
                selected = set()
                # Non-session or malformed source rows may sort between valid
                # sessions. Select by the exchange calendar, not raw row span.
                for sample in assessment.samples:
                    first_session = ordinal[sample.input_start_date]
                    target_session = ordinal[sample.target_date]
                    if (target_session - first_session != policy.lookback
                            or ordinal[sample.input_end_date] != target_session - 1):
                        raise AssertionError("Approved sample does not span exactly lookback+1 sessions")
                    selected.update(valid_positions[session_dates[position]]
                                    for position in range(first_session, target_session + 1))
                valid_count = sum(bar.valid for bar in assessment.bars)
                endpoint_count = sum(bar.input_eligible for bar in assessment.bars)
                sample_count = len(assessment.samples)
                kept_count = len(selected)
                totals.update(valid_candidate_rows=valid_count, assessed_candidate_rows=len(rows),
                              input_eligible_endpoints=endpoint_count, samples=sample_count, stored_rows=kept_count)
                segment_ids, segment, previous_ordinal = {}, 0, None
                if selected:
                    destination.execute(insert_catalog.format("instruments"), tuple(item.get(key) for key in CATALOG_COLUMNS))
                    for index in sorted(selected):
                        bar = bars[index]
                        current = ordinal[bar.date]
                        if previous_ordinal is None or current != previous_ordinal + 1:
                            segment += 1
                        segment_ids[index] = segment
                        previous_ordinal = current
                    payload = []
                    for index in sorted(selected):
                        row, bar = rows[index], bars[index]
                        if not bar.valid:
                            raise AssertionError("Approved sample contains a hard-invalid bar")
                        payload.append(tuple(row[key] for key in BAR_COLUMNS) +
                                       (segment_ids[index], row["source_rowid"], json_text(bar.flags)))
                    destination.executemany("INSERT INTO daily_bars VALUES(" + ",".join("?" for _ in range(18)) + ")", payload)
                    sample_rows = []
                    for sample in assessment.samples:
                        if segment_ids[sample.input_start_index] != segment_ids[sample.target_index]:
                            raise AssertionError("Approved sample crosses a stored segment boundary")
                        sample_rows.append((symbol, exchange, sample.input_start_date, sample.input_end_date,
                                            sample.target_date, int(sample.target_up), segment_ids[sample.target_index]))
                    destination.executemany("INSERT INTO training_samples VALUES(?,?,?,?,?,?,?)", sample_rows)
                    status = "retained"
                    totals.update(retained_instruments=1)
                else:
                    status = "no_eligible_samples"
                audit = []
                for index, row in enumerate(rows):
                    bar = bars[index]
                    flags.update(bar.flags)
                    reasons.update(bar.hard_errors)
                    decision = "kept" if index in selected else "hard_invalid" if not bar.valid else "not_in_approved_window"
                    counts.update({decision: 1})
                    if decision != "kept" or bar.flags:
                        detail = tuple(bar.hard_errors) + tuple(bar.flags)
                        if decision == "not_in_approved_window":
                            detail += ("NO_APPROVED_30_PLUS_1_WINDOW",)
                        audit.append((symbol, exchange, row["trade_date"], row["source_rowid"], decision, json_text(detail)))
                destination.executemany("INSERT INTO bar_audit VALUES(?,?,?,?,?,?)", audit)
            else:
                counts.update({"instrument_" + status: source_count})
            destination.execute("INSERT INTO instrument_audit VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (symbol, exchange, status, json_text(classification_reasons), source_count, valid_count,
                 kept_count, endpoint_count, sample_count, first, last, json_text(candidates)))
            destination.commit()
            if number % progress_every == 0 or number == len(keys):
                print(json_text({"market": market, "keys_done": number, "keys_total": len(keys),
                                 "stored_rows": totals["stored_rows"], "samples": totals["samples"],
                                 "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        for (exchange, symbol), row in catalog.items():
            if (exchange, symbol) not in key_set:
                destination.execute("INSERT INTO instrument_audit VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (symbol, exchange, "no_exact_source_bars", json_text(("NO_EXACT_SOURCE_BARS",)),
                     0, 0, 0, 0, 0, None, None, "[]"))
        destination.executescript("""
            CREATE INDEX idx_daily_bars_date ON daily_bars(trade_date);
            CREATE INDEX idx_training_samples_target ON training_samples(target_date);
            CREATE INDEX idx_bar_audit_decision ON bar_audit(decision);
        """)
        if sum(counts.values()) != totals["source_rows"]:
            raise AssertionError("Source row accounting does not balance")
        if destination.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise AssertionError("Cleaned database foreign key check failed")
        if destination.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise AssertionError("Cleaned database integrity check failed")
        source.close()
        source = None
        after = source_fingerprints(source_path)
        if before != after:
            raise RuntimeError("Source changed during build; output is not published as complete")
        summary = {"market": market, "source": str(source_path), "output": str(output_path),
                   "as_of_exclusive": as_of.isoformat(), "policy": asdict(policy), "totals": dict(totals),
                   "row_decisions": dict(counts), "hard_error_occurrences": dict(reasons),
                   "flag_occurrences": dict(flags), "source_unchanged": True,
                   "integrity": "ok", "elapsed_seconds": round(time.monotonic() - started, 1)}
        write_metadata(destination, {"build_status": "complete", "summary": summary})
        destination.commit()
        destination.close()
        destination = None
        if output_path.exists():
            raise FileExistsError("Refusing to overwrite an output created during the build")
        building.rename(output_path)
        output_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--market", choices=("domestic", "us", "all"), default="all")
    parser.add_argument("--as-of", type=date.fromisoformat, required=True, help="Exclude this date and later")
    parser.add_argument("--min-krw", type=float, default=1_000_000_000)
    parser.add_argument("--min-usd", type=float, default=1_000_000)
    parser.add_argument("--min-median-volume", type=float, default=10_000)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(argv)
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    results = []
    for market in ("domestic", "us") if args.market == "all" else (args.market,):
        policy = QualityPolicy(min_median_turnover=args.min_krw if market == "domestic" else args.min_usd,
                               min_median_volume=args.min_median_volume)
        results.append(build_database(args.source_dir / f"{market}_daily.sqlite3",
                                      args.output_dir / f"{market}_daily_clean.sqlite3", market=market,
                                      as_of=args.as_of, policy=policy, progress_every=args.progress_every))
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
