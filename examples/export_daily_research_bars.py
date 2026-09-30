"""Export selected clean daily bars as read-only pretraining input.

The export is a new research artifact. It never edits the source database,
fills a missing session, or represents a daily bar as an observed minute bar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import NamedTemporaryFile

from dockdack.market_schedule import session_on
from dockdack.models import Market


def _positive_ohlcv(row: sqlite3.Row) -> bool:
    try:
        values = tuple(Decimal(str(row[field])) for field in
                       ("open", "high", "low", "close", "volume"))
    except (InvalidOperation, TypeError, ValueError):
        return False
    op, hi, lo, cl, volume = values
    return (all(value.is_finite() for value in values)
            and min(op, hi, lo, cl) > 0 and volume >= 0
            and hi >= max(op, lo, cl) and lo <= min(op, hi, cl))


def export_daily(*, database: Path, market: Market, exchange: str,
                 symbols: tuple[str, ...], from_date: date, through_date: date,
                 output: Path, now: datetime | None = None) -> dict:
    if not symbols or len(symbols) > 100 or len(set(symbols)) != len(symbols):
        raise ValueError("서로 다른 종목 1~100개를 지정해야 합니다.")
    if not exchange or exchange != exchange.strip():
        raise ValueError("거래소 코드를 지정해야 합니다.")
    if from_date > through_date:
        raise ValueError("시작일이 종료일보다 늦습니다.")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("현재 시각에는 시간대가 필요합니다.")
    database = database.resolve(strict=True)
    output = output.resolve()
    receipt = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() or receipt.exists():
        raise FileExistsError("기존 연구 자료는 덮어쓰지 않습니다.")

    stats = {"source_rows": 0, "quality_flagged_skipped": 0,
             "invalid_ohlcv_skipped": 0, "closed_session_skipped": 0,
             "uncompleted_session_skipped": 0, "accepted_bars": 0}
    rows = []
    uri = database.as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True, timeout=30)) as connection:
        connection.row_factory = sqlite3.Row
        for symbol in symbols:
            result = connection.execute(
                """SELECT trade_date,open,high,low,close,volume,currency,quality_flags
                   FROM daily_bars WHERE symbol=? AND exchange=?
                   AND trade_date BETWEEN ? AND ? ORDER BY trade_date""",
                (symbol, exchange, from_date.isoformat(), through_date.isoformat()),
            )
            for row in result:
                stats["source_rows"] += 1
                try:
                    flags = json.loads(row["quality_flags"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("원본 일봉 품질 표시를 읽을 수 없습니다.") from exc
                if flags:
                    stats["quality_flagged_skipped"] += 1
                    continue
                if not _positive_ohlcv(row):
                    stats["invalid_ohlcv_skipped"] += 1
                    continue
                day = date.fromisoformat(row["trade_date"])
                session = session_on(market, day)
                if session is None:
                    stats["closed_session_skipped"] += 1
                    continue
                if session.closed >= now:
                    stats["uncompleted_session_skipped"] += 1
                    continue
                expected_currency = "KRW" if market is Market.DOMESTIC else "USD"
                if row["currency"] != expected_currency:
                    raise ValueError("일봉 통화가 시장과 일치하지 않습니다.")
                rows.append({
                    "market": market.value, "exchange": exchange,
                    "symbol": symbol, "timestamp": session.closed.isoformat(),
                    "bar_minutes": 1440,
                    **{field: str(row[field]) for field in
                       ("open", "high", "low", "close", "volume")},
                })

    if not rows:
        raise ValueError("내보낼 유효한 완료 일봉이 없습니다.")
    rows.sort(key=lambda row: (row["market"], row["exchange"],
                               row["symbol"], row["timestamp"]))
    stats["accepted_bars"] = len(rows)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    info = {
        "source": "clean_daily_sqlite_read_only", "source_database": str(database),
        "market": market.value, "exchange": exchange, "symbols": list(symbols),
        "from_date": from_date.isoformat(), "through_date": through_date.isoformat(),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "bar_kind": "observed_daily_not_minute", "sha256": hashlib.sha256(payload).hexdigest(),
        **stats,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = None
    try:
        with NamedTemporaryFile(prefix="daily-research-", suffix=".tmp",
                                dir=output.parent, delete=False) as handle:
            stage = Path(handle.name)
            handle.write(payload)
        if output.exists() or receipt.exists():
            raise FileExistsError("출력 경로가 내보내는 중 생성됐습니다.")
        stage.rename(output)
        receipt.write_text(json.dumps(info, ensure_ascii=False, sort_keys=True, indent=2)
                           + "\n", encoding="utf-8")
    finally:
        if stage is not None and stage.exists():
            stage.unlink()
    return info


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--market", choices=("domestic", "us"), required=True)
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--from-date", type=date.fromisoformat, required=True)
    parser.add_argument("--through-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    symbols = tuple(part.strip().upper() for part in args.symbols.split(",") if part.strip())
    info = export_daily(database=args.database, market=Market(args.market),
                        exchange=args.exchange, symbols=symbols,
                        from_date=args.from_date, through_date=args.through_date,
                        output=args.output)
    print(json.dumps({"output": str(args.output.resolve()), **info},
                     ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
