"""Collect a small, read-only Kiwoom DEMO minute-bar research snapshot.

This command never reads an account or sends an order. It preserves the
broker's bar timestamp, excludes a still-forming bar in the broker adapter,
and drops every conflicting duplicate timestamp instead of choosing a price.
The output is a new JSONL research artifact, not an operating chart cache.
US collection currently stops before any API call because the broker's
``cntr_tm`` clock basis, including observed 24-hour-plus values, is unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

from dockdack.broker.kiwoom import KiwoomBroker
from dockdack.config import KiwoomConfig
from dockdack.models import Market, TradingMode


_SESSION_HOURS = {
    Market.DOMESTIC: (ZoneInfo("Asia/Seoul"), time(9, 0), time(15, 30)),
    Market.US: (ZoneInfo("America/New_York"), time(9, 30), time(16, 0)),
}


def _session_membership(market: Market, stamp: datetime,
                        interval_minutes: int) -> str:
    """Keep only a bar safely inside regular wall-clock hours.

    Kiwoom does not specify whether ``cntr_tm`` labels an interval's start
    or end. Both possibilities must lie strictly inside the session. This is
    not a holiday/early-close calendar check.
    """
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("분봉 시각의 시간대가 누락됐습니다.")
    zone, opening, closing = _SESSION_HOURS[market]
    expected = stamp.replace(tzinfo=zone)
    if stamp.utcoffset() != expected.utcoffset():
        raise ValueError("분봉 시각의 UTC 오프셋이 해당 시장 현지 시각과 다릅니다.")
    day = stamp.date()
    start = datetime.combine(day, opening, tzinfo=zone)
    end = datetime.combine(day, closing, tzinfo=zone)
    edge = timedelta(minutes=interval_minutes)
    if stamp < start or stamp > end:
        return "outside"
    if not start + edge < stamp < end - edge:
        return "boundary"
    return "inside"


def _rows(broker, market: Market, exchange: str, symbols: tuple[str, ...],
          interval_minutes: int, max_pages: int, index_code: str | None = None):
    records = []
    outside_count = 0
    boundary_count = 0
    page_counts = {}
    truncated_symbols = []
    pagination_known = True
    def add_bars(symbol, row_exchange, bars):
        nonlocal outside_count, boundary_count, pagination_known
        key = f"INDEX:{symbol}" if row_exchange == "INDEX" else symbol
        page_count = getattr(bars, "page_count", None)
        truncated = getattr(bars, "truncated", None)
        if type(page_count) is not int or type(truncated) is not bool:
            pagination_known = False
        page_counts[key] = page_count
        if truncated is True:
            truncated_symbols.append(key)
        for bar in bars:
            membership = _session_membership(market, bar.timestamp, interval_minutes)
            if membership == "outside":
                outside_count += 1
                continue
            if membership == "boundary":
                boundary_count += 1
                continue
            values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
            if any(value is None or not value.is_finite() for value in values):
                raise ValueError("분봉 가격·거래량이 유한한 값이어야 합니다.")
            records.append({
                "market": market.value, "exchange": row_exchange, "symbol": symbol,
                "timestamp": bar.timestamp.isoformat(), "bar_minutes": interval_minutes,
                "open": str(bar.open), "high": str(bar.high), "low": str(bar.low),
                "close": str(bar.close), "volume": str(bar.volume),
            })

    for symbol in symbols:
        if market is Market.DOMESTIC:
            bars = broker.minute_bars_domestic(
                symbol, exchange=exchange, interval_minutes=interval_minutes,
                max_pages=max_pages)
        else:
            bars = broker.minute_bars_us(
                symbol, exchange=exchange, interval_minutes=interval_minutes,
                max_pages=max_pages)
        add_bars(symbol, exchange, bars)
    if index_code is not None:
        bars = broker.minute_bars_domestic_index(
            index_code, interval_minutes=interval_minutes, max_pages=max_pages)
        add_bars(index_code, "INDEX", bars)
    by_key = {}
    conflicts = set()
    exact_duplicates = 0
    for row in records:
        key = (row["market"], row["exchange"], row["symbol"], row["timestamp"])
        old = by_key.get(key)
        if old is None:
            by_key[key] = row
        elif old == row:
            exact_duplicates += 1
        else:
            conflicts.add(key)
    for key in conflicts:
        by_key.pop(key, None)
    accepted = tuple(sorted(by_key.values(), key=lambda row: (
        row["market"], row["exchange"], row["symbol"], row["timestamp"])))
    return accepted, {"raw_bars": len(records) + outside_count + boundary_count,
                      "outside_regular_hours_dropped": outside_count,
                      "ambiguous_session_boundary_dropped": boundary_count,
                      "chart_page_counts": page_counts,
                      "chart_truncated_symbols": truncated_symbols,
                      "chart_pagination_metadata_known": pagination_known,
                      "chart_truncated": bool(truncated_symbols) if pagination_known else None,
                      "exact_duplicates": exact_duplicates,
                      "conflicting_timestamps_dropped": len(conflicts),
                      "accepted_bars": len(accepted)}


def collect(*, market: Market, exchange: str, symbols: tuple[str, ...],
            interval_minutes: int, max_pages: int, output: Path, broker=None,
            index_code: str | None = None):
    if (len(symbols) > 25 or len(set(symbols)) != len(symbols)
            or (not symbols and index_code is None)):
        raise ValueError("서로 다른 종목 1~25개 또는 국내 업종지수 코드를 지정해야 합니다.")
    if interval_minutes not in {1, 3, 5, 10, 15, 30, 45, 60}:
        raise ValueError("키움이 지원하는 분봉 간격을 지정해야 합니다.")
    if not 1 <= max_pages <= 20:
        raise ValueError("종목당 조회 페이지는 1~20개로 제한합니다.")
    if index_code is not None:
        if market is not Market.DOMESTIC or index_code not in {
                "001", "002", "003", "004", "101", "201", "302", "701"}:
            raise ValueError("업종지수 수집은 국내 시장의 지원 코드만 허용합니다.")
    if (not symbols and exchange != "INDEX") or (symbols and exchange == "INDEX"):
        raise ValueError("지수만 수집할 때는 거래소 INDEX, 주식은 실제 거래소를 지정해야 합니다.")
    output = output.resolve()
    receipt = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() or receipt.exists():
        raise FileExistsError("기존 분봉 연구 자료는 덮어쓰지 않습니다.")
    if broker is None:
        config = KiwoomConfig.from_env(TradingMode.DEMO, market=market)
        broker = KiwoomBroker(config)
    if broker.mode is not TradingMode.DEMO:
        raise ValueError("분봉 연구 수집기는 모의 데이터 조회만 허용합니다.")
    accepted, stats = _rows(broker, market, exchange, symbols, interval_minutes,
                            max_pages, index_code)
    if not accepted:
        raise ValueError("검증 가능한 완료 분봉이 없어 빈 자료를 저장하지 않습니다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode("utf-8")
                       for row in accepted)
    result = {
        "source": "kiwoom_rest_demo_minute_chart", "market": market.value,
        "exchange": exchange, "symbols": list(symbols),
        "index_code": index_code,
        "index_price_basis": ("ka20005_signed_abs_divided_by_100_points; demo_sample_verified_2026-09-29"
                              if index_code else None),
        "interval_minutes": interval_minutes, "max_pages_per_symbol": max_pages,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_meaning": "broker_cntr_tm; conservative complete-bar filter in adapter",
        "regular_session_filter": True,
        "regular_session_filter_basis": "exchange_local_wall_clock_with_one_interval_boundary_exclusion",
        "regular_hours_local": ("09:00-15:30 Asia/Seoul" if market is Market.DOMESTIC
                                else "09:30-16:00 America/New_York"),
        "holiday_and_early_close_calendar_verified": False,
        "sha256": hashlib.sha256(payload).hexdigest(), **stats,
    }
    stage = None
    try:
        with NamedTemporaryFile(prefix="minute-research-", suffix=".jsonl.tmp",
                                dir=output.parent, delete=False) as handle:
            stage = Path(handle.name)
            handle.write(payload)
        if output.exists() or receipt.exists():
            raise FileExistsError("출력 경로가 수집 중 생성됐습니다.")
        stage.rename(output)
        receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
                           + "\n", encoding="utf-8")
    finally:
        if stage is not None and stage.exists():
            stage.unlink()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=("domestic", "us"), required=True)
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--symbols", default="",
                        help="쉼표로 구분한 종목코드/티커, 최대 25개")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--index-code", help="국내 업종지수 분봉을 함께 수집 (예: 201=KOSPI200)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-network", action="store_true",
                        help="모의 분봉 REST 조회를 실제로 실행")
    args = parser.parse_args(argv)
    if not args.allow_network:
        parser.error("네트워크 조회를 실행하려면 --allow-network가 필요합니다.")
    symbols = tuple(part.strip().upper() for part in args.symbols.split(",") if part.strip())
    result = collect(market=Market(args.market), exchange=args.exchange,
                     symbols=symbols, interval_minutes=args.interval_minutes,
                     max_pages=args.max_pages, output=args.output,
                     index_code=args.index_code)
    print(json.dumps({"output": str(args.output.resolve()), **result},
                     ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
