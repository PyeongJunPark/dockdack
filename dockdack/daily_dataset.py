"""Resumable full-universe Kiwoom daily OHLCV dataset collection."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dockdack.config import KiwoomConfig
from dockdack.daily_generations import DailyGenerationStore, GenerationConflict, canonical_json
from dockdack.exceptions import BrokerAPIError
from dockdack.kiwoom import KiwoomBroker, _domestic_daily_bar, _us_daily_bar
from dockdack.models import DailyBar, DomesticExchange, Market, StockInfo, TradingMode, USExchange


# Every domestic catalog category that represents a security supported by the stock APIs.
# Gold spot (80) is excluded because it uses a separate gold chart TR rather than ka10081.
DEFAULT_DOMESTIC_MARKET_CODES = (
    "0",   # KOSPI
    "10",  # KOSDAQ
    "30",  # K-OTC
    "50",  # KONEX
    "60",  # ETN
    "70",  # loss-limited ETN
    "90",  # volatility ETN
    "2",   # infrastructure funds
    "3",   # ELW
    "4",   # mutual funds
    "5",   # warrants
    "6",   # REITs
    "7",   # warrant certificates
    "8",   # ETF
    "9",   # high-yield funds
)

SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    exchange: str
    name: str


@dataclass(frozen=True, slots=True)
class CollectionResult:
    market: Market
    database: Path
    instrument_count: int
    completed: int
    failed: int
    bars: int
    pending_generations: int = 0


class DailyDatasetStore:
    """Published daily history and resumable, separately staged generations."""

    def __init__(self, path: Path, market: Market) -> None:
        self.path = path.resolve()
        self.market = market
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=60)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA busy_timeout=60000")
        self._create_schema()
        self.generations = DailyGenerationStore(self.connection)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DailyDatasetStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS instruments (
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                name TEXT NOT NULL,
                english_name TEXT,
                listing_market TEXT,
                catalog_market_code TEXT,
                is_etf INTEGER,
                raw_json TEXT NOT NULL,
                discovered_at TEXT NOT NULL,
                PRIMARY KEY (symbol, exchange)
            );

            CREATE TABLE IF NOT EXISTS daily_bars (
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open TEXT,
                high TEXT,
                low TEXT,
                close TEXT,
                volume INTEGER,
                trade_value TEXT,
                change TEXT,
                change_rate TEXT,
                adjustment_type TEXT,
                adjustment_rate TEXT,
                currency TEXT NOT NULL,
                collected_at TEXT NOT NULL,
                PRIMARY KEY (symbol, exchange, trade_date),
                FOREIGN KEY (symbol, exchange)
                    REFERENCES instruments(symbol, exchange)
            );

            CREATE INDEX IF NOT EXISTS idx_daily_bars_date
                ON daily_bars(trade_date);

            CREATE TABLE IF NOT EXISTS collection_progress (
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                status TEXT NOT NULL,
                pages_fetched INTEGER NOT NULL DEFAULT 0,
                rows_seen INTEGER NOT NULL DEFAULT 0,
                earliest_date TEXT,
                latest_date TEXT,
                cont_yn TEXT,
                next_key TEXT,
                error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (symbol, exchange),
                FOREIGN KEY (symbol, exchange)
                    REFERENCES instruments(symbol, exchange)
            );
            """
        )
        now = _now()
        values = {
            "schema_version": SCHEMA_VERSION,
            "market": self.market.value,
            "source": "Kiwoom REST API",
            "created_or_opened_at": now,
        }
        self.connection.executemany(
            """
            INSERT INTO metadata(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            values.items(),
        )
        self.connection.commit()

    def save_catalog(self, stocks: Iterable[StockInfo]) -> int:
        now = _now()
        rows: dict[tuple[str, str], tuple[Any, ...]] = {}
        for stock in stocks:
            exchange = "KRX" if self.market is Market.DOMESTIC else stock.exchange
            if not stock.symbol or not exchange:
                continue
            if self.market is Market.US and exchange not in {"NA", "ND", "NY"}:
                # usa06012 officially accepts only AMEX, NASDAQ, and NYSE.
                continue
            raw = dict(stock.raw)
            market_code = raw.get("marketCode") if self.market is Market.DOMESTIC else raw.get("stex_tp")
            rows[(stock.symbol, exchange)] = (
                stock.symbol,
                exchange,
                stock.name,
                stock.english_name,
                stock.exchange,
                str(market_code or ""),
                None if stock.is_etf is None else int(stock.is_etf),
                json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
                now,
            )
        self.connection.executemany(
            """
            INSERT INTO instruments(
                symbol, exchange, name, english_name, listing_market,
                catalog_market_code, is_etf, raw_json, discovered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, exchange) DO UPDATE SET
                name=excluded.name,
                english_name=excluded.english_name,
                listing_market=excluded.listing_market,
                catalog_market_code=excluded.catalog_market_code,
                is_etf=excluded.is_etf,
                raw_json=excluded.raw_json,
                discovered_at=excluded.discovered_at
            """,
            rows.values(),
        )
        self.connection.commit()
        return len(rows)

    def instruments(self, limit: int | None = None) -> tuple[Instrument, ...]:
        sql = "SELECT symbol, exchange, name FROM instruments ORDER BY symbol, exchange"
        parameters: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            parameters = (limit,)
        return tuple(Instrument(**dict(row)) for row in self.connection.execute(sql, parameters))

    def progress(self, instrument: Instrument) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM collection_progress
            WHERE symbol=? AND exchange=?
            """,
            (instrument.symbol, instrument.exchange),
        ).fetchone()

    def start_fresh_pass(self, instrument: Instrument) -> None:
        self.connection.execute(
            """
            INSERT INTO collection_progress(
                symbol, exchange, status, pages_fetched, rows_seen,
                cont_yn, next_key, error, updated_at
            ) VALUES (?, ?, 'running', 0, 0, NULL, NULL, NULL, ?)
            ON CONFLICT(symbol, exchange) DO UPDATE SET
                status='running', cont_yn=NULL, next_key=NULL,
                error=NULL, updated_at=excluded.updated_at
            """,
            (instrument.symbol, instrument.exchange, _now()),
        )
        self.connection.commit()

    def save_page(
        self,
        instrument: Instrument,
        bars: Sequence[DailyBar],
        *,
        has_next: bool,
        cont_yn: str | None,
        next_key: str | None,
    ) -> None:
        now = _now()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.executemany(
                """
                INSERT INTO daily_bars(
                    symbol, exchange, trade_date, open, high, low, close,
                    volume, trade_value, change, change_rate,
                    adjustment_type, adjustment_rate, currency, collected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, exchange, trade_date) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume,
                    trade_value=excluded.trade_value,
                    change=excluded.change,
                    change_rate=excluded.change_rate,
                    adjustment_type=excluded.adjustment_type,
                    adjustment_rate=excluded.adjustment_rate,
                    currency=excluded.currency,
                    collected_at=excluded.collected_at
                """,
                (_bar_row(bar, now) for bar in bars),
            )
            dates = [bar.trade_date.isoformat() for bar in bars]
            previous = self.progress(instrument)
            prior_earliest = str(previous["earliest_date"] or "") if previous else ""
            prior_latest = str(previous["latest_date"] or "") if previous else ""
            earliest = min([value for value in (prior_earliest, *dates) if value], default=None)
            latest = max([value for value in (prior_latest, *dates) if value], default=None)
            status = "running" if has_next else "complete"
            self.connection.execute(
                """
                INSERT INTO collection_progress(
                    symbol, exchange, status, pages_fetched, rows_seen,
                    earliest_date, latest_date, cont_yn, next_key, error, updated_at
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(symbol, exchange) DO UPDATE SET
                    status=excluded.status,
                    pages_fetched=collection_progress.pages_fetched + 1,
                    rows_seen=collection_progress.rows_seen + excluded.rows_seen,
                    earliest_date=excluded.earliest_date,
                    latest_date=excluded.latest_date,
                    cont_yn=excluded.cont_yn,
                    next_key=excluded.next_key,
                    error=NULL,
                    updated_at=excluded.updated_at
                """,
                (
                    instrument.symbol,
                    instrument.exchange,
                    status,
                    len(bars),
                    earliest,
                    latest,
                    cont_yn if has_next else None,
                    next_key if has_next else None,
                    now,
                ),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def mark_partial(self, instrument: Instrument) -> None:
        self.connection.execute(
            """
            UPDATE collection_progress
            SET status='partial', updated_at=?
            WHERE symbol=? AND exchange=?
            """,
            (_now(), instrument.symbol, instrument.exchange),
        )
        self.connection.commit()

    def mark_error(self, instrument: Instrument, message: str) -> None:
        self.connection.execute(
            """
            INSERT INTO collection_progress(
                symbol, exchange, status, error, updated_at
            ) VALUES (?, ?, 'error', ?, ?)
            ON CONFLICT(symbol, exchange) DO UPDATE SET
                status='error', error=excluded.error, updated_at=excluded.updated_at
            WHERE collection_progress.status != 'complete'
            """,
            (instrument.symbol, instrument.exchange, message[:2_000], _now()),
        )
        self.connection.commit()

    def stats(self) -> Mapping[str, int]:
        result = {
            "instruments": int(self.connection.execute("SELECT COUNT(*) FROM instruments").fetchone()[0]),
            "bars": int(self.connection.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]),
            "complete": 0,
            "error": 0,
            "partial": 0,
            "running": 0,
        }
        for row in self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM collection_progress GROUP BY status"
        ):
            result[str(row["status"])] = int(row["count"])
        for row in self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM daily_generations WHERE active=1 GROUP BY status"
        ):
            result[f"staged_{row['status']}"] = int(row["count"])
        return result


def collect_daily_dataset(
    broker: KiwoomBroker,
    market: Market,
    output_dir: Path,
    *,
    domestic_market_codes: Sequence[str] = DEFAULT_DOMESTIC_MARKET_CODES,
    max_symbols: int | None = None,
    max_pages_per_symbol: int | None = None,
    retries: int = 3,
    refresh_complete: bool = False,
) -> CollectionResult:
    database = output_dir.resolve() / f"{market.value}_daily.sqlite3"
    with DailyDatasetStore(database, market) as store:
        print(f"[{market.value}] 종목 목록을 키움에서 조회합니다.", flush=True)
        if market is Market.DOMESTIC:
            stocks = broker.list_domestic_stocks(
                market_codes=domestic_market_codes,
                max_pages=100,
            )
        else:
            stocks = broker.list_us_stocks(exchange=USExchange.ALL, max_pages=100)
        catalog_count = store.save_catalog(stocks)
        instruments = store.instruments(max_symbols)
        print(
            f"[{market.value}] 종목 {catalog_count:,}개 발견, "
            f"이번 실행 대상 {len(instruments):,}개, DB={database}",
            flush=True,
        )

        failed = 0
        completed = 0
        for index, instrument in enumerate(instruments, 1):
            progress = store.progress(instrument)
            is_complete = progress is not None and progress["status"] == "complete"
            if (is_complete and not refresh_complete
                    and not store.generations.pending(instrument.symbol, instrument.exchange)):
                completed += 1
                continue

            try:
                published = _collect_instrument(
                    broker,
                    market,
                    store,
                    instrument,
                    cont_yn=None,
                    next_key=None,
                    known_latest="",
                    max_pages=max_pages_per_symbol,
                    retries=retries,
                )
                if published:
                    completed += 1
                if index == 1 or index % 25 == 0 or index == len(instruments):
                    stats = store.stats()
                    print(
                        f"[{market.value}] {index:,}/{len(instruments):,} 종목, "
                        f"완료 {stats['complete']:,}, 일봉 {stats['bars']:,}, 오류 {stats['error']:,}, "
                        f"별도 수집 중 {sum(value for key, value in stats.items() if key.startswith('staged_')):,}",
                        flush=True,
                    )
            except KeyboardInterrupt:
                print(f"[{market.value}] 사용자 중단: 저장된 다음 페이지부터 재개할 수 있습니다.", flush=True)
                raise
            except Exception as exc:
                failed += 1
                # A failed refresh must not relabel the previous valid publication.
                if not is_complete and not isinstance(exc, GenerationConflict):
                    store.mark_error(instrument, f"{type(exc).__name__}: {exc}")
                print(
                    f"[{market.value}] {instrument.symbol}/{instrument.exchange} 오류: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        stats = store.stats()
        return CollectionResult(
            market=market,
            database=database,
            instrument_count=stats["instruments"],
            completed=stats["complete"],
            failed=failed,
            bars=stats["bars"],
            pending_generations=sum(value for key, value in stats.items() if key.startswith("staged_")),
        )


def _collect_instrument(
    broker: KiwoomBroker,
    market: Market,
    store: DailyDatasetStore,
    instrument: Instrument,
    *,
    cont_yn: str | None,
    next_key: str | None,
    known_latest: str,
    max_pages: int | None,
    retries: int,
) -> bool:
    # Legacy continuation/overlap hints cannot establish an adjustment basis.
    # Resume only a new generation with a verified first-page anchor.
    del cont_yn, next_key, known_latest
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive or None")
    client = broker._http_for(market)
    if market is Market.DOMESTIC:
        api_id = "ka10081"
        path = "/api/dostk/chart"
        body = {
            "stk_cd": instrument.symbol,
            "base_dt": date.today().strftime("%Y%m%d"),
            "upd_stkpc_tp": "1",
        }
        response_key = "stk_dt_pole_chart_qry"
    else:
        if instrument.exchange not in {item.value for item in USExchange if item is not USExchange.ALL}:
            raise ValueError(f"지원하지 않는 미국 거래소 코드: {instrument.exchange!r}")
        api_id = "usa06012"
        path = "/api/us/chart"
        body = {
            "stex_tp": instrument.exchange,
            "stk_cd": instrument.symbol,
            # strt_dt is deliberately omitted: the API then starts at the newest bar.
            "upd_stkpc_tp": "1",
            "exrt_appl_tp": "0",
        }
        response_key = "result_list"

    generations = store.generations
    generation = generations.claim(instrument.symbol, instrument.exchange)
    generation_id = generation["id"]

    def request_page(continuation: str | None = None) -> tuple[tuple[DailyBar, ...], str | None, str]:
        generations.renew(generation_id)
        page = _request_with_retry(
            client,
            api_id=api_id,
            path=path,
            body=body,
            cont_yn="Y" if continuation else None,
            next_key=continuation,
            retries=retries,
        )
        bars = _validated_page(page.body, response_key, instrument, market)
        flag = str(page.cont_yn or "N").strip().upper()
        if flag not in {"Y", "N"}:
            raise ValueError("Invalid continuation flag")
        token = str(page.next_key or "").strip() or None
        if flag == "Y" and (not token or not bars):
            raise ValueError("A continued response requires bars and a nonempty continuation key")
        token = token if flag == "Y" else None
        anchor = canonical_json({"bars": [_bar_row(bar, "") for bar in bars], "has_next": token is not None})
        return bars, token, anchor

    try:
        # Reuse the original domestic reference date across capped runs/midnight.
        if generation["request_json"]:
            body = json.loads(generation["request_json"])
        first_bars, first_token, anchor = request_page()
        if generation["anchor"] is not None and generation["anchor"] != anchor:
            generations.release(generation_id, superseded=True, error="First page changed; full restart")
            generation = generations.claim(instrument.symbol, instrument.exchange)
            generation_id = generation["id"]
        pages_this_run = 0
        if generation["pages"] == 0:
            generations.append(generation_id, [_bar_row(bar, _now()) for bar in first_bars],
                               anchor=anchor, request_json=canonical_json(body), next_key=first_token)
            pages_this_run += 1
        while True:
            generation = generations.get(generation_id)
            if generation["final_page"]:
                # Detect an adjustment revision during traversal before publishing.
                _, _, final_anchor = request_page()
                generations.publish(generation_id, verified_anchor=final_anchor, timestamp=_now())
                return True
            if max_pages is not None and pages_this_run >= max_pages:
                generations.release(generation_id)
                return False
            bars, token, _ = request_page(generation["next_key"])
            generations.append(generation_id, [_bar_row(bar, _now()) for bar in bars],
                               anchor=None, request_json=canonical_json(body), next_key=token)
            pages_this_run += 1
    except BaseException as exc:
        generations.release(generation_id, error=f"{type(exc).__name__}: {exc}")
        raise


def _validated_page(body: Mapping[str, Any], response_key: str,
                    instrument: Instrument, market: Market) -> tuple[DailyBar, ...]:
    """Reject missing/malformed success payloads, never silently discard records."""
    if not isinstance(body, Mapping) or response_key not in body or not isinstance(body[response_key], list):
        raise ValueError(f"키움 응답에 필수 배열 {response_key}가 없습니다.")

    def identity(value: Mapping[str, Any]) -> None:
        if "stk_cd" in value:
            symbol = str(value["stk_cd"]).strip().upper()
            if market is Market.DOMESTIC and len(symbol) == 7 and symbol.startswith("A"):
                symbol = symbol[1:]
            if symbol != instrument.symbol:
                raise ValueError("Chart response symbol does not match the requested instrument")
        if "stex_tp" in value and str(value["stex_tp"]).strip() != instrument.exchange:
            raise ValueError("Chart response exchange does not match the requested instrument")

    identity(body)
    result: list[DailyBar] = []
    seen: dict[date, tuple[Any, ...]] = {}
    volume_field = "trde_qty" if market is Market.DOMESTIC else "acc_trde_qty"
    required = ("open_pric", "high_pric", "low_pric", "cur_prc", volume_field)
    for row in body[response_key]:
        if not isinstance(row, dict):
            raise ValueError("Chart response contains a non-object record")
        identity(row)
        text_date = str(row.get("dt", ""))
        if len(text_date) != 8 or not text_date.isascii() or not text_date.isdigit():
            raise ValueError("Chart record requires an eight-digit trade date")
        for field in required:
            value = row.get(field)
            if value is None or isinstance(value, bool) or not str(value).strip():
                raise ValueError(f"Chart record is missing required field: {field}")
            number = Decimal(str(value).strip().replace(",", ""))
            if not number.is_finite():
                raise ValueError(f"Chart record has a non-finite value: {field}")
            if field == volume_field and (number < 0 or number != number.to_integral_value() or number > 2**63 - 1):
                raise ValueError("Volume must be a nonnegative SQLite-sized integer")
        bar = (_domestic_daily_bar(row, instrument.symbol, DomesticExchange.KRX)
               if market is Market.DOMESTIC else _us_daily_bar(row, instrument.symbol, USExchange(instrument.exchange), False))
        if not (bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high):
            raise ValueError("Chart OHLC prices are inconsistent")
        for value in (bar.trade_value, bar.change, bar.change_rate, bar.adjustment_rate):
            if value is not None and not value.is_finite():
                raise ValueError("Chart optional numeric value is non-finite")
        normalized = _bar_row(bar, "")
        if bar.trade_date in seen:
            if seen[bar.trade_date] != normalized:
                raise ValueError("Conflicting duplicate date in chart response")
            continue
        if result and bar.trade_date >= result[-1].trade_date:
            raise ValueError("Daily chart must be ordered newest to oldest")
        seen[bar.trade_date] = normalized
        result.append(bar)
    return tuple(result)


def _request_with_retry(
    client: Any,
    *,
    api_id: str,
    path: str,
    body: Mapping[str, Any],
    cont_yn: str | None,
    next_key: str | None,
    retries: int,
) -> Any:
    for attempt in range(retries + 1):
        try:
            return client.request(
                api_id=api_id,
                path=path,
                body=body,
                cont_yn=cont_yn,
                next_key=next_key,
            )
        except BrokerAPIError:
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def _bar_row(bar: DailyBar, collected_at: str) -> tuple[Any, ...]:
    return (
        bar.symbol,
        bar.exchange,
        bar.trade_date.isoformat(),
        _decimal_text(bar.open),
        _decimal_text(bar.high),
        _decimal_text(bar.low),
        _decimal_text(bar.close),
        _integer(bar.volume),
        _decimal_text(bar.trade_value),
        _decimal_text(bar.change),
        _decimal_text(bar.change_rate),
        bar.adjustment_type,
        _decimal_text(bar.adjustment_rate),
        bar.currency,
        collected_at,
    )


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _integer(value: Decimal | None) -> int | None:
    return None if value is None else int(value)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _broker_for(markets: Sequence[Market], mode: TradingMode | str | None) -> KiwoomBroker:
    if set(markets) == {Market.DOMESTIC, Market.US}:
        return KiwoomBroker.from_env(mode)
    selected = markets[0]
    config = KiwoomConfig.from_env(mode, market=selected)
    if selected is Market.DOMESTIC:
        return KiwoomBroker(config)
    return KiwoomBroker(config, us_config=config)


def _positive_or_none(value: int, label: str) -> int | None:
    if value < 0:
        raise ValueError(f"{label}은 0 이상이어야 합니다.")
    return value or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="키움 REST API에서 국내/미국 전 종목의 최장 일봉 OHLCV를 수집합니다."
    )
    parser.add_argument("--market", choices=("domestic", "us", "all"), default="all")
    parser.add_argument("--mode", choices=("demo", "real"), default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/kiwoom_daily"))
    parser.add_argument(
        "--domestic-market-codes",
        default=",".join(DEFAULT_DOMESTIC_MARKET_CODES),
        help="쉼표로 구분한 ka10099 시장 코드",
    )
    parser.add_argument("--max-symbols", type=int, default=0, help="0이면 전 종목")
    parser.add_argument(
        "--max-pages-per-symbol",
        type=int,
        default=0,
        help="0이면 연속조회가 끝날 때까지",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--refresh-complete",
        action="store_true",
        help="완료 종목의 전체 수정주가 이력을 별도 수집한 뒤 원자적으로 교체 (부분 수집은 별도 보존)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.retries < 0:
        raise SystemExit("--retries는 0 이상이어야 합니다.")
    try:
        max_symbols = _positive_or_none(args.max_symbols, "--max-symbols")
        max_pages = _positive_or_none(args.max_pages_per_symbol, "--max-pages-per-symbol")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    markets = (
        (Market.DOMESTIC, Market.US)
        if args.market == "all"
        else (Market(args.market),)
    )
    codes = tuple(code.strip() for code in args.domestic_market_codes.split(",") if code.strip())
    if Market.DOMESTIC in markets and not codes:
        raise SystemExit("국내 시장 코드가 하나 이상 필요합니다.")

    broker = _broker_for(markets, args.mode)
    results: list[CollectionResult] = []
    try:
        for market in markets:
            results.append(
                collect_daily_dataset(
                    broker,
                    market,
                    args.output_dir,
                    domestic_market_codes=codes,
                    max_symbols=max_symbols,
                    max_pages_per_symbol=max_pages,
                    retries=args.retries,
                    refresh_complete=args.refresh_complete,
                )
            )
    except KeyboardInterrupt:
        return 130

    for result in results:
        print(
            f"[{result.market.value}] DB={result.database} | 종목 {result.instrument_count:,} | "
            f"공개 완료 {result.completed:,} | 일봉 {result.bars:,} | "
            f"별도 미완료 {result.pending_generations:,} | 이번 실행 오류 {result.failed:,}",
            flush=True,
        )
    return 0 if all(result.failed == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
