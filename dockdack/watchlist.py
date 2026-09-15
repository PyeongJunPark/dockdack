"""Persistent watchlist, immutable one-shot rules, and a durable order-intent journal."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import uuid4

from dockdack.cli import identify_symbol
from dockdack.gui_service import Instrument
from dockdack.history import DailyBar, DailyHistory, day_count
from dockdack.models import Market, OrderSide, Quote, TradingMode


class TriggerKind(str, Enum):
    PRICE_GE = "price_ge"
    PRICE_LE = "price_le"
    SMA_GE = "sma_ge"
    SMA_LE = "sma_le"
    EXTERNAL = "external"


TRIGGER_LABELS = {TriggerKind.PRICE_GE: "현재가 ≥ 지정 가격", TriggerKind.PRICE_LE: "현재가 ≤ 지정 가격",
                  TriggerKind.SMA_GE: "현재가 ≥ N일 이동평균", TriggerKind.SMA_LE: "현재가 ≤ N일 이동평균",
                  TriggerKind.EXTERNAL: "외부 매매 신호"}
STATUS_LABELS = {"ready": "대기", "paused": "비활성", "submitting": "전송 여부 확인 필요",
                 "accepted": "접수 · 체결 대기", "unknown": "접수 여부 확인 필요", "rejected": "거절",
                 "filled": "체결 확인", "cancelled": "취소 확인", "not_sent": "전송 전 중지", "reviewed": "수동 확인 완료",
                 "expired": "신호 만료", "superseded": "새 신호로 대체"}
PENDING = ("submitting", "accepted", "unknown")
MAX_WATCH_ITEMS = 500
EVENT_CATEGORIES = frozenset({"system", "monitor", "signal", "order"})

# Keep classification in SQLite so older processes' untyped INSERTs are indexed
# too. Explicit categories override this fallback in the same write transaction.
_EVENT_CATEGORY_SQL = """CASE
    WHEN e.message LIKE '주문 전송 의도 기록%'
      OR e.message LIKE '접수 · 체결 대기 ·%'
      OR e.message LIKE '접수 여부 확인 필요 ·%'
      OR e.message LIKE '체결 확인 ·%'
      OR e.message LIKE '취소 확인 ·%'
      OR e.message LIKE '거절 ·%'
      OR e.message LIKE '전송 전 중지 ·%'
      OR e.message LIKE '수동 확인 완료 ·%' THEN 'order'
    WHEN e.message LIKE '외부 신호%'
      OR e.message LIKE '규칙 등록%'
      OR e.message LIKE '외부 매매 신호%'
      OR e.message LIKE '현재가 ≥%'
      OR e.message LIKE '현재가 ≤%'
      OR e.message LIKE '자동주문 보류:%'
      OR e.message LIKE '정규장 시간이 아니므로%' THEN 'signal'
    WHEN e.message LIKE '완료 일봉%'
      OR e.message LIKE '차트 JSON 내보내기%'
      OR e.message LIKE '조회/확인 실패:%'
      OR e.message LIKE '시세/차트 조회 완료%' THEN 'monitor'
    ELSE 'system' END"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def positive(value: Decimal, label: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise ValueError(f"{label}은 0보다 큰 유한한 숫자여야 합니다.")
    return value


def instrument_key(instrument: Instrument) -> str:
    market, symbol = identify_symbol(instrument.symbol)
    venues = {"KRX"} if market is Market.DOMESTIC else {"ND", "NY", "NA"}
    if market is not instrument.market or symbol != instrument.symbol or instrument.exchange not in venues:
        raise ValueError("모의투자 관심종목의 시장·종목·거래소가 올바르지 않습니다.")
    return f"{market.value}:{instrument.exchange}:{symbol}"


@dataclass(frozen=True)
class WatchItem:
    instrument: Instrument
    name: str = ""
    days: int = 30

    def __post_init__(self):
        instrument_key(self.instrument)
        day_count(self.days)

    @property
    def id(self) -> str:
        return instrument_key(self.instrument)


@dataclass(frozen=True)
class TriggerRule:
    id: str
    watch_id: str
    kind: TriggerKind
    side: OrderSide
    quantity: int
    max_notional: Decimal
    threshold: Decimal | None = None
    period: int = 20
    status: str = "ready"

    def __post_init__(self):
        if not isinstance(self.kind, TriggerKind) or not isinstance(self.side, OrderSide):
            raise ValueError("트리거 종류와 매수·매도 방향을 확인하세요.")
        if type(self.quantity) is not int or not 1 <= self.quantity <= 999_999_999:
            raise ValueError("주문 수량은 1~999999999의 정수여야 합니다.")
        positive(self.max_notional, "주문금액 상한")
        if self.kind in {TriggerKind.PRICE_GE, TriggerKind.PRICE_LE}:
            positive(self.threshold, "트리거 가격")
        if type(self.period) is not int or not 2 <= self.period <= 999:
            raise ValueError("이동평균 기간은 2~999 거래일이어야 합니다.")
        if self.status not in STATUS_LABELS:
            raise ValueError("알 수 없는 규칙 상태입니다.")

    @classmethod
    def create(cls, watch: WatchItem, kind: str, side: str, quantity: int, max_notional: Decimal,
               threshold: Decimal | None = None, period: int = 20):
        return cls(uuid4().hex, watch.id, TriggerKind(kind), OrderSide(side), quantity,
                   max_notional, threshold, period)

    @property
    def description(self) -> str:
        if self.kind is TriggerKind.EXTERNAL:
            return "외부 매매 신호 · 유효기간 내 1회"
        value = str(self.threshold) if self.kind in {TriggerKind.PRICE_GE, TriggerKind.PRICE_LE} else f"{self.period}일"
        return f"{TRIGGER_LABELS[self.kind]} ({value})"


@dataclass(frozen=True)
class MarketSnapshot:
    quote: Quote
    history: DailyHistory
    fetched_at: datetime


class WatchStore:
    """One connection per operation; transactions also coordinate separate app processes.

    Rules never automatically return to ready after an attempt. The submission intent is
    committed before HTTP, so a crash cannot make a previously sent rule eligible again.
    """

    def __init__(self, path: Path | str, *, seed_defaults: bool = False, mode: TradingMode = TradingMode.DEMO, storage_scope: str | None = None):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = TradingMode(mode)
        self._storage_scope = storage_scope or ("demo" if self._mode is TradingMode.DEMO else "unconfigured")
        # Bind before any schema migration. Untagged legacy data is DEMO only;
        # selecting REAL can never relabel the existing demo order ledger.
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("PRAGMA user_version").fetchone()[0] not in (0, 1, 2, 3):
                raise ValueError("지원하지 않는 관심종목 DB 버전입니다. 원본 파일을 보존하세요.")
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'app_environment' in tables:
                bound = db.execute("SELECT mode FROM app_environment WHERE singleton=1").fetchone()
                if not bound or bound[0] != self._mode.value:
                    raise ValueError("저장소의 모의/실전 환경이 선택한 환경과 다릅니다. 기록을 혼합하지 않습니다.")
                columns = {row[1] for row in db.execute('PRAGMA table_info(app_environment)')}
                if 'scope' not in columns:
                    if self._mode is TradingMode.REAL:
                        raise ValueError("실전 저장소의 인증 범위가 확인되지 않습니다.")
                    db.execute("ALTER TABLE app_environment ADD COLUMN scope TEXT NOT NULL DEFAULT 'demo'")
                if db.execute("SELECT scope FROM app_environment WHERE singleton=1").fetchone()[0] != self._storage_scope:
                    raise ValueError("저장소와 현재 API 인증 범위가 다릅니다. 서로 다른 계정 기록을 혼합하지 않습니다.")
            else:
                if tables and self._mode is TradingMode.REAL:
                    raise ValueError("환경 정보가 없는 기존 기록은 실전 저장소로 사용할 수 없습니다.")
                db.execute("CREATE TABLE app_environment (singleton INTEGER PRIMARY KEY CHECK(singleton=1), mode TEXT NOT NULL CHECK(mode IN ('demo','real')), scope TEXT NOT NULL)")
                db.execute("INSERT INTO app_environment VALUES (1,?,?)", (self._mode.value, self._storage_scope))
        with self.connection() as db:
            if db.execute("PRAGMA user_version").fetchone()[0] not in (0, 1, 2, 3):
                raise ValueError("지원하지 않는 관심종목 DB 버전입니다. 원본 파일을 보존하세요.")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    id TEXT PRIMARY KEY, market TEXT NOT NULL, symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL, name TEXT NOT NULL, days INTEGER NOT NULL, active INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE IF NOT EXISTS rules (
                    id TEXT PRIMARY KEY, watch_id TEXT NOT NULL REFERENCES watchlist(id), kind TEXT NOT NULL,
                    side TEXT NOT NULL, quantity INTEGER NOT NULL, max_notional TEXT NOT NULL,
                    threshold TEXT, period INTEGER NOT NULL, status TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                    rule_id TEXT PRIMARY KEY REFERENCES rules(id), watch_id TEXT NOT NULL,
                    status TEXT NOT NULL, price TEXT NOT NULL, started_at TEXT NOT NULL,
                    order_number TEXT NOT NULL DEFAULT '', message TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, symbol TEXT NOT NULL, message TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS event_categories (
                    event_id INTEGER PRIMARY KEY REFERENCES events(id),
                    category TEXT NOT NULL CHECK(category IN ('system','monitor','signal','order')));
                CREATE TABLE IF NOT EXISTS event_category_index (
                    event_id INTEGER PRIMARY KEY REFERENCES events(id),
                    category TEXT NOT NULL CHECK(category IN ('system','monitor','signal','order')));
                CREATE INDEX IF NOT EXISTS event_category_index_by_category ON event_category_index(category,event_id DESC);
                CREATE TABLE IF NOT EXISTS order_execution_snapshots (
                    rule_id TEXT PRIMARY KEY REFERENCES attempts(rule_id),
                    filled_quantity TEXT NOT NULL, remaining_quantity TEXT NOT NULL,
                    fill_price TEXT, observed_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS order_fill_recovery (
                    rule_id TEXT PRIMARY KEY REFERENCES attempts(rule_id),
                    status TEXT NOT NULL, message TEXT NOT NULL, checked_at TEXT NOT NULL,
                    source_api TEXT NOT NULL DEFAULT '', price_basis TEXT NOT NULL DEFAULT '',
                    order_date TEXT NOT NULL DEFAULT '', order_time TEXT NOT NULL DEFAULT '',
                    fill_time TEXT NOT NULL DEFAULT '', reported_fill_price TEXT);
                CREATE TABLE IF NOT EXISTS snapshots (watch_id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS turnover_ranks (
                    market TEXT NOT NULL, rank INTEGER NOT NULL, watch_id TEXT NOT NULL,
                    turnover TEXT NOT NULL, currency TEXT NOT NULL, fetched_at TEXT NOT NULL,
                    PRIMARY KEY(market, rank));
                CREATE INDEX IF NOT EXISTS turnover_ranks_by_watch ON turnover_ranks(watch_id);
                CREATE TABLE IF NOT EXISTS chart_exports (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, path TEXT NOT NULL, digest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chart_export_members (
                    export_id TEXT NOT NULL REFERENCES chart_exports(id), watch_id TEXT NOT NULL,
                    PRIMARY KEY(export_id, watch_id));
                CREATE TABLE IF NOT EXISTS external_signals (
                    source_id TEXT NOT NULL, signal_id TEXT NOT NULL, payload TEXT NOT NULL,
                    rule_id TEXT UNIQUE REFERENCES rules(id), watch_id TEXT NOT NULL,
                    generated_at TEXT NOT NULL, expires_at TEXT NOT NULL, export_id TEXT NOT NULL,
                    received_at TEXT NOT NULL, decision TEXT NOT NULL, status TEXT NOT NULL,
                    PRIMARY KEY(source_id, signal_id));
                CREATE INDEX IF NOT EXISTS external_by_watch ON external_signals(watch_id, source_id);
                CREATE TABLE IF NOT EXISTS managed_watchlist (watch_id TEXT PRIMARY KEY REFERENCES watchlist(id));
                CREATE TABLE IF NOT EXISTS history_cache (watch_id TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS test_decisions (export_id TEXT NOT NULL, watch_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(export_id,watch_id));
                CREATE TABLE IF NOT EXISTS ranking_runs (
                    market TEXT NOT NULL, slot TEXT NOT NULL, status TEXT NOT NULL,
                    attempts INTEGER NOT NULL, attempted_at TEXT NOT NULL, detail TEXT NOT NULL,
                    PRIMARY KEY(market,slot));
                PRAGMA user_version=3;
            """)
            db.execute(f"""CREATE TRIGGER IF NOT EXISTS classify_inserted_event
                       AFTER INSERT ON events BEGIN
                           INSERT OR IGNORE INTO event_category_index(event_id,category)
                           VALUES(NEW.id,{_EVENT_CATEGORY_SQL.replace('e.message', 'NEW.message')});
                       END""")
            for operation in ("INSERT", "UPDATE"):
                db.execute(f"""CREATE TRIGGER IF NOT EXISTS index_event_category_{operation.lower()}
                           AFTER {operation} ON event_categories BEGIN
                               INSERT INTO event_category_index(event_id,category) VALUES(NEW.event_id,NEW.category)
                               ON CONFLICT(event_id) DO UPDATE SET category=excluded.category;
                           END""")
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM settings WHERE key='event_category_index_v1'").fetchone():
                # Add only missing metadata; keep all original rows and explicit categories.
                db.execute(f"""INSERT OR IGNORE INTO event_category_index(event_id,category)
                           SELECT e.id,COALESCE(c.category,{_EVENT_CATEGORY_SQL}) FROM events e
                           LEFT JOIN event_categories c ON c.event_id=e.id""")
                db.execute("INSERT INTO settings VALUES('event_category_index_v1','1')")
            if not db.execute("SELECT 1 FROM settings WHERE key='initialized'").fetchone():
                if seed_defaults:
                    for market, symbol, exchange, name in (
                        (Market.DOMESTIC, "005930", "KRX", "삼성전자"), (Market.DOMESTIC, "000660", "KRX", "SK하이닉스"),
                        (Market.US, "AAPL", "ND", "애플"), (Market.US, "GOOGL", "ND", "알파벳 A"),
                    ):
                        item = WatchItem(Instrument(market, symbol, exchange), name)
                        db.execute("INSERT OR IGNORE INTO watchlist VALUES (?, ?, ?, ?, ?, ?, 1)",
                                   (item.id, market.value, symbol, exchange, name, item.days))
                db.execute("INSERT INTO settings VALUES ('initialized', '1')")

    @property
    def mode(self) -> TradingMode:
        return self._mode

    @property
    def storage_scope(self) -> str:
        return self._storage_scope

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def items(self) -> tuple[WatchItem, ...]:
        with self.connection() as db:
            return tuple(WatchItem(Instrument(Market(r["market"]), r["symbol"], r["exchange"]), r["name"], r["days"])
                         for r in db.execute("SELECT * FROM watchlist WHERE active=1 ORDER BY rowid"))

    def chart_export_rows(self, watch_ids: set[str] | None = None) -> tuple[dict, ...]:
        """One consistent read for export metadata and only the requested snapshots.

        Raw snapshot JSON is decoded by the exporter per stock, preserving its
        existing error isolation (one corrupt cache does not abort the whole file).
        """
        if watch_ids is not None:
            if not isinstance(watch_ids, (set, frozenset)) or any(not isinstance(key, str) for key in watch_ids):
                raise ValueError("내보낼 관심종목 ID는 문자열 집합이어야 합니다.")
            if not watch_ids:
                return ()
        sql = """SELECT w.*,s.data AS snapshot_data,h.fetched_at AS history_fetched_at,
                        t.rank AS turnover_rank,t.turnover,t.fetched_at AS ranking_fetched_at
                 FROM watchlist w LEFT JOIN snapshots s ON s.watch_id=w.id
                 LEFT JOIN history_cache h ON h.watch_id=w.id
                 LEFT JOIN turnover_ranks t ON t.watch_id=w.id AND t.market=w.market
                 WHERE w.active=1"""
        params = ()
        if watch_ids is not None:
            params = tuple(sorted(watch_ids))
            sql += " AND w.id IN (" + ",".join("?" for _ in params) + ")"
        with self.connection() as db:
            return tuple(dict(row) for row in db.execute(sql + " ORDER BY w.rowid", params))

    def save_item(self, item: WatchItem):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT active FROM watchlist WHERE id=?", (item.id,)).fetchone()
            count = db.execute("SELECT COUNT(*) FROM watchlist WHERE active=1").fetchone()[0]
            if (not current or not current[0]) and count >= MAX_WATCH_ITEMS:
                raise ValueError(f"관심종목은 최대 {MAX_WATCH_ITEMS}개입니다. 조회 간격과 API 호출 제한을 고려하세요.")
            db.execute("""INSERT INTO watchlist VALUES (?, ?, ?, ?, ?, ?, 1)
                       ON CONFLICT(id) DO UPDATE SET name=excluded.name, days=excluded.days, active=1""",
                       (item.id, item.instrument.market.value, item.instrument.symbol, item.instrument.exchange, item.name, item.days))
            # Explicit user edits pin a stock; legacy stocks are also preserved on migration.
            db.execute("DELETE FROM managed_watchlist WHERE watch_id=?", (item.id,))

    def add_ranked(self, rankings, days: int = 30):
        day_count(days)
        items = [(WatchItem(Instrument(r.market, r.symbol, r.exchange), r.name, days), r) for r in rankings]
        if not items or len({item.id for item, _ in items}) != len(items):
            raise ValueError("순위 목록이 비어 있거나 종목이 중복됩니다.")
        now = utc_now().isoformat()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            active = {r[0] for r in db.execute("SELECT id FROM watchlist WHERE active=1")}
            if len(active | {item.id for item, _ in items}) > MAX_WATCH_ITEMS:
                raise ValueError(f"추가 후 관심종목이 {MAX_WATCH_ITEMS}개를 넘습니다. 기존 종목을 먼저 정리하세요.")
            # Refresh only ranking membership. Existing interests, N, rules and order history survive.
            for market in {r.market.value for _, r in items}:
                db.execute("DELETE FROM turnover_ranks WHERE market=?", (market,))
            for item, rank in items:
                new = not db.execute("SELECT 1 FROM watchlist WHERE id=?", (item.id,)).fetchone()
                db.execute("""INSERT INTO watchlist VALUES (?, ?, ?, ?, ?, ?, 1)
                           ON CONFLICT(id) DO UPDATE SET name=excluded.name, active=1""",
                           (item.id, item.instrument.market.value, item.instrument.symbol, item.instrument.exchange, item.name, item.days))
                if new:
                    db.execute("INSERT INTO managed_watchlist VALUES (?)", (item.id,))
                db.execute("INSERT INTO turnover_ranks VALUES (?, ?, ?, ?, ?, ?)",
                           (rank.market.value, rank.rank, item.id, str(rank.turnover), rank.currency, now))
            self._insert_event(db, "SYSTEM", f"시장별 거래대금 상위 {len(items)}종목 추가/갱신 · 기존 관심종목 유지",
                               category="system", at=datetime.fromisoformat(now))

    def replace_ranked(self, market: Market, rankings, protected_symbols: set[str], days: int = 30):
        pairs = [(WatchItem(Instrument(r.market, r.symbol, r.exchange), r.name, days), r) for r in rankings]
        if len(pairs) != 100 or len({i.id for i, _ in pairs}) != 100 or any(r.market is not market for _, r in pairs):
            raise ValueError("해당 시장의 서로 다른 100종목이 필요합니다.")
        incoming = {i.id for i, _ in pairs}
        now = utc_now().isoformat()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            removable = []
            for row in db.execute("SELECT w.* FROM watchlist w JOIN managed_watchlist m ON m.watch_id=w.id WHERE w.active=1 AND w.market=?", (market.value,)):
                if row["id"] in incoming or row["symbol"] in protected_symbols:
                    continue
                if db.execute("SELECT 1 FROM attempts WHERE watch_id=? AND status IN ('submitting','unknown','accepted')", (row["id"],)).fetchone():
                    continue
                if db.execute("SELECT 1 FROM rules WHERE watch_id=? AND status='ready' AND kind!='external'", (row["id"],)).fetchone():
                    continue
                removable.append(row["id"])
            active = {r[0] for r in db.execute("SELECT id FROM watchlist WHERE active=1")}
            if len((active - set(removable)) | incoming) > MAX_WATCH_ITEMS:
                raise ValueError("보호 종목을 포함한 관심목록이 500개를 넘습니다. 기존 목록을 유지합니다.")
            for key in removable:
                db.execute("UPDATE watchlist SET active=0 WHERE id=?", (key,))
                db.execute("UPDATE rules SET status='paused' WHERE watch_id=? AND status='ready'", (key,))
                db.execute("UPDATE external_signals SET status='paused' WHERE watch_id=? AND rule_id IN (SELECT id FROM rules WHERE status='paused')", (key,))
            db.execute("DELETE FROM turnover_ranks WHERE market=?", (market.value,))
            for item, rank in pairs:
                new = not db.execute("SELECT 1 FROM watchlist WHERE id=?", (item.id,)).fetchone()
                db.execute("""INSERT INTO watchlist VALUES (?, ?, ?, ?, ?, ?, 1)
                           ON CONFLICT(id) DO UPDATE SET name=excluded.name,active=1""",
                           (item.id, market.value, item.instrument.symbol, item.instrument.exchange, item.name, item.days))
                if new:
                    db.execute("INSERT INTO managed_watchlist VALUES (?)", (item.id,))
                db.execute("INSERT INTO turnover_ranks VALUES (?, ?, ?, ?, ?, ?)",
                           (market.value, rank.rank, item.id, str(rank.turnover), rank.currency, now))
            self._insert_event(db, "SYSTEM",
                               f"{market.value} 정시 TOP100 재선정 · 자동등록 순위이탈 {len(removable)}개 비활성 · 수동/보호 종목 유지",
                               category="system", at=datetime.fromisoformat(now))

    def restrict_to_common(self, market: Market, eligible: set[tuple[str, str]],
                           protected_symbols: set[str]) -> dict[str, int]:
        """Apply a verified common-stock universe, preserving history and order intents.

        This explicit restriction includes legacy/manual interests. Ineligible held or
        pending stocks remain visible for reconciliation, but all their ready rules
        are paused too. Callers must obtain complete eligibility/protection data first;
        this method neither queries a broker nor cancels/submits any order.
        """
        if not isinstance(market, Market) or not isinstance(eligible, (set, frozenset)) or not eligible:
            raise ValueError("보통주 분류 결과가 비어 있거나 시장/목록 형식이 올바르지 않습니다.")
        if not isinstance(protected_symbols, (set, frozenset)) or any(not isinstance(s, str) or not s for s in protected_symbols):
            raise ValueError("보호 종목 목록을 확인할 수 없어 기존 관심종목을 유지합니다.")
        for key in eligible:
            if not isinstance(key, tuple) or len(key) != 2 or any(not isinstance(v, str) for v in key):
                raise ValueError("보통주 목록은 (종목코드, 거래소) 쌍이어야 합니다.")
            instrument_key(Instrument(market, key[0], key[1]))
        counts = {"eligible": 0, "removed": 0, "protected": 0, "paused_rules": 0, "paused_signals": 0}
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT * FROM watchlist WHERE active=1 AND market=?", (market.value,)).fetchall()
            for row in rows:
                if (row["symbol"], row["exchange"]) in eligible:
                    counts["eligible"] += 1
                    continue
                key = row["id"]
                # Pause linked signals before changing the rules, so historical
                # filled/accepted/expired signals retain their original status.
                counts["paused_signals"] += db.execute(
                    """UPDATE external_signals SET status='paused' WHERE watch_id=? AND status!='paused'
                       AND rule_id IN (SELECT id FROM rules WHERE watch_id=? AND status='ready')""",
                    (key, key),
                ).rowcount
                counts["paused_rules"] += db.execute(
                    "UPDATE rules SET status='paused' WHERE watch_id=? AND status='ready'", (key,),
                ).rowcount
                pending = db.execute(
                    "SELECT 1 FROM attempts WHERE watch_id=? AND status IN ('submitting','unknown','accepted')", (key,),
                ).fetchone()
                if row["symbol"] in protected_symbols or pending:
                    counts["protected"] += 1
                else:
                    db.execute("UPDATE watchlist SET active=0 WHERE id=?", (key,))
                    counts["removed"] += 1
            self._insert_event(db, "SYSTEM",
                f"{market.value} 보통주 제한 적용 · 유지 {counts['eligible']}개 · 제외 {counts['removed']}개 · "
                f"보유/미확정 보호 {counts['protected']}개 · 규칙 {counts['paused_rules']}개 / 신호 {counts['paused_signals']}개 비활성 · 이력 보존",
                category="system")
        return counts

    def adopt_ranked_management(self, market: Market | None = None) -> int:
        """Explicitly adopt active ranked legacy interests, never unrelated manuals."""
        if market is not None and not isinstance(market, Market):
            raise ValueError("자동 순위관리로 전환할 시장을 확인하세요.")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            adopted = db.execute(
                """INSERT OR IGNORE INTO managed_watchlist(watch_id)
                   SELECT w.id FROM watchlist w JOIN turnover_ranks t ON t.watch_id=w.id AND t.market=w.market
                   WHERE w.active=1""" + (" AND w.market=?" if market is not None else ""),
                (market.value,) if market is not None else (),
            ).rowcount
            self._insert_event(db, "SYSTEM",
                f"{market.value if market is not None else '전체 시장'} 기존 순위 종목 {adopted}개 자동 재선정 관리로 전환 · 순위 외 수동종목 유지",
                category="system")
        return adopted

    def claim_ranking_run(self, market: Market, slot: str, now: datetime) -> bool:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM ranking_runs WHERE market=? AND slot=?", (market.value, slot)).fetchone()
            if row and (row["status"] != "failed" or row["attempts"] >= 3 or
                        (now - datetime.fromisoformat(row["attempted_at"])).total_seconds() < 60):
                return False
            db.execute("""INSERT INTO ranking_runs VALUES(?,?,'running',1,?,'') ON CONFLICT(market,slot)
                       DO UPDATE SET status='running',attempts=attempts+1,attempted_at=excluded.attempted_at,detail=''""",
                       (market.value, slot, now.isoformat()))
            return True

    def finish_ranking_run(self, market: Market, slot: str, status: str, detail: str):
        if status not in {"done", "failed"}:
            raise ValueError("잘못된 재선정 상태")
        with self.connection() as db:
            db.execute("UPDATE ranking_runs SET status=?,detail=? WHERE market=? AND slot=?", (status, detail, market.value, slot))

    def rankings(self):
        with self.connection() as db:
            return tuple(dict(r) for r in db.execute("SELECT * FROM turnover_ranks ORDER BY market, rank"))

    def external_for_rule(self, rule_id: str):
        with self.connection() as db:
            row = db.execute("SELECT * FROM external_signals WHERE rule_id=?", (rule_id,)).fetchone()
            return dict(row) if row else None

    def expire_external(self, now: datetime):
        with self.connection() as db:
            rows = db.execute("""SELECT e.rule_id,e.expires_at FROM external_signals e JOIN rules r ON r.id=e.rule_id
                               WHERE r.status='ready'""").fetchall()
            for row in rows:
                if datetime.fromisoformat(row["expires_at"]) <= now:
                    db.execute("UPDATE rules SET status='expired' WHERE id=? AND status='ready'", (row["rule_id"],))
                    db.execute("UPDATE external_signals SET status='expired' WHERE rule_id=?", (row["rule_id"],))

    def remove_item(self, watch_id: str):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM attempts WHERE watch_id=? AND status IN ('submitting','accepted','unknown')", (watch_id,)).fetchone():
                raise ValueError("미확정/미체결 주문이 있습니다. 주문 내역을 확인하고 차단을 해제한 뒤 종목을 제외하세요.")
            db.execute("UPDATE watchlist SET active=0 WHERE id=?", (watch_id,))
            db.execute("UPDATE rules SET status='paused' WHERE watch_id=? AND status='ready'", (watch_id,))
            db.execute("UPDATE external_signals SET status='paused' WHERE watch_id=? AND rule_id IN (SELECT id FROM rules WHERE status='paused')", (watch_id,))
        self.event(watch_id, "관심종목 제외 · 규칙 비활성화 · 주문 이력 보존", category="system")

    @staticmethod
    def _rule(row) -> TriggerRule:
        return TriggerRule(row["id"], row["watch_id"], TriggerKind(row["kind"]), OrderSide(row["side"]),
                           row["quantity"], Decimal(row["max_notional"]),
                           Decimal(row["threshold"]) if row["threshold"] is not None else None,
                           row["period"], row["status"])

    def rules(self, watch_id: str | None = None) -> tuple[TriggerRule, ...]:
        with self.connection() as db:
            sql = "SELECT r.* FROM rules r JOIN watchlist w ON r.watch_id=w.id WHERE w.active=1"
            params = () if watch_id is None else (watch_id,)
            return tuple(self._rule(row) for row in db.execute(sql + (" AND watch_id=?" if params else "") + " ORDER BY r.rowid", params))

    def add_rule(self, rule: TriggerRule):
        if rule.status != "ready":
            raise ValueError("새 규칙은 대기 상태여야 합니다.")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM watchlist WHERE id=? AND active=1", (rule.watch_id,)).fetchone():
                raise ValueError("관심종목을 먼저 등록하세요.")
            if db.execute("SELECT COUNT(*) FROM rules WHERE watch_id=? AND status='ready'", (rule.watch_id,)).fetchone()[0] >= 10:
                raise ValueError("종목별 대기 규칙은 최대 10개입니다.")
            for row in db.execute("SELECT * FROM rules WHERE watch_id=? AND status='ready'", (rule.watch_id,)):
                existing = self._rule(row)
                if (existing.kind, existing.side, existing.quantity, existing.max_notional, existing.threshold, existing.period) == (
                        rule.kind, rule.side, rule.quantity, rule.max_notional, rule.threshold, rule.period):
                    raise ValueError("동일한 대기 규칙이 이미 있습니다.")
            db.execute("INSERT INTO rules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (rule.id, rule.watch_id, rule.kind.value, rule.side.value, rule.quantity,
                        str(rule.max_notional), str(rule.threshold) if rule.threshold is not None else None, rule.period, rule.status))
        self.event(rule.watch_id, f"규칙 등록 · {rule.description} · {rule.side.value} {rule.quantity}주 · 주문금액 상한 {rule.max_notional}", category="signal")

    def pause_rule(self, rule_id: str):
        with self.connection() as db:
            db.execute("UPDATE rules SET status='paused' WHERE id=? AND status='ready'", (rule_id,))
            db.execute("UPDATE external_signals SET status='paused' WHERE rule_id=? AND rule_id IN (SELECT id FROM rules WHERE status='paused')", (rule_id,))

    def attempts(self, watch_id: str | None = None, *, pending_only=False) -> tuple[dict, ...]:
        with self.connection() as db:
            sql, params = "SELECT * FROM attempts WHERE 1=1", []
            if watch_id is not None:
                sql += " AND watch_id=?"
                params.append(watch_id)
            if pending_only:
                sql += " AND status IN ('submitting', 'accepted', 'unknown')"
            return tuple(dict(row) for row in db.execute(sql + " ORDER BY rowid", params))

    def claim(self, rule: TriggerRule, price: Decimal, now: datetime) -> bool:
        positive(price, "주문 단가")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM rules WHERE id=? AND status='ready'", (rule.id,)).fetchone()
            if not row or self._rule(row) != rule:
                return False
            if not db.execute("SELECT 1 FROM watchlist WHERE id=? AND active=1", (rule.watch_id,)).fetchone():
                return False
            if db.execute("SELECT 1 FROM attempts WHERE watch_id=? AND status IN ('submitting','accepted','unknown')", (rule.watch_id,)).fetchone():
                return False
            db.execute("INSERT INTO attempts (rule_id, watch_id, status, price, started_at) VALUES (?, ?, 'submitting', ?, ?)",
                       (rule.id, rule.watch_id, str(price), now.isoformat()))
            db.execute("UPDATE rules SET status='submitting' WHERE id=?", (rule.id,))
            db.execute("UPDATE external_signals SET status='submitting' WHERE rule_id=?", (rule.id,))
            self._insert_event(db, rule.watch_id,
                               f"주문 전송 의도 기록 · {rule.id} · {rule.side.value} {rule.quantity}주 · 참조 현재가 {price} (체결가 아님)",
                               category="order", at=now)
        return True

    def finish(self, rule_id: str, status: str, message: str, order_number: str = ""):
        allowed = {"accepted": ("submitting",), "unknown": ("submitting",), "rejected": ("submitting",),
                   "not_sent": ("submitting",), "filled": ("accepted",), "cancelled": ("accepted",), "reviewed": PENDING}
        if status not in allowed:
            raise ValueError("허용하지 않는 주문 상태 변경입니다.")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT * FROM attempts WHERE rule_id=?", (rule_id,)).fetchone()
            if not current or current["status"] not in allowed[status]:
                raise ValueError("이미 처리되었거나 변경할 수 없는 주문 상태입니다.")
            db.execute("UPDATE attempts SET status=?, message=?, order_number=? WHERE rule_id=?",
                       (status, message, order_number or current["order_number"], rule_id))
            db.execute("UPDATE rules SET status=? WHERE id=?", (status, rule_id))
            db.execute("UPDATE external_signals SET status=? WHERE rule_id=?", (status, rule_id))
            self._insert_event(db, current["watch_id"], f"{STATUS_LABELS[status]} · {message}", category="order")

    def mark_reviewed(self, rule_id: str, confirmation: str):
        if confirmation != "CHECKED_ORDER_HISTORY":
            raise ValueError("영웅문 주문·체결 내역을 직접 확인해야 차단을 해제할 수 있습니다.")
        self.finish(rule_id, "reviewed", "사용자가 주문·체결 내역을 확인하고 종목 차단을 해제함 · 기존 규칙은 재실행되지 않음")

    @staticmethod
    def _insert_event(db, symbol: str, message: str, *, category: str | None = None,
                      at: datetime | None = None):
        if category is not None and category not in EVENT_CATEGORIES:
            raise ValueError("알 수 없는 기록 분류입니다.")
        event_id = db.execute("INSERT INTO events(time,symbol,message) VALUES(?,?,?)",
                              ((at or utc_now()).isoformat(), symbol, message)).lastrowid
        if category is not None:
            db.execute("""INSERT INTO event_categories(event_id,category) VALUES(?,?)
                       ON CONFLICT(event_id) DO UPDATE SET category=excluded.category""", (event_id, category))

    def event(self, symbol: str, message: str, *, category: str | None = None):
        with self.connection() as db:
            self._insert_event(db, symbol, message, category=category)

    def events(self, limit=200, *, category: str | None = None):
        if category is not None and category not in EVENT_CATEGORIES:
            raise ValueError("알 수 없는 기록 분류입니다.")
        with self.connection() as db:
            if category is None:
                sql = """SELECT e.*,c.category FROM events e JOIN event_category_index c ON c.event_id=e.id
                         ORDER BY e.id DESC LIMIT ?"""
                params = (limit,)
            else:
                sql = """SELECT e.*,c.category FROM event_category_index c INDEXED BY event_category_index_by_category
                         JOIN events e ON e.id=c.event_id WHERE c.category=? ORDER BY c.event_id DESC LIMIT ?"""
                params = (category, limit)
            return tuple(dict(r) for r in db.execute(sql, params))

    def event_heads(self) -> dict[str, dict]:
        """Cheap category high-water marks; no event-body scans or broker requests."""
        with self.connection() as db:
            result = {}
            for category in sorted(EVENT_CATEGORIES):
                row = db.execute("""SELECT e.id,e.time FROM event_category_index c
                                  JOIN events e ON e.id=c.event_id WHERE c.category=?
                                  ORDER BY c.event_id DESC LIMIT 1""", (category,)).fetchone()
                result[category] = dict(row) if row else {"id": 0, "time": None}
            return result

    def order_history(self, limit=500) -> tuple[dict, ...]:
        """Durable order intents/results, never inferred from monitoring messages.

        accepted only means acknowledgement, not a fill. reference_price is the
        pre-submission quote; missing execution prices remain NULL, including old
        filled orders for which the earlier application saved no fill metadata.
        Inactive watchlist members are included to preserve historical trades.

        ``limit=None`` returns the complete ledger in chronological order for
        recovery and P/L accounting, not only the newest UI page.
        """
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("주문 내역 개수는 양의 정수 또는 None이어야 합니다.")
        with self.connection() as db:
            sql = """
                SELECT a.rule_id,a.watch_id,a.started_at,a.status,a.order_number,a.message,
                       a.price AS reference_price,w.symbol,w.name,w.market,w.exchange,
                       CASE w.market WHEN 'domestic' THEN 'KRW' WHEN 'us' THEN 'USD' END AS currency,
                       r.side,r.quantity,x.filled_quantity,x.remaining_quantity,x.fill_price,x.observed_at,
                       f.status AS recovery_status,f.message AS recovery_message,f.checked_at AS recovery_checked_at,
                       f.source_api AS recovery_source_api,f.price_basis AS recovery_price_basis,
                       f.order_date AS recovery_order_date,f.order_time AS recovery_order_time,
                       f.fill_time AS recovery_fill_time,f.reported_fill_price AS recovery_reported_fill_price
                FROM attempts a JOIN rules r ON r.id=a.rule_id JOIN watchlist w ON w.id=a.watch_id
                LEFT JOIN order_execution_snapshots x ON x.rule_id=a.rule_id
                LEFT JOIN order_fill_recovery f ON f.rule_id=a.rule_id
            """
            sql += " ORDER BY a.started_at,a.rowid" if limit is None else " ORDER BY a.started_at DESC,a.rowid DESC LIMIT ?"
            return tuple(dict(row) for row in db.execute(sql, () if limit is None else (limit,)))

    def record_execution(self, rule_id: str, *, filled_quantity: Decimal,
                         remaining_quantity: Decimal, fill_price: Decimal | None,
                         observed_at: datetime):
        """Save an already-fetched broker execution snapshot; never changes order state."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._record_execution(db, rule_id, filled_quantity=filled_quantity, remaining_quantity=remaining_quantity,
                                   fill_price=fill_price, observed_at=observed_at)

    @staticmethod
    def _record_execution(db, rule_id, *, filled_quantity, remaining_quantity, fill_price, observed_at):
        for value in (filled_quantity, remaining_quantity):
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ValueError("체결/잔여 수량이 올바르지 않습니다.")
        if observed_at.tzinfo is None:
            raise ValueError("체결 확인 시각에 시간대가 필요합니다.")
        # Missing/zero/invalid broker price is unknown, never the order's quote.
        if (filled_quantity == 0 or not isinstance(fill_price, Decimal)
                or not fill_price.is_finite() or fill_price <= 0):
            fill_price = None
        rule = db.execute("SELECT r.quantity FROM rules r JOIN attempts a ON a.rule_id=r.id WHERE r.id=?", (rule_id,)).fetchone()
        if not rule or filled_quantity + remaining_quantity > rule[0]:
            raise ValueError("체결/잔여 수량이 저장된 원주문 수량과 다릅니다.")
        old = db.execute("SELECT * FROM order_execution_snapshots WHERE rule_id=?", (rule_id,)).fetchone()
        if old:
            old_quantity = Decimal(old["filled_quantity"])
            if filled_quantity < old_quantity:
                raise ValueError("기존에 확인한 체결 수량보다 적은 과거 응답으로 덮어쓰지 않습니다.")
            if filled_quantity == old_quantity and fill_price is None and old["fill_price"] is not None:
                # Same cumulative quantity with an omitted price does not erase
                # already-confirmed evidence. More fills need a new known price.
                fill_price = Decimal(old["fill_price"])
        db.execute("""INSERT INTO order_execution_snapshots VALUES (?,?,?,?,?)
                       ON CONFLICT(rule_id) DO UPDATE SET filled_quantity=excluded.filled_quantity,
                           remaining_quantity=excluded.remaining_quantity,fill_price=excluded.fill_price,
                           observed_at=excluded.observed_at""",
                   (rule_id, str(filled_quantity), str(remaining_quantity),
                    str(fill_price) if fill_price is not None else None, observed_at.isoformat()))

    def record_fill_recovery(self, rule_id: str, *, status: str, message: str, checked_at: datetime,
                             source_api="", price_basis="", order_date="", order_time="", fill_time="",
                             reported_fill_price=None, filled_quantity=None, remaining_quantity=None, fill_price=None):
        """Atomically enrich evidence/provenance only; never alters attempts or rules."""
        if status not in {"enriched", "not_found", "ambiguous", "price_unknown", "quantity_conflict", "error"}:
            raise ValueError("체결가 보완 상태가 올바르지 않습니다.")
        if checked_at.tzinfo is None:
            raise ValueError("체결가 보완 시각에 시간대가 필요합니다.")
        reported = (str(reported_fill_price) if isinstance(reported_fill_price, Decimal)
                    and reported_fill_price.is_finite() and reported_fill_price > 0 else None)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT status,message FROM order_fill_recovery WHERE rule_id=?", (rule_id,)).fetchone()
            if filled_quantity is not None or remaining_quantity is not None:
                self._record_execution(db, rule_id, filled_quantity=filled_quantity, remaining_quantity=remaining_quantity,
                                       fill_price=fill_price, observed_at=checked_at)
            db.execute("""INSERT INTO order_fill_recovery VALUES (?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(rule_id) DO UPDATE SET status=excluded.status,message=excluded.message,
                         checked_at=excluded.checked_at,source_api=excluded.source_api,price_basis=excluded.price_basis,
                         order_date=excluded.order_date,order_time=excluded.order_time,fill_time=excluded.fill_time,
                         reported_fill_price=excluded.reported_fill_price""",
                       (rule_id, status, message, checked_at.isoformat(), source_api, price_basis,
                        order_date, order_time, fill_time, reported))
            if not old or (old["status"], old["message"]) != (status, message):
                watch = db.execute("SELECT watch_id FROM attempts WHERE rule_id=?", (rule_id,)).fetchone()
                self._insert_event(db, watch[0], f"체결가 보완 · {message}", category="order", at=checked_at)

    def save_snapshot(self, item: WatchItem, snapshot: MarketSnapshot):
        quote, history = snapshot.quote, snapshot.history
        payload = {"at": snapshot.fetched_at.isoformat(), "name": quote.name, "price": str(quote.price),
                   "change": str(quote.change) if quote.change is not None else None,
                   "rate": str(quote.change_rate) if quote.change_rate is not None else None,
                   "days": history.requested_days,
                   "bars": [[bar.day.isoformat(), *(str(getattr(bar, key)) for key in ("open", "high", "low", "close", "volume"))]
                            for bar in history.bars]}
        with self.connection() as db:
            db.execute("INSERT INTO snapshots VALUES (?, ?) ON CONFLICT(watch_id) DO UPDATE SET data=excluded.data",
                       (item.id, json.dumps(payload, ensure_ascii=False)))

    def cached_snapshot(self, item: WatchItem) -> MarketSnapshot | None:
        with self.connection() as db:
            row = db.execute("SELECT data FROM snapshots WHERE watch_id=?", (item.id,)).fetchone()
        if not row:
            return None
        return self.decode_snapshot(item, row[0])

    @staticmethod
    def decode_snapshot(item: WatchItem, data: str) -> MarketSnapshot:
        payload, inst = json.loads(data), item.instrument
        bars = tuple(DailyBar(date.fromisoformat(row[0]), *(Decimal(v) for v in row[1:])) for row in payload["bars"])
        quote = Quote(inst.market, inst.symbol, payload["name"], inst.exchange, Decimal(payload["price"]), inst.currency,
                      Decimal(payload["change"]) if payload["change"] is not None else None,
                      Decimal(payload["rate"]) if payload["rate"] is not None else None)
        return MarketSnapshot(quote, DailyHistory(inst.market, inst.symbol, inst.exchange, inst.currency, payload["days"], bars),
                              datetime.fromisoformat(payload["at"]))


def default_store(mode: TradingMode = TradingMode.DEMO, *, scope: str = "unconfigured") -> WatchStore:
    selected = TradingMode(mode)
    folder = Path(__file__).resolve().parent.parent / ".dockdack"
    if selected is TradingMode.REAL:
        if scope != "unconfigured" and (len(scope) != 64 or any(c not in "0123456789abcdef" for c in scope)):
            raise ValueError("실전 저장소의 인증 범위 식별자가 올바르지 않습니다.")
        folder = folder / "real" / scope
    return WatchStore(folder / "watchlist.sqlite3", seed_defaults=True, mode=selected,
                      storage_scope=scope if selected is TradingMode.REAL else "demo")
