"""Independent exchange slots: open minus ten minutes, then local whole hours."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from uuid import uuid4

from dockdack.history import market_time
from dockdack.models import Market

# Known CSAT date (Busan education-office calendar). The bundled XKRX calendar
# lacks this special session; do not invent exchange hours before KRX verification.
UNVERIFIED_SESSIONS = {(Market.DOMESTIC, date(2026, 11, 19))}
EXTRA_CLOSURES = {(Market.DOMESTIC, date(2026, 6, 3)), (Market.DOMESTIC, date(2026, 7, 17))}


@dataclass(frozen=True)
class Session:
    opened: datetime
    closed: datetime

    def slots(self):
        yield self.opened - timedelta(minutes=10)
        value = self.opened.replace(minute=0, second=0, microsecond=0)
        if value < self.opened:
            value += timedelta(hours=1)
        while value < self.closed:
            yield value
            value += timedelta(hours=1)


@lru_cache(maxsize=8)
def calendar_for(market: Market, year: int):
    import exchange_calendars as calendars
    return calendars.get_calendar("XKRX" if market is Market.DOMESTIC else "XNYS",
                                  start=f"{year-1}-12-01", end=f"{year+1}-01-31")


@lru_cache(maxsize=512)
def session_on(market: Market, day: date) -> Session | None:
    if (market, day) in EXTRA_CLOSURES:
        return None
    if (market, day) in UNVERIFIED_SESSIONS:
        raise ValueError(f"{day} 특별 개장시간은 KRX 공지 확인 후 캘린더 갱신이 필요합니다.")
    calendar = calendar_for(market, day.year)
    if not calendar.is_session(day.isoformat()):
        return None
    opened = calendar.session_open(day.isoformat()).to_pydatetime()
    closed = calendar.session_close(day.isoformat()).to_pydatetime()
    return Session(market_time(market, opened), market_time(market, closed))


def is_open(market: Market, now: datetime) -> bool:
    session = session_on(market, market_time(market, now).date())
    return session is not None and session.opened <= now < session.closed


def ranking_slot(market: Market, now: datetime) -> datetime | None:
    """Latest eligible slot, not a backlog, including missed starts/long sweeps.

    The pre-open exception authorizes ranking preparation only, never orders.
    A slot remains eligible until the next slot or market close. Holidays and
    overnight times return None independently for each exchange's local day.
    """
    session = session_on(market, market_time(market, now).date())
    if session is None or not session.opened - timedelta(minutes=10) <= now < session.closed:
        return None
    return max(slot for slot in session.slots() if slot <= now)


def ranking_allowed(market: Market, now: datetime) -> bool:
    return ranking_slot(market, now) is not None


class RankingScheduler:
    """Call tick from the broker worker only. Persistent claims deduplicate GUI restarts.

    Latest-slot catch-up avoids missing updates during long chart sweeps. Failures
    retry at most three times, >=60 seconds apart. A ten-minute durable lease lets
    a new process reclaim a crashed worker without repeating a completed slot.
    """

    def __init__(self, service, store, *, clock, stopped=lambda: False, days=31):
        if type(days) is not int or days < 1:
            raise ValueError("순위 종목 차트 일수는 양의 정수여야 합니다.")
        self.service, self.store, self.clock, self.stopped = service, store, clock, stopped
        self.days = days
        self.started = None
        self.errors = {}
        with self.store.connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS ranking_leases (
                market TEXT NOT NULL, slot TEXT NOT NULL, owner TEXT NOT NULL,
                expires_at TEXT NOT NULL, PRIMARY KEY(market,slot))""")

    @property
    def markets(self):
        return tuple(Market)

    def start(self):
        self.started = self.clock().replace(second=0, microsecond=0)

    def stop(self):
        self.started = None

    def _slot(self, market, now):
        slot = ranking_slot(market, now)
        return slot.astimezone(timezone.utc).isoformat() if slot is not None else None

    @staticmethod
    def _retryable(row, lease, now):
        if row is None:
            return True
        if row["status"] == "done" or row["attempts"] >= 3:
            return False
        age = (now - datetime.fromisoformat(row["attempted_at"])).total_seconds()
        if row["status"] == "failed":
            return age >= 60
        if row["status"] == "running":
            return (now >= datetime.fromisoformat(lease["expires_at"]) if lease else age >= 600)
        return False

    def _due_slot(self, market, slot, now):
        with self.store.connection() as db:
            row = db.execute("SELECT * FROM ranking_runs WHERE market=? AND slot=?", (market.value, slot)).fetchone()
            lease = db.execute("SELECT * FROM ranking_leases WHERE market=? AND slot=?", (market.value, slot)).fetchone()
        return self._retryable(row, lease, now)

    def _claim(self, market, slot, now):
        owner = uuid4().hex
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM ranking_runs WHERE market=? AND slot=?", (market.value, slot)).fetchone()
            lease = db.execute("SELECT * FROM ranking_leases WHERE market=? AND slot=?", (market.value, slot)).fetchone()
            if not self._retryable(row, lease, now):
                return None
            db.execute("""INSERT INTO ranking_runs VALUES(?,?,'running',1,?,'')
                ON CONFLICT(market,slot) DO UPDATE SET status='running',attempts=attempts+1,
                attempted_at=excluded.attempted_at,detail=''""", (market.value, slot, now.isoformat()))
            db.execute("""INSERT INTO ranking_leases VALUES(?,?,?,?)
                ON CONFLICT(market,slot) DO UPDATE SET owner=excluded.owner,expires_at=excluded.expires_at""",
                (market.value, slot, owner, (now + timedelta(minutes=10)).isoformat()))
        return owner

    def _owns(self, market, slot, owner):
        with self.store.connection() as db:
            row = db.execute("SELECT owner FROM ranking_leases WHERE market=? AND slot=?", (market.value, slot)).fetchone()
        return row is not None and row["owner"] == owner

    def _finish(self, market, slot, owner, status, detail):
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT owner FROM ranking_leases WHERE market=? AND slot=?", (market.value, slot)).fetchone()
            if row is None or row["owner"] != owner:
                return
            db.execute("UPDATE ranking_runs SET status=?,detail=? WHERE market=? AND slot=?",
                       (status, detail, market.value, slot))
            db.execute("DELETE FROM ranking_leases WHERE market=? AND slot=?", (market.value, slot))

    def _error(self, market, exc):
        message = f"{market.value} 거래량 TOP100 재선정 실패 · 기존 목록 유지: {exc}"
        if self.errors.get(market) != message:
            self.store.event("SYSTEM", message, category="system")
            self.errors[market] = message

    def _refresh_market(self, market, guard):
        rankings = self.service.top_volume(market, 100)
        protected = self.service.protected_symbols(market)
        guard()
        self.store.replace_ranked(market, rankings, protected, days=self.days, separate_holdings=True)

    def due(self):
        if self.started is None or self.stopped():
            return False
        now = self.clock()
        for market in self.markets:
            try:
                slot = self._slot(market, now)
                if slot is not None and self._due_slot(market, slot, now):
                    return True
            except Exception as exc:
                self._error(market, exc)
        return False

    def tick(self):
        changed = False
        if self.started is None or self.stopped():
            return changed
        for market in self.markets:
            try:
                now = self.clock()
                slot = self._slot(market, now)
                if slot is None:
                    continue
                owner = self._claim(market, slot, now)
                if owner is None:
                    continue
                try:
                    def guard():
                        if (self.started is None or self.stopped() or not ranking_allowed(market, self.clock())
                                or not self._owns(market, slot, owner)):
                            raise InterruptedError("중지·장 종료·다른 작업 인계로 재선정 결과를 적용하지 않음")
                    guard()
                    self._refresh_market(market, guard)
                    self._finish(market, slot, owner, "done", "거래량 TOP100 재선정 완료")
                    changed = True
                except Exception as exc:
                    self._finish(market, slot, owner, "failed", str(exc))
                    raise
                self.errors.pop(market, None)
            except Exception as exc:
                self._error(market, exc)
        return changed
