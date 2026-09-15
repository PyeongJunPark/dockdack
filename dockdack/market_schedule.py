"""Exchange-calendar slots: session open, then local whole hours before close."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

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
        yield self.opened
        value = self.opened.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
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


class RankingScheduler:
    """Call tick from the broker worker only. Persistent claims deduplicate GUI restarts.

    Late requests get five minutes' grace, not an unbounded backlog. Failures retry
    at most three times, >=60 seconds apart, and never outside the same open slot.
    """

    def __init__(self, service, store, *, clock, stopped=lambda: False):
        self.service, self.store, self.clock, self.stopped = service, store, clock, stopped
        self.started = None
        self.errors = {}

    def start(self):
        self.started = self.clock().replace(second=0, microsecond=0)

    def stop(self):
        self.started = None

    def due(self):
        if self.started is None or self.stopped():
            return False
        now = self.clock()
        for market in Market:
            try:
                session = session_on(market, market_time(market, now).date())
                if not session or not session.opened <= now < session.closed:
                    continue
                slots = [s for s in session.slots() if self.started <= s <= now]
                if slots and (now - slots[-1]).total_seconds() <= 300:
                    with self.store.connection() as db:
                        row = db.execute("SELECT * FROM ranking_runs WHERE market=? AND slot=?",
                                         (market.value, slots[-1].astimezone(timezone.utc).isoformat())).fetchone()
                    if row is None or (row["status"] == "failed" and row["attempts"] < 3 and
                                       (now-datetime.fromisoformat(row["attempted_at"])).total_seconds() >= 60):
                        return True
            except Exception as exc:
                message = f"{market.value} 거래일 캘린더 오류 · 재선정 차단: {exc}"
                if self.errors.get(market) != message:
                    self.store.event("SYSTEM", message, category="system")
                    self.errors[market] = message
        return False

    def tick(self):
        changed = False
        if self.started is None or self.stopped():
            return changed
        for market in Market:
            try:
                now = self.clock()
                session = session_on(market, market_time(market, now).date())
                if not session or not session.opened <= now < session.closed:
                    continue
                slots = [s for s in session.slots() if self.started <= s <= now]
                if not slots or (now - slots[-1]).total_seconds() > 300:
                    continue
                slot = slots[-1].astimezone(timezone.utc).isoformat()
                if not self.store.claim_ranking_run(market, slot, now):
                    continue
                try:
                    rankings = self.service.top_turnover(market, 100)
                    protected = self.service.protected_symbols(market)
                    if self.stopped() or not is_open(market, self.clock()):
                        raise InterruptedError("중지되었거나 장이 종료되어 재선정 결과를 적용하지 않음")
                    self.store.replace_ranked(market, rankings, protected)
                    self.store.finish_ranking_run(market, slot, "done", "TOP100 재선정 완료")
                    changed = True
                except Exception as exc:
                    self.store.finish_ranking_run(market, slot, "failed", str(exc))
                    raise
                self.errors.pop(market, None)
            except Exception as exc:
                message = f"{market.value} 정시 재선정 실패 · 기존 목록 유지: {exc}"
                if self.errors.get(market) != message:
                    self.store.event("SYSTEM", message, category="system")
                    self.errors[market] = message
        return changed
