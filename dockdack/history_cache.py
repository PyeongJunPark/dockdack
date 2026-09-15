"""Durable adjusted daily bars: full backfill once, then incremental recent bars."""

import json
from datetime import date, datetime, timedelta
from decimal import Decimal

from dockdack.history import DailyBar, DailyHistory, market_time
from dockdack.market_schedule import session_on


class HistoryCache:
    def __init__(self, store, service, clock):
        self.store, self.service, self.clock = store, service, clock
        self.ignore_legacy = set()

    def invalidate(self, watch_id):
        with self.store.connection() as db:
            db.execute("DELETE FROM history_cache WHERE watch_id=?", (watch_id,))
        self.ignore_legacy.add(watch_id)

    def load(self, item):
        with self.store.connection() as db:
            row = db.execute("SELECT * FROM history_cache WHERE watch_id=?", (item.id,)).fetchone()
        if row:
            data = json.loads(row["data"])
            bars = tuple(DailyBar(date.fromisoformat(b[0]), *(Decimal(v) for v in b[1:])) for b in data["bars"])
            inst = item.instrument
            return DailyHistory(inst.market, inst.symbol, inst.exchange, inst.currency, data["capacity"], bars), datetime.fromisoformat(row["fetched_at"])
        if item.id not in self.ignore_legacy:
            # Preserve already downloaded histories. Legacy timestamps may include up
            # to five minutes of memory cache; force a recent-bar check on first use.
            snapshot = self.store.cached_snapshot(item)
            if snapshot:
                return snapshot.history, snapshot.fetched_at - timedelta(minutes=5)
        return None

    def save(self, item, history, fetched_at):
        payload = {"capacity": history.requested_days,
                   "bars": [[b.day.isoformat(), *(str(getattr(b, k)) for k in ("open", "high", "low", "close", "volume"))] for b in history.bars]}
        with self.store.connection() as db:
            db.execute("INSERT INTO history_cache VALUES(?,?,?) ON CONFLICT(watch_id) DO UPDATE SET fetched_at=excluded.fetched_at,data=excluded.data",
                       (item.id, fetched_at.isoformat(), json.dumps(payload)))

    def get(self, item, days):
        inst, now = item.instrument, self.clock()
        today = market_time(inst.market, now).date()
        session = session_on(inst.market, today)
        cached = self.load(item)
        capacity = days
        full = not cached or cached[0].requested_days < days
        old, old_at = cached if cached else (None, None)
        if old:
            capacity = max(days, old.requested_days)
        # Finalize today's bar once after close; otherwise refresh live data at most
        # once per five minutes. Completed bars survive process restarts.
        crossed_close = bool(session and old_at and old_at < session.closed + timedelta(seconds=30) <= now)
        same_day = bool(old_at and market_time(inst.market, old_at).date() == today)
        if not full and same_day and not crossed_close and 0 <= (now-old_at).total_seconds() < 300:
            return self._trim(old, days)
        if not full and same_day and session and old.bars[-1].day == today and session.closed + timedelta(seconds=30) <= old_at <= now:
            return self._trim(old, days)
        if not full and not session and old_at and 0 <= (now-old_at).total_seconds() < 86400:
            return self._trim(old, days)
        request_days = capacity if full else 1
        if not full and not same_day:
            # Keep a completed overlap even when the last cached bar was intraday.
            # Missing sessions are backfilled without reloading the entire history.
            last_day = old.bars[-1].day
            missing, cursor = 0, last_day + timedelta(days=1)
            while cursor <= today and missing < capacity:
                if session_on(inst.market, cursor):
                    missing += 1
                cursor += timedelta(days=1)
            previous_session = session_on(inst.market, last_day)
            overlap = 2 if previous_session and old_at < previous_session.closed + timedelta(seconds=30) else 1
            request_days = min(capacity, max(2, missing+overlap))
            if missing == 0 and not session and previous_session and previous_session.closed + timedelta(seconds=30) <= old_at <= now:
                return self._trim(old, days)
        fresh = self.service.history(inst, request_days)
        self._validate(inst, fresh)
        fetched_at = self.clock()
        if not full:
            prior = {b.day: b for b in old.bars}
            revised = False
            for bar in fresh.bars:
                previous = prior.get(bar.day)
                previous_session = session_on(inst.market, bar.day)
                if previous and previous != bar and previous_session and previous_session.closed + timedelta(seconds=30) <= old_at:
                    revised = True
            if revised:
                fresh = self.service.history(inst, capacity)
                self._validate(inst, fresh)
                fetched_at, full = self.clock(), True
                self.store.event(item.id, "완료 일봉 변경 감지 · 수정주가 과거 구간 재동기화", category="monitor")
        merged = {} if full else {bar.day: bar for bar in old.bars}
        merged.update((bar.day, bar) for bar in fresh.bars)
        history = DailyHistory(inst.market, inst.symbol, inst.exchange, inst.currency, capacity,
                               tuple(merged[d] for d in sorted(merged))[-capacity:])
        self.save(item, history, fetched_at)
        return self._trim(history, days)

    @staticmethod
    def _trim(history, days):
        return DailyHistory(history.market, history.symbol, history.exchange, history.currency, days, history.bars[-days:])

    @staticmethod
    def _validate(inst, history):
        if (history.market, history.symbol, history.exchange, history.currency) != (inst.market, inst.symbol, inst.exchange, inst.currency):
            raise ValueError("증분 일봉의 종목/시장/통화가 다릅니다.")
        if not history.bars or any(a.day >= b.day for a, b in zip(history.bars, history.bars[1:])):
            raise ValueError("증분 일봉의 날짜/개수를 확인할 수 없습니다.")
