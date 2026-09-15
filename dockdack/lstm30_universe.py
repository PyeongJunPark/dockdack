"""Explicit-market TOP100 ownership for the DEMO LSTM dashboard.

The broker's top_turnover contract already verifies common-equity classification
and refuses a short result. Ranking refreshes preserve the existing WatchStore
ledger, manual interests, held names and pending/unresolved orders.
"""

from datetime import datetime, timezone
from decimal import Decimal

from dockdack.gui_service import Instrument
from dockdack.history import market_time
from dockdack.market_schedule import RankingScheduler, is_open, session_on
from dockdack.models import Market, TradingMode
from dockdack.universe import RankedStock
from dockdack.watchlist import WatchItem, utc_now


class LSTM30Universe:
    def __init__(self, service, store, *, ranked_markets=(), baseline_items=None,
                 clock=utc_now, stopped=lambda: False, on_change=None):
        self.service, self.store, self.clock = service, store, clock
        self.ranked_markets = frozenset(Market(market) for market in ranked_markets)
        self.stopped, self.on_change = stopped, on_change
        baseline = tuple(store.items() if baseline_items is None else baseline_items)
        self.baseline_ids = frozenset(item.id for item in baseline)
        self.approved_ids = frozenset(item.id for item in store.items())
        if self.approved_ids != self.baseline_ids:
            raise ValueError("초기 LSTM30 관심목록과 승인된 기준 목록이 다릅니다.")
        self.initialized = not self.ranked_markets
        self._ready_markets = set()
        self._ranked_ids = {}
        self.updated_at = {}
        self.errors = {}
        self._check_demo()
        # The exact baseline was explicitly accepted above; upgrade its legacy
        # lookback without changing managed/manual ownership or order history.
        with self.store.connection() as db:
            db.executemany("UPDATE watchlist SET days=31 WHERE id=? AND days<31",
                           ((item.id,) for item in baseline))

    def _check_demo(self):
        if self.service.mode is not TradingMode.DEMO or self.store.mode is not TradingMode.DEMO:
            raise ValueError("LSTM30 TOP100은 모의투자 전용입니다.")

    def items(self):
        return self.store.items()

    def validate_active(self):
        self._check_demo()
        actual = self.items()
        if frozenset(item.id for item in actual) != self.approved_ids:
            raise ValueError("검증된 LSTM30 순위 갱신 이외의 관심종목 변경을 차단했습니다.")
        if any(item.days < 31 for item in actual):
            raise ValueError("LSTM30 관심종목은 최소 31개 일봉을 요청해야 합니다.")

    @staticmethod
    def _validate_rankings(market, rankings):
        rows = tuple(rankings)
        if len(rows) != 100:
            raise ValueError(f"{market.value} 보통주 순위가 {len(rows)}개입니다. 100개를 확정하지 않아 기존 목록을 유지합니다.")
        keys, ranks = set(), set()
        for row in rows:
            if not isinstance(row, RankedStock) or row.market is not market:
                raise ValueError("해당 시장의 검증된 보통주 순위 응답이 필요합니다.")
            item = WatchItem(Instrument(row.market, row.symbol, row.exchange), row.name, 31)
            if (row.currency != item.instrument.currency or not isinstance(row.turnover, Decimal)
                    or not row.turnover.is_finite() or row.turnover < 0 or type(row.rank) is not int):
                raise ValueError("순위의 통화·거래대금·순번을 확인할 수 없습니다.")
            keys.add(item.id)
            ranks.add(row.rank)
        if len(keys) != 100 or ranks != set(range(1, 101)):
            raise ValueError("서로 다른 보통주 100개와 1~100 순번이 필요합니다.")
        return rows, frozenset(keys)

    def refresh(self, market, *, require_open=False):
        market = Market(market)
        if market not in self.ranked_markets:
            raise ValueError("이 시장의 TOP100 재선정은 승인되지 않았습니다.")
        self.validate_active()
        if self.stopped():
            raise InterruptedError("감시 중지로 순위 갱신을 하지 않습니다.")
        try:
            # Classification and pagination are fail-closed in the service's
            # top_turnover implementation; never make up the missing names.
            rankings, incoming = self._validate_rankings(market, self.service.top_turnover(market, 100))
            protected = self.service.protected_symbols(market)
            if (not isinstance(protected, (set, frozenset))
                    or any(not isinstance(symbol, str) or not symbol for symbol in protected)):
                raise ValueError("전체 보유/미체결 보호 종목을 확인할 수 없습니다.")
            if self.stopped():
                raise InterruptedError("중지 요청으로 순위 결과를 적용하지 않습니다.")
            if require_open and not is_open(market, self.clock()):
                raise InterruptedError("정규장이 종료되어 순위 결과를 적용하지 않습니다.")
            self.validate_active()
            allowed = self.approved_ids | incoming
            self.store.replace_ranked(market, rankings, set(protected), days=31)
            current = self.items()
            if not {item.id for item in current} <= allowed:
                raise ValueError("순위 교체 중 승인되지 않은 종목이 추가되었습니다.")
            # replace_ranked preserves existing N. Upgrade legacy N without
            # save_item(), which would turn all managed stocks into manual pins.
            with self.store.connection() as db:
                db.executemany("UPDATE watchlist SET days=31 WHERE id=? AND days<31",
                               ((item.id,) for item in current))
            current = self.items()
            self.approved_ids = frozenset(item.id for item in current)
            self._ranked_ids = {**self._ranked_ids, market: incoming}
            self.updated_at = {**self.updated_at, market: self.clock().isoformat()}
            self._ready_markets.add(market)
            self.initialized = self._ready_markets == self.ranked_markets
            self.errors.pop(market, None)
            if self.on_change is not None:
                self.on_change(current)
            return current
        except Exception as exc:
            self.errors[market] = str(exc)
            raise

    def bootstrap(self):
        self.validate_active()
        for market in sorted(self.ranked_markets, key=lambda value: value.value):
            self.refresh(market)
        return self.items()

    def status(self):
        items = self.items()
        return {
            "ranked_markets": sorted(market.value for market in self.ranked_markets),
            "initialized": self.initialized,
            "markets": {
                market.value: {
                    "ranked_count": len(self._ranked_ids.get(market, ())),
                    "active_count": sum(item.instrument.market is market for item in items),
                    "retained_extra_count": sum(item.instrument.market is market and item.id not in self._ranked_ids.get(market, ()) for item in items),
                    "updated_at": self.updated_at.get(market), "error": self.errors.get(market, ""),
                } for market in Market
            },
        }


class ScopedRankingScheduler(RankingScheduler):
    """Main's exchange slots and durable retry claims, for approved markets only."""

    def __init__(self, service, store, *, universe, clock=utc_now, stopped=lambda: False, on_error=None):
        super().__init__(service, store, clock=clock, stopped=stopped)
        self.universe, self.on_error = universe, on_error

    def _slot(self, market, now):
        session = session_on(market, market_time(market, now).date())
        if not session or not session.opened <= now < session.closed:
            return None
        slots = [slot for slot in session.slots() if self.started <= slot <= now]
        if not slots or (now - slots[-1]).total_seconds() > 300:
            return None
        return slots[-1].astimezone(timezone.utc).isoformat()

    def _error(self, market, exc):
        message = f"{market.value} TOP100 갱신 실패 · 기존 목록 유지 · 자동주문 OFF: {exc}"
        if self.errors.get(market) != message:
            self.store.event("SYSTEM", message, category="system")
            self.errors[market] = message
        if self.on_error is not None:
            self.on_error(market, exc)

    def record_bootstrap(self):
        """A successful initial fetch also satisfies this session's current slot."""
        if self.started is None or self.stopped() or not self.universe.initialized:
            return
        now = self.clock()
        for market in sorted(self.universe.ranked_markets, key=lambda value: value.value):
            slot = self._slot(market, now)
            if slot is not None and self.store.claim_ranking_run(market, slot, now):
                self.store.finish_ranking_run(market, slot, "done", "초기 LSTM30 TOP100 확정으로 현재 개장/정시 슬롯 처리 완료")

    def due(self):
        if self.started is None or self.stopped() or not self.universe.initialized:
            return False
        now = self.clock()
        for market in sorted(self.universe.ranked_markets, key=lambda value: value.value):
            try:
                slot = self._slot(market, now)
                if slot is None:
                    continue
                with self.store.connection() as db:
                    row = db.execute("SELECT * FROM ranking_runs WHERE market=? AND slot=?", (market.value, slot)).fetchone()
                if row is None or (row["status"] == "failed" and row["attempts"] < 3
                                   and (now-datetime.fromisoformat(row["attempted_at"])).total_seconds() >= 60):
                    return True
            except Exception as exc:
                self._error(market, exc)
        return False

    def tick(self):
        changed = False
        if self.started is None or self.stopped() or not self.universe.initialized:
            return changed
        for market in sorted(self.universe.ranked_markets, key=lambda value: value.value):
            try:
                now = self.clock()
                slot = self._slot(market, now)
                if slot is None or not self.store.claim_ranking_run(market, slot, now):
                    continue
                try:
                    if not is_open(market, self.clock()):
                        raise InterruptedError("정규장이 종료되어 순위 결과를 적용하지 않습니다.")
                    self.universe.refresh(market, require_open=True)
                    self.store.finish_ranking_run(market, slot, "done", "LSTM30 보통주 TOP100 · 31개 일봉 갱신 완료")
                    changed = True
                except Exception as exc:
                    self.store.finish_ranking_run(market, slot, "failed", str(exc))
                    raise
                self.errors.pop(market, None)
            except Exception as exc:
                self._error(market, exc)
        return changed
