"""Explicit-market TOP100 ownership for the DEMO LSTM dashboard.

The broker's top_volume contract already verifies common-equity classification
and refuses a short result. Ranking refreshes preserve the existing WatchStore
ledger, manual interests and pending/unresolved orders. Holdings outside the
ranking are monitored by the independent exit scan, not retained as buy interests.
"""

from decimal import Decimal
from time import monotonic

from dockdack.gui_service import Instrument
from dockdack.market_schedule import RankingScheduler, is_open, ranking_allowed
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
        self._validated_at = float("-inf")
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

    def validate_active(self, *, force=True):
        self._check_demo()
        if not force and monotonic() - self._validated_at < 1:
            return
        actual = self.items()
        if frozenset(item.id for item in actual) != self.approved_ids:
            raise ValueError("검증된 LSTM30 순위 갱신 이외의 관심종목 변경을 차단했습니다.")
        if any(item.days < 31 for item in actual):
            raise ValueError("LSTM30 관심종목은 최소 31개 일봉을 요청해야 합니다.")
        self._validated_at = monotonic()

    def ready_for_open_markets(self):
        """A closed US session must not block an independently ready KR session."""
        now = self.clock()
        return all(market in self._ready_markets for market in self.ranked_markets if is_open(market, now))

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
                    or not row.turnover.is_finite() or row.turnover < 0 or type(row.rank) is not int
                    or row.ranking_basis != "volume" or type(row.volume) is not int or row.volume < 0):
                raise ValueError("순위의 통화·거래량·순번을 확인할 수 없습니다.")
            keys.add(item.id)
            ranks.add(row.rank)
        if len(keys) != 100 or ranks != set(range(1, 101)):
            raise ValueError("서로 다른 보통주 100개와 1~100 순번이 필요합니다.")
        return rows, frozenset(keys)

    def refresh(self, market, *, require_open=False, guard=None):
        market = Market(market)
        if market not in self.ranked_markets:
            raise ValueError("이 시장의 TOP100 재선정은 승인되지 않았습니다.")
        self.validate_active()
        if self.stopped():
            raise InterruptedError("감시 중지로 순위 갱신을 하지 않습니다.")
        try:
            if not ranking_allowed(market, self.clock()) or (require_open and not is_open(market, self.clock())):
                raise InterruptedError("개장 10분 전 준비/정규장 시간이 아니어서 순위를 조회하지 않습니다.")
            # Classification and pagination are fail-closed in the service's
            # top_volume implementation; never make up the missing names.
            rankings, incoming = self._validate_rankings(market, self.service.top_volume(market, 100))
            protected = self.service.protected_symbols(market)
            if (not isinstance(protected, (set, frozenset))
                    or any(not isinstance(symbol, str) or not symbol for symbol in protected)):
                raise ValueError("전체 보유/미체결 보호 종목을 확인할 수 없습니다.")
            if self.stopped():
                raise InterruptedError("중지 요청으로 순위 결과를 적용하지 않습니다.")
            if not ranking_allowed(market, self.clock()) or (require_open and not is_open(market, self.clock())):
                raise InterruptedError("정규장이 종료되어 순위 결과를 적용하지 않습니다.")
            if guard is not None:
                guard()
            self.validate_active()
            allowed = self.approved_ids | incoming
            self.store.replace_ranked(market, rankings, set(protected), days=31, separate_holdings=True)
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
        """Prepare only eligible markets, using the same durable slots as refresh.

        Closed markets are deferred, not queried or treated as an error. A
        restart can adopt a persisted, classified volume list without issuing a
        second request for an already completed slot.
        """
        self.validate_active()
        self._restore_ranked_state()
        scheduler = ScopedRankingScheduler(self.service, self.store, universe=self,
                                           clock=self.clock, stopped=self.stopped)
        scheduler.start()
        scheduler.tick()
        return self.items()

    def _restore_ranked_state(self):
        rows = self.store.rankings()
        active = {item.id for item in self.items()}
        for market in self.ranked_markets:
            current = [row for row in rows if row["market"] == market.value
                       and row.get("ranking_basis") == "volume" and row["watch_id"] in active]
            if len(current) == 100 and {row["rank"] for row in current} == set(range(1, 101)):
                self._ranked_ids[market] = frozenset(row["watch_id"] for row in current)
                self._ready_markets.add(market)
                self.updated_at[market] = max(row["fetched_at"] for row in current)
        self.initialized = self._ready_markets == self.ranked_markets

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
                    "ranking_basis": "volume", "ranking_allowed": ranking_allowed(market, self.clock()),
                } for market in Market
            },
        }


class ScopedRankingScheduler(RankingScheduler):
    """Main's exchange slots and durable retry claims, for approved markets only."""

    def __init__(self, service, store, *, universe, clock=utc_now, stopped=lambda: False, on_error=None):
        super().__init__(service, store, clock=clock, stopped=stopped)
        self.universe, self.on_error = universe, on_error

    @property
    def markets(self):
        return tuple(sorted(self.universe.ranked_markets, key=lambda value: value.value))

    def _error(self, market, exc):
        message = f"{market.value} 거래량 TOP100 갱신 실패 · 기존 목록 유지 · 다음 재시도 대기: {exc}"
        if self.errors.get(market) != message:
            self.store.event("SYSTEM", message, category="system")
            self.errors[market] = message
        if self.on_error is not None:
            self.on_error(market, exc)

    def record_bootstrap(self):
        """Compatibility hook: bootstrap now owns/finishes its durable claims."""
        return None

    def _refresh_market(self, market, guard):
        self.universe.refresh(market, guard=guard)
