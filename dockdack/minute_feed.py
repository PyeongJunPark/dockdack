"""Demand-driven, DEMO-only domestic five-minute chart feed.

Kiwoom ka10080 documents a calendar ``base_dt`` and backwards pagination,
not an intraday "since timestamp" cursor.  We therefore refresh the newest
page once per completed five-minute slot and merge bars by their broker label.
This module does not import account, order, or trading-engine code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Callable, Protocol
from zoneinfo import ZoneInfo

from dockdack.market_schedule import session_on
from dockdack.models import DomesticExchange, Market, MinuteBar, TradingMode


KST = ZoneInfo("Asia/Seoul")
INTERVAL = timedelta(minutes=5)
MAX_REFRESH_DURATION = timedelta(seconds=30)
MAX_PAGES = 5
MAX_COUNT = 70
RETRY_DELAY = timedelta(seconds=20)
MAX_SLOT_ATTEMPTS = 3
_SOURCE = "kiwoom_rest_demo_ka10080"


class MinuteFeedUnavailable(ValueError):
    """Do not use this feed to make an order decision."""


class _DomesticMinuteBroker(Protocol):
    mode: TradingMode

    def minute_bars_domestic(self, symbol: str, *, exchange: DomesticExchange,
                             interval_minutes: int, base_date: date, adjusted: bool,
                             as_of: datetime, max_pages: int): ...


@dataclass(frozen=True, slots=True)
class MinuteFeedSnapshot:
    symbol: str
    exchange: str
    session_date: date
    bars: tuple[MinuteBar, ...]  # Chronological, contiguous, requested count.
    fetched_at_utc: datetime
    last_bar_label: datetime
    sha256: str
    jsonl_path: Path
    receipt_path: Path
    page_count: int
    truncated: bool
    from_cache: bool

    @property
    def market(self) -> Market:
        return Market.DOMESTIC


@dataclass(frozen=True, slots=True)
class _Stored:
    bars: tuple[MinuteBar, ...]
    fetched_at: datetime
    sha256: str
    page_count: int
    truncated: bool


class DomesticMinuteFeed:
    """Return verified current-session bars for one needed watchlist symbol.

    A single instance should be shared across model workers.  It serializes
    refreshes, so nine models requesting one symbol/slot do not make nine API
    calls.  The cache is an integrity-checked JSONL plus receipt, not an order
    authorization.  Every call rechecks session, freshness, and contiguous
    window; a network failure never falls back to stale bars.
    """

    def __init__(self, broker: _DomesticMinuteBroker, cache_dir: Path | str,
                 *, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        if broker.mode is not TradingMode.DEMO:
            raise MinuteFeedUnavailable("실전 환경에서는 분봉 실험 피드를 사용하지 않습니다.")
        self.broker = broker
        self.cache_dir = Path(cache_dir)
        self.clock = clock
        self._lock = RLock()
        self._memory: dict[tuple[str, date], _Stored] = {}
        # Negative results are shared by the model group, but a late broker
        # candle or short outage can recover within the same five-minute slot.
        self._insufficient: dict[tuple[str, date, datetime, int], tuple[datetime, int]] = {}
        self._failed_slot: dict[tuple[str, date, datetime], tuple[str, datetime, int]] = {}

    def get_complete_bars(self, symbol: str, exchange: DomesticExchange | str,
                          now: datetime | None = None, count: int = 20) -> MinuteFeedSnapshot:
        """Get ``count`` contiguous, finalized five-minute bars of today's KRX session.

        `now` is injectable for an already captured decision clock.  A caller
        still must revalidate its order immediately before transmission; this
        read-only snapshot is never itself an order permission.
        """
        if self.broker.mode is not TradingMode.DEMO:
            raise MinuteFeedUnavailable("실전 환경에서는 분봉 실험 피드를 사용하지 않습니다.")
        if not isinstance(symbol, str) or len(symbol) != 6 or not symbol.isascii() or not symbol.isdigit():
            raise MinuteFeedUnavailable("국내 분봉에는 6자리 종목코드가 필요합니다.")
        if exchange not in (DomesticExchange.KRX, "KRX"):
            raise MinuteFeedUnavailable("분봉 피드는 모의투자 KRX 종목만 지원합니다.")
        if type(count) is not int or not 1 <= count <= MAX_COUNT:
            raise ValueError(f"필요 분봉 수는 1~{MAX_COUNT}개여야 합니다.")
        at = self.clock() if now is None else now
        if not isinstance(at, datetime) or at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("분봉 판단시각에는 시간대 정보가 있어야 합니다.")
        local = at.astimezone(KST)
        try:
            session = session_on(Market.DOMESTIC, local.date())
        except Exception as exc:
            raise MinuteFeedUnavailable("국내 거래소 세션을 검증할 수 없습니다.") from exc
        if session is None or not session.opened <= local < session.closed:
            raise MinuteFeedUnavailable("현재 국내 정규장이 아니므로 분봉 매매 판단을 중단합니다.")
        # Because the chart label's start/end convention is undocumented,
        # permit only labels that are a full bar away from both session edges.
        if local <= session.opened + 2 * INTERVAL:
            raise MinuteFeedUnavailable("장 초반에는 검증 가능한 완료 분봉이 부족합니다.")
        key = (symbol, local.date())
        path, receipt_path = self._paths(symbol, local.date())
        needed_slot = self._complete_slot(local)
        slot_key = (symbol, local.date(), needed_slot)
        with self._lock:
            if len(self._memory) > 256:
                self._memory = {key: value for key, value in self._memory.items()
                                if key[1] == local.date()}
                self._insufficient = {key: value for key, value in self._insufficient.items()
                                      if key[1] == local.date()}
                self._failed_slot = {key: value for key, value in self._failed_slot.items()
                                     if key[1] == local.date()}
            # Verify the on-disk receipt even for a warm in-process cache.  A
            # changed/deleted JSONL after the previous model's read cannot be
            # carried forward under an old, apparently valid hash.
            in_memory = self._memory.get(key)
            stored = self._read(path, receipt_path, symbol, local.date())
            if in_memory is not None and stored is None:
                raise MinuteFeedUnavailable("분봉 캐시가 조회 사이에 사라졌습니다.")
            if stored is not None:
                self._memory[key] = stored
            if stored is not None:
                if (stored.fetched_at.utcoffset() != timedelta(0)
                        or stored.fetched_at > at.astimezone(timezone.utc)
                        or self._merge((), stored.bars, symbol, local, session) != stored.bars):
                    raise MinuteFeedUnavailable("분봉 캐시 시각·장 구간·식별자가 유효하지 않습니다.")
            if stored is not None and stored.fetched_at >= needed_slot:
                selected = self._select(stored.bars, count, local, session)
                if selected is not None:
                    return self._snapshot(symbol, local.date(), selected, stored,
                                          path, receipt_path, from_cache=True)
            failure = self._failed_slot.get(slot_key)
            if failure is not None and (failure[2] >= MAX_SLOT_ATTEMPTS
                                        or at - failure[1] < RETRY_DELAY):
                raise MinuteFeedUnavailable(failure[0])
            shortage_key = (*slot_key, count)
            shortage = self._insufficient.get(shortage_key)
            if shortage is not None and (shortage[1] >= MAX_SLOT_ATTEMPTS
                                         or at - shortage[0] < RETRY_DELAY):
                raise MinuteFeedUnavailable("이번 완료 슬롯에는 요청한 연속 분봉이 부족합니다.")

            # A newer page contains recent bars; it is not a broker-supported
            # incremental cursor.  Only look farther back if the requested
            # contiguous suffix is not present in the latest page/cache.
            old = stored.bars if stored is not None else ()
            last_query = None
            for pages in (1, 2, MAX_PAGES):
                try:
                    result = self.broker.minute_bars_domestic(
                        symbol, exchange=DomesticExchange.KRX, interval_minutes=5,
                        base_date=local.date(), adjusted=True, as_of=at,
                        max_pages=pages)
                    page_count = getattr(result, "page_count", None)
                    truncated = getattr(result, "truncated", None)
                    if type(page_count) is not int or not 1 <= page_count <= pages or type(truncated) is not bool:
                        raise MinuteFeedUnavailable("분봉 조회 페이지 정보를 확인할 수 없습니다.")
                    merged = self._merge(old, tuple(result), symbol, local, session)
                except Exception as exc:
                    message = f"이번 완료 슬롯의 분봉 조회·검증이 실패했습니다: {type(exc).__name__}"
                    self._failed_slot[slot_key] = (
                        message, at, (failure[2] if failure is not None else 0) + 1)
                    raise MinuteFeedUnavailable(message) from exc
                last_query = (merged, page_count, truncated)
                selected = self._select(merged, count, local, session)
                if selected is None:
                    if not truncated:
                        break
                    continue
                fetched_at = self.clock().astimezone(timezone.utc)
                # Data must not be labelled as current after an unusually
                # long blocked broker request or after a session transition.
                if fetched_at - at.astimezone(timezone.utc) > MAX_REFRESH_DURATION:
                    raise MinuteFeedUnavailable("분봉 조회 중 판단시각이 지나 재조회가 필요합니다.")
                saved = self._write(path, receipt_path, symbol, local.date(),
                                    merged, fetched_at, page_count, truncated)
                self._memory[key] = saved
                self._failed_slot.pop(slot_key, None)
                self._insufficient.pop(shortage_key, None)
                return self._snapshot(symbol, local.date(), selected, saved,
                                      path, receipt_path, from_cache=False)
            if last_query is not None:
                merged, page_count, truncated = last_query
                if merged:
                    fetched_at = self.clock().astimezone(timezone.utc)
                    if fetched_at - at.astimezone(timezone.utc) <= MAX_REFRESH_DURATION:
                        self._memory[key] = self._write(
                            path, receipt_path, symbol, local.date(), merged,
                            fetched_at, page_count, truncated)
            self._insufficient[shortage_key] = (
                at, (shortage[1] if shortage is not None else 0) + 1)
            raise MinuteFeedUnavailable("현재 장의 최신 연속 5분봉이 부족하거나 지연됐습니다.")

    @staticmethod
    def _complete_slot(local: datetime) -> datetime:
        # A label at 10:00 cannot be trusted until 10:05.  Compare against the
        # start of its eligibility slot rather than fetching again per model.
        minute = (local.minute // 5) * 5
        return local.replace(minute=minute, second=0, microsecond=0)

    def _paths(self, symbol: str, day: date) -> tuple[Path, Path]:
        path = self.cache_dir / f"domestic-KRX-{symbol}-{day:%Y%m%d}-5m.jsonl"
        return path, path.with_suffix(path.suffix + ".receipt.json")

    @staticmethod
    def _validate_bar(bar: MinuteBar, symbol: str, local: datetime, session) -> bool:
        if (not isinstance(bar, MinuteBar) or bar.market is not Market.DOMESTIC
                or bar.symbol != symbol or bar.exchange != "KRX" or bar.currency != "KRW"):
            raise MinuteFeedUnavailable("분봉 시장·거래소·종목 식별자가 요청과 다릅니다.")
        label = bar.timestamp
        if label.tzinfo is None or label.utcoffset() != local.utcoffset():
            raise MinuteFeedUnavailable("분봉 시각의 국내 거래소 시간대를 확인할 수 없습니다.")
        label = label.astimezone(KST)
        if label.date() != session.opened.date():
            return False
        if label + INTERVAL > local:
            return False
        if not session.opened + INTERVAL < label < session.closed - INTERVAL:
            return False
        if (label.second or label.microsecond or label.minute % 5):
            raise MinuteFeedUnavailable("분봉 시각이 5분 간격에 맞지 않습니다.")
        values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        if (any(not isinstance(value, Decimal) or not value.is_finite() for value in values)
                or min(values[:4]) <= 0 or bar.volume < 0
                or bar.high < max(bar.open, bar.close, bar.low)
                or bar.low > min(bar.open, bar.close)):
            raise MinuteFeedUnavailable("분봉 OHLCV 값이 유효하지 않습니다.")
        return True

    @classmethod
    def _merge(cls, old: tuple[MinuteBar, ...], incoming: tuple[MinuteBar, ...],
               symbol: str, local: datetime, session) -> tuple[MinuteBar, ...]:
        by_label: dict[datetime, MinuteBar] = {}
        for bar in (*old, *incoming):
            if not cls._validate_bar(bar, symbol, local, session):
                continue
            label = bar.timestamp.astimezone(timezone.utc)
            prior = by_label.get(label)
            if prior is not None and cls._bar_row(prior) != cls._bar_row(bar):
                raise MinuteFeedUnavailable("동일 시각 분봉의 값이 달라 판단을 중단합니다.")
            by_label[label] = bar
        return tuple(by_label[key] for key in sorted(by_label))

    @staticmethod
    def _select(bars: tuple[MinuteBar, ...], count: int, local: datetime,
                session) -> tuple[MinuteBar, ...] | None:
        valid = tuple(bar for bar in bars
                      if DomesticMinuteFeed._validate_bar(bar, bars[-1].symbol, local, session)) if bars else ()
        if len(valid) < count:
            return None
        selected = valid[-count:]
        if any(right.timestamp - left.timestamp != INTERVAL
               for left, right in zip(selected, selected[1:])):
            return None
        if (selected[-1].timestamp.astimezone(KST)
                < DomesticMinuteFeed._complete_slot(local) - INTERVAL):
            return None
        return selected

    @staticmethod
    def _bar_row(bar: MinuteBar) -> dict[str, str | int]:
        return {
            "market": "domestic", "exchange": "KRX", "symbol": bar.symbol,
            "timestamp": bar.timestamp.astimezone(KST).isoformat(),
            "bar_minutes": 5, "open": str(bar.open), "high": str(bar.high),
            "low": str(bar.low), "close": str(bar.close), "volume": str(bar.volume),
        }

    @classmethod
    def _payload(cls, bars: tuple[MinuteBar, ...]) -> bytes:
        return b"".join((json.dumps(cls._bar_row(bar), sort_keys=True,
                                   ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                        for bar in bars)

    @classmethod
    def _write(cls, path: Path, receipt_path: Path, symbol: str, day: date,
               bars: tuple[MinuteBar, ...], fetched_at: datetime,
               page_count: int, truncated: bool) -> _Stored:
        if not bars:
            raise MinuteFeedUnavailable("빈 분봉 캐시는 저장하지 않습니다.")
        payload = cls._payload(bars)
        digest = sha256(payload).hexdigest()
        receipt = {
            "format": "dockdack-domestic-minute-feed-v1", "source": _SOURCE,
            "market": "domestic", "exchange": "KRX", "symbol": symbol,
            "session_date": day.isoformat(), "interval_minutes": 5,
            "fetched_at_utc": fetched_at.isoformat(),
            "last_bar_label": bars[-1].timestamp.isoformat(),
            "bar_count": len(bars), "sha256": digest,
            "page_count": page_count, "truncated": truncated,
            "cursor_used": False,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        stage_data = stage_receipt = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix=".minute-data-", delete=False) as out:
                stage_data = Path(out.name)
                out.write(payload)
            with NamedTemporaryFile(dir=path.parent, prefix=".minute-receipt-", delete=False) as out:
                stage_receipt = Path(out.name)
                out.write((json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
            os.replace(stage_data, path)
            os.replace(stage_receipt, receipt_path)
        finally:
            for stage in (stage_data, stage_receipt):
                if stage is not None and stage.exists():
                    stage.unlink()
        return _Stored(bars, fetched_at, digest, page_count, truncated)

    @classmethod
    def _read(cls, path: Path, receipt_path: Path, symbol: str,
              day: date) -> _Stored | None:
        if not path.exists() and not receipt_path.exists():
            return None
        if not path.is_file() or not receipt_path.is_file() or path.is_symlink() or receipt_path.is_symlink():
            raise MinuteFeedUnavailable("분봉 캐시 데이터와 영수증이 일치하지 않습니다.")
        payload = path.read_bytes()
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt["format"] != "dockdack-domestic-minute-feed-v1"
                    or receipt["source"] != _SOURCE
                    or receipt["market"] != "domestic"
                    or receipt["exchange"] != "KRX"
                    or receipt["symbol"] != symbol
                    or receipt["session_date"] != day.isoformat()
                    or receipt["interval_minutes"] != 5
                    or receipt["cursor_used"] is not False
                    or receipt["sha256"] != sha256(payload).hexdigest()):
                raise ValueError("receipt mismatch")
            rows = [json.loads(line) for line in payload.splitlines()]
            bars = tuple(cls._parse_row(row, symbol) for row in rows)
            if (len(bars) != receipt["bar_count"] or not bars
                    or bars[-1].timestamp.isoformat() != receipt["last_bar_label"]
                    or any(a.timestamp >= b.timestamp for a, b in zip(bars, bars[1:]))):
                raise ValueError("bars mismatch")
            fetched_at = datetime.fromisoformat(receipt["fetched_at_utc"])
            if fetched_at.tzinfo is None or fetched_at.utcoffset() != timedelta(0):
                raise ValueError("receipt fetch time must be UTC")
            pages = receipt["page_count"]
            truncated = receipt["truncated"]
            if type(pages) is not int or not 1 <= pages <= MAX_PAGES or type(truncated) is not bool:
                raise ValueError("invalid pagination")
            return _Stored(bars, fetched_at, receipt["sha256"], pages, truncated)
        except (OSError, UnicodeError, KeyError, TypeError, ValueError, InvalidOperation) as exc:
            raise MinuteFeedUnavailable("분봉 캐시 무결성 확인에 실패했습니다.") from exc

    @staticmethod
    def _parse_row(row: dict, symbol: str) -> MinuteBar:
        if (set(row) != {"market", "exchange", "symbol", "timestamp", "bar_minutes",
                         "open", "high", "low", "close", "volume"}
                or row["market"] != "domestic" or row["exchange"] != "KRX"
                or row["symbol"] != symbol or row["bar_minutes"] != 5):
            raise ValueError("cache row identity")
        stamp = datetime.fromisoformat(row["timestamp"])
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("cache row clock")
        return MinuteBar(Market.DOMESTIC, symbol, "KRX", stamp,
                         *(Decimal(row[field]) for field in
                           ("open", "high", "low", "close", "volume")), "KRW")

    @staticmethod
    def _snapshot(symbol: str, day: date, bars: tuple[MinuteBar, ...],
                  stored: _Stored, path: Path, receipt_path: Path,
                  *, from_cache: bool) -> MinuteFeedSnapshot:
        return MinuteFeedSnapshot(symbol, "KRX", day, bars, stored.fetched_at,
                                  bars[-1].timestamp, stored.sha256, path,
                                  receipt_path, stored.page_count,
                                  stored.truncated, from_cache)
