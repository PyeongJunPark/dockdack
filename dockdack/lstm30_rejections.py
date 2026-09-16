"""Persistent per-instrument quarantine for confirmed LSTM30 order rejections.

A confirmed broker rejection is not an unknown order outcome. Other instruments
can continue, but the rejected instrument must not retry through a new signal,
another order path, or a process restart during the same market-local day.
"""

from datetime import datetime

from dockdack.history import market_time
from dockdack.watchlist import WatchItem


def rejected_today(store, instrument, now):
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("주문 거절 확인에는 시간대가 있는 현재 시각이 필요합니다.")
    today = market_time(instrument.market, now).date()
    for attempt in store.attempts(WatchItem(instrument).id):
        if attempt["status"] != "rejected":
            continue
        try:
            started = datetime.fromisoformat(attempt["started_at"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("거절 주문의 발생 시각을 확인할 수 없습니다.") from exc
        if started.tzinfo is None or started.utcoffset() is None:
            raise ValueError("거절 주문의 발생 시간대를 확인할 수 없습니다.")
        if market_time(instrument.market, started).date() == today:
            return True
    return False
