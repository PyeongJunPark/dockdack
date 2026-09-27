"""Immutable, model-owned holding periods for experimental DEMO lots.

The account-wide closing liquidator is intentionally not used here.  A policy
applies only to confirmed fills whose persisted strategy ID names that model.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from dockdack.history import market_time
from dockdack.market_schedule import session_on
from dockdack.models import Market


@dataclass(frozen=True)
class ModelExitSchedule:
    # Number of *later exchange sessions* after the first observed fill session.
    sessions_after_fill: int
    # preclose: final five minutes; elapsed: any regular-session time on/after due date.
    timing: str

    def __post_init__(self):
        if (type(self.sessions_after_fill) is not int
                or not 0 <= self.sessions_after_fill <= 30
                or self.timing not in {"preclose", "elapsed"}):
            raise ValueError("Invalid model exit schedule")


MODEL_EXIT_SCHEDULES = {
    "mark1-4-prototype": ModelExitSchedule(0, "preclose"),
    "mark1-5-prototype": ModelExitSchedule(0, "preclose"),
    "mark1-6-prototype": ModelExitSchedule(0, "preclose"),
    "mark1-7-prototype": ModelExitSchedule(0, "preclose"),
    "mark1-8-prototype": ModelExitSchedule(0, "preclose"),
    "mark1-9-prototype": ModelExitSchedule(0, "preclose"),
    "mark1-10-prototype": ModelExitSchedule(0, "preclose"),
    # H3/H5 count the observed fill session as day one. Once the final
    # session starts, the signal may close the confirmed lot at any price.
    "mark1-11-prototype": ModelExitSchedule(2, "elapsed"),
    "mark1-12-prototype": ModelExitSchedule(4, "elapsed"),
}


def model_exit_schedule(strategy_id: str) -> ModelExitSchedule | None:
    return MODEL_EXIT_SCHEDULES.get(strategy_id)


def _filled_session_day(market: Market, observed_at: datetime) -> date:
    """Use a conservative observed fill day, never the earlier order intent."""
    local_day = market_time(market, observed_at).date()
    for offset in range(370):
        day = local_day + timedelta(days=offset)
        if session_on(market, day) is not None:
            return day
    raise ValueError("No exchange session after observed model fill")


def _target_day(market: Market, first_day: date, later_sessions: int) -> date:
    count = 0
    for offset in range(1, 370):
        day = first_day + timedelta(days=offset)
        if session_on(market, day) is None:
            continue
        count += 1
        if count == later_sessions:
            return day
    raise ValueError("Model exit horizon exceeds exchange calendar")


def timed_exit_due(lot: dict, market: Market | str, now: datetime) -> bool:
    """True only inside this model's own regular-session exit window.

    Unknown fill observation or disabled/unknown strategy stays false. An
    overdue fill may exit at the next valid window rather than being stranded.
    """
    market = Market(market)
    spec = model_exit_schedule(lot.get("strategy_id"))
    observed = lot.get("buy_fill_observed_at")
    if spec is None or not isinstance(observed, str) or not observed:
        return False
    try:
        fill_time = datetime.fromisoformat(observed.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if (fill_time.tzinfo is None or fill_time.utcoffset() is None
            or now.tzinfo is None or now.utcoffset() is None
            or now.astimezone(timezone.utc) < fill_time.astimezone(timezone.utc)):
        return False
    first = _filled_session_day(market, fill_time)
    target = first if spec.sessions_after_fill == 0 else _target_day(market, first, spec.sessions_after_fill)
    today = market_time(market, now).date()
    if today < target:
        return False
    session = session_on(market, today)
    if session is None or not session.opened <= now < session.closed:
        return False
    return spec.timing == "elapsed" or now >= session.closed - timedelta(minutes=5)
