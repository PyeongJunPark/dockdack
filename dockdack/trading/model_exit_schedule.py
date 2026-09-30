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
    # minute_elapsed: observed-fill-based intraday timeout, with same-session
    # preclose escape so a short-horizon prototype never intentionally carries
    # overnight because its horizon ran past the closing bell.
    timing: str
    minutes_after_fill: int | None = None

    def __post_init__(self):
        if (type(self.sessions_after_fill) is not int
                or not 0 <= self.sessions_after_fill <= 30
                or self.timing not in {"preclose", "elapsed", "minute_elapsed"}
                or (self.timing == "minute_elapsed") != (self.minutes_after_fill is not None)
                or (self.minutes_after_fill is not None
                    and (type(self.minutes_after_fill) is not int
                         or not 1 <= self.minutes_after_fill <= 390))):
            raise ValueError("Invalid model exit schedule")


@dataclass(frozen=True)
class PlannedModelExit:
    """Earliest strategy exit session, not an order or a guaranteed fill."""

    day: date
    timing: str
    at: datetime | None = None


MODEL_EXIT_SCHEDULES = {
    "mark1-3-prototype": ModelExitSchedule(0, "preclose"),
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
    # Target/horizon daily-bar experiments: fill session is day one. If the
    # upper target has not closed the confirmed lot, attempt its exit in the
    # final five minutes of the Hth exchange session. This is an order window,
    # not a promise of a close-price fill.
    "mark1-23-prototype": ModelExitSchedule(9, "preclose"),
    "mark1-24-prototype": ModelExitSchedule(19, "preclose"),
    "mark1-25-prototype": ModelExitSchedule(19, "preclose"),
    "mark1-26-prototype": ModelExitSchedule(19, "preclose"),
    "mark1-27-prototype": ModelExitSchedule(9, "preclose"),
    "mark1-28-prototype": ModelExitSchedule(9, "preclose"),
    **{f"mark1-{number}-prototype": ModelExitSchedule(0, "minute_elapsed", horizon * 5)
       for number, horizon in ((29, 3), (30, 3), (31, 3),
                               (32, 6), (33, 6), (34, 6),
                               (35, 12), (36, 12), (37, 12))},
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


def _observed_fill(lot: dict) -> tuple[ModelExitSchedule, datetime] | None:
    spec = model_exit_schedule(lot.get("strategy_id"))
    observed = lot.get("buy_fill_observed_at")
    if spec is None or not isinstance(observed, str) or not observed:
        return None
    try:
        fill_time = datetime.fromisoformat(observed.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if fill_time.tzinfo is None or fill_time.utcoffset() is None:
        return None
    return spec, fill_time


def _planned_exit(spec: ModelExitSchedule, fill_time: datetime, market: Market) -> PlannedModelExit:
    if spec.timing == "minute_elapsed":
        due_at = fill_time + timedelta(minutes=spec.minutes_after_fill)
        local_day = market_time(market, fill_time).date()
        first_session = session_on(market, local_day)
        if first_session is not None and first_session.opened <= fill_time < first_session.closed:
            earliest = max(fill_time, min(due_at, first_session.closed - timedelta(minutes=5)))
            return PlannedModelExit(local_day, spec.timing, earliest)
        # An out-of-session observation must never display an earlier exit.
        for offset in range(370):
            session = session_on(market, local_day + timedelta(days=offset))
            if session is None:
                continue
            earliest = max(due_at, session.opened)
            if earliest < session.closed:
                return PlannedModelExit(market_time(market, earliest).date(), spec.timing, earliest)
        raise ValueError("No regular-session minute exit after observed fill")
    first = _filled_session_day(market, fill_time)
    target = first if spec.sessions_after_fill == 0 else _target_day(market, first, spec.sessions_after_fill)
    return PlannedModelExit(target, spec.timing)


def planned_model_exit(lot: dict, market: Market | str) -> PlannedModelExit | None:
    """Read-only exchange-local date for a confirmed model lot, if verifiable."""
    market = Market(market)
    observed = _observed_fill(lot)
    return _planned_exit(*observed, market) if observed is not None else None


def timed_exit_due(lot: dict, market: Market | str, now: datetime) -> bool:
    """True only inside this model's own regular-session exit window.

    Unknown fill observation or disabled/unknown strategy stays false. An
    overdue fill may exit at the next valid window rather than being stranded.
    """
    market = Market(market)
    observed = _observed_fill(lot)
    if observed is None:
        return False
    spec, fill_time = observed
    if (now.tzinfo is None or now.utcoffset() is None
            or now.astimezone(timezone.utc) < fill_time.astimezone(timezone.utc)):
        return False
    if spec.timing == "minute_elapsed":
        if market is not Market.DOMESTIC or spec.minutes_after_fill is None:
            return False
        fill_session = session_on(market, market_time(market, fill_time).date())
        today = market_time(market, now).date()
        session = session_on(market, today)
        if session is None or not session.opened <= now < session.closed:
            return False
        due_at = fill_time + timedelta(minutes=spec.minutes_after_fill)
        if fill_session is None or not fill_session.opened <= fill_time < fill_session.closed:
            # Observation time is not broker fill time. A confirmed fill first
            # seen before open, after close, or on a holiday still gets a
            # future regular-session exit after its full observed-time horizon.
            return now >= due_at
        return now >= due_at or now >= session.closed - timedelta(minutes=5)
    planned = _planned_exit(spec, fill_time, market)
    target = planned.day
    today = market_time(market, now).date()
    if today < target:
        return False
    session = session_on(market, today)
    if session is None or not session.opened <= now < session.closed:
        return False
    return planned.timing == "elapsed" or now >= session.closed - timedelta(minutes=5)
