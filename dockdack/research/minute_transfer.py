"""Daily-bar pretraining followed by genuine completed five-minute-bar adaptation.

This is an offline, research-only experiment. A daily candle is treated as a
generic OHLCV token, *not* represented as an observed minute candle. The model
is adapted and evaluated on real five-minute bars before a paper signal can be
emitted. None of the classes here has a broker or order API.

Input JSONL: one completed bar per line with market (domestic/us), symbol,
timestamp (ISO-8601 with the exchange's local UTC offset), open, high, low,
close, volume, and bar_minutes (1440 for daily, 5 for five-minute). The minute
timestamp is the broker's five-minute bar label: its start/end convention is
not assumed. The daily timestamp must equal that session's close.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from itertools import chain
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dockdack.market_schedule import session_on
from dockdack.models import Market


BAR_FIELDS = frozenset({"market", "symbol", "timestamp", "open", "high", "low",
                        "close", "volume", "bar_minutes"})
BAR_OPTIONAL_FIELDS = frozenset({"exchange"})
LOOKBACK = 20
HORIZON = 3
FEATURES = 5
ARCHITECTURES = ("linear", "conv", "gru")
MARKET_ZONE = {"domestic": "Asia/Seoul", "us": "America/New_York"}
MAX_PILOT_BARS_PER_DOMAIN = 100_000


@dataclass(frozen=True)
class Bar:
    market: str
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_minutes: int
    exchange: str = ""

    @property
    def session_date(self) -> date:
        return self.timestamp.date()


@dataclass(frozen=True)
class Sample:
    features: np.ndarray
    market: str
    symbol: str
    first_session: date
    entry_session: date
    exit_session: date
    decision_at: datetime
    entry_at: datetime
    exit_at: datetime
    net_return: float
    exchange: str = ""

    @property
    def label(self) -> float:
        return float(self.net_return > 0)


class OutsideRegularSession(ValueError):
    """An otherwise valid broker bar falls outside a regular session."""


def _market_session(bar: Bar):
    from zoneinfo import ZoneInfo

    local = bar.timestamp.astimezone(ZoneInfo(MARKET_ZONE[bar.market]))
    if local.replace(tzinfo=None) != bar.timestamp.replace(tzinfo=None):
        raise ValueError(f"{bar.market} timestamp is not in the exchange local zone")
    session = session_on(Market(bar.market), local.date())
    if session is None:
        raise OutsideRegularSession(f"{bar.market} bar is outside a regular session: {bar.timestamp}")
    return session


def _bar_from_mapping(value: dict, *, kind: str, as_of: datetime) -> Bar:
    if not BAR_FIELDS <= set(value) or set(value) - BAR_FIELDS - BAR_OPTIONAL_FIELDS:
        raise ValueError(f"JSONL requires {sorted(BAR_FIELDS)}; only exchange is optional")
    market = value["market"]
    if market not in MARKET_ZONE:
        raise ValueError("market must be domestic or us")
    symbol = value["symbol"]
    if not isinstance(symbol, str) or not symbol.strip() or symbol != symbol.strip():
        raise ValueError("symbol must be a nonempty, trimmed string")
    exchange = value.get("exchange", "")
    if not isinstance(exchange, str) or exchange != exchange.strip():
        raise ValueError("exchange must be a trimmed string when present")
    try:
        timestamp = datetime.fromisoformat(value["timestamp"])
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must contain an explicit UTC offset")
    expected = 1440 if kind == "daily" else 5
    if type(value["bar_minutes"]) is not int or value["bar_minutes"] != expected:
        raise ValueError(f"{kind} bar_minutes must equal {expected}")
    # The broker's minute label may denote start or end. Requiring one entire
    # extra interval after the label is conservative in both conventions.
    grace = timedelta(minutes=5) if kind == "minute" else timedelta()
    if timestamp + grace >= as_of:
        raise ValueError("bar is still forming, not finalized, or in the future")
    numbers = []
    for name in ("open", "high", "low", "close", "volume"):
        raw = value[name]
        if isinstance(raw, bool):
            raise ValueError(f"{name} must be numeric")
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(number) or (number <= 0 if name != "volume" else number < 0):
            raise ValueError(f"{name} must be finite and nonnegative/positive")
        numbers.append(number)
    op, hi, lo, cl, volume = numbers
    if hi < max(op, lo, cl) or lo > min(op, hi, cl):
        raise ValueError("inconsistent OHLC")
    bar = Bar(market, symbol, timestamp, op, hi, lo, cl, volume, expected, exchange)
    session = _market_session(bar)
    if kind == "daily":
        if timestamp != session.closed:
            raise ValueError("daily timestamp must be that exchange session close")
    else:
        offset = timestamp - session.opened
        if not timedelta() <= offset <= session.closed - session.opened:
            raise OutsideRegularSession("five-minute bar label is outside a regular session")
        if offset.total_seconds() % 300 != 0:
            raise ValueError("five-minute broker bar label must align to an in-session slot")
    return bar


def load_bars(path: str | Path, *, kind: str,
              as_of: datetime | None = None,
              stats: dict | None = None) -> tuple[Bar, ...]:
    """Read-only JSONL loader; regular-only and strict on malformed/live bars.

    The broker snapshot may contain pre/post-market minute bars. Such bars are
    excluded, and an optional stats dictionary records the exact count. Price,
    timezone, duplicate and completion errors within regular hours fail closed.
    """
    if kind not in {"daily", "minute"}:
        raise ValueError("kind must be daily or minute")
    as_of = as_of or datetime.now(timezone.utc)
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    result = []
    seen = set()
    rows_seen = excluded = 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            rows_seen += 1
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL row must be an object")
                bar = _bar_from_mapping(value, kind=kind, as_of=as_of)
            except OutsideRegularSession as exc:
                if kind == "minute":
                    excluded += 1
                    continue
                raise ValueError(f"{path}:{number}: {exc}") from exc
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            key = (bar.market, bar.exchange, bar.symbol, bar.timestamp)
            if key in seen:
                raise ValueError(f"{path}:{number}: duplicate market/symbol/timestamp")
            seen.add(key)
            result.append(bar)
    if not result:
        raise ValueError(f"{path}: no complete {kind} bars")
    if stats is not None:
        stats.update({"jsonl_rows": rows_seen, "regular_session_excluded": excluded,
                      "accepted_bars": len(result)})
    return tuple(result)


def _consecutive(previous: Bar, current: Bar, *, kind: str) -> bool:
    if kind == "minute":
        return (previous.session_date == current.session_date
                and current.timestamp - previous.timestamp == timedelta(minutes=5))
    # Do not silently join a symbol across a missing exchange session. A real
    # suspension/missing observation starts a new run, not a fabricated candle.
    cursor = previous.session_date + timedelta(days=1)
    while cursor < current.session_date:
        if session_on(Market(previous.market), cursor) is not None:
            return False
        cursor += timedelta(days=1)
    return current.session_date > previous.session_date


def _features(history: tuple[Bar, ...] | list[Bar]) -> np.ndarray:
    if not history:
        raise ValueError("empty history")
    arr = np.asarray([(b.open, b.high, b.low, b.close, b.volume)
                      for b in history], dtype=np.float64)
    op, high, low, close, volume = arr.T
    last_close = close[-1]
    log_volume = np.log1p(volume)
    result = np.column_stack((np.log(close / last_close),
                              np.log(high / close), np.log(low / close),
                              np.log(close / op),
                              log_volume - np.median(log_volume))).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite causal features")
    return result


def make_samples(bars: Iterable[Bar], *, kind: str, lookback: int = LOOKBACK,
                 horizon: int = HORIZON, fee_bps_per_side: float = 2.,
                 slippage_bps_per_side: float = 8.) -> tuple[Sample, ...]:
    """Features stop at t; entry/exit proxy use later observed bar opens.

    Five-minute labels/history must all fit the same complete exchange session.
    A split/window never jumps over an unavailable bar or exchange session.
    Minute entry is t+3, leaving t+1 and t+2 as two latency bars because the
    broker's timestamp start/end convention is unverified and the last input
    bar is accepted only one interval after its label. Daily entry is t+1.
    """
    if kind not in {"daily", "minute"} or lookback < 2 or horizon < 1:
        raise ValueError("invalid kind/lookback/horizon")
    for value in (fee_bps_per_side, slippage_bps_per_side):
        if not math.isfinite(value) or not 0 <= value < 10_000:
            raise ValueError("fee and slippage bps must be finite, nonnegative and below 10000")
    groups: dict[tuple[str, str, str], list[Bar]] = {}
    for bar in bars:
        if bar.bar_minutes != (1440 if kind == "daily" else 5):
            raise ValueError("bar interval does not match kind")
        groups.setdefault((bar.market, bar.exchange, bar.symbol), []).append(bar)
    samples = []
    for (market, exchange, symbol), group in groups.items():
        group.sort(key=lambda b: b.timestamp)
        run = [0] * len(group)
        for index in range(1, len(group)):
            if group[index].timestamp <= group[index - 1].timestamp:
                raise ValueError("duplicate or unordered symbol timestamps")
            run[index] = run[index - 1] + int(not _consecutive(group[index - 1], group[index], kind=kind))
        latency = 3 if kind == "minute" else 1
        for t in range(lookback - 1, len(group) - horizon - latency):
            first, entry_i, exit_i = t - lookback + 1, t + latency, t + latency + horizon
            if run[first] != run[exit_i]:
                continue
            history = group[first:t + 1]
            entry, exit_bar = group[entry_i], group[exit_i]
            if kind == "minute" and history[0].session_date != exit_bar.session_date:
                continue
            # Minute timestamps below are BAR LABELS, not asserted wall-clock
            # entry/exit instants. The return uses only their observed OPENs.
            entry_at = (entry.timestamp if kind == "minute"
                        else session_on(Market(market), entry.session_date).opened)
            exit_at = (exit_bar.timestamp if kind == "minute"
                       else session_on(Market(market), exit_bar.session_date).opened)
            if not (history[-1].timestamp < entry_at < exit_at):
                raise ValueError("entry/exit proxy has an invalid chronological boundary")
            fee = fee_bps_per_side / 10_000.
            slip = slippage_bps_per_side / 10_000.
            net = (exit_bar.open * (1 - slip) * (1 - fee)
                   / (entry.open * (1 + slip) * (1 + fee)) - 1.)
            samples.append(Sample(_features(history), market, symbol,
                                  history[0].session_date, entry.session_date,
                                  exit_bar.session_date, history[-1].timestamp,
                                  entry_at, exit_at, net, exchange))
    return tuple(sorted(samples, key=lambda s: (s.entry_at, s.market, s.symbol)))


def split_sessions(samples: Iterable[Sample], *, purge_sessions: int = 3,
                   train_fraction: float = .70, val_fraction: float = .15
                   ) -> dict[str, tuple[Sample, ...]]:
    """Chronological exchange-date split with blank sessions around boundaries.

    Validation/test features may reference prior history, which is observable,
    but their entry and label exit must be inside their split. The purge removes
    both sides of each boundary globally across symbols/markets.
    """
    rows = tuple(samples)
    if type(purge_sessions) is not int or purge_sessions < 0:
        raise ValueError("purge_sessions must be nonnegative")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1
            and train_fraction + val_fraction < 1):
        raise ValueError("invalid split fractions")
    days = sorted({date_value for row in rows
                   for date_value in (row.entry_session, row.exit_session)})
    if len(days) < 3:
        raise ValueError("at least three independent sessions required")
    a, b = int(len(days) * train_fraction), int(len(days) * (train_fraction + val_fraction))
    if not (0 < a < b < len(days)):
        raise ValueError("insufficient sessions for train/val/test")
    lookup = {day: index for index, day in enumerate(days)}
    ranges = {"train": (0, a - purge_sessions),
              "val": (a + purge_sessions, b - purge_sessions),
              "test": (b + purge_sessions, len(days))}
    result = {}
    for name, (start, stop) in ranges.items():
        result[name] = tuple(row for row in rows
                             if start <= lookup[row.entry_session]
                             and lookup[row.exit_session] < stop)
    return result


class LinearBarNet(nn.Module):
    def __init__(self, lookback: int):
        super().__init__()
        self.lookback = lookback
        self.classifier = nn.Linear(lookback * FEATURES, 1)

    def forward(self, x):
        return self.classifier(x.flatten(start_dim=1)).flatten()


class ConvBarNet(nn.Module):
    def __init__(self, lookback: int):
        super().__init__()
        self.lookback = lookback
        self.conv = nn.Sequential(nn.Conv1d(FEATURES, 12, kernel_size=3), nn.GELU(),
                                  nn.Conv1d(12, 12, kernel_size=3), nn.GELU())
        self.classifier = nn.Linear(12, 1)

    def forward(self, x):
        return self.classifier(self.conv(x.transpose(1, 2)).mean(dim=2)).flatten()


class RecurrentBarNet(nn.Module):
    def __init__(self, lookback: int):
        super().__init__()
        self.lookback = lookback
        self.gru = nn.GRU(FEATURES, 12, batch_first=True)
        self.classifier = nn.Linear(12, 1)

    def forward(self, x):
        _, state = self.gru(x)
        return self.classifier(state[-1]).flatten()


def build_model(architecture: str, lookback: int = LOOKBACK) -> nn.Module:
    if lookback < 5 and architecture == "conv":
        raise ValueError("convolution needs at least five completed bars")
    types = {"linear": LinearBarNet, "conv": ConvBarNet, "gru": RecurrentBarNet}
    if architecture not in types:
        raise ValueError("unknown transfer architecture")
    return types[architecture](lookback)


def fit_scaler(train: tuple[Sample, ...]) -> tuple[np.ndarray, np.ndarray]:
    if not train:
        raise ValueError("no training samples")
    matrix = np.stack([sample.features for sample in train]).astype(np.float64)
    mean = matrix.mean(axis=(0, 1)).astype(np.float32)
    scale = np.maximum(matrix.std(axis=(0, 1)).astype(np.float32), 1e-5)
    return mean, scale


def _tensor(rows: tuple[Sample, ...], mean: np.ndarray, scale: np.ndarray):
    x = np.stack([row.features for row in rows]).astype(np.float32)
    x = np.clip((x - mean) / scale, -8, 8)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite scaled features")
    y = np.asarray([row.label for row in rows], dtype=np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def _fit(model: nn.Module, rows: tuple[Sample, ...], mean: np.ndarray,
         scale: np.ndarray, *, epochs: int, learning_rate: float, seed: int):
    x, y = _tensor(rows, mean, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.001)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for batch in torch.randperm(len(x), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x[batch]), y[batch])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
    model.eval()


def _probabilities(model: nn.Module, rows: tuple[Sample, ...],
                   mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if not rows:
        return np.empty(0, dtype=np.float64)
    x, _ = _tensor(rows, mean, scale)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(x)).cpu().numpy().astype(np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("nonfinite inference probability")
    return probabilities


def _choose_threshold(rows: tuple[Sample, ...], probabilities: np.ndarray,
                      *, min_trades: int) -> float | None:
    # Threshold search is validation-only; zero eligible positive candidates
    # means the paper signal explicitly abstains instead of manufacturing one.
    choices = []
    returns = np.asarray([row.net_return for row in rows])
    for threshold in (.50, .55, .60):
        selected = probabilities >= threshold
        if int(selected.sum()) < min_trades:
            continue
        total = float(returns[selected].sum())
        if total > 0:
            choices.append((total, threshold))
    return max(choices)[1] if choices else None


def _report(rows: tuple[Sample, ...], probabilities: np.ndarray,
            threshold: float | None) -> dict:
    selected = np.zeros(len(rows), dtype=bool) if threshold is None else probabilities >= threshold
    net = np.asarray([row.net_return for row in rows])
    return {"samples": len(rows), "candidate_count": int(selected.sum()),
            "candidate_mean_net_return": (float(net[selected].mean()) if selected.any() else None),
            "candidate_total_net_return": (float(net[selected].sum()) if selected.any() else None),
            "all_mean_net_return": (float(net.mean()) if len(net) else None),
            "unit": "fraction per latency-adjusted bar-open to later bar-open historical proxy; not actual fills"}


def _cap_training_rows(rows: tuple[Sample, ...], maximum: int) -> tuple[Sample, ...]:
    if len(rows) <= maximum:
        return rows
    # Preserve early and late market regimes without preferentially retaining
    # only the oldest rows when the CPU/memory training cap is reached.
    indices = np.linspace(0, len(rows) - 1, maximum, dtype=np.int64)
    return tuple(rows[int(index)] for index in indices)


def train_transfer(daily: tuple[Bar, ...], minute: tuple[Bar, ...], *,
                   lookback: int = LOOKBACK, horizon: int = HORIZON,
                   fee_bps_per_side: float = 2., slippage_bps_per_side: float = 8.,
                   purge_sessions: int = 3,
                   pretrain_epochs: int = 4, adapt_epochs: int = 4,
                   min_samples_per_split: int = 30, min_validation_trades: int = 5,
                   max_train_samples: int = 50_000, seed: int = 20260929):
    """Fit three distinct models; return in-memory artifacts and honest reports.

    No synthetic augmentation, brokerage, account access, automatic promotion
    or model-to-order bridge is present. Cross-market candidates are research
    only; caller must run each market independently for meaningful review.
    """
    if not daily or not minute or {b.market for b in daily} != {b.market for b in minute}:
        raise ValueError("daily/minute inputs must contain the same one market")
    if len({b.market for b in daily}) != 1:
        raise ValueError("train one exchange market at a time")
    if any(b.exchange == "INDEX" for b in chain(daily, minute)):
        raise ValueError("stock transfer candidates cannot train on benchmark-index bars")
    if len(daily) > MAX_PILOT_BARS_PER_DOMAIN or len(minute) > MAX_PILOT_BARS_PER_DOMAIN:
        raise ValueError("offline CPU pilot accepts at most 100000 bars per domain")
    if min(pretrain_epochs, adapt_epochs, min_samples_per_split,
           min_validation_trades, max_train_samples) < 1:
        raise ValueError("epochs/sample limits must be positive")
    daily_samples = make_samples(daily, kind="daily", lookback=lookback,
                                 horizon=horizon, fee_bps_per_side=fee_bps_per_side,
                                 slippage_bps_per_side=slippage_bps_per_side)
    minute_samples = make_samples(minute, kind="minute", lookback=lookback,
                                  horizon=horizon, fee_bps_per_side=fee_bps_per_side,
                                  slippage_bps_per_side=slippage_bps_per_side)
    daily_split = split_sessions(daily_samples, purge_sessions=purge_sessions)
    minute_split = split_sessions(minute_samples, purge_sessions=purge_sessions)
    for domain, splits in (("daily", daily_split), ("minute", minute_split)):
        for name, rows in splits.items():
            if len(rows) < min_samples_per_split:
                raise ValueError(f"{domain} {name} has {len(rows)} samples, needs {min_samples_per_split}")
    daily_train = _cap_training_rows(daily_split["train"], max_train_samples)
    minute_train = _cap_training_rows(minute_split["train"], max_train_samples)
    # Independent domain splits do not establish cross-domain chronology. A
    # later daily label must never pretrain a model evaluated on earlier
    # minute observations, even when each domain is internally purged.
    latest_daily_pretraining_exit = max(row.exit_at for row in daily_train)
    earliest_minute_adaptation_entry = min(row.entry_at for row in minute_train)
    if latest_daily_pretraining_exit >= earliest_minute_adaptation_entry:
        raise ValueError(
            "future daily pretraining label overlaps minute adaptation: "
            f"{latest_daily_pretraining_exit.isoformat()} >= "
            f"{earliest_minute_adaptation_entry.isoformat()}"
        )
    minute_adaptation_identities = frozenset(
        (row.exchange, row.symbol) for row in minute_train
    )
    if any(not exchange or not symbol for exchange, symbol in minute_adaptation_identities):
        raise ValueError("minute adaptation needs explicit stock exchange and symbol identities")
    daily_mean, daily_scale = fit_scaler(daily_train)
    minute_mean, minute_scale = fit_scaler(minute_train)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    artifacts = {}
    for index, architecture in enumerate(ARCHITECTURES):
        random.seed(seed + index)
        np.random.seed(seed + index)
        torch.manual_seed(seed + index)
        model = build_model(architecture, lookback)
        _fit(model, daily_train, daily_mean, daily_scale,
             epochs=pretrain_epochs, learning_rate=.002, seed=seed + index)
        _fit(model, minute_train, minute_mean, minute_scale,
             epochs=adapt_epochs, learning_rate=.0005, seed=seed + index + 1000)
        val_prob = _probabilities(model, minute_split["val"], minute_mean, minute_scale)
        threshold = _choose_threshold(minute_split["val"], val_prob,
                                      min_trades=min_validation_trades)
        test_prob = _probabilities(model, minute_split["test"], minute_mean, minute_scale)
        artifacts[architecture] = {
            "model": model, "minute_mean": minute_mean, "minute_scale": minute_scale,
            "threshold": threshold, "market": daily[0].market,
            "minute_adaptation_identities": minute_adaptation_identities,
            "validation": _report(minute_split["val"], val_prob, threshold),
            "test": _report(minute_split["test"], test_prob, threshold),
        }
    summary = {
        "market": daily[0].market, "lookback": lookback, "horizon_bars": horizon,
        "bar_minutes": 5, "fee_bps_per_side": fee_bps_per_side,
        "slippage_bps_per_side": slippage_bps_per_side,
        "purge_sessions": purge_sessions, "split_counts": {
            domain: {name: len(rows) for name, rows in splits.items()}
            for domain, splits in (("daily", daily_split), ("minute", minute_split))},
        "cross_domain_chronology": {
            "latest_daily_pretraining_label_exit": latest_daily_pretraining_exit.isoformat(),
            "earliest_minute_adaptation_entry": earliest_minute_adaptation_entry.isoformat(),
            "daily_pretraining_strictly_before_minute_adaptation": True,
        },
        "minute_adaptation_identities": [
            {"exchange": exchange, "symbol": symbol}
            for exchange, symbol in sorted(minute_adaptation_identities)
        ],
        "label": "daily_next_session_open_or_minute_two_intervening_bars_then_bar_open_entry_to_later_bar_open_exit",
        "pretraining_domain": "daily_bar_as_generic_OHLCV_token",
        "adaptation_domain": "declared_completed_five_minute_bars_from_JSONL",
        "source_authenticity_verified": False,
        "domain_mismatch": True, "research_only": True,
        "deployment_allowed": False, "order_routing_connected": False,
        "reasons": ["daily and five-minute market microstructure differ",
                    "historical next-open proxy is not an executable fill guarantee",
                    "requires forward paper validation and market-specific review"],
    }
    return artifacts, summary


def paper_probability(artifact: dict, completed_minute_bars: Iterable[Bar], *,
                      architecture: str, as_of: datetime) -> dict:
    """Pure research inference, not an order recommendation or broker action."""
    history = tuple(completed_minute_bars)
    if architecture not in ARCHITECTURES or not history:
        raise ValueError("unknown model or empty completed history")
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    if any(b.bar_minutes != 5 or b.timestamp + timedelta(minutes=5) >= as_of
           for b in history):
        raise ValueError("paper inference requires completed historical five-minute bars")
    if artifact.get("market") not in (None, history[-1].market):
        raise ValueError("paper data market differs from the trained market")
    if ((history[-1].exchange, history[-1].symbol)
            not in artifact.get("minute_adaptation_identities", ())):
        raise ValueError("paper stock/exchange was not in minute adaptation training")
    if len({(b.market, b.exchange, b.symbol, b.session_date) for b in history}) != 1:
        raise ValueError("one market, exchange, symbol and session required")
    if any(not _consecutive(left, right, kind="minute")
           for left, right in zip(history, history[1:])):
        raise ValueError("missing or duplicate five-minute bar in paper history")
    for bar in history:
        session = _market_session(bar)
        offset = bar.timestamp - session.opened
        if not (timedelta() <= offset <= session.closed - session.opened
                and offset.total_seconds() % 300 == 0):
            raise ValueError("paper bar is outside an aligned regular-session slot")
    model = artifact["model"]
    types = {"linear": LinearBarNet, "conv": ConvBarNet, "gru": RecurrentBarNet}
    if not isinstance(model, types[architecture]):
        raise ValueError("paper architecture does not match loaded model")
    lookback = len(history)
    if lookback != model_lookback(model):
        raise ValueError("paper history length differs from model lookback")
    x = (_features(history) - artifact["minute_mean"]) / artifact["minute_scale"]
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(torch.from_numpy(
            np.clip(x, -8, 8)[None].astype(np.float32)))).item())
    threshold = artifact["threshold"]
    return {"market": history[-1].market, "exchange": history[-1].exchange,
            "symbol": history[-1].symbol,
            "as_of": history[-1].timestamp.isoformat(),
            "probability_proxy": probability, "validation_threshold": threshold,
            "paper_candidate": threshold is not None and probability >= threshold,
            "earliest_entry_proxy_bar_offset": 3,
            "domain_mismatch": True, "research_only": True,
            "deployment_allowed": False, "order_routing_connected": False}


def model_lookback(model: nn.Module) -> int:
    if isinstance(model, (LinearBarNet, ConvBarNet, RecurrentBarNet)):
        return model.lookback
    raise ValueError("unsupported research model")


def load_paper_artifact(bundle: str | Path, architecture: str) -> tuple[dict, dict]:
    """Load only a declared research-only bundle for offline paper inference."""
    folder = Path(bundle)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != "dockdack-minute-transfer-research-v1"
            or manifest.get("safety") != {
                "research_only": True, "deployment_allowed": False,
                "order_routing_connected": False, "historical_profitability_claim": False}
            or manifest.get("study", {}).get("domain_mismatch") is not True
            or architecture not in ARCHITECTURES):
        raise ValueError("not an approved offline research bundle")
    declared_identities = manifest["study"].get("minute_adaptation_identities")
    if (not isinstance(declared_identities, list) or not declared_identities
            or any(not isinstance(row, dict) or set(row) != {"exchange", "symbol"}
                   or any(not isinstance(row[key], str) or not row[key].strip()
                          or row[key] != row[key].strip()
                          for key in ("exchange", "symbol"))
                   for row in declared_identities)):
        raise ValueError("research bundle lacks valid minute adaptation stock identities")
    minute_adaptation_identities = frozenset(
        (row["exchange"], row["symbol"]) for row in declared_identities
    )
    if (len(minute_adaptation_identities) != len(declared_identities)
            or sorted(declared_identities, key=lambda row: (row["exchange"], row["symbol"]))
            != declared_identities):
        raise ValueError("research bundle has duplicate or unordered adaptation identities")
    record = manifest["models"][architecture]
    state_file = record["state_file"]
    if state_file != f"{architecture}.pt":
        raise ValueError("unexpected model state file")
    path = folder / state_file
    if sha256_file(path) != record["state_sha256"]:
        raise ValueError("model state hash mismatch")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    lookback = manifest["study"]["lookback"]
    if payload["architecture"] != architecture or payload["lookback"] != lookback:
        raise ValueError("model state identity mismatch")
    model = build_model(architecture, lookback)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    mean = payload["mean"].numpy().astype(np.float32)
    scale = payload["scale"].numpy().astype(np.float32)
    if (mean.shape != (FEATURES,) or scale.shape != (FEATURES,)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError("invalid minute training scaler")
    threshold = record["validation_threshold"]
    if threshold is not None and threshold not in (.50, .55, .60):
        raise ValueError("unexpected validation threshold")
    return {"model": model, "minute_mean": mean, "minute_scale": scale,
            "threshold": threshold, "market": manifest["market"],
            "minute_adaptation_identities": minute_adaptation_identities}, manifest


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
