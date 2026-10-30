"""Completed daily OHLCV as ordered generic bars; real five-minute inference.

This deliberately does *not* adapt weights, fit a scaler, select a threshold, or
evaluate profitability on five-minute observations. A daily candle is one
generic bar token, never a synthetic observation of intraday microstructure.
The proxy target is the cost-adjusted direction from the third *later* daily
open to a still later daily open. Applying that representation to real five-
minute bars is an unvalidated domain transfer, even if daily holdout metrics
look attractive. The module imports no account, order, or network API.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch

from dockdack.market_schedule import session_on
from dockdack.models import Market, MinuteBar
from dockdack.research.minute_transfer import (
    ARCHITECTURES, FEATURES, Bar, Sample, _consecutive, _fit,
    _probabilities, build_model, fit_scaler, model_lookback, split_sessions,
)


FORMAT = "dockdack-daily-proxy-minute-v1"
FEATURE_SCHEMA = "range-normalized-relative-ohlcv5-v2"
ENTRY_LATENCY_BARS = 3
MAX_DAILY_BARS = 250_000
STOP_FRACTION = -0.02
TAKE_FRACTION = 0.03


@dataclass(frozen=True)
class DailyProxyConfig:
    model_id: str
    bundle_name: str
    architecture: str
    lookback: int
    horizon: int


DAILY_PROXY_CONFIGS = {
    f"mark1-{number}-prototype": DailyProxyConfig(
        model_id=f"mark1-{number}-prototype",
        bundle_name=f"daily-proxy-domestic-{lookback}x{horizon}-segment-safe-20260930",
        architecture=architecture, lookback=lookback, horizon=horizon,
    )
    for number, (lookback, horizon, architecture) in enumerate(
        ((lookback, horizon, architecture)
         for lookback, horizon in ((20, 3), (20, 6), (30, 12))
         for architecture in ARCHITECTURES), start=29)
}


def _relative_bar_features(history: tuple[Bar, ...] | list[Bar]) -> np.ndarray:
    """Causal five-channel shape representation without a daily/5m scale unit.

    All divisors come only from the already completed input window. The price
    channels are normalized by that window's median high/low log range; the
    volume channel uses its own robust log-volume dispersion. This reduces,
    but does not establish away, the frequency-domain mismatch.
    """
    if not history:
        raise ValueError("empty bar history")
    matrix = np.asarray([(b.open, b.high, b.low, b.close, b.volume)
                         for b in history], dtype=np.float64)
    op, high, low, close, volume = matrix.T
    if (not np.isfinite(matrix).all() or min(op.min(), high.min(), low.min(), close.min()) <= 0
            or volume.min() < 0):
        raise ValueError("invalid OHLCV in feature window")
    price_range = float(np.median(np.log(high / low)))
    price_range = max(price_range, 1e-5)
    log_volume = np.log1p(volume)
    volume_median = np.median(log_volume)
    volume_scale = max(float(1.4826 * np.median(np.abs(log_volume - volume_median))), .1)
    features = np.column_stack((
        np.log(close / close[-1]) / price_range,
        np.log(high / close) / price_range,
        np.log(low / close) / price_range,
        np.log(close / op) / price_range,
        (log_volume - volume_median) / volume_scale,
    )).astype(np.float32)
    if features.shape != (len(history), FEATURES) or not np.isfinite(features).all():
        raise ValueError("nonfinite relative OHLCV features")
    return features


def make_daily_proxy_samples(
    daily: Iterable[Bar], *, lookback: int, horizon: int,
    fee_bps_per_side: float, slippage_bps_per_side: float,
) -> tuple[Sample, ...]:
    """Causal daily-only proxy labels with third-later-open entry.

    Session gaps split a symbol run; no suspended/missing day is synthesized.
    Returns are historical open-price proxies after both-side costs, not fills.
    """
    if lookback < 5 or horizon < 1 or horizon > 24:
        raise ValueError("lookback must be >=5 and horizon 1..24")
    if not all(math.isfinite(v) and 0 <= v < 10_000 for v in
               (fee_bps_per_side, slippage_bps_per_side)):
        raise ValueError("finite, nonnegative cost bps below 10000 required")
    groups: dict[tuple[str, str, str], list[Bar]] = {}
    for bar in daily:
        if bar.bar_minutes != 1440 or not bar.exchange or bar.exchange == "INDEX":
            raise ValueError("only identified, completed stock daily bars are accepted")
        session = session_on(Market(bar.market), bar.session_date)
        if session is None or bar.timestamp != session.closed:
            raise ValueError("daily timestamp must equal official session close")
        groups.setdefault((bar.market, bar.exchange, bar.symbol), []).append(bar)
    samples: list[Sample] = []
    for (market, exchange, symbol), group in groups.items():
        group.sort(key=lambda bar: bar.timestamp)
        run = [0] * len(group)
        for i in range(1, len(group)):
            if group[i].timestamp <= group[i - 1].timestamp:
                raise ValueError("duplicate symbol/date in daily input")
            run[i] = run[i - 1] + int(not _consecutive(group[i - 1], group[i], kind="daily"))
        for t in range(lookback - 1, len(group) - ENTRY_LATENCY_BARS - horizon):
            first = t - lookback + 1
            entry_i = t + ENTRY_LATENCY_BARS
            exit_i = entry_i + horizon
            if run[first] != run[exit_i]:
                continue
            history = group[first:t + 1]
            entry, exit_bar = group[entry_i], group[exit_i]
            entry_at = session_on(Market(market), entry.session_date).opened
            exit_at = session_on(Market(market), exit_bar.session_date).opened
            if not history[-1].timestamp < entry_at < exit_at:
                raise ValueError("invalid proxy chronology")
            fee = fee_bps_per_side / 10_000
            slip = slippage_bps_per_side / 10_000
            net = (exit_bar.open * (1 - fee) * (1 - slip)
                   / (entry.open * (1 + fee) * (1 + slip)) - 1)
            samples.append(Sample(_relative_bar_features(history), market, symbol,
                                  history[0].session_date, entry.session_date,
                                  exit_bar.session_date, history[-1].timestamp,
                                  entry_at, exit_at, net, exchange))
    return tuple(sorted(samples, key=lambda s: (s.entry_at, s.exchange, s.symbol)))


def _cap(rows: tuple[Sample, ...], maximum: int) -> tuple[Sample, ...]:
    if len(rows) <= maximum:
        return rows
    # Equally spaced chronological subsampling avoids oldest-only training.
    indices = np.linspace(0, len(rows) - 1, maximum, dtype=np.int64)
    return tuple(rows[int(i)] for i in indices)


def _validation_threshold(rows: tuple[Sample, ...], probabilities: np.ndarray,
                          *, min_candidates: int, min_candidate_days: int) -> float | None:
    """Validation-only top-quintile activity gate; never optimize on test PnL."""
    if len(rows) < min_candidates or not np.isfinite(probabilities).all():
        return None
    if len(probabilities) != len(rows) or float(np.ptp(probabilities)) <= 1e-7:
        return None
    threshold = float(np.quantile(probabilities, .80))
    selected = probabilities >= threshold
    days = {row.entry_session for row, use in zip(rows, selected) if use}
    if int(selected.sum()) < min_candidates or len(days) < min_candidate_days:
        return None
    return threshold


def _metrics(rows: tuple[Sample, ...], probabilities: np.ndarray,
             threshold: float | None) -> dict:
    selected = (np.zeros(len(rows), dtype=bool) if threshold is None
                else probabilities >= threshold)
    net = np.array([r.net_return for r in rows], dtype=np.float64)
    labels = net > 0
    return {
        "samples": len(rows), "symbols": len({r.symbol for r in rows}),
        "sessions": len({r.entry_session for r in rows}),
        "selected": int(selected.sum()),
        "selected_sessions": len({r.entry_session for r, use in zip(rows, selected) if use}),
        "base_positive_fraction": float(labels.mean()) if len(rows) else None,
        "selected_positive_fraction": float(labels[selected].mean()) if selected.any() else None,
        "selected_mean_net_proxy": float(net[selected].mean()) if selected.any() else None,
        "selected_total_net_proxy": float(net[selected].sum()) if selected.any() else None,
        "unit": "fraction of cost-adjusted historical daily next-open proxy; not intraday fills",
    }


def train_daily_proxy(
    daily: tuple[Bar, ...], *, lookback: int, horizon: int,
    fee_bps_per_side: float = 2., slippage_bps_per_side: float = 8.,
    purge_sessions: int = 3, epochs: int = 4, max_train_samples: int = 50_000,
    min_samples_per_split: int = 100, min_validation_candidates: int = 20,
    min_candidate_days: int = 10, min_training_symbols: int = 10,
    seed: int = 20260930,
) -> tuple[dict, dict]:
    """Fit three architectures using *only* completed daily bars.

    Demo experimental eligibility means metadata/threshold contract complete;
    it does not assert financial edge or demonstrated five-minute behavior.
    """
    if not daily or len(daily) > MAX_DAILY_BARS:
        raise ValueError("daily-only training requires 1..250000 completed bars")
    markets = {bar.market for bar in daily}
    if len(markets) != 1:
        raise ValueError("train one market at a time")
    if len({bar.exchange for bar in daily}) != 1:
        raise ValueError("train one exchange at a time")
    if len({(bar.exchange, bar.symbol) for bar in daily}) < min_training_symbols:
        raise ValueError("too few distinct training stocks for a pooled daily model")
    if min(epochs, max_train_samples, min_samples_per_split,
           min_validation_candidates, min_candidate_days, min_training_symbols) < 1:
        raise ValueError("positive training limits required")
    samples = make_daily_proxy_samples(
        daily, lookback=lookback, horizon=horizon,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side)
    splits = split_sessions(samples, purge_sessions=purge_sessions)
    for name, rows in splits.items():
        if len(rows) < min_samples_per_split:
            raise ValueError(f"{name} has only {len(rows)} daily proxy samples")
    train = _cap(splits["train"], max_train_samples)
    mean, scale = fit_scaler(train)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    artifacts = {}
    for index, architecture in enumerate(ARCHITECTURES):
        random.seed(seed + index)
        np.random.seed(seed + index)
        torch.manual_seed(seed + index)
        model = build_model(architecture, lookback)
        _fit(model, train, mean, scale, epochs=epochs,
             learning_rate=.002, seed=seed + index)
        val_prob = _probabilities(model, splits["val"], mean, scale)
        threshold = _validation_threshold(
            splits["val"], val_prob,
            min_candidates=min_validation_candidates,
            min_candidate_days=min_candidate_days)
        test_prob = _probabilities(model, splits["test"], mean, scale)
        artifacts[architecture] = {
            "model": model, "mean": mean, "scale": scale,
            "threshold": threshold, "market": daily[0].market,
            "exchange": daily[0].exchange, "lookback": lookback,
            "horizon": horizon,
            "training_symbols": frozenset(bar.symbol for bar in daily),
            "validation": _metrics(splits["val"], val_prob, threshold),
            "daily_test": _metrics(splits["test"], test_prob, threshold),
        }
    summary = {
        "market": daily[0].market, "exchanges": sorted({b.exchange for b in daily}),
        "training_symbols": sorted({b.symbol for b in daily}),
        "daily_input_bars": len(daily), "lookback": lookback,
        "horizon_bars": horizon, "feature_schema": FEATURE_SCHEMA,
        "train_domain": "completed_daily_bars_as_generic_ordered_OHLCV_tokens_only",
        "inference_domain": "actual_completed_five_minute_bars_without_retraining",
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_session_bounds": {
            name: {"entry_first": min(r.entry_session for r in rows).isoformat(),
                   "label_exit_last": max(r.exit_session for r in rows).isoformat()}
            for name, rows in splits.items()},
        "purge_sessions": purge_sessions,
        "costs": {"fee_bps_per_side": fee_bps_per_side,
                  "slippage_bps_per_side": slippage_bps_per_side},
        "proxy_label": "completed bar t; entry t+3 open; exit entry+horizon open; both-side costs",
        "threshold_policy": "validation top quintile; minimum candidate/sample/day counts; no test PnL selection",
        "minute_performance_tested": False,
        "domain_mismatch": True,
        "risk": "daily-to-minute frequency transfer is unverified; daily holdout is not a minute backtest",
        "exit_policy": {
            "entry_delay_completed_5m_bars": ENTRY_LATENCY_BARS,
            "max_hold_completed_5m_bars_after_fill": horizon,
            "emergency_stop_fraction": STOP_FRACTION,
            "take_profit_fraction": TAKE_FRACTION,
            "force_flat_before_session_close": True,
            "stop_take_note": "risk override independent of the proxy training label",
        },
    }
    return artifacts, summary


def _validate_minute_history(history: tuple[Bar, ...], *, as_of: datetime,
                             market: str, exchange: str, lookback: int) -> None:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("aware as_of required")
    if len(history) != lookback:
        raise ValueError("exactly the trained number of bars required")
    first = history[0]
    if (market != first.market or exchange != first.exchange
            or len({(b.market, b.exchange, b.symbol, b.session_date) for b in history}) != 1):
        raise ValueError("market/exchange/symbol/session mismatch")
    if any(b.bar_minutes != 5 or b.timestamp + timedelta(minutes=5) > as_of
           for b in history):
        raise ValueError("only completed actual five-minute bars may be inferred")
    if as_of - history[-1].timestamp > timedelta(minutes=10):
        raise ValueError("stale minute snapshot cannot authorize a candidate")
    if any(not _consecutive(left, right, kind="minute")
           for left, right in zip(history, history[1:])):
        raise ValueError("gap, duplicate, or unordered minute bars")
    for bar in history:
        session = session_on(Market(bar.market), bar.session_date)
        if (session is None or not session.opened <= bar.timestamp < session.closed
                or (bar.timestamp - session.opened).total_seconds() % 300):
            raise ValueError("bar outside aligned regular trading session")


def _actual_minute_bars(values: Iterable[Bar | MinuteBar]) -> tuple[Bar, ...]:
    """Convert a validated broker-feed snapshot without inventing any bars."""
    converted = []
    for value in values:
        if isinstance(value, MinuteBar):
            if value.market is not Market.DOMESTIC or value.currency != "KRW":
                raise ValueError("daily proxy currently supports domestic KRW five-minute bars only")
            converted.append(Bar(
                market=value.market.value, exchange=value.exchange,
                symbol=value.symbol, timestamp=value.timestamp,
                open=float(value.open), high=float(value.high),
                low=float(value.low), close=float(value.close),
                volume=float(value.volume), bar_minutes=5,
            ))
        elif isinstance(value, Bar):
            converted.append(value)
        else:
            raise TypeError("expected research Bar or broker MinuteBar")
    return tuple(converted)


def infer_daily_proxy(artifact: dict, completed_minute_bars: Iterable[Bar | MinuteBar], *,
                      architecture: str, as_of: datetime) -> dict:
    """Pure inference; candidate and schedule contract only, never an order."""
    history = _actual_minute_bars(completed_minute_bars)
    if architecture not in ARCHITECTURES:
        raise ValueError("unknown architecture")
    model = artifact["model"]
    lookback = model_lookback(model)
    if not isinstance(model, type(build_model(architecture, lookback))):
        raise ValueError("model architecture differs from requested architecture")
    market, exchange = artifact["market"], artifact["exchange"]
    # Demand two subsequently completed bars so an always-latest feed can
    # actually reach the t+3 entry moment. Neither later bar enters features.
    _validate_minute_history(history, as_of=as_of, market=market,
                             exchange=exchange, lookback=lookback + 2)
    signal_window = history[:-2]
    horizon = artifact["horizon"]
    if not isinstance(horizon, int) or not 1 <= horizon <= 24:
        raise ValueError("invalid model hold horizon")
    session = session_on(Market(market), signal_window[-1].session_date)
    earliest = signal_window[-1].timestamp + timedelta(minutes=5 * ENTRY_LATENCY_BARS)
    planned_exit = earliest + timedelta(minutes=5 * horizon)
    # Broker labels may be bar start or end; reserve another completed slot
    # before market close rather than placing an entry that cannot be closed.
    lifecycle_fits_session = planned_exit + timedelta(minutes=5) < session.closed
    decision_ready = as_of >= earliest
    x = (_relative_bar_features(signal_window) - artifact["mean"]) / artifact["scale"]
    if not np.isfinite(x).all():
        raise ValueError("nonfinite model features")
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(torch.from_numpy(
            np.clip(x, -8, 8)[None].astype(np.float32)))).item())
    if not math.isfinite(probability):
        raise ValueError("nonfinite model output")
    threshold = artifact["threshold"]
    symbol_trained = history[-1].symbol in artifact["training_symbols"]
    return {
        "market": market, "exchange": exchange, "symbol": history[-1].symbol,
        "signal_bar_label": signal_window[-1].timestamp.isoformat(),
        "probability_proxy": probability, "validation_threshold": threshold,
        "candidate": bool(threshold is not None and probability >= threshold
                          and lifecycle_fits_session and symbol_trained
                          and decision_ready),
        "earliest_order_at": earliest.isoformat(),
        "max_hold_bars": horizon, "emergency_stop_fraction": STOP_FRACTION,
        "take_profit_fraction": TAKE_FRACTION,
        "force_flat_before_session_close": True,
        "lifecycle_fits_session": lifecycle_fits_session,
        "decision_ready_now": decision_ready,
        "latency_bars_observed_not_used_as_features": 2,
        "symbol_in_daily_training_universe": symbol_trained,
        "minute_performance_tested": False,
        "domain_mismatch": True,
    }


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_daily_proxy_artifact(bundle: str | Path, architecture: str) -> tuple[dict, dict]:
    """Verify manifest identity and model hash before inference."""
    folder = Path(bundle)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != FORMAT or architecture not in ARCHITECTURES
            or manifest.get("study", {}).get("feature_schema") != FEATURE_SCHEMA
            or manifest.get("study", {}).get("minute_performance_tested") is not False
            or manifest.get("study", {}).get("domain_mismatch") is not True
            or manifest.get("safety", {}).get("real_money_allowed") is not False
            or manifest.get("safety", {}).get("demo_experimental_only") is not True
            or manifest.get("safety", {}).get("minute_profitability_claim") is not False
            or manifest.get("safety", {}).get("model_does_not_place_orders") is not True):
        raise ValueError("invalid daily-proxy minute bundle")
    record = manifest["models"][architecture]
    state_file = record["state_file"]
    if state_file != f"{architecture}.pt":
        raise ValueError("unexpected model state file")
    state_path = folder / state_file
    if sha256_file(state_path) != record["state_sha256"]:
        raise ValueError("model hash mismatch")
    payload = torch.load(state_path, weights_only=True, map_location="cpu")
    lookback = manifest["study"]["lookback"]
    horizon = manifest["study"]["horizon_bars"]
    if (payload["architecture"] != architecture or payload["lookback"] != lookback
            or payload["horizon"] != horizon):
        raise ValueError("model identity mismatch")
    model = build_model(architecture, lookback)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    mean = payload["mean"].numpy().astype(np.float32)
    scale = payload["scale"].numpy().astype(np.float32)
    if (mean.shape != (FEATURES,) or scale.shape != (FEATURES,)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError("invalid trained scaler")
    threshold = record["validation_threshold"]
    if threshold is not None and (not isinstance(threshold, (int, float))
                                  or not math.isfinite(threshold)
                                  or not 0 < threshold < 1):
        raise ValueError("invalid validation threshold")
    expected_policy = {
        "entry_delay_completed_5m_bars": ENTRY_LATENCY_BARS,
        "max_hold_completed_5m_bars_after_fill": horizon,
        "emergency_stop_fraction": STOP_FRACTION,
        "take_profit_fraction": TAKE_FRACTION,
        "force_flat_before_session_close": True,
        "stop_take_note": "risk override independent of the proxy training label",
    }
    if (manifest["study"].get("exit_policy") != expected_policy
            or record.get("demo_experimental_eligible") is not bool(
                threshold is not None
                and manifest.get("source", {}).get("daily_export_receipt_verified") is True)):
        raise ValueError("bundle validation gate or exit policy differs from trained contract")
    exchanges = manifest["study"]["exchanges"]
    if len(exchanges) != 1 or not isinstance(exchanges[0], str):
        raise ValueError("bundle must identify one exchange")
    training_symbols = manifest["study"].get("training_symbols")
    if (not isinstance(training_symbols, list) or not training_symbols
            or len(training_symbols) != len(set(training_symbols))
            or any(not isinstance(symbol, str) or len(symbol) != 6
                   or not symbol.isascii() or not symbol.isdigit()
                   for symbol in training_symbols)):
        raise ValueError("bundle lacks valid training symbol identities")
    return {"model": model, "mean": mean, "scale": scale,
            "threshold": threshold, "market": manifest["market"],
            "exchange": exchanges[0], "horizon": horizon,
            "training_symbols": frozenset(training_symbols)}, manifest


def load_configured_daily_proxy(models_root: str | Path,
                                model_id: str) -> tuple[dict, dict]:
    """Load one mark1.29–1.37 identity without silently swapping weights."""
    config = DAILY_PROXY_CONFIGS[model_id]
    artifact, manifest = load_daily_proxy_artifact(
        Path(models_root) / config.bundle_name, config.architecture)
    study = manifest["study"]
    record = manifest["models"][config.architecture]
    if (manifest["market"] != "domestic" or study["lookback"] != config.lookback
            or study["horizon_bars"] != config.horizon
            or record["model_id"] != model_id):
        raise ValueError("configured prototype identity does not match model bundle")
    return artifact, manifest
