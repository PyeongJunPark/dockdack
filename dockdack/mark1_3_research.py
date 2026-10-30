"""Mark1.3 research only: pre-open signals from completed daily bars.

The decision for session t+1 uses bars through t. OPEN and CLOSE of t+1 are
read only to score a fixed open-to-close proxy after the decision. No broker,
order, GUI, Mark1.2 model, or operational database is imported or modified.
"""

from __future__ import annotations

from bisect import bisect_right
from contextlib import closing
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dockdack.clean_daily_dataset import load_sessions, normalized_turnover, open_source
from dockdack.dataset_identity import classify_instrument
from dockdack.dataset_quality import QualityPolicy, assess_series


LOOKBACK = 30
EMBARGO_SESSIONS = 30
FEATURE_NAMES = (
    "completed_close_return_5", "completed_close_return_20",
    "completed_return_volatility_5", "completed_return_volatility_20",
    "completed_bar_body", "completed_open_gap",
    "completed_range_mean_5", "completed_volume_vs_median_20",
)


@dataclass(frozen=True)
class ResearchSamples:
    features: np.ndarray
    target_dates: np.ndarray
    target_ordinals: np.ndarray
    symbol_ids: np.ndarray
    entry_open: np.ndarray
    exit_close: np.ndarray
    source: dict

    @property
    def observed(self) -> np.ndarray:
        return np.isfinite(self.entry_open) & np.isfinite(self.exit_close)


def features_through_t(history: np.ndarray) -> np.ndarray:
    """Eight finite features from exactly 30 completed OHLCV bars."""
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 3 or history.shape[1:] != (LOOKBACK, 5):
        raise ValueError("Expected [samples,30,5] completed OHLCV history")
    if (not np.isfinite(history).all() or np.any(history[:, :, :4] <= 0)
            or np.any(history[:, :, 4] < 0)):
        raise ValueError("Historical prices and volumes must be finite and valid")
    opened, high, low, close, volume = (history[:, :, column] for column in range(5))
    log_close = np.log(close)
    returns = np.diff(log_close, axis=1)
    values = np.column_stack((
        log_close[:, -1] - log_close[:, -6],
        log_close[:, -1] - log_close[:, -21],
        returns[:, -5:].std(axis=1),
        returns[:, -20:].std(axis=1),
        close[:, -1] / opened[:, -1] - 1,
        opened[:, -1] / close[:, -2] - 1,
        np.log(high[:, -5:] / low[:, -5:]).mean(axis=1),
        np.log1p(volume[:, -1]) - np.log1p(np.median(volume[:, -20:], axis=1)),
    ))
    if values.shape[1] != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError("Nonfinite causal features")
    return values.astype(np.float32)


def fixed_open_close_returns(entry_open: np.ndarray, exit_close: np.ndarray,
                             *, cost_bps: float = 20) -> tuple[np.ndarray, np.ndarray]:
    """Hypothetical t+1 auction OPEN entry and same-session CLOSE exit.

    Half of the declared roundtrip cost is applied to each side. Unobserved
    outcomes remain NaN; they are never silently counted as flat trades.
    """
    if not math.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("Roundtrip cost must be finite and between 0 and 10000 bps")
    opened, closed = np.broadcast_arrays(np.asarray(entry_open, dtype=np.float64),
                                           np.asarray(exit_close, dtype=np.float64))
    observed = np.isfinite(opened) & np.isfinite(closed)
    if np.any(opened[observed] <= 0) or np.any(closed[observed] <= 0):
        raise ValueError("Observed entry and exit prices must be positive")
    half = cost_bps / 20_000
    gross = np.full(opened.shape, np.nan, dtype=np.float64)
    net = np.full(opened.shape, np.nan, dtype=np.float64)
    gross[observed] = closed[observed] / opened[observed] - 1
    net[observed] = closed[observed] * (1 - half) / (opened[observed] * (1 + half)) - 1
    return gross, net


def _candidate_rows(db, market: str):
    rows = [dict(row) for row in db.execute("SELECT * FROM instruments")]
    eligible = []
    for row in rows:
        status, _ = classify_instrument(row, market)
        if status == "eligible":
            eligible.append((row["symbol"], row["exchange"]))
    return eligible, len(rows)


def load_raw_daily_candidates(database: Path, market: str, *, start: str,
                              train_end: str, test_end: str, max_symbols: int = 8,
                              seed: int = 42, session_dates=None,
                              policy: QualityPolicy | None = None) -> ResearchSamples:
    """Read raw bars in mode=ro; select candidates using t-only eligibility.

    Historical catalog classification is imperfect and not point-in-time. A
    symbol is retained based only on training-period eligible endpoints, not
    on target-day activity or future returns. Missing/untradable t+1 outcomes
    remain candidates with NaN outcome prices.
    """
    if market not in {"domestic", "us"} or type(max_symbols) is not int or max_symbols < 1:
        raise ValueError("A supported market and positive max_symbols are required")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    first, train_last, last = (date.fromisoformat(value) for value in (start, train_end, test_end))
    if not first < train_last < last:
        raise ValueError("Expected start < train_end < test_end")
    if policy is None:
        policy = QualityPolicy(min_median_turnover=1_000_000_000 if market == "domestic" else 1_000_000)
    if session_dates is None:
        session_dates, _ = load_sessions(market, first, last + timedelta(days=1))
    sessions = tuple(session_dates)
    if not sessions or sessions[0] < start or sessions[-1] > test_end or sessions != tuple(sorted(set(sessions))):
        raise ValueError("Session calendar must contain unique ordered dates within the requested range")
    session_index = {day: ordinal for ordinal, day in enumerate(sessions)}
    features, dates, ordinals, symbols, opens, closes = [], [], [], [], [], []
    selected, missing_targets, catalog_count = [], 0, 0
    with closing(open_source(database)) as db:
        db.execute("BEGIN")
        metadata_market = db.execute("SELECT value FROM metadata WHERE key='market'").fetchone()
        if metadata_market is None or metadata_market[0] != market:
            raise ValueError("Raw database market metadata does not match the requested market")
        candidates, catalog_count = _candidate_rows(db, market)
        candidates.sort(key=lambda key: hashlib.sha256(
            f"{seed}:{key[1]}:{key[0]}".encode()).hexdigest())
        for symbol, exchange in candidates:
            if len(selected) == max_symbols:
                break
            raw = [dict(row) for row in db.execute(
                "SELECT trade_date AS date,open,high,low,close,volume,trade_value,currency "
                "FROM daily_bars WHERE symbol=? AND exchange=? AND trade_date>=? AND trade_date<=? "
                "ORDER BY trade_date", (symbol, exchange, start, test_end))]
            if not raw:
                continue
            for row in raw:
                row["liquidity_turnover"] = normalized_turnover(row["trade_value"], market)
            assessment = assess_series(raw, market=market, as_of=last + timedelta(days=1),
                                       session_dates=sessions, policy=policy)
            by_session = {session_index[bar.date]: bar for bar in assessment.bars if bar.valid}
            per_symbol = []
            for bar in assessment.bars:
                if not bar.input_eligible:
                    continue
                t = session_index[bar.date]
                if t + 1 >= len(sessions):
                    continue
                target_day = sessions[t + 1]
                if target_day > test_end:
                    continue
                history = [by_session.get(index) for index in range(t - LOOKBACK + 1, t + 1)]
                if any(past is None for past in history):
                    continue
                target = by_session.get(t + 1)
                observed = target is not None and target.volume > 0
                historic = np.asarray([(past.open, past.high, past.low, past.close, past.volume)
                                       for past in history], dtype=np.float64)
                per_symbol.append((historic, target_day, t + 1,
                                   target.open if observed else np.nan,
                                   target.close if observed else np.nan))
            if sum(day <= train_end for _, day, *_ in per_symbol) < 30:
                continue
            symbol_id = len(selected)
            selected.append({"symbol_id": symbol_id, "symbol": symbol, "exchange": exchange,
                             "preopen_candidates": len(per_symbol)})
            for history, target_day, target_ordinal, entry, exit_price in per_symbol:
                features.append(history)
                dates.append(target_day)
                ordinals.append(target_ordinal)
                symbols.append(symbol_id)
                opens.append(entry)
                closes.append(exit_price)
                missing_targets += not np.isfinite(entry)
    if not selected:
        raise ValueError("No symbols have 30 training-period t-only eligible candidates")
    # Feature construction sees only completed t histories. The outcome arrays
    # are assembled afterward and never passed to features_through_t.
    values = features_through_t(np.stack(features))
    return ResearchSamples(values, np.asarray(dates, dtype="U10"),
                           np.asarray(ordinals, dtype=np.int32),
                           np.asarray(symbols, dtype=np.int32),
                           np.asarray(opens, dtype=np.float64),
                           np.asarray(closes, dtype=np.float64),
                           {"database": str(Path(database).resolve()), "market": market,
                            "catalog_rows": catalog_count, "selected_symbols": selected,
                            "eligible_preopen_candidates": len(dates),
                            "unobserved_or_untradable_target": missing_targets,
                            "selection": "catalog candidate and >=30 t-only eligible training endpoints; seeded symbol hash",
                            "quality_policy": vars(policy), "read_only": True})


def chronological_splits(samples: ResearchSamples, *, train_end: str,
                         validation_end: str, test_end: str,
                         sessions: tuple[str, ...], embargo_sessions: int = EMBARGO_SESSIONS):
    if type(embargo_sessions) is not int or embargo_sessions < LOOKBACK:
        raise ValueError("Embargo must span at least the 30-bar feature window")
    if not train_end < validation_end < test_end:
        raise ValueError("Expected train_end < validation_end < test_end")
    cuts = [bisect_right(sessions, day) - 1 for day in (train_end, validation_end, test_end)]
    if cuts[0] < 0 or cuts[-1] >= len(sessions) or cuts != sorted(cuts):
        raise ValueError("Split dates fall outside the session calendar")
    target = samples.target_ordinals
    splits = {
        "train": np.flatnonzero(target <= cuts[0]),
        "validation": np.flatnonzero((target > cuts[0] + embargo_sessions) & (target <= cuts[1])),
        "test": np.flatnonzero((target > cuts[1] + embargo_sessions) & (target <= cuts[2])),
    }
    if any(not len(indices) for indices in splits.values()):
        raise ValueError("Empty chronological split after embargo")
    return splits


class DailyNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 32), nn.ReLU(),
                                    nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def fit_neural_candidate(features: np.ndarray, net_returns: np.ndarray,
                         train: np.ndarray, validation: np.ndarray, *, seed: int = 42,
                         epochs: int = 12, batch_size: int = 512):
    """Fit train-only scaling and a small CPU net; validation chooses epoch."""
    if not 1 <= epochs <= 100 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive and bounded")
    train = train[np.isfinite(net_returns[train])]
    validation = validation[np.isfinite(net_returns[validation])]
    if len(train) < 20 or len(validation) < 5:
        raise ValueError("At least 20 train and 5 validation observed outcomes are required")
    mean = features[train].mean(axis=0).astype(np.float32)
    scale = features[train].std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1
    scaled = np.clip((features - mean) / scale, -8, 8).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("Nonfinite standardized features")
    torch.manual_seed(seed)
    model = DailyNetRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.SmoothL1Loss(beta=1.0)
    x = torch.from_numpy(scaled)
    y = torch.from_numpy((net_returns * 100).astype(np.float32))
    best_loss, best_epoch, best_state = float("inf"), 0, None
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        for positions in torch.randperm(len(train), generator=generator).split(batch_size):
            indices = train[positions.numpy()]
            optimizer.zero_grad()
            loss = criterion(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            value = float(criterion(model(x[validation]), y[validation]))
        if value < best_loss:
            best_loss, best_epoch = value, epoch
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prediction = model(x).numpy().astype(np.float64) / 100
    if not np.isfinite(prediction).all():
        raise ValueError("Neural candidate produced nonfinite predictions")
    checkpoint = {"state_dict": best_state, "mean": mean, "scale": scale,
                  "feature_names": FEATURE_NAMES, "research_only": True,
                  "deployment_allowed": False}
    training = {"best_epoch": best_epoch, "validation_huber_percentage_points": best_loss,
                "train_observed": len(train), "validation_observed": len(validation)}
    return prediction, checkpoint, training


def validation_coverage_threshold(prediction: np.ndarray, baseline: np.ndarray,
                                  validation: np.ndarray) -> float:
    """Use only score ranks and baseline frequency, with a zero-return floor."""
    fraction = float(np.mean(baseline[validation]))
    if fraction <= 0:
        return 0.0
    quantile = float(np.quantile(prediction[validation], 1 - fraction))
    return max(0.0, quantile)


def prediction_distribution(prediction: np.ndarray, indices: np.ndarray) -> dict:
    """Descriptive score distribution; never uses target outcomes."""
    values = prediction[indices]
    quantiles = np.quantile(values, (0, .01, .05, .25, .5, .75, .95, .99, 1))
    return {"count": len(values), "mean": float(values.mean()),
            "positive_fraction": float(np.mean(values > 0)),
            "quantiles": {name: float(value) for name, value in zip(
                ("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"),
                quantiles)}}


def strategy_metrics(samples: ResearchSamples, indices: np.ndarray, signals: np.ndarray,
                     gross: np.ndarray, net: np.ndarray) -> dict:
    indices = np.asarray(indices, dtype=np.int64)
    chosen = indices[np.asarray(signals[indices], dtype=bool)]
    observed = chosen[np.isfinite(net[chosen])]
    sessions = np.unique(samples.target_dates[indices])
    day_returns = np.zeros(len(sessions), dtype=np.float64)
    active_days = 0
    for offset, day in enumerate(sessions):
        day_signals = observed[samples.target_dates[observed] == day]
        if len(day_signals):
            day_returns[offset] = net[day_signals].mean()
            active_days += 1
    return {"eligible_candidates": len(indices), "signals": len(chosen),
            "coverage": len(chosen) / len(indices),
            "signals_per_session": len(chosen) / len(sessions),
            "observed_signals": len(observed), "unobserved_signals": len(chosen) - len(observed),
            "observed_signal_fraction": len(observed) / len(chosen) if len(chosen) else None,
            "sessions": len(sessions), "observed_active_sessions": active_days,
            "mean_gross_return_observed": float(gross[observed].mean()) if len(observed) else None,
            "mean_net_return_observed": float(net[observed].mean()) if len(observed) else None,
            "median_net_return_observed": float(np.median(net[observed])) if len(observed) else None,
            "positive_net_fraction_observed": float(np.mean(net[observed] > 0)) if len(observed) else None,
            "equal_weight_session_compound_observed_only":
                (0.0 if not len(chosen) else
                 float(np.prod(1 + day_returns) - 1) if len(chosen) == len(observed)
                 else None)}


def run_experiment(samples: ResearchSamples, splits: dict[str, np.ndarray], *,
                   cost_bps: float = 20, seed: int = 42, epochs: int = 12,
                   train_cap: int = 20_000, cost_grid_bps=(0, 10, 20, 40, 80)):
    gross, net = fixed_open_close_returns(samples.entry_open, samples.exit_close, cost_bps=cost_bps)
    train = splits["train"]
    if train_cap < 1:
        raise ValueError("train_cap must be positive")
    if len(train) > train_cap:
        selected = np.sort(np.random.default_rng(seed).choice(len(train), train_cap, replace=False))
        train = train[selected]
    prediction, checkpoint, training = fit_neural_candidate(
        samples.features, net, train, splits["validation"], seed=seed, epochs=epochs)
    always = np.ones(len(samples.features), dtype=bool)
    momentum = samples.features[:, 0] > 0
    threshold = validation_coverage_threshold(prediction, momentum, splits["validation"])
    neural = prediction > threshold  # Always also requires predicted net > 0.
    validation_momentum_fraction = float(np.mean(momentum[splits["validation"]]))
    validation_quantile = float(np.quantile(
        prediction[splits["validation"]], 1 - validation_momentum_fraction))
    score_diagnostics = {
        "threshold_provenance": {
            "validation_momentum_fraction": validation_momentum_fraction,
            "unfloored_validation_quantile_net_return": validation_quantile,
            "zero_floor_active": validation_quantile < 0,
            "actual_threshold_net_return": threshold,
        },
        "distribution_by_split": {
            name: prediction_distribution(prediction, splits[name])
            for name in ("train", "validation", "test")},
    }
    policies = {"no_trade": np.zeros(len(always), dtype=bool), "always": always,
                "momentum_5": momentum, "neural_positive": neural}
    results = {}
    for name in ("validation", "test"):
        results[name] = {policy: strategy_metrics(samples, splits[name], mask, gross, net)
                         for policy, mask in policies.items()}
    grid = {}
    for cost in tuple(dict.fromkeys((*cost_grid_bps, cost_bps))):
        gross_at_cost, net_at_cost = fixed_open_close_returns(
            samples.entry_open, samples.exit_close, cost_bps=cost)
        grid[str(cost)] = {
            policy: strategy_metrics(samples, splits["test"], mask, gross_at_cost, net_at_cost)
            for policy, mask in policies.items()}
    # Post-hoc diagnostics only: these unfloored cutoffs can accept predicted
    # nonpositive net returns, so they are not candidate trading policies.
    diagnostic_sweep = []
    for target_coverage in (.01, .05, .10, .25, .50):
        relaxed = float(np.quantile(prediction[splits["validation"]], 1 - target_coverage))
        selected = prediction > relaxed
        diagnostic_sweep.append({
            "target_validation_coverage": target_coverage,
            "threshold_net_return": relaxed,
            "can_select_nonpositive_prediction": relaxed < 0,
            "validation_acceptance": float(np.mean(selected[splits["validation"]])),
            "test": strategy_metrics(samples, splits["test"], selected, gross, net),
        })
    report = {
        "experiment": "mark1-3-daily-preopen-open-close-proxy-v1",
        "research_only": True, "deployment_allowed": False,
        "decision": "after completed session t, before scheduled t+1 opening auction",
        "features": list(FEATURE_NAMES), "entry": "t+1 OPEN proxy",
        "exit": "t+1 CLOSE fixed proxy; no intraday barrier or path claim",
        "cost": {"roundtrip_bps": cost_bps, "method": "half cost on entry and exit notional",
                 "sensitivity": "Policies held fixed; models are not retrained for each grid cost"},
        "neural_rule": "predicted net return > max(0, validation score quantile matching momentum-5 frequency)",
        "neural_threshold_net_return": threshold,
        "score_diagnostics": score_diagnostics,
        "diagnostic_only_unfloored_validation_coverage_sweep": diagnostic_sweep,
        "training": training, "splits": {key: {"candidates": len(value),
                    "first_target": min(samples.target_dates[value].tolist()),
                    "last_target": max(samples.target_dates[value].tolist())}
                    for key, value in splits.items()},
        "results": results, "test_cost_grid_bps": grid, "source": samples.source,
        "limitations": [
            "Historical research reuse; the test period is not a pristine unseen market experiment.",
            "Current catalog membership and subtype classification can create survivorship bias.",
            "Observed open/close prices do not prove opening-auction or closing-auction fills.",
            "Missing or zero-volume t+1 bars remain in the signal denominator but have unknown returns.",
            "The equal-weight compound proxy is withheld when selected signals have missing outcomes.",
            "Spread, impact, queue position and real fees are approximated by one declared roundtrip cost.",
            "Equal-weight session returns have no capital, integer-share, concurrent-position or capacity constraints.",
            "Only task A (t+1 open-to-close) is implemented; no t+2-open task or intraday path is inferred.",
            "Overlapping symbols and training examples are dependent; no live or paper execution is verified.",
        ],
    }
    return report, checkpoint
