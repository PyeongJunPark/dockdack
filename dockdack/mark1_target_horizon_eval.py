"""Research-only sizing-free evaluation for target-price/horizon candidates.

These utilities never read a broker or operating ledger.  A nonoverlapping
block summary is intentionally a *hypothetical* fixed-allocation diagnostic:
it books returns only at block end and cannot measure intraperiod drawdown or
prove that a daily high would have filled a resting limit order.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RidgeReturnModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    coefficients: np.ndarray

    def predict(self, features: object) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != len(self.coefficients)
                or not np.isfinite(values).all()):
            raise ValueError("finite two-dimensional features with the fitted width required")
        normalized = np.clip((values - self.feature_mean) / self.feature_scale, -8., 8.)
        prediction = self.target_mean + normalized @ self.coefficients
        if not np.isfinite(prediction).all():
            raise ValueError("nonfinite ridge prediction")
        return prediction


def fit_ridge_return(features: object, net_returns: object, *, alpha: float = 0.05) -> RidgeReturnModel:
    """Fit a small deterministic baseline using *training rows only*.

    Callers must pass only the chronological training split.  The mean and
    scale are never fitted on tune, selection or test observations.
    """
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(net_returns, dtype=np.float64)
    if (values.ndim != 2 or len(values) < 2 or values.shape[1] < 1
            or targets.shape != (len(values),) or not np.isfinite(values).all()
            or not np.isfinite(targets).all() or np.any(targets < -1.)
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("finite training features, returns and positive ridge alpha required")
    center = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-6)
    normalized = np.clip((values - center) / scale, -8., 8.)
    target_mean = float(targets.mean())
    width = normalized.shape[1]
    covariance = normalized.T @ normalized / len(values)
    cross = normalized.T @ (targets - target_mean) / len(values)
    coefficients = np.linalg.solve(covariance + alpha * np.eye(width), cross)
    if not np.isfinite(coefficients).all():
        raise ValueError("nonfinite ridge fit")
    return RidgeReturnModel(center, scale, target_mean, coefficients)


def nonoverlap_block_summary(
    scores: object,
    net_returns: object,
    target_session_ordinals: object,
    symbol_ids: object,
    hits: object,
    eligible: object,
    *,
    horizon: int,
    top_k: int = 3,
    allocation: float = 0.1,
    score_floor: float = -math.inf,
    anchor_ordinal: int | None = None,
) -> dict:
    """Rank candidates at most once per H-session block and book exits.

    Entries are spaced H sessions apart even when a target is reached early;
    freed cash is *not* reinvested before the next block.  Each selected name
    receives ``allocation`` of start-of-block equity, with the rest in cash.
    This deliberately prevents overlapping holdings and leverage but ignores
    integer shares, interim mark-to-market, holidays in execution and fills.
    It is a comparison diagnostic, not a backtest of actual brokerage orders.
    """
    score = np.asarray(scores, dtype=np.float64)
    net = np.asarray(net_returns, dtype=np.float64)
    ordinal = np.asarray(target_session_ordinals)
    symbol = np.asarray(symbol_ids)
    hit = np.asarray(hits)
    keep = np.asarray(eligible)
    count = len(score) if score.ndim == 1 else -1
    if (count < 1 or net.shape != (count,) or ordinal.shape != (count,)
            or symbol.shape != (count,) or hit.shape != (count,)
            or keep.shape != (count,) or keep.dtype.kind != "b"
            or hit.dtype.kind != "b" or ordinal.dtype.kind not in "iu"
            or symbol.dtype.kind not in "iu" or np.any(ordinal < 0)
            or np.any(symbol < 0) or not np.isfinite(score[keep]).all()
            or not np.isfinite(net[keep]).all() or np.any(net[keep] < -1.)):
        raise ValueError("aligned, finite eligible events and integer identities required")
    if (type(horizon) is not int or horizon < 1 or type(top_k) is not int or top_k < 1
            or not math.isfinite(allocation) or not 0 < allocation <= 1 / top_k
            or math.isnan(score_floor)):
        raise ValueError("invalid nonoverlap horizon, top-K, allocation or score floor")
    if anchor_ordinal is None:
        if not keep.any():
            raise ValueError("no eligible events")
        anchor = int(ordinal[keep].min())
    elif type(anchor_ordinal) is int and anchor_ordinal >= 0:
        anchor = anchor_ordinal
    else:
        raise ValueError("anchor ordinal must be nonnegative")
    considered = np.flatnonzero(keep & (ordinal >= anchor)
                                & ((ordinal - anchor) % horizon == 0))
    days = np.unique(ordinal[considered])
    if not len(days):
        return {"blocks": 0, "trades": 0, "target_hits": 0,
                "mean_net_return": None, "hit_rate": None,
                "total_return": 0.0, "max_realized_block_drawdown": 0.0,
                "selected_indices": [], "block_returns": []}
    equity = peak = 1.0
    worst_drawdown = 0.0
    selected: list[int] = []
    block_returns: list[float] = []
    for day in days:
        rows = considered[ordinal[considered] == day]
        if len(np.unique(symbol[rows])) != len(rows):
            raise ValueError("duplicate same-symbol events on one entry session")
        # Primary key is descending model score; symbol identity resolves ties.
        ordering = np.lexsort((symbol[rows], -score[rows]))
        chosen = rows[ordering][:top_k]
        chosen = chosen[score[chosen] >= score_floor]
        selected.extend(int(row) for row in chosen)
        block_return = float(allocation * net[chosen].sum())
        if not math.isfinite(block_return) or block_return <= -1.0:
            raise ValueError("invalid fully invested block return")
        block_returns.append(block_return)
        equity *= 1.0 + block_return
        peak = max(peak, equity)
        worst_drawdown = max(worst_drawdown, 1.0 - equity / peak)
    chosen_array = np.asarray(selected, dtype=np.int64)
    return {"blocks": int(len(days)), "trades": int(len(selected)),
            "target_hits": int(hit[chosen_array].sum()),
            "mean_net_return": float(net[chosen_array].mean()) if len(selected) else None,
            "hit_rate": float(hit[chosen_array].mean()) if len(selected) else None,
            "total_return": float(equity - 1.0),
            "max_realized_block_drawdown": float(worst_drawdown),
            "selected_indices": selected, "block_returns": block_returns}
