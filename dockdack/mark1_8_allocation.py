"""Research-only Mark1.8 direct-utility, long-only daily allocation experiment.

The model sees only 30 completed bars through t. Outcomes are hypothetical
t+1 open-to-same-close returns. No inference here is suitable for live orders.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dockdack.mark1_4_evolution import EvolutionSamples, normalize_windows


class AllocationNet(nn.Module):
    """Small nonlinear 30-bar network with separate rank and sizing heads."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(150, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(),
        )
        self.head = nn.Linear(32, 2)

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.head(self.backbone(windows.reshape(-1, 150)))
        # Score is a ranking score, NOT a calibrated chance of profit.
        return torch.sigmoid(output[:, 0]), 0.1 * torch.sigmoid(output[:, 1])


@dataclass(frozen=True)
class AllocationFit:
    scores: np.ndarray
    sizes: np.ndarray
    model: AllocationNet
    history: tuple[dict, ...]
    training_rows: int
    cuda_name: str


def _valid_rows(samples: EvolutionSamples, indices: np.ndarray) -> np.ndarray:
    rows = np.asarray(indices)
    if (rows.ndim != 1 or len(rows) == 0 or
            not np.issubdtype(rows.dtype, np.integer) or
            np.any(rows < 0) or np.any(rows >= len(samples.windows)) or
            len(np.unique(rows)) != len(rows)):
        raise ValueError("indices must be unique, nonempty, in-range integers")
    return rows.astype(np.int64, copy=False)


def _net_outcomes(samples: EvolutionSamples, rows: np.ndarray,
                  cost_bps: float) -> tuple[np.ndarray, np.ndarray]:
    opened = np.asarray(samples.entry_open[rows], dtype=np.float64)
    closed = np.asarray(samples.exit_close[rows], dtype=np.float64)
    valid = np.isfinite(opened) & np.isfinite(closed) & (opened > 0) & (closed > 0)
    net = np.full(len(rows), np.nan, dtype=np.float32)
    half = cost_bps / 20_000.0
    net[valid] = (closed[valid] * (1.0 - half) /
                  (opened[valid] * (1.0 + half)) - 1.0).astype(np.float32)
    return net, valid


def fit_allocation(samples: EvolutionSamples, train_indices: np.ndarray, *,
                   seed: int, epochs: int = 12, batch_days: int = 16,
                   cost_bps: float = 20.0) -> AllocationFit:
    """Fit only observed training outcomes on CUDA; refuse CPU fallback.

    A differentiable soft top-10 allocates at most 10% per candidate. The
    reward is net daily portfolio return after the declared roundtrip cost,
    with downside and turnover penalties and a small direction auxiliary.
    This surrogate is *not* the exact simulator used for evaluation.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Mark1.8 training requires CUDA; CPU fallback is forbidden")
    if type(seed) is not int or seed < 0 or type(epochs) is not int or epochs < 1:
        raise ValueError("seed must be nonnegative and epochs positive")
    if type(batch_days) is not int or batch_days < 1:
        raise ValueError("batch_days must be positive")
    if not math.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("Invalid transaction cost")
    rows = _valid_rows(samples, train_indices)
    outcomes, valid = _net_outcomes(samples, rows, cost_bps)
    rows = rows[valid]
    targets = outcomes[valid]
    if len(rows) < 100:
        raise ValueError("Too few observed training candidates")
    order = np.argsort(np.asarray(samples.target_ordinals)[rows], kind="stable")
    rows, targets = rows[order], targets[order]
    ordinals = np.asarray(samples.target_ordinals)[rows]
    _, starts, counts = np.unique(ordinals, return_index=True, return_counts=True)
    day_slices = [np.arange(start, start + count) for start, count in zip(starts, counts)]
    # Each sample is normalized solely by its own completed window.
    features = normalize_windows(samples.windows).reshape(len(samples.windows), 150)
    rng = np.random.default_rng(seed)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = AllocationNet().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4,
                                      weight_decay=1e-3)
        history: list[dict] = []
        for epoch in range(epochs):
            model.train()
            day_order = rng.permutation(len(day_slices))
            losses, rewards, downsides = [], [], []
            for offset in range(0, len(day_order), batch_days):
                chosen = day_order[offset:offset + batch_days]
                batch_rows = np.concatenate([rows[day_slices[d]] for d in chosen])
                batch_targets = np.concatenate([targets[day_slices[d]] for d in chosen])
                lengths = [len(day_slices[d]) for d in chosen]
                x = torch.as_tensor(features[batch_rows], device="cuda")
                y = torch.as_tensor(batch_targets, device="cuda")
                scores, sizes = model(x)
                daily_return, daily_turnover = [], []
                start = 0
                for length in lengths:
                    stop = start + length
                    day_scores = scores[start:stop]
                    # The detached kth boundary permits gradients to all
                    # candidates near the 10-name selection frontier.
                    k = min(10, length)
                    kth = torch.topk(day_scores.detach(), k).values[-1]
                    gate = torch.sigmoid((day_scores - kth) / 0.04)
                    budget = sizes[start:stop] * gate
                    budget = budget / torch.clamp(budget.sum(), min=1.0)
                    daily_return.append((budget * y[start:stop]).sum())
                    daily_turnover.append(2.0 * budget.sum())
                    start = stop
                returns = torch.stack(daily_return)
                turnover = torch.stack(daily_turnover)
                downside = torch.sqrt(torch.mean(F.relu(-returns).square()) + 1e-10)
                utility = (returns.mean() - 0.25 * downside -
                           0.0002 * turnover.mean())
                direction = F.binary_cross_entropy(scores, (y > 0).float())
                loss = -100.0 * utility + 0.03 * direction
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                rewards.append(float(returns.mean().detach().cpu()))
                downsides.append(float(downside.detach().cpu()))
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                            "soft_daily_net_return": float(np.mean(rewards)),
                            "soft_downside": float(np.mean(downsides))})
        model.eval()
        all_scores = np.empty(len(features), dtype=np.float32)
        all_sizes = np.empty(len(features), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(features), 4096):
                x = torch.as_tensor(features[start:start + 4096], device="cuda")
                s, w = model(x)
                all_scores[start:start + len(x)] = s.cpu().numpy()
                all_sizes[start:start + len(x)] = w.cpu().numpy()
    if (not np.isfinite(all_scores).all() or not np.isfinite(all_sizes).all() or
            np.any(all_sizes < 0) or np.any(all_sizes > .1 + 1e-6)):
        raise ValueError("Nonfinite or oversized allocation prediction")
    return AllocationFit(all_scores, all_sizes, model.cpu(), tuple(history),
                         len(rows), torch.cuda.get_device_name(0))


def simulate_allocations(samples: EvolutionSamples, indices: np.ndarray,
                         scores: np.ndarray, sizes: np.ndarray, *,
                         threshold: float, cost_bps: float = 20.0,
                         initial_equity: float = 10_000_000.0) -> dict:
    """Independent exact cash ledger with integer shares and 10-name cap.

    A selected missing t+1 outcome invalidates exact equity from that date;
    it is never silently assigned zero P&L. Scores/sizes are t-only outputs.
    """
    rows = _valid_rows(samples, indices)
    score = np.asarray(scores, dtype=np.float64)
    size = np.asarray(sizes, dtype=np.float64)
    count = len(samples.windows)
    if (score.shape != (count,) or size.shape != (count,) or
            not np.isfinite(score[rows]).all() or not np.isfinite(size[rows]).all() or
            np.any(size[rows] < 0) or np.any(size[rows] > .1 + 1e-8)):
        raise ValueError("Need finite scores and allocation fractions in [0, 0.1]")
    if (not math.isfinite(threshold) or not math.isfinite(cost_bps) or
            not 0 <= cost_bps < 10_000 or not math.isfinite(initial_equity) or
            initial_equity <= 0):
        raise ValueError("Invalid threshold, cost, or equity")
    ordinals = np.asarray(samples.target_ordinals)
    dates = np.asarray(samples.target_dates)
    symbols = np.asarray(samples.symbol_ids)
    opened = np.asarray(samples.entry_open, dtype=np.float64)
    closed = np.asarray(samples.exit_close, dtype=np.float64)
    ordered = rows[np.argsort(ordinals[rows], kind="stable")]
    unique, starts, counts = np.unique(ordinals[ordered], return_index=True,
                                       return_counts=True)
    ranges = {int(day): ordered[a:a + n] for day, a, n in zip(unique, starts, counts)}
    half = cost_bps / 20_000.0
    equity = float(initial_equity)
    peak = equity
    max_drawdown = 0.0
    exact = True
    signals = trades = active_days = unresolved = 0
    daily: list[dict] = []
    for day in range(int(unique[0]), int(unique[-1]) + 1):
        day_rows = ranges.get(day, np.empty(0, dtype=np.int64))
        if len(day_rows) and not np.all(dates[day_rows] == dates[day_rows[0]]):
            raise ValueError("Inconsistent date for session ordinal")
        eligible = day_rows[score[day_rows] > threshold]
        ranking = np.lexsort((symbols[eligible], -score[eligible]))
        selected = eligible[ranking[:10]]
        if len(set(symbols[selected])) != len(selected):
            raise ValueError("Duplicate selected symbol on one session")
        signals += len(selected)
        active_days += bool(len(selected))
        valid = (np.isfinite(opened[selected]) & np.isfinite(closed[selected]) &
                 (opened[selected] > 0) & (closed[selected] > 0))
        entry = {"ordinal": day, "date": str(dates[day_rows[0]]) if len(day_rows) else None,
                 "signals": len(selected), "selected_symbol_ids":
                     [int(value) for value in symbols[selected]],
                 "sizes": [float(value) for value in size[selected]],
                 "return": None, "equity": None, "drawdown": None,
                 "shares": None}
        if not valid.all():
            unresolved += int((~valid).sum())
            exact = False
            daily.append(entry)
            continue
        if exact:
            start_equity = equity
            gross_entry = opened[selected] * (1 + half)
            quantities = np.floor(size[selected] * start_equity / gross_entry).astype(np.int64)
            debit = float(np.dot(quantities, gross_entry))
            if debit > start_equity + 1e-7 or len(selected) > 10:
                raise AssertionError("Cash or position limit violated")
            proceeds = float(np.dot(quantities, closed[selected] * (1 - half)))
            equity += proceeds - debit
            if equity < -1e-7:
                raise AssertionError("Negative cash-only equity")
            trades += int(np.count_nonzero(quantities))
            peak = max(peak, equity)
            drawdown = equity / peak - 1.0
            max_drawdown = min(max_drawdown, drawdown)
            entry.update({"return": equity / start_equity - 1.0,
                          "equity": equity, "drawdown": drawdown,
                          "shares": [int(q) for q in quantities]})
        daily.append(entry)
    return {"sessions": len(daily), "signals": signals,
            "active_sessions": active_days, "executed_trades": trades if exact else None,
            "unresolved_selected_outcomes": unresolved,
            "incomplete_data": not exact,
            "final_equity": equity if exact else None,
            "compound_net_return": equity / initial_equity - 1 if exact else None,
            "max_drawdown": max_drawdown if exact else None,
            "threshold": threshold, "cost_bps": cost_bps,
            "initial_equity": initial_equity,
            "daily": daily,
            "sizing": "model output 0..10% of start-day equity, floor integer shares, max 10 names, cash only",
            "fills": "hypothetical t+1 open buy and same-session close sell; unverified"}
