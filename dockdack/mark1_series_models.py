"""Research-only 30-bar neural scorers for the Mark1.5--1.7 series.

Every feature ends at session t.  The sole supervised target is the later
session t+1 open-to-close outcome, read only at explicitly supplied training
indices.  This module has no broker, GUI, or operational database imports.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dockdack.mark1_4_evolution import EvolutionSamples, normalize_windows


VARIANTS = ("mark1.5", "mark1.6", "mark1.7")
SCORE_UNITS = {
    "mark1.5": "predicted_t_plus_1_net_open_to_close_return_percent",
    "mark1.6": "estimated_probability_positive_t_plus_1_net_open_to_close_return",
    "mark1.7": "unscaled_same_session_pairwise_ranking_score",
}


@dataclass(frozen=True)
class SeriesFit:
    scores: np.ndarray
    artifact: dict
    state: dict[str, np.ndarray]


class _RecurrentScorer(nn.Module):
    def __init__(self, hidden: int, cell: str) -> None:
        super().__init__()
        if cell == "lstm":
            self.encoder = nn.LSTM(5, hidden, batch_first=True)
        elif cell == "gru":
            self.encoder = nn.GRU(5, hidden, batch_first=True)
        else:
            raise ValueError("Recurrent cell must be lstm or gru")
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, final = self.encoder(values)
        hidden = final[0][-1] if isinstance(final, tuple) else final[-1]
        return self.head(hidden).squeeze(-1)


class _ImageScorer(nn.Module):
    """A small CNN on a 30 x 5 OHLCV time/price-volume tile.

    This is a representation experiment, not a reproduction of the 15 x 15
    technical-indicator image in Sezer and Ozbayoglu.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        channels = max(8, hidden // 2)
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(3, 3), padding=1), nn.GELU(),
            nn.Conv2d(channels, hidden, kernel_size=(3, 3), padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(hidden, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values.unsqueeze(1)).squeeze(-1)


class _CrossSectionScorer(nn.Module):
    """Attention across t-only candidate windows on the *same* target date."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        if hidden % 4:
            raise ValueError("Attention hidden width must be divisible by four")
        self.embedding = nn.Sequential(
            nn.Linear(150, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, values: torch.Tensor,
                padding: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(values.reshape(*values.shape[:2], 150))
        attended, _ = self.attention(embedded, embedded, embedded,
                                     key_padding_mask=padding, need_weights=False)
        return self.head(embedded + attended).squeeze(-1)


def _device(requested: str, *, test_only_allow_cpu: bool) -> str:
    if requested == "cpu" and test_only_allow_cpu:
        return "cpu"
    if requested != "cuda":
        raise ValueError("Mark1.5--1.7 training requires CUDA; CPU is test-only")
    if not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return "cuda"


def _groups(ordinals: np.ndarray) -> dict[int, np.ndarray]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for row, ordinal in enumerate(ordinals):
        grouped[int(ordinal)].append(row)
    return {day: np.asarray(rows, dtype=np.int64)
            for day, rows in grouped.items()}


def _padded_batch(day_rows: list[np.ndarray], values: np.ndarray):
    width = max(map(len, day_rows))
    ids = np.zeros((len(day_rows), width), dtype=np.int64)
    padding = np.ones((len(day_rows), width), dtype=bool)
    for index, rows in enumerate(day_rows):
        ids[index, :len(rows)] = rows
        padding[index, :len(rows)] = False
    return values[ids], padding


def _score_all(model: nn.Module, values: np.ndarray, variant: str,
               ordinals: np.ndarray, device: str, batch_size: int,
               day_batch_size: int) -> np.ndarray:
    model.eval()
    scores = np.empty(len(values), dtype=np.float32)
    with torch.inference_mode():
        if variant != "mark1.7":
            for start in range(0, len(values), batch_size):
                batch = torch.as_tensor(values[start:start + batch_size],
                                        device=device)
                output = model(batch)
                if variant == "mark1.6":
                    output = torch.sigmoid(output)
                scores[start:start + len(batch)] = output.cpu().numpy()
        else:
            groups = list(_groups(ordinals).values())
            for start in range(0, len(groups), day_batch_size):
                rows = groups[start:start + day_batch_size]
                padded_values, padding = _padded_batch(rows, values)
                output = model(
                    torch.as_tensor(padded_values, device=device),
                    torch.as_tensor(padding, device=device),
                ).cpu().numpy()
                for batch_index, indices in enumerate(rows):
                    scores[indices] = output[batch_index, :len(indices)]
    if not np.isfinite(scores).all():
        raise ValueError("Model produced nonfinite scores")
    return scores


def fit_series_variant(samples: EvolutionSamples, train_indices: np.ndarray,
                       variant: str, *, seed: int = 41, device: str = "cuda",
                       epochs: int = 5, batch_size: int = 512,
                       day_batch_size: int = 12, hidden: int = 32,
                       recurrent_cell: str = "lstm", cost_bps: float = 20.0,
                       test_only_allow_cpu: bool = False,
                       ) -> SeriesFit:
    """Fit one model and score all windows without reading nontrain outcomes.

    Mark1.5 regresses expected net percent, Mark1.6 classifies positive net
    return, and Mark1.7 learns same-day pairwise relative order.  Only
    completed t windows are model inputs; scaling is fitted on observed
    2018--2020 training windows and then frozen.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(epochs) is not int or not 1 <= epochs <= 500:
        raise ValueError("epochs must be in [1,500]")
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if type(day_batch_size) is not int or day_batch_size < 1:
        raise ValueError("day_batch_size must be positive")
    if type(hidden) is not int or hidden < 8 or hidden > 256 or hidden % 4:
        raise ValueError("hidden must be a multiple of four in [8,256]")
    if recurrent_cell not in {"lstm", "gru"}:
        raise ValueError("recurrent_cell must be lstm or gru")
    if type(test_only_allow_cpu) is not bool:
        raise ValueError("test_only_allow_cpu must be a bool")
    if not math.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps must be finite in [0,10000)")
    rows = np.asarray(train_indices)
    if (rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer)
            or np.any(rows < 0) or np.any(rows >= len(samples.windows))
            or len(np.unique(rows)) != len(rows)):
        raise ValueError("train_indices must be unique in-range integers")
    rows = rows.astype(np.int64)
    opened = np.asarray(samples.entry_open[rows], dtype=np.float64)
    closed = np.asarray(samples.exit_close[rows], dtype=np.float64)
    observed = np.isfinite(opened) & np.isfinite(closed) & (opened > 0) & (closed > 0)
    labeled = rows[observed]
    if len(labeled) < 3:
        raise ValueError("At least three observed training outcomes are required")
    half = cost_bps / 20_000
    target = np.full(len(samples.windows), np.nan, dtype=np.float32)
    target[labeled] = (100 * (
        closed[observed] * (1 - half) /
        (opened[observed] * (1 + half)) - 1
    )).astype(np.float32)
    raw = normalize_windows(samples.windows)
    mean = raw[labeled].mean(axis=(0, 1), dtype=np.float64)
    std = raw[labeled].std(axis=(0, 1), dtype=np.float64)
    std = np.where(std >= 1e-6, std, 1.0)
    values = np.clip((raw - mean) / std, -6, 6).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Scaled features are nonfinite")
    runtime = _device(device, test_only_allow_cpu=test_only_allow_cpu)
    observed_target = target[labeled].astype(np.float64)
    low, high = np.quantile(observed_target, [0.005, 0.995]).tolist()
    clipped = np.clip(target, low, high)
    rng = np.random.default_rng(seed)
    cuda_devices = [torch.cuda.current_device()] if runtime == "cuda" else []
    history = []
    pair_count = 0
    ordinals = np.asarray(samples.target_ordinals, dtype=np.int64)
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if runtime == "cuda":
            torch.cuda.manual_seed_all(seed)
        # Construct only after seeding: initial weights are seed-specific.
        if variant == "mark1.7":
            model: nn.Module = _CrossSectionScorer(hidden)
        elif variant == "mark1.6":
            model = _ImageScorer(hidden)
        else:
            model = _RecurrentScorer(hidden, recurrent_cell)
        model = model.to(runtime)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                      weight_decay=1e-3)
        if variant == "mark1.7":
            all_groups = _groups(ordinals)
            train_days = np.unique(ordinals[labeled])
            usable = []
            is_labeled = np.zeros(len(samples.windows), dtype=bool)
            is_labeled[labeled] = True
            for day in train_days:
                group = all_groups[int(day)]
                positions = np.flatnonzero(is_labeled[group])
                if len(positions) >= 2 and np.unique(target[group[positions]]).size >= 2:
                    usable.append((group, positions))
            if not usable:
                raise ValueError("Pairwise attention requires unequal outcomes on a training day")
        for _ in range(epochs):
            model.train()
            weighted_loss = 0.0
            seen = 0
            if variant == "mark1.7":
                day_order = rng.permutation(len(usable))
                for start in range(0, len(usable), day_batch_size):
                    chosen = day_order[start:start + day_batch_size]
                    groups = [usable[int(i)][0] for i in chosen]
                    padded_values, padding = _padded_batch(groups, values)
                    output = model(
                        torch.as_tensor(padded_values, device=runtime),
                        torch.as_tensor(padding, device=runtime),
                    )
                    left_batch, left_pos, right_pos, signs = [], [], [], []
                    for batch_id, pick in enumerate(chosen):
                        group, positions = usable[int(pick)]
                        permuted = rng.permutation(positions)
                        alternate = np.roll(permuted, int(rng.integers(1, len(permuted))))
                        sign = np.sign(target[group[permuted]] - target[group[alternate]])
                        valid = sign != 0
                        left_batch.extend([batch_id] * int(valid.sum()))
                        left_pos.extend(permuted[valid].tolist())
                        right_pos.extend(alternate[valid].tolist())
                        signs.extend(sign[valid].tolist())
                    if not signs:
                        continue
                    b = torch.as_tensor(left_batch, device=runtime)
                    left = torch.as_tensor(left_pos, device=runtime)
                    right = torch.as_tensor(right_pos, device=runtime)
                    y = torch.as_tensor(signs, dtype=torch.float32, device=runtime)
                    loss = F.softplus(-y * (output[b, left] - output[b, right])).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    count = len(signs)
                    pair_count += count
                    weighted_loss += float(loss.detach()) * count
                    seen += count
            else:
                shuffled = rng.permutation(labeled)
                for start in range(0, len(labeled), batch_size):
                    selected = shuffled[start:start + batch_size]
                    x = torch.as_tensor(values[selected], device=runtime)
                    prediction = model(x)
                    if variant == "mark1.6":
                        y = torch.as_tensor((target[selected] > 0).astype(np.float32),
                                            device=runtime)
                        loss = F.binary_cross_entropy_with_logits(prediction, y)
                    else:
                        y = torch.as_tensor(clipped[selected], device=runtime)
                        loss = F.huber_loss(prediction, y, delta=2.0)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    weighted_loss += float(loss.detach()) * len(selected)
                    seen += len(selected)
            if seen == 0 or not math.isfinite(weighted_loss):
                raise ValueError("Training objective is empty or nonfinite")
            history.append(weighted_loss / seen)
        scores = _score_all(model, values, variant, ordinals, runtime,
                            batch_size, day_batch_size)
        state = {name: value.detach().cpu().numpy().copy()
                 for name, value in model.state_dict().items()}
    artifact = {
        "variant": variant, "research_only": True,
        "deployment_allowed": False, "seed": seed,
        "score_unit": SCORE_UNITS[variant],
        "input": "30 completed OHLCV bars through t; t+1 outcome for training label only",
        "label": "next session open-to-close net return after assumed roundtrip cost",
        "architecture": (
            f"{recurrent_cell.upper()} 30x5 sequence regressor" if variant == "mark1.5" else
            "30x5 price-volume tile 2D CNN binary net-return classifier" if variant == "mark1.6" else
            "same-session cross-sectional attention pairwise ranker"
        ),
        "training_objective": (
            "Huber regression on clipped net return percent" if variant == "mark1.5" else
            "binary cross entropy: net return > 0" if variant == "mark1.6" else
            "same-session pairwise logistic ranking"
        ),
        "not_paper_reproduction": True,
        "train_rows_requested": len(rows),
        "train_rows_labeled": len(labeled),
        "excluded_train_outcomes": int((~observed).sum()),
        "train_first_target": str(min(samples.target_dates[rows])),
        "train_last_target": str(max(samples.target_dates[rows])),
        "feature_scaler": {"mean": mean.tolist(), "std": std.tolist(),
                           "fitted_on": "observed training rows only",
                           "post_scale_clip": [-6, 6]},
        "target_clip_percent_from_train_only": [float(low), float(high)],
        "training_loss_history": history,
        "train_pair_instances": pair_count if variant == "mark1.7" else None,
        "device": runtime, "epochs": epochs, "batch_size": batch_size,
        "test_only_cpu_override": runtime == "cpu",
        "day_batch_size": day_batch_size, "hidden": hidden,
        "recurrent_cell": recurrent_cell if variant == "mark1.5" else None,
        "cost_bps": float(cost_bps),
    }
    return SeriesFit(scores=scores, artifact=artifact, state=state)
