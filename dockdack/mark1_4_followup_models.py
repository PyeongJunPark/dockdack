"""Research-only daily-bar scorers for the Mark1.4 follow-up experiments.

The only labels read by ``fit_score_variant`` are next-session outcomes at the
supplied training indices. Every score is computed from the preceding 30
*completed* OHLCV bars. Calibration, portfolio selection, and later-period
evaluation belong to the research runner, not to this module. No broker or
operational database interface is imported here.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dockdack.mark1_4_evolution import EvolutionSamples, normalize_windows


Variant = Literal["e1_rank", "e2_uncertainty", "e5_features"]
VARIANTS = ("e1_rank", "e2_uncertainty", "e5_features")

FEATURE_NAMES = (
    "close_log_return_1", "close_log_return_3", "close_log_return_5",
    "close_log_return_10", "close_log_return_20", "close_log_return_29",
    "close_return_volatility_5", "close_return_volatility_10",
    "close_return_volatility_20", "close_return_volatility_29",
    "mean_log_high_low_5", "mean_log_high_low_10",
    "mean_log_high_low_20", "mean_log_high_low_30",
    "mean_log_close_open_5", "mean_log_close_open_20",
    "last_open_gap_log", "last_log_volume_vs_20day_mean",
    "last_log_volume_vs_30day_median", "median_log_dollar_volume_20",
)


def extract_engineered_features(windows: np.ndarray) -> np.ndarray:
    """Derive t-only trend, volatility and liquidity features from 30 bars."""
    raw = np.asarray(windows, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (30, 5):
        raise ValueError("Expected [samples,30,5] OHLCV windows")
    if (not np.isfinite(raw).all() or np.any(raw[:, :, :4] <= 0)
            or np.any(raw[:, :, 4] < 0)):
        raise ValueError("Window prices must be positive and volume nonnegative")
    opened, high, low, close, volume = (raw[:, :, i] for i in range(5))
    log_close = np.log(close)
    log_returns = np.diff(log_close, axis=1)
    log_range = np.log(high / low)
    log_body = np.log(close / opened)
    log_volume = np.log1p(volume)
    columns = []
    for lookback in (1, 3, 5, 10, 20, 29):
        columns.append(log_close[:, -1] - log_close[:, -lookback - 1])
    for lookback in (5, 10, 20, 29):
        columns.append(np.std(log_returns[:, -lookback:], axis=1))
    for lookback in (5, 10, 20, 30):
        columns.append(np.mean(log_range[:, -lookback:], axis=1))
    for lookback in (5, 20):
        columns.append(np.mean(log_body[:, -lookback:], axis=1))
    columns.extend((
        np.log(opened[:, -1] / close[:, -2]),
        log_volume[:, -1] - np.mean(log_volume[:, -20:], axis=1),
        log_volume[:, -1] - np.median(log_volume, axis=1),
        np.median(np.log1p(close[:, -20:] * volume[:, -20:]), axis=1),
    ))
    features = np.stack(columns, axis=1).astype(np.float32)
    if features.shape[1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError("Engineered features are nonfinite or misaligned")
    return features


class _SmallScorer(nn.Module):
    def __init__(self, inputs: int, variant: Variant) -> None:
        super().__init__()
        if variant == "e5_features":
            self.net = nn.Sequential(
                nn.Linear(inputs, 16), nn.Tanh(),
                nn.Linear(16, 8), nn.Tanh(), nn.Linear(8, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(inputs, 8), nn.Tanh(),
                nn.Linear(8, 2 if variant == "e2_uncertainty" else 1),
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def _valid_training_rows(samples: EvolutionSamples, indices: np.ndarray) -> np.ndarray:
    rows = np.asarray(indices)
    if (rows.ndim != 1 or len(rows) == 0 or not np.issubdtype(rows.dtype, np.integer)
            or np.any(rows < 0) or np.any(rows >= len(samples.windows))
            or len(np.unique(rows)) != len(rows)):
        raise ValueError("train_indices must be nonempty, unique in-range integers")
    return rows.astype(np.int64, copy=False)


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return requested


def _same_day_pairs(rows: np.ndarray, ordinals: np.ndarray,
                    returns: np.ndarray, rng: np.random.Generator
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw O(N), never O(N squared), ordinal-matched ranking pairs."""
    ordered = rows[np.argsort(ordinals[rows], kind="stable")]
    days, starts, counts = np.unique(ordinals[ordered], return_index=True,
                                     return_counts=True)
    left_parts, right_parts, signs = [], [], []
    for _, start, count in zip(days, starts, counts):
        if count < 2:
            continue
        group = rng.permutation(ordered[start:start + count])
        alternate = np.roll(group, int(rng.integers(1, count)))
        sign = np.sign(returns[group] - returns[alternate])
        different = sign != 0
        if np.any(different):
            left_parts.append(group[different])
            right_parts.append(alternate[different])
            signs.append(sign[different].astype(np.float32))
    if not left_parts:
        raise ValueError("Ranking requires two unequal observed outcomes on a training date")
    left = np.concatenate(left_parts)
    right = np.concatenate(right_parts)
    targets = np.concatenate(signs)
    order = rng.permutation(len(left))
    return left[order], right[order], targets[order]


def _prediction_chunks(model: nn.Module, features: np.ndarray, *, device: str,
                       batch_size: int = 4096) -> np.ndarray:
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            values = torch.as_tensor(features[start:start + batch_size],
                                     dtype=torch.float32, device=device)
            parts.append(model(values).cpu().numpy())
    return np.concatenate(parts, axis=0)


def fit_score_variant(samples: EvolutionSamples, train_indices: np.ndarray,
                      variant: Variant, *, seed: int = 41,
                      device: str = "auto", epochs: int = 12,
                      batch_size: int = 1024, cost_bps: float = 20.0
                      ) -> tuple[np.ndarray, dict]:
    """Fit one fixed small model and score *all* t-only candidate windows.

    The target is a hypothetical t+1 open-to-close net return in **percent**,
    with half the declared round-trip cost applied on entry and exit. Only
    ``train_indices`` outcomes are accessed; later outcomes cannot affect the
    fit, feature scaler, or predicted scores. For E2, a Gaussian mean and
    aleatoric standard deviation are fitted, and the ranking score is the
    conservative mean-minus-one-standard-deviation estimate.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(epochs) is not int or not 1 <= epochs <= 500:
        raise ValueError("epochs must be in [1,500]")
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError("batch_size must be at least two")
    if not math.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps must be finite and nonnegative")
    rows = _valid_training_rows(samples, train_indices)
    runtime_device = _device(device)
    # This is the only read of target prices in the entire function.
    opened = np.asarray(samples.entry_open[rows], dtype=np.float64)
    closed = np.asarray(samples.exit_close[rows], dtype=np.float64)
    labeled = np.isfinite(opened) & np.isfinite(closed) & (opened > 0) & (closed > 0)
    labeled_rows = rows[labeled]
    if len(labeled_rows) < 2:
        raise ValueError("At least two observed training outcomes are required")
    half = cost_bps / 20_000
    target_percent = np.full(len(samples.windows), np.nan, dtype=np.float32)
    target_percent[labeled_rows] = (100.0 * (
        closed[labeled] * (1 - half) / (opened[labeled] * (1 + half)) - 1
    )).astype(np.float32)

    if variant == "e5_features":
        raw_features = extract_engineered_features(samples.windows)
        feature_names = FEATURE_NAMES
    else:
        raw_features = normalize_windows(samples.windows).reshape(len(samples.windows), -1)
        feature_names = None
    # Fit scaling on labeled training features only, never all-period moments.
    mean = raw_features[labeled_rows].mean(axis=0, dtype=np.float64)
    std = raw_features[labeled_rows].std(axis=0, dtype=np.float64)
    std = np.where(std >= 1e-6, std, 1.0)
    features = np.clip((raw_features - mean) / std, -6.0, 6.0).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite scaled feature")

    # Winsorization bounds are estimated from training targets alone. Ranking
    # uses only pair order and does not need a target magnitude cap.
    observed_target = target_percent[labeled_rows].astype(np.float64)
    clip_low, clip_high = np.quantile(observed_target, [0.005, 0.995]).tolist()
    if clip_low >= clip_high:
        clip_low, clip_high = float(observed_target.min()), float(observed_target.max())
    if clip_low >= clip_high and variant != "e1_rank":
        raise ValueError("Regression requires unequal training outcomes")
    clipped_target = np.clip(target_percent, clip_low, clip_high)

    rng = np.random.default_rng(seed)
    # Torch's RNG is scoped so this research fit does not perturb application
    # RNG state. CUDA reductions can still vary at the last bit across devices.
    cuda_devices = [torch.cuda.current_device()] if runtime_device == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if runtime_device == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = _SmallScorer(features.shape[1], variant).to(runtime_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                      weight_decay=1e-3)
        history: list[float] = []
        pair_count_last_epoch = 0
        for _ in range(epochs):
            model.train()
            weighted_loss = 0.0
            seen = 0
            if variant == "e1_rank":
                left, right, signs = _same_day_pairs(
                    labeled_rows, np.asarray(samples.target_ordinals),
                    target_percent, rng)
                pair_count_last_epoch = len(left)
                for start in range(0, len(left), batch_size):
                    stop = start + batch_size
                    x_left = torch.as_tensor(features[left[start:stop]],
                                             device=runtime_device)
                    x_right = torch.as_tensor(features[right[start:stop]],
                                              device=runtime_device)
                    labels = torch.as_tensor(signs[start:stop], device=runtime_device)
                    score_difference = (model(x_left).squeeze(1)
                                        - model(x_right).squeeze(1))
                    loss = F.softplus(-labels * score_difference).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    count = len(labels)
                    weighted_loss += float(loss.detach()) * count
                    seen += count
            else:
                shuffled = rng.permutation(labeled_rows)
                for start in range(0, len(shuffled), batch_size):
                    selected = shuffled[start:start + batch_size]
                    x = torch.as_tensor(features[selected], device=runtime_device)
                    y = torch.as_tensor(clipped_target[selected], device=runtime_device)
                    prediction = model(x)
                    if variant == "e2_uncertainty":
                        mu = prediction[:, 0]
                        log_sigma = prediction[:, 1].clamp(-3.0, 3.0)
                        loss = (0.5 * torch.square((y - mu) * torch.exp(-log_sigma))
                                + log_sigma).mean()
                    else:
                        loss = F.huber_loss(prediction[:, 0], y, delta=2.0)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    count = len(selected)
                    weighted_loss += float(loss.detach()) * count
                    seen += count
            if seen == 0 or not math.isfinite(weighted_loss):
                raise ValueError("Nonfinite or empty training objective")
            history.append(weighted_loss / seen)

        predictions = _prediction_chunks(model, features, device=runtime_device)
        state = {key: value.detach().cpu().numpy().tolist()
                 for key, value in model.state_dict().items()}

    if variant == "e2_uncertainty":
        mean_percent = predictions[:, 0].astype(np.float32)
        sigma_percent = np.exp(np.clip(predictions[:, 1], -3.0, 3.0)).astype(np.float32)
        scores = (mean_percent - sigma_percent).astype(np.float32)
        components = {
            "predicted_mean_percent": mean_percent.tolist(),
            "predicted_sigma_percent": sigma_percent.tolist(),
        }
    else:
        scores = predictions[:, 0].astype(np.float32)
        components = None
    if not np.isfinite(scores).all():
        raise ValueError("Model produced nonfinite scores")
    artifact = {
        "variant": variant,
        "research_only": True,
        "deployment_allowed": False,
        "score_uses_only_completed_30_bars": True,
        "selection_or_calibration_in_this_module": False,
        "seed": seed,
        "device": runtime_device,
        "epochs": epochs,
        "batch_size": batch_size,
        "cost_bps": float(cost_bps),
        "train_rows_requested": len(rows),
        "train_rows_labeled": len(labeled_rows),
        "excluded_train_outcomes": int((~labeled).sum()),
        "train_first_target": str(min(samples.target_dates[rows])),
        "train_last_target": str(max(samples.target_dates[rows])),
        "label": "100 * (t+1 close*(1-half-cost)/(t+1 open*(1+half-cost)) - 1)",
        "target_clip_percent_from_train_only": [float(clip_low), float(clip_high)],
        "architecture": (
            "20-to-16-to-8-to-1 tanh MLP" if variant == "e5_features" else
            "150-to-8-to-2 tanh mean/log-sigma MLP" if variant == "e2_uncertainty" else
            "150-to-8-to-1 tanh same-session pairwise ranker"
        ),
        "score_formula": (
            "mean_minus_one_sigma" if variant == "e2_uncertainty" else
            "predicted_net_return_percent" if variant == "e5_features" else
            "unscaled_pairwise_ranking_score"
        ),
        "training_objective": (
            "Gaussian negative log likelihood on clipped net-return percent" if variant == "e2_uncertainty" else
            "Huber net-return-percent regression on clipped training targets" if variant == "e5_features" else
            "same-session pairwise logistic ranking; O(N) sampled pairs per epoch"
        ),
        "train_pair_count_last_epoch": pair_count_last_epoch if variant == "e1_rank" else None,
        "feature_names": list(feature_names) if feature_names is not None else None,
        "feature_scaler": {"mean": mean.tolist(), "std": std.tolist(),
                           "fitted_on": "observed training rows only",
                           "post_scale_clip": [-6.0, 6.0]},
        "model_state": state,
        "training_loss_history": history,
        "prediction_components": components,
        "warning": "Historical proxy only. Neither opening nor closing auction fills are verified.",
    }
    return scores, artifact
