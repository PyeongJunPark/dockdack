"""Thirty-completed-daily-bar LSTM inference, independent of order execution.

The target is a next-trading-day *close* gain of at least 1% from the final
input close. It is not a prediction of an intraday barrier being touched, and
does not promise a 1% return from a later execution price. Take-profit and
stop-loss decisions belong to a separate, deterministic execution policy.

Input columns are open, high, low, close, volume, in chronological order.
Feature normalization uses only the supplied 30 bars; no 31st bar, fitted
population scaler, or future observation is required. There is no clipping.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


LOOKBACK = 30
TARGET = "next_trading_day_close_return_ge_1pct"
FEATURE_NAMES = (
    "log_open_to_first_close",
    "log_high_to_first_close",
    "log_low_to_first_close",
    "log_close_to_first_close",
    "centered_log1p_volume",
    "close_log_change_first_zero",
    "log_high_to_low",
)


def _inference_backend(device: str | torch.device):
    """Avoid the Windows CUDA cuDNN LSTM shutdown failure; restore flags on exit."""
    if sys.platform == "win32" and torch.device(device).type == "cuda":
        return torch.backends.cudnn.flags(enabled=False)
    return nullcontext()


def _check_shape(raw: Tensor) -> None:
    if not isinstance(raw, Tensor):
        raise ValueError("raw must be a torch.Tensor with shape [batch, 30, 5]")
    if raw.ndim != 3 or tuple(raw.shape[1:]) != (LOOKBACK, 5) or raw.shape[0] < 1:
        raise ValueError("raw must have shape [nonempty batch, 30, 5]")
    if not raw.is_floating_point():
        raise ValueError("raw OHLCV must use a floating-point tensor dtype")


def validate_windows(raw: Tensor) -> None:
    """Reject malformed raw windows at ingestion, before batched training.

    Value checks synchronize CUDA tensors, so the model's hot forward path
    only checks tensor shape/type. Training data loaders must validate raw
    prices once; :class:`Predictor` always validates its inference boundary.
    """
    _check_shape(raw)
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("OHLCV must contain only finite values")
    prices, volume = raw[..., :4], raw[..., 4]
    if not bool((prices > 0).all()) or not bool((volume >= 0).all()):
        raise ValueError("OHLC prices must be positive and volume nonnegative")
    open_, high, low, close = prices.unbind(dim=-1)
    valid_range = (high >= low) & (high >= open_) & (high >= close)
    valid_range &= (low <= open_) & (low <= close)
    if not bool(valid_range.all()):
        raise ValueError("OHLC high/low bounds are inconsistent")


def window_features(raw: Tensor) -> Tensor:
    """Convert prevalidated raw ``[B, 30, 5]`` OHLCV to ``[B, 30, 7]``.

    Normalizing across this *input* window is intentional: all 30 completed
    bars are available at prediction time. Volume uses log1p, so zero volume
    is supported. Lower-precision raw inputs are promoted before logarithms.
    Call ``validate_windows`` when accepting untrusted raw data.
    """
    _check_shape(raw)
    if raw.dtype not in (torch.float32, torch.float64):
        raw = raw.float()
    log_prices = raw[..., :4].log()
    relative_prices = log_prices - log_prices[:, :1, 3:4]
    log_volume = raw[..., 4:5].log1p()
    centered_volume = log_volume - log_volume.mean(dim=1, keepdim=True)
    log_close = log_prices[..., 3:4]
    close_changes = torch.cat(
        (torch.zeros_like(log_close[:, :1]), log_close[:, 1:] - log_close[:, :-1]),
        dim=1,
    )
    log_range = log_prices[..., 1:2] - log_prices[..., 2:3]
    return torch.cat((relative_prices, centered_volume, close_changes, log_range), dim=-1)


class CandleLSTM(nn.Module):
    """A conventional LSTM classifier returning one unnormalized logit/window."""

    def __init__(self, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (float, int)):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a finite number in [0, 1)")
        self.lstm = nn.LSTM(
            input_size=len(FEATURE_NAMES),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, raw: Tensor) -> Tensor:
        sequence, _ = self.lstm(window_features(raw))
        return self.head(sequence[:, -1]).squeeze(-1)


class Predictor:
    """Load a compatible local checkpoint and predict from exactly 30 bars.

    ``buy_threshold`` is the model-probability decision threshold, not an
    expected percentage return. The event being predicted is defined by
    ``TARGET``. This object reads neither broker credentials nor positions
    and never places an order. Windows CUDA inference temporarily bypasses
    cuDNN; the model architecture, weights and caller's backend flags are kept.
    """

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("metadata"), dict):
            raise ValueError("checkpoint requires a metadata dictionary")
        metadata = dict(checkpoint["metadata"])
        if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
            raise ValueError("unsupported checkpoint schema_version")
        if type(metadata.get("lookback")) is not int or metadata["lookback"] != LOOKBACK:
            raise ValueError("checkpoint must use exactly 30 input bars")
        if metadata.get("feature_names") not in (list(FEATURE_NAMES), FEATURE_NAMES):
            raise ValueError("checkpoint feature_names do not match the 30-bar feature contract")
        if metadata.get("target") != TARGET:
            raise ValueError("checkpoint target is incompatible with the 1% close-gain classifier")
        if metadata.get("market") not in ("domestic", "us"):
            raise ValueError("checkpoint market must be domestic or us")
        threshold = metadata.get("buy_threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)):
            raise ValueError("buy_threshold must be a finite probability strictly between 0 and 1")
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("buy_threshold must be a finite probability strictly between 0 and 1")
        architecture = metadata.get("architecture")
        if not isinstance(architecture, dict) or set(architecture) != {"hidden_size", "num_layers", "dropout"}:
            raise ValueError("checkpoint architecture must specify hidden_size, num_layers, dropout")
        self.model = CandleLSTM(**architecture)
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint requires model_state_dict")
        try:
            self.model.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("checkpoint weights do not match its architecture") from exc
        self.model.to(self.device).eval()
        self.metadata = metadata
        self.market = metadata["market"]
        self.buy_threshold = float(threshold)

    @torch.inference_mode()
    def predict(self, bars: Any) -> dict[str, float | bool]:
        try:
            raw = torch.as_tensor(bars, dtype=torch.float32, device="cpu")
        except (ValueError, TypeError, RuntimeError, OverflowError) as exc:
            raise ValueError("bars must be a numeric [30, 5] OHLCV array") from exc
        if raw.ndim != 2 or tuple(raw.shape) != (LOOKBACK, 5):
            raise ValueError("prediction requires exactly 30 chronological OHLCV bars with shape [30, 5]")
        raw = raw.unsqueeze(0)
        validate_windows(raw)
        with _inference_backend(self.device):
            probability = float(self.model(raw.to(self.device)).sigmoid().item())
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("model returned a nonfinite or invalid probability")
        return {
            "probability_ge_1pct": probability,
            "buy_threshold": self.buy_threshold,
            "predicts_gain": probability >= self.buy_threshold,
        }
