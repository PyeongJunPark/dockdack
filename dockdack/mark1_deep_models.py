"""Causal features and deeper four-outcome classifiers for mark_1 research.

Input is exactly 30 *completed* OHLCV bars and one candidate entry price. No
target-session high, low, close or volume is accepted. Window statistics use
only the completed history, independently for every sample. Four logits refer
to take-only, stop-only, both-touch and neither; neither an architecture nor
the four-way loss resolves intraday ordering. Both-touch remains a failure.

ResNets use 1-D residual block layouts (2,2,2,2) and (3,4,6,3), inspired by the
image ResNet family, NOT exact paper reproductions. Inception is inspired by
InceptionTime with shorter kernels for 30-day windows. GroupNorm, GELU, dual
mean/max pooling, dropout, and the separate price-query head are implementation
choices. No convolution interprets the query as a completed candle.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


LOOKBACK = 30
TARGET = "daily_high_ge_entry_1pct_and_low_gt_entry_minus_0_9pct_conservative"
CLASS_NAMES = ("take_only", "stop_only", "both_touch", "neither")
MODEL_NAMES = ("mlp_deep", "resnet18", "resnet34", "inception")
FEATURE_NAMES = (
    "vol_scaled_log_open_or_entry_over_last_close",
    "vol_scaled_log_high_over_last_close",
    "vol_scaled_log_low_over_last_close",
    "vol_scaled_log_close_over_last_close",
    "standardized_log1p_volume", "vol_scaled_close_log_return",
    "vol_scaled_log_high_low_range", "candle_body_fraction",
    "close_range_position", "upper_wick_fraction", "lower_wick_fraction",
    "log_close_or_entry_level_div10", "log1p_volume_level_div20",
    "log_historical_return_scale_div5", "historical_mean_return_over_scale",
    "vol_scaled_open_gap", "historical_bar_mask", "entry_query_mask",
)


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _check_features(features: Tensor, input_size: int, sequence_length: int) -> None:
    if (not isinstance(features, Tensor) or not features.is_floating_point()
            or features.ndim != 3 or features.shape[0] < 1
            or tuple(features.shape[1:]) != (sequence_length, input_size)):
        raise ValueError(
            f"features must be floating tensors [nonempty batch,{sequence_length},{input_size}]"
        )


def validate_features(features: Tensor, *, input_size: int = 18,
                      sequence_length: int = 31) -> None:
    """Check finite values once on ingestion, not in the GPU training hot path."""
    _positive_integer(input_size, "input_size")
    _positive_integer(sequence_length, "sequence_length")
    _check_features(features, input_size, sequence_length)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("features must contain only finite values")


def features_from_history(history: Tensor, entry_prices: Tensor, *,
                          validate: bool = False) -> Tensor:
    """Return [B,31,18] features without changing either input.

    Population standard deviation of the 29 observed close log returns is
    clamped to [.003,.3]. Historical volume log1p standard deviation has a
    floor of 1. All channels are bounded to [-12,12] with fixed limits; no
    cross-sample, future-day or learned normalization statistics are used.
    Zero-range candles have zero body/wicks and neutral close-range position.
    ``validate=False`` assumes values were already checked at ingestion.
    """
    if (not isinstance(history, Tensor) or not history.is_floating_point()
            or history.ndim != 3 or history.shape[0] < 1
            or tuple(history.shape[1:]) != (LOOKBACK, 5)):
        raise ValueError("history must be floating tensors [nonempty batch,30,5] in OHLCV order")
    if (not isinstance(entry_prices, Tensor) or not entry_prices.is_floating_point()
            or entry_prices.ndim != 1 or len(entry_prices) != len(history)):
        raise ValueError("entry_prices must be floating tensors [batch]")
    if history.device != entry_prices.device or history.dtype != entry_prices.dtype:
        raise ValueError("history and entry prices must have the same dtype and device")
    prices, volume = history[..., :4], history[..., 4]
    if validate:
        if (not bool(torch.isfinite(history).all())
                or not bool(torch.isfinite(entry_prices).all())
                or bool((prices <= 0).any()) or bool((volume < 0).any())
                or bool((entry_prices <= 0).any())
                or bool((prices[..., 1] < prices.amax(dim=-1)).any())
                or bool((prices[..., 2] > prices.amin(dim=-1)).any())):
            raise ValueError("Invalid historical OHLCV or entry prices")

    log_prices = prices.log()
    log_close = log_prices[..., 3]
    close_returns = log_close[:, 1:] - log_close[:, :-1]
    scale = close_returns.std(dim=1, unbiased=False).clamp(min=.003, max=.3)
    trend = close_returns.mean(dim=1) / scale
    log_scale = scale.log() / 5
    reference = log_close[:, -1]
    result = history.new_zeros((len(history), LOOKBACK + 1, len(FEATURE_NAMES)))
    result[:, :LOOKBACK, :4] = (log_prices - reference[:, None, None]) / scale[:, None, None]
    log_volume = volume.log1p()
    result[:, :LOOKBACK, 4] = (
        (log_volume - log_volume.mean(dim=1, keepdim=True))
        / log_volume.std(dim=1, unbiased=False, keepdim=True).clamp(min=1)
    )
    result[:, 1:LOOKBACK, 5] = close_returns / scale[:, None]
    result[:, :LOOKBACK, 6] = (log_prices[..., 1] - log_prices[..., 2]) / scale[:, None]
    open_, high, low, close = prices.unbind(dim=-1)
    spread = high - low
    denominator = spread.clamp(min=torch.finfo(history.dtype).tiny)
    result[:, :LOOKBACK, 7] = (close - open_) / denominator
    result[:, :LOOKBACK, 8] = torch.where(spread > 0, 2 * ((close - low) / denominator) - 1, 0)
    result[:, :LOOKBACK, 9] = (high - torch.maximum(open_, close)) / denominator
    result[:, :LOOKBACK, 10] = (torch.minimum(open_, close) - low) / denominator
    result[:, :LOOKBACK, 11] = log_close / 10
    result[:, :LOOKBACK, 12] = log_volume / 20
    result[:, :, 13] = log_scale[:, None]
    result[:, :, 14] = trend[:, None]
    result[:, 1:LOOKBACK, 15] = (log_prices[:, 1:, 0] - log_close[:, :-1]) / scale[:, None]
    result[:, :LOOKBACK, 16] = 1
    result[:, LOOKBACK, 0] = (entry_prices.log() - reference) / scale
    result[:, LOOKBACK, 11] = entry_prices.log() / 10
    result[:, LOOKBACK, 17] = 1
    result = result.clamp(-12, 12)
    if validate:
        validate_features(result)
    return result


def success_logit(logits: Tensor) -> Tensor:
    """One-vs-rest logit whose sigmoid is four-way softmax take-only mass."""
    if (not isinstance(logits, Tensor) or not logits.is_floating_point()
            or logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] != len(CLASS_NAMES)):
        raise ValueError("logits must be floating tensors [nonempty batch,4]")
    return logits[:, 0] - torch.logsumexp(logits[:, 1:], dim=1)


def _norm(channels: int) -> nn.GroupNorm:
    groups = next(number for number in (8, 4, 2, 1)
                  if channels % number == 0 and channels // number >= 2)
    return nn.GroupNorm(groups, channels)


class _Classifier(nn.Module):
    def __init__(self, input_size: int, sequence_length: int, width: int, dropout: float):
        super().__init__()
        for name, value in (("input_size", input_size), ("sequence_length", sequence_length),
                            ("width", width)):
            _positive_integer(value, name)
        if sequence_length < 3:
            raise ValueError("sequence_length must include at least two historical bars and a query")
        if width < 4:
            raise ValueError("width must be at least 4")
        if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("dropout must be finite in [0,1)")
        self.input_size, self.sequence_length, self.width = input_size, sequence_length, width
        self.dropout_probability = float(dropout)

    def _split(self, features: Tensor) -> tuple[Tensor, Tensor]:
        _check_features(features, self.input_size, self.sequence_length)
        return features[:, :-1], features[:, -1]


class _QueryHead(nn.Module):
    def __init__(self, history_size: int, query_size: int, width: int, dropout: float):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(query_size, width * 2), nn.GELU(),
                                   nn.LayerNorm(width * 2))
        self.classifier = nn.Sequential(
            nn.Linear(history_size + width * 2, width * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width * 4, len(CLASS_NAMES)),
        )

    def forward(self, history: Tensor, query: Tensor) -> Tensor:
        return self.classifier(torch.cat((history, self.query(query)), dim=1))


class DeepMLP(_Classifier):
    """Four hidden history layers plus a separate candidate-query head."""
    def __init__(self, input_size=18, sequence_length=31, width=48, dropout=.2):
        super().__init__(input_size, sequence_length, width, dropout)
        dimensions = (input_size * (sequence_length - 1), width * 8, width * 4, width * 4, width * 2)
        layers = [nn.Flatten(start_dim=1)]
        for incoming, outgoing in zip(dimensions[:-1], dimensions[1:]):
            layers.extend((nn.Linear(incoming, outgoing), nn.LayerNorm(outgoing), nn.GELU(),
                           nn.Dropout(dropout)))
        self.history_encoder = nn.Sequential(*layers)
        self.head = _QueryHead(width * 2, input_size, width, dropout)

    def forward(self, features: Tensor) -> Tensor:
        history, query = self._split(features)
        return self.head(self.history_encoder(history), query)


class _ResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, stride: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
            _norm(outgoing), nn.GELU(),
            nn.Conv1d(outgoing, outgoing, 3, padding=1, bias=False), _norm(outgoing),
        )
        self.shortcut = (nn.Identity() if incoming == outgoing and stride == 1 else
                         nn.Sequential(nn.Conv1d(incoming, outgoing, 1, stride=stride, bias=False),
                                       _norm(outgoing)))
        self.activation = nn.GELU()

    def forward(self, history: Tensor) -> Tensor:
        return self.activation(self.main(history) + self.shortcut(history))


class ResidualClassifier(_Classifier):
    """Historical-only 1-D ResNet, retaining a distinct price-query embedding."""
    def __init__(self, block_counts, input_size=18, sequence_length=31, width=48, dropout=.2):
        super().__init__(input_size, sequence_length, width, dropout)
        if tuple(block_counts) not in ((2, 2, 2, 2), (3, 4, 6, 3)):
            raise ValueError("Unsupported residual block layout")
        self.block_counts = tuple(block_counts)
        self.stem = nn.Sequential(nn.Conv1d(input_size, width, 3, padding=1, bias=False),
                                  _norm(width), nn.GELU())
        incoming, stages = width, []
        for stage_index, count in enumerate(block_counts):
            outgoing = width * 2 ** stage_index
            for block_index in range(count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                stages.append(_ResidualBlock(incoming, outgoing, stride))
                incoming = outgoing
        self.history_encoder = nn.Sequential(*stages)
        self.head = _QueryHead(incoming * 2, input_size, width, dropout)

    def forward(self, features: Tensor) -> Tensor:
        history, query = self._split(features)
        encoded = self.history_encoder(self.stem(history.transpose(1, 2)))
        pooled = torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=1)
        return self.head(pooled, query)


class _InceptionModule(nn.Module):
    def __init__(self, incoming: int, width: int):
        super().__init__()
        self.bottleneck = nn.Conv1d(incoming, width, 1, bias=False)
        self.branches = nn.ModuleList(nn.Conv1d(width, width, kernel, padding=kernel // 2,
                                                bias=False) for kernel in (3, 7, 15))
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                         nn.Conv1d(incoming, width, 1, bias=False))
        self.output = nn.Sequential(_norm(width * 4), nn.GELU())

    def forward(self, history: Tensor) -> Tensor:
        reduced = self.bottleneck(history)
        branches = [branch(reduced) for branch in self.branches]
        return self.output(torch.cat((*branches, self.pool_branch(history)), dim=1))


class _InceptionResidualGroup(nn.Module):
    def __init__(self, incoming: int, width: int):
        super().__init__()
        outgoing = width * 4
        self.modules_sequence = nn.Sequential(_InceptionModule(incoming, width),
                                             _InceptionModule(outgoing, width),
                                             _InceptionModule(outgoing, width))
        self.shortcut = nn.Sequential(nn.Conv1d(incoming, outgoing, 1, bias=False), _norm(outgoing))
        self.activation = nn.GELU()

    def forward(self, history: Tensor) -> Tensor:
        return self.activation(self.modules_sequence(history) + self.shortcut(history))


class InceptionClassifier(_Classifier):
    """Six multiscale modules, residual connections every three modules."""
    def __init__(self, input_size=18, sequence_length=31, width=48, dropout=.2):
        super().__init__(input_size, sequence_length, width, dropout)
        self.history_encoder = nn.Sequential(_InceptionResidualGroup(input_size, width),
                                             _InceptionResidualGroup(width * 4, width))
        self.head = _QueryHead(width * 8, input_size, width, dropout)

    def forward(self, features: Tensor) -> Tensor:
        history, query = self._split(features)
        encoded = self.history_encoder(history.transpose(1, 2))
        pooled = torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=1)
        return self.head(pooled, query)


def build_model(name: str, input_size: int = 18, sequence_length: int = 31,
                width: int = 48, dropout: float = .2) -> nn.Module:
    """Build an uncalibrated four-logit classifier; callers own loss and labels."""
    if not isinstance(name, str) or name not in MODEL_NAMES:
        raise ValueError(f"name must be one of {', '.join(MODEL_NAMES)}")
    options = dict(input_size=input_size, sequence_length=sequence_length, width=width, dropout=dropout)
    if name in ("resnet18", "resnet34"):
        blocks = (2, 2, 2, 2) if name == "resnet18" else (3, 4, 6, 3)
        return ResidualClassifier(blocks, **options)
    return {"mlp_deep": DeepMLP, "inception": InceptionClassifier}[name](**options)


def parameter_count(model: nn.Module) -> int:
    """Trainable parameter count, not bar count or generated sample count."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
