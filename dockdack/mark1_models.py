"""Small comparable sequence classifiers for the mark_1 experiment.

Every architecture consumes precomputed ``[batch, 31, 9]`` features and emits
one *logit* per sample, suitable for ``BCEWithLogitsLoss``. The final token is
the entry-price query, not a completed candle: callers must mask information
that would only become known after entry. Feature construction, labels,
chronological splits, probability calibration and trading are intentionally
outside this module. Different architectures need not have identical parameter
counts; report :func:`parameter_count` alongside measured validation results.

Validate feature values once at an ingestion/inference boundary with
``validate_features``. Forward passes check shape and dtype, but do not
synchronize the GPU to recheck already validated finite values every batch.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
import sys

import torch
from torch import Tensor, nn


MODEL_NAMES = ("mlp", "lstm", "gru", "tcn", "transformer")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _check_hyperparameters(input_size: int, sequence_length: int,
                         hidden_size: int, dropout: float) -> None:
    for name, value in (("input_size", input_size), ("sequence_length", sequence_length),
                        ("hidden_size", hidden_size)):
        _positive_integer(value, name)
    if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout) or not 0 <= dropout < 1):
        raise ValueError("dropout must be a finite number in [0, 1)")


def _check_features(features: Tensor, input_size: int, sequence_length: int) -> None:
    if not isinstance(features, Tensor):
        raise ValueError("features must be a floating-point torch.Tensor")
    if (features.ndim != 3 or features.shape[0] < 1
            or tuple(features.shape[1:]) != (sequence_length, input_size)):
        raise ValueError(
            f"features must have shape [nonempty batch, {sequence_length}, {input_size}]"
        )
    if not features.is_floating_point():
        raise ValueError("features must use a floating-point tensor dtype")


def validate_features(features: Tensor, *, input_size: int = 9,
                      sequence_length: int = 31) -> None:
    """Validate at a trusted-data boundary; finite checking may synchronize CUDA."""
    _positive_integer(input_size, "input_size")
    _positive_integer(sequence_length, "sequence_length")
    _check_features(features, input_size, sequence_length)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("features must contain only finite values")


def _rnn_backend(device: str | torch.device):
    """Bypass the known Windows CUDA RNN shutdown failure without global changes."""
    if sys.platform == "win32" and torch.device(device).type == "cuda":
        return torch.backends.cudnn.flags(enabled=False)
    return nullcontext()


class _SequenceClassifier(nn.Module):
    def __init__(self, input_size: int, sequence_length: int,
                 hidden_size: int, dropout: float):
        super().__init__()
        _check_hyperparameters(input_size, sequence_length, hidden_size, dropout)
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.dropout_probability = float(dropout)

    def _validate(self, features: Tensor) -> None:
        _check_features(features, self.input_size, self.sequence_length)


class MLPClassifier(_SequenceClassifier):
    """Flattened-window baseline: the query and every historical token are explicit."""

    def __init__(self, input_size: int = 9, sequence_length: int = 31,
                 hidden_size: int = 64, dropout: float = 0.15):
        super().__init__(input_size, sequence_length, hidden_size, dropout)
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size * sequence_length, 2 * hidden_size),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        self._validate(features)
        return self.network(features).squeeze(-1)


class RecurrentClassifier(_SequenceClassifier):
    """Two-layer LSTM or GRU, using the hidden state after the final query token."""

    def __init__(self, cell: str, input_size: int = 9, sequence_length: int = 31,
                 hidden_size: int = 64, dropout: float = 0.15):
        super().__init__(input_size, sequence_length, hidden_size, dropout)
        if cell not in ("lstm", "gru"):
            raise ValueError("cell must be 'lstm' or 'gru'")
        self.cell = cell
        cell_type = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = cell_type(input_size, hidden_size, num_layers=2,
                             dropout=float(dropout), batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, features: Tensor) -> Tensor:
        self._validate(features)
        with _rnn_backend(features.device):
            sequence, _ = self.rnn(features)
        return self.head(sequence[:, -1]).squeeze(-1)


class _CausalResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3,
                               dilation=dilation, padding=2 * dilation)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3,
                               dilation=dilation, padding=2 * dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor) -> Tensor:
        length = features.shape[-1]
        hidden = self.dropout(self.activation(self.conv1(features)[..., :length]))
        hidden = self.dropout(self.activation(self.conv2(hidden)[..., :length]))
        return self.activation(features + hidden)


class TemporalCNNClassifier(_SequenceClassifier):
    """Causal residual TCN with a receptive field covering the full input window."""

    def __init__(self, input_size: int = 9, sequence_length: int = 31,
                 hidden_size: int = 64, dropout: float = 0.15):
        super().__init__(input_size, sequence_length, hidden_size, dropout)
        # Two kernel-3 convolutions per block add 4*dilation positions.
        dilations = []
        receptive_field, dilation = 1, 1
        while receptive_field < sequence_length:
            dilations.append(dilation)
            receptive_field += 4 * dilation
            dilation *= 2
        self.dilations = tuple(dilations)
        self.receptive_field = receptive_field
        self.projection = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.blocks = nn.Sequential(*(
            _CausalResidualBlock(hidden_size, dilation, dropout) for dilation in dilations
        ))
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, features: Tensor) -> Tensor:
        self._validate(features)
        hidden = self.blocks(self.projection(features.transpose(1, 2)))
        return self.head(hidden[:, :, -1]).squeeze(-1)


class TransformerClassifier(_SequenceClassifier):
    """Two-layer encoder with sinusoidal positions and a final-query readout.

    Full attention is intentional: every input token, including the candidate
    entry price, is available when scoring the query. No outcome-day future
    observation may be supplied as a feature.
    """

    def __init__(self, input_size: int = 9, sequence_length: int = 31,
                 hidden_size: int = 64, dropout: float = 0.15):
        super().__init__(input_size, sequence_length, hidden_size, dropout)
        heads = next(candidate for candidate in (4, 2, 1) if hidden_size % candidate == 0)
        self.attention_heads = heads
        self.projection = nn.Linear(input_size, hidden_size)
        position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_size)
        )
        encoding = torch.zeros(sequence_length, hidden_size)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies[:hidden_size // 2])
        self.register_buffer("positional_encoding", encoding.unsqueeze(0), persistent=True)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=heads, dim_feedforward=2 * hidden_size,
            dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(hidden_size),
                                            enable_nested_tensor=False)
        # TransformerEncoder clones the supplied layer; initialize the clones
        # independently instead of starting both layers with identical weights.
        for encoder_layer in self.encoder.layers:
            nn.init.xavier_uniform_(encoder_layer.self_attn.in_proj_weight)
            nn.init.zeros_(encoder_layer.self_attn.in_proj_bias)
            for linear in (encoder_layer.self_attn.out_proj,
                           encoder_layer.linear1, encoder_layer.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, features: Tensor) -> Tensor:
        self._validate(features)
        hidden = self.projection(features)
        hidden = self.input_dropout(hidden + self.positional_encoding.to(dtype=hidden.dtype))
        return self.head(self.encoder(hidden)[:, -1]).squeeze(-1)


def build_model(name: str, input_size: int = 9, sequence_length: int = 31,
                hidden_size: int = 64, dropout: float = 0.15) -> nn.Module:
    """Construct an architecture by a stable lower-case name; output is not sigmoid."""
    if not isinstance(name, str) or name not in MODEL_NAMES:
        raise ValueError(f"name must be one of {', '.join(MODEL_NAMES)}")
    options = dict(input_size=input_size, sequence_length=sequence_length,
                   hidden_size=hidden_size, dropout=dropout)
    if name in ("lstm", "gru"):
        return RecurrentClassifier(cell=name, **options)
    architectures = {"mlp": MLPClassifier, "tcn": TemporalCNNClassifier,
                     "transformer": TransformerClassifier}
    return architectures[name](**options)


def parameter_count(model: nn.Module) -> int:
    """Return trainable parameter count, excluding non-trainable positional buffers."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
