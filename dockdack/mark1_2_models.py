"""Comparable history/query classifiers for the separate Mark1.2 experiment.

All seven architectures consume the same [batch,31,18] representation and
return uncalibrated logits in the same four-class order. The last token is a
candidate price query, never a completed candle: history encoders see only the
first thirty tokens. Features and the MLP/ResNet implementations are reused
without modifying the frozen Mark1 research modules.

RNN-family parameters remain FP32. Their forward pass disables autocast and,
only around CUDA recurrence, cuDNN, avoiding the previously reproduced Windows
CUDA RNN/dropout native shutdown problem. CNN/ResNet callers may use AMP.
No data access, training loop, calibration, inference bundle, or order policy
is implemented by this module.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn

from dockdack.mark1_deep_models import (
    CLASS_NAMES, FEATURE_NAMES, LOOKBACK, TARGET, _Classifier, _QueryHead, _norm,
    build_model as _deep_model, features_from_history, parameter_count,
    success_logit, validate_features,
)


MODEL_NAMES = ("mlp", "rnn", "lstm", "gru", "cnn", "resnet18", "resnet34")


class RecurrentClassifier(_Classifier):
    """Two-layer unidirectional history encoder with a separate query head."""

    def __init__(self, cell: str, input_size=18, sequence_length=31, width=48, dropout=.2):
        super().__init__(input_size, sequence_length, width, dropout)
        if cell not in ("rnn", "lstm", "gru"):
            raise ValueError("cell must be rnn, lstm, or gru")
        cell_type = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.cell = cell
        self.history_encoder = cell_type(input_size, width * 2, num_layers=2,
                                         dropout=float(dropout), batch_first=True)
        self.head = _QueryHead(width * 2, input_size, width, dropout)

    def forward(self, features: Tensor) -> Tensor:
        history, query = self._split(features)
        if next(self.parameters()).dtype != torch.float32:
            raise ValueError("Mark1.2 recurrent model parameters must remain float32; use AMP, not model.half()")
        # Restore both local contexts even on errors. No backend setting is
        # changed at import/build time or for CPU recurrence/convolutions.
        with torch.autocast(device_type=features.device.type, enabled=False):
            backend = torch.backends.cudnn.flags(enabled=False) if features.is_cuda else nullcontext()
            with backend:
                sequence, _ = self.history_encoder(history.float())
            return self.head(sequence[:, -1], query.float())


class CNNClassifier(_Classifier):
    """Three plain Conv1d stages over history, with mean/max pooling."""

    def __init__(self, input_size=18, sequence_length=31, width=48, dropout=.2):
        super().__init__(input_size, sequence_length, width, dropout)
        channels = (input_size, width, width * 2, width * 4)
        layers = []
        for index, (incoming, outgoing) in enumerate(zip(channels[:-1], channels[1:])):
            kernel = 5 if index == 0 else 3
            layers.extend((nn.Conv1d(incoming, outgoing, kernel, stride=1 if index == 0 else 2,
                                     padding=kernel // 2, bias=False), _norm(outgoing), nn.GELU()))
        self.history_encoder = nn.Sequential(*layers)
        self.head = _QueryHead(width * 8, input_size, width, dropout)

    def forward(self, features: Tensor) -> Tensor:
        history, query = self._split(features)
        encoded = self.history_encoder(history.transpose(1, 2))
        pooled = torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=1)
        return self.head(pooled, query)


def build_model(name: str, input_size: int = 18, sequence_length: int = 31,
                width: int = 48, dropout: float = .2) -> nn.Module:
    """Build a four-logit model under the fixed common Mark1.2 input contract."""
    if not isinstance(name, str) or name not in MODEL_NAMES:
        raise ValueError(f"name must be one of {', '.join(MODEL_NAMES)}")
    if type(input_size) is not int or input_size != len(FEATURE_NAMES):
        raise ValueError("Mark1.2 comparisons require exactly 18 common input features")
    if type(sequence_length) is not int or sequence_length != LOOKBACK + 1:
        raise ValueError("Mark1.2 comparisons require exactly 30 history tokens plus one query")
    options = dict(input_size=input_size, sequence_length=sequence_length, width=width, dropout=dropout)
    if name in ("mlp", "resnet18", "resnet34"):
        return _deep_model("mlp_deep" if name == "mlp" else name, **options)
    if name == "cnn":
        return CNNClassifier(**options)
    return RecurrentClassifier(name, **options)


__all__ = ["CLASS_NAMES", "FEATURE_NAMES", "LOOKBACK", "TARGET", "MODEL_NAMES", "build_model",
           "features_from_history", "parameter_count", "success_logit", "validate_features"]
