"""Mark_1 inference: historical daily bars plus a candidate-entry query.

This predicts a conservative WHOLE-session OHLC event. It is not a calibrated
claim about the unseen remainder of an arbitrary intraday price path. Neither
this module nor loading a checkpoint enables monitoring or submits orders.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
from pathlib import Path

import torch

from dockdack.mark1_data import FEATURE_NAMES, TARGET, features_from_history
from dockdack.mark1_metrics import calibrated_probability
from dockdack.mark1_models import MODEL_NAMES, build_model, validate_features


LOOKBACK = 30
SEQUENCE_LENGTH = 31
BUY_THRESHOLD = 0.5


class Predictor:
    """Strict, non-ordering inference contract for a selected Mark_1 model."""

    def __init__(self, checkpoint: str | Path, device="cpu"):
        self.device = torch.device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
            raise ValueError("Mark_1 checkpoint requires metadata")
        metadata = dict(payload["metadata"])
        if (type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1
                or metadata.get("strategy_version") != "mark_1"
                or metadata.get("target") != TARGET
                or not isinstance(metadata.get("market"), str)
                or metadata.get("market") not in {"domestic", "us"}
                or type(metadata.get("lookback")) is not int or metadata["lookback"] != LOOKBACK
                or type(metadata.get("sequence_length")) is not int or metadata["sequence_length"] != SEQUENCE_LENGTH
                or metadata.get("feature_names") != list(FEATURE_NAMES)):
            raise ValueError("Incompatible Mark_1 feature/target/market metadata")
        for key, expected in (("buy_threshold", 0.5), ("take_profit_pct", 1.0), ("stop_loss_pct", 0.9)):
            value = metadata.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value != expected:
                raise ValueError(f"Mark_1 requires {key}={expected}")
        name, config = metadata.get("model_name"), metadata.get("model_config")
        if name not in MODEL_NAMES or not isinstance(config, dict):
            raise ValueError("Unknown Mark_1 model architecture")
        if (set(config) != {"input_size", "sequence_length", "hidden_size", "dropout"}
                or type(config["input_size"]) is not int or config["input_size"] != len(FEATURE_NAMES)
                or type(config["sequence_length"]) is not int or config["sequence_length"] != SEQUENCE_LENGTH
                or type(config["hidden_size"]) is not int or not 4 <= config["hidden_size"] <= 512):
            raise ValueError("Incompatible Mark_1 model configuration")
        calibration = metadata.get("calibration")
        if not isinstance(calibration, dict) or calibration.get("method") != "platt_monotone":
            raise ValueError("A held-out Mark_1 probability calibration is required")
        if type(calibration.get("fit_samples")) is not int or calibration["fit_samples"] < 1:
            raise ValueError("Calibration must report positive held-out sample count")
        # Validate scale/offset even before the first inference request.
        calibrated_probability([0.0], calibration)
        self.model = build_model(name, **config)
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Mark_1 checkpoint requires weights")
        if any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
               or not bool(torch.isfinite(value).all())
               for value in state.values()):
            raise ValueError("Mark_1 checkpoint contains invalid weights")
        try:
            self.model.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("Mark_1 weights do not match the architecture") from exc
        self.model.to(self.device).eval()
        self.metadata, self.market = metadata, metadata["market"]
        self.buy_threshold = BUY_THRESHOLD

    @torch.inference_mode()
    def predict(self, bars, current_price):
        if isinstance(current_price, bool):
            raise ValueError("Current price must be positive and finite")
        try:
            entry = Decimal(str(current_price))
            if not entry.is_finite() or entry <= 0:
                raise ValueError("Current price must be positive and finite")
            raw = torch.as_tensor(bars, dtype=torch.float32)
            quote = torch.tensor([float(entry)], dtype=torch.float32)
        except (InvalidOperation, ValueError, TypeError, RuntimeError, OverflowError) as exc:
            raise ValueError("Invalid Mark_1 history or current price") from exc
        if raw.shape != (LOOKBACK, 5):
            raise ValueError("Mark_1 needs exactly 30 completed OHLCV bars plus the current-price query")
        features = features_from_history(raw.unsqueeze(0), quote, validate=True)
        validate_features(features)
        logit = float(self.model(features.to(self.device)).item())
        probability = float(calibrated_probability([logit], self.metadata["calibration"])[0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Mark_1 returned an invalid probability")
        return {"probability_success": probability, "predicts_success": probability > BUY_THRESHOLD,
                "buy_threshold": BUY_THRESHOLD, "target": TARGET,
                "intraday_path_verified": False, "model_name": self.metadata["model_name"]}
