"""Read-only, research-only inference for a completed deep Mark_1 ensemble.

The contract is 30 completed OHLCV bars plus one candidate price, with a
conservative whole-session target. This is NOT the probability of profit
before loss after an arbitrary intraday entry. The module neither imports
broker/GUI code nor changes deployment, monitoring, or order settings.

CUDA inference follows the training evaluator's BF16 autocast. CPU inference
uses FP32 and can differ numerically, including near the strict 50% boundary.
Research qualification never grants deployment permission.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import pickle
import re

import numpy as np
import torch

from .mark1_deep_data import FOLDS
from .mark1_deep_models import (
    CLASS_NAMES, FEATURE_NAMES, MODEL_NAMES, TARGET, build_model,
    features_from_history, success_logit, validate_features,
)
from .mark1_metrics import calibrated_probability


SEEDS = (42, 43, 44)
FOLD = "walk_2024"
BUY_THRESHOLD = .5
_ENSEMBLE_METHOD = "mean three raw success logits, then independent calibration-year positive-slope Platt"


def _json_value(value):
    """Normalize tuple/list differences between JSON and safe torch metadata."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Ensemble contract metadata must be finite JSON-compatible values") from exc


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError(f"A completed ensemble requires readable {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return _json_value(value)


def _number_equals(value, expected, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value != expected:
        raise ValueError(f"Deep Mark_1 requires {name}={expected}")


def _validate_calibration(value, expected_count=None):
    if (not isinstance(value, dict) or value.get("method") != "platt_monotone"
            or type(value.get("fit_samples")) is not int or value["fit_samples"] < 1):
        raise ValueError("A fitted held-out monotone Platt calibration is required")
    if expected_count is not None and value["fit_samples"] != expected_count:
        raise ValueError("Calibration sample count disagrees with the declared fold")
    calibrated_probability([0.], value)


def _model_config(value):
    if (not isinstance(value, dict)
            or set(value) != {"input_size", "sequence_length", "width", "dropout"}
            or type(value["input_size"]) is not int or value["input_size"] != 18
            or type(value["sequence_length"]) is not int or value["sequence_length"] != 31
            or type(value["width"]) is not int or not 4 <= value["width"] <= 256
            or type(value["dropout"]) not in (int, float)
            or not math.isfinite(value["dropout"]) or not 0 <= value["dropout"] < 1):
        raise ValueError("Incompatible deep Mark_1 model configuration")
    return dict(value)


class DeepPredictor:
    """Load a completed three-seed research ensemble, never a deployed model."""

    def __init__(self, run_market_directory: str | Path, device="cpu"):
        try:
            self.device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("Research inference device must be CPU or CUDA") from exc
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Research inference supports only CPU or CUDA")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("Requested CUDA is unavailable")
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("Research CUDA inference requires BF16 support")
        self.inference_precision = "bf16_cuda" if self.device.type == "cuda" else "float32_cpu"
        folder = Path(run_market_directory).resolve()
        summary, source = _read_json(folder / "summary.json"), _read_json(folder / "source.json")
        market, selected = summary.get("market"), summary.get("selected")
        if (not isinstance(market, str) or market not in ("domestic", "us")
                or not isinstance(selected, str) or selected not in MODEL_NAMES
                or type(summary.get("research_qualified")) is not bool):
            raise ValueError("Incompatible ensemble market, architecture, or research qualification")
        if (source.get("market") != market or source.get("target") != TARGET
                or type(source.get("version")) is not int or source["version"] != 2
                or not isinstance(source.get("database_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", source["database_sha256"])):
            raise ValueError("Incompatible source database provenance")
        protocol = summary.get("protocol")
        if (not isinstance(protocol, dict) or protocol.get("version") != "mark1-deep-v1"
                or protocol.get("ensemble_seeds") != list(SEEDS)
                or protocol.get("folds") != FOLDS
                or protocol.get("ensemble") != _ENSEMBLE_METHOD
                or protocol.get("threshold_rule") != "strictly_greater"
                or protocol.get("both_touch") != "stop_first"):
            raise ValueError("Incompatible ensemble protocol")
        for key, expected in (("threshold", .5), ("take", .01), ("stop", .009)):
            _number_equals(protocol.get(key), expected, key)
        config = _model_config(protocol.get("model_config"))
        calibration = summary.get("ensemble_calibration")
        _validate_calibration(calibration)

        seed_results = summary.get("seed_results")
        if (not isinstance(seed_results, list) or len(seed_results) != len(SEEDS)
                or any(not isinstance(result, dict) or type(result.get("seed")) is not int
                       or result["seed"] != seed or result.get("architecture") != selected
                       for seed, result in zip(SEEDS, seed_results))):
            raise ValueError("Ensemble summary must identify exactly seeds 42,43,44 of the selected architecture")
        gates = summary.get("fold_qualification")
        ensemble_gate = summary.get("ensemble_qualification")
        if (not isinstance(gates, dict) or set(gates) != set(FOLDS)
                or any(not isinstance(value, dict) or type(value.get("qualified")) is not bool
                       for value in gates.values())
                or not isinstance(ensemble_gate, dict) or type(ensemble_gate.get("qualified")) is not bool
                or summary["research_qualified"] != (ensemble_gate["qualified"]
                    and all(value["qualified"] for value in gates.values()))):
            raise ValueError("Inconsistent ensemble research qualification flags")

        models, common_context = [], None
        for seed in SEEDS:
            path = folder / FOLD / f"{selected}-{seed}" / "model.pt"
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as exc:
                raise ValueError(f"Cannot safely load research checkpoint {path.parent.name}") from exc
            if not isinstance(checkpoint, dict):
                raise ValueError("Research checkpoint must be a dictionary")
            if (checkpoint.get("architecture") != selected
                    or type(checkpoint.get("seed")) is not int or checkpoint["seed"] != seed
                    or checkpoint.get("target") != TARGET
                    or _json_value(checkpoint.get("feature_names")) != list(FEATURE_NAMES)
                    or _json_value(checkpoint.get("class_names")) != list(CLASS_NAMES)
                    or checkpoint.get("research_only") is not True
                    or checkpoint.get("intraday_path_verified") is not False
                    or type(checkpoint.get("best_epoch")) is not int or checkpoint["best_epoch"] < 1
                    or _model_config(checkpoint.get("model_config")) != config
                    or _json_value(checkpoint.get("protocol")) != protocol):
                raise ValueError("Research checkpoint feature/target/architecture/protocol mismatch")
            for key, expected in (("threshold", .5), ("take_profit_pct", 1.), ("stop_loss_pct", .9)):
                _number_equals(checkpoint.get(key), expected, key)
            context = checkpoint.get("context")
            if (not isinstance(context, dict) or context.get("market") != market
                    or context.get("fold") != FOLD or _json_value(context.get("source")) != source
                    or not isinstance(context.get("splits"), dict)):
                raise ValueError("Checkpoint source/market/fold contract mismatch")
            context = _json_value(context)
            if common_context is not None and context != common_context:
                raise ValueError("Ensemble checkpoints disagree on training/evaluation membership")
            common_context = context
            splits = context["splits"]
            if (set(splits) != {"train", "tune", "calibration", "selection"}
                    or any(not isinstance(part, dict) or type(part.get("samples")) is not int
                           or part["samples"] < 1 for part in splits.values())):
                raise ValueError("Checkpoint requires nonempty declared development splits")
            _validate_calibration(calibration, splits["calibration"]["samples"])
            _validate_calibration(checkpoint.get("calibration"), splits["calibration"]["samples"])
            state = checkpoint.get("state_dict")
            if (not isinstance(state, dict) or not state
                    or any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
                           or not bool(torch.isfinite(value).all()) for value in state.values())):
                raise ValueError("Research checkpoint contains invalid or non-finite weights")
            # Initial random parameters are overwritten; do not alter the
            # caller's CPU RNG just by loading an inference-only ensemble.
            with torch.random.fork_rng(devices=[]):
                model = build_model(selected, **config)
            try:
                model.load_state_dict(state, strict=True)
            except (RuntimeError, TypeError) as exc:
                raise ValueError("Research checkpoint weights do not match its architecture") from exc
            models.append(model.to(self.device).float().eval())

        self.models = tuple(models)
        self.calibration = dict(calibration)
        self.market, self.model_name = market, selected
        self.research_qualified = summary["research_qualified"]
        self.buy_threshold = BUY_THRESHOLD
        self.metadata = dict(market=market, model_name=selected, seeds=list(SEEDS),
                             feature_names=list(FEATURE_NAMES), class_names=list(CLASS_NAMES),
                             target=TARGET, model_config=config, source=source,
                             research_only=True, deployment_allowed=False,
                             research_qualified=self.research_qualified)

    @torch.inference_mode()
    def predict(self, bars, current_price):
        if isinstance(current_price, (bool, np.bool_)):
            raise ValueError("Candidate entry must be finite and positive")
        try:
            entry = Decimal(str(current_price))
            if not entry.is_finite() or entry <= 0:
                raise ValueError("Candidate entry must be finite and positive")
            raw = torch.as_tensor(bars)
            if raw.dtype == torch.bool or raw.is_complex():
                raise ValueError("Historical OHLCV must be real numeric values, not boolean/complex")
            raw = raw.to(device="cpu", dtype=torch.float32)
            quote = torch.tensor([float(entry)], dtype=torch.float32)
        except (InvalidOperation, ValueError, TypeError, RuntimeError, OverflowError) as exc:
            raise ValueError("Invalid research OHLCV history or candidate entry") from exc
        if tuple(raw.shape) != (30, 5):
            raise ValueError("Research inference needs exactly 30 completed OHLCV bars plus one price query")
        # Compute causal FP32 features on the selected device, as the training
        # evaluator does, before entering the CUDA BF16 model context.
        features = features_from_history(raw[None].to(self.device), quote.to(self.device), validate=True)
        validate_features(features)
        logits = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            for model in self.models:
                value = success_logit(model(features).float())
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Research ensemble returned a non-finite success logit")
                logits.append(float(value.item()))
        # Python floats are float64, matching the evaluator's ensemble mean.
        raw_logit = math.fsum(logits) / len(logits)
        probability = float(calibrated_probability([raw_logit], self.calibration)[0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Research ensemble returned an invalid probability")
        return dict(probability_success=probability, predicts_success=probability > BUY_THRESHOLD,
                    buy_threshold=BUY_THRESHOLD, target=TARGET, model_name=self.model_name,
                    market=self.market, intraday_path_verified=False, research_only=True,
                    research_qualified=self.research_qualified, deployment_allowed=False,
                    inference_precision=self.inference_precision)
