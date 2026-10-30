"""Research target-hit and timeout policies; never connect to an order path.

This is deliberately separate from Mark1 prototype inference and the GUI.
Screen a predeclared L/H/p grid on train/tune, compare a few model families
on calibration/selection, freeze one policy, then read test outcomes once.
Daily HIGH touching a target is only a fill proxy. No profitability claim is
made from these samples or from the hypothetical H-spaced blocks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from dockdack.mark1_2_data import load_dataset
from dockdack.mark1_target_horizon_data import load_horizon_bank
from dockdack.mark1_target_horizon_eval import (fit_ridge_return,
                                                nonoverlap_block_summary)
from dockdack.mark1_target_horizon_models import (LOOKBACKS, MODEL_FAMILIES,
                                                  build_model, sequence_features)
from dockdack.research_artifacts import sha256_file, write_new_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "mark1" / "target-horizon"
SPLITS = ("train", "tune", "calibration", "selection", "test")
BASELINE = (20, 10, 3.0)
GRID_LOOKBACKS = (10, 20, 30)
GRID_HORIZONS = (5, 10, 20)
GRID_TARGETS = (2.0, 3.0, 4.0)
FAMILIES = MODEL_FAMILIES + ("catboost",)


def _sha_array(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_npz(path: Path, **arrays: np.ndarray) -> str:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return sha256_file(path)


def _safe_output(path: Path) -> Path:
    output = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if not output.is_relative_to(root) or output == root or output.exists():
        raise ValueError("output must be a new child directory under outputs/mark1/target-horizon")
    return output


def _compact(summary: dict) -> dict:
    """Keep metrics and a receipt without bloating every grid trial report."""
    chosen = np.asarray(summary["selected_indices"], dtype=np.int64)
    return {key: value for key, value in summary.items()
            if key not in ("selected_indices", "block_returns")} | {
        "selected_bank_positions_sha256": _sha_array(chosen),
    }


def _split_samples(dataset, *, train_cap: int, eval_cap: int, seed: int) -> np.ndarray:
    """Reproducible sampling within already approved chronological splits."""
    if train_cap < 0 or eval_cap < 0:
        raise ValueError("sample caps must be nonnegative (0 means all)")
    rng = np.random.default_rng(seed)
    chosen = []
    for name in SPLITS:
        source = np.asarray(dataset.splits[name], dtype=np.int64)
        if not len(source):
            raise ValueError(f"approved {name} split is empty")
        cap = train_cap if name == "train" else eval_cap
        if cap and len(source) > cap:
            source = np.sort(rng.choice(source, size=cap, replace=False))
        chosen.append(source)
    combined = np.sort(np.concatenate(chosen))
    if len(np.unique(combined)) != len(combined):
        raise ValueError("approved split memberships overlap")
    return combined


class FeatureCache:
    """Materialize only the requested split from the sealed completed-bar bank."""

    def __init__(self, bank, dataset, *, batch_size: int = 2048):
        self.bank = bank
        self.dataset = dataset
        self.batch_size = batch_size
        self._cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    def get(self, lookback: int, split: str) -> tuple[np.ndarray, np.ndarray]:
        if split not in SPLITS:
            raise ValueError("unknown split")
        key = (lookback, split)
        if key not in self._cache:
            positions = np.flatnonzero(self.bank.split_names == split).astype(np.int64)
            chunks = []
            for start in range(0, len(positions), self.batch_size):
                subset = positions[start:start + self.batch_size]
                raw = self.bank.histories(self.dataset, lookback=lookback,
                                          indices=subset)
                # The only new-session observation used at training is its
                # known OPEN. Runtime must provide the current quote instead.
                chunks.append(sequence_features(raw, lookback,
                                                self.bank.entry_prices[subset]))
            if not chunks:
                raise ValueError(f"no {split} samples")
            self._cache[key] = positions, np.concatenate(chunks)
        return self._cache[key]


def _eligible_features(cache: FeatureCache, lookback: int, split: str,
                       outcomes) -> tuple[np.ndarray, np.ndarray]:
    positions, features = cache.get(lookback, split)
    keep = outcomes.eligible[positions]
    return positions[keep], features[keep]


def _score_array(bank, positions: np.ndarray, values: np.ndarray) -> np.ndarray:
    if values.shape != (len(positions),) or not np.isfinite(values).all():
        raise ValueError("prediction must be finite and aligned with bank positions")
    result = np.zeros(len(bank.sample_indices), dtype=np.float64)
    result[positions] = values
    return result


def _evaluate(bank, outcomes, positions: np.ndarray, scores: np.ndarray, *,
              horizon: int, top_k: int, allocation: float,
              anchor: int, score_floor: float = -math.inf) -> dict:
    eligible = np.zeros(len(bank.sample_indices), dtype=bool)
    eligible[positions] = True
    summary = nonoverlap_block_summary(
        _score_array(bank, positions, scores), outcomes.net_return,
        bank.entry_ordinals, bank.symbol_ids, outcomes.hit, eligible,
        horizon=horizon, top_k=top_k, allocation=allocation,
        score_floor=score_floor, anchor_ordinal=anchor,
    )
    # Normalize across H on the *same chronological period*, including cash
    # sessions, rather than rewarding short horizons for having more blocks.
    span = int(np.max(bank.split_end_ordinals[positions])
               - np.min(bank.split_start_ordinals[positions]) + 1)
    if span <= 0:
        raise ValueError("nonpositive chronological comparison span")
    summary["period_sessions"] = span
    summary["per_session_log_growth"] = math.log1p(summary["total_return"]) / span
    return summary


def _anchor(bank, split: str) -> int:
    dates = bank.entry_ordinals[bank.split_names == split]
    if not len(dates):
        raise ValueError(f"no {split} entry dates")
    return int(dates.min())


def _momentum_rule(features: np.ndarray) -> np.ndarray:
    """Fixed, unfitted close/open plus gap momentum in the completed window."""
    return np.asarray((features[..., 0] + features[..., 3]).sum(axis=1),
                      dtype=np.float64)


def _quality(summary: dict, min_trades: int) -> tuple[float, float, float, float]:
    trades = summary["trades"]
    return (float(trades >= min_trades), float(summary["per_session_log_growth"]),
            float(summary["mean_net_return"] if summary["mean_net_return"] is not None
                  else -1.0), float(trades))


def _eligibility_counts(bank, outcomes, split: str, lookback: int,
                        horizon: int) -> dict:
    """Count nonexclusive reasons; path causes cannot be separated in this bank."""
    member = bank.split_names == split
    history_boundary = bank.entry_ordinals - lookback < bank.split_start_ordinals
    horizon_boundary = bank.entry_ordinals + horizon - 1 > bank.split_end_ordinals
    path_invalid = ~bank.valid_prefix[:, horizon - 1]
    ineligible = member & ~outcomes.eligible
    return {
        "sampled": int(member.sum()),
        "eligible": int((member & outcomes.eligible).sum()),
        "ineligible": int(ineligible.sum()),
        "reason_counts_are_nonexclusive": True,
        "insufficient_completed_history_inside_split": int((member & history_boundary).sum()),
        "outcome_crosses_split_boundary": int((member & horizon_boundary).sum()),
        "future_path_missing_zero_volume_or_segment_discontinuity":
            int((member & path_invalid).sum()),
        "future_path_subcauses_not_separable_from_current_bank": True,
    }


def _screen(bank, cache: FeatureCache, destination: Path, *,
            cost_bps: float, slippage_bps: float, top_k: int,
            allocation: float, min_trades: int) -> list[dict]:
    records: list[dict] = []
    for lookback in GRID_LOOKBACKS:
        for horizon in GRID_HORIZONS:
            for target_pct in GRID_TARGETS:
                train = bank.outcomes(target_pct=target_pct, horizon=horizon,
                                      cost_bps=cost_bps, slippage_bps=slippage_bps,
                                      split_name="train", lookback=lookback)
                tune = bank.outcomes(target_pct=target_pct, horizon=horizon,
                                     cost_bps=cost_bps, slippage_bps=slippage_bps,
                                     split_name="tune", lookback=lookback)
                train_positions, train_x = _eligible_features(cache, lookback, "train", train)
                tune_positions, tune_x = _eligible_features(cache, lookback, "tune", tune)
                if len(train_positions) < 2 or len(tune_positions) == 0:
                    records.append({"lookback": lookback, "horizon": horizon,
                                    "target_pct": target_pct,
                                    "status": "insufficient_eligible_events",
                                    "train_count": len(train_positions),
                                    "tune_count": len(tune_positions),
                                    "eligibility": {
                                        "train": _eligibility_counts(bank, train, "train",
                                                                     lookback, horizon),
                                        "tune": _eligibility_counts(bank, tune, "tune",
                                                                    lookback, horizon)}})
                    continue
                ridge = fit_ridge_return(train_x.reshape(len(train_x), -1),
                                         train.net_return[train_positions], alpha=0.05)
                ridge_scores = ridge.predict(tune_x.reshape(len(tune_x), -1))
                rule_scores = _momentum_rule(tune_x)
                trial = {"lookback": lookback, "horizon": horizon,
                         "target_pct": target_pct, "status": "completed",
                         "train_count": len(train_positions),
                         "tune_count": len(tune_positions),
                         "cost_bps_round_trip": cost_bps,
                         "slippage_bps_per_side": slippage_bps,
                         "eligibility": {
                             "train": _eligibility_counts(bank, train, "train",
                                                          lookback, horizon),
                             "tune": _eligibility_counts(bank, tune, "tune",
                                                         lookback, horizon)},
                         "methods": {}}
                for name, scores in (("ridge_return", ridge_scores),
                                     ("fixed_momentum", rule_scores)):
                    summary = _evaluate(bank, tune, tune_positions, scores,
                                        horizon=horizon, top_k=top_k,
                                        allocation=allocation, anchor=_anchor(bank, "tune"))
                    trial["methods"][name] = _compact(summary)
                stem = f"L{lookback}-H{horizon}-P{target_pct:g}-ridge"
                path = destination / f"{stem}.npz"
                trial["ridge_checkpoint"] = path.name
                trial["ridge_checkpoint_sha256"] = _write_npz(
                    path, feature_mean=ridge.feature_mean,
                    feature_scale=ridge.feature_scale,
                    target_mean=np.asarray(ridge.target_mean),
                    coefficients=ridge.coefficients,
                )
                records.append(trial)
    return records


def _shortlist(records: list[dict], count: int, min_trades: int) -> list[tuple[int, int, float]]:
    if count < 1:
        raise ValueError("candidate count must be positive")
    baseline = BASELINE
    ranked = []
    for trial in records:
        if trial["status"] != "completed":
            continue
        combo = (trial["lookback"], trial["horizon"], trial["target_pct"])
        best = max((_quality(value, min_trades) for value in trial["methods"].values()))
        ranked.append((best, combo))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    selected = [baseline]
    for _, combo in ranked:
        if combo not in selected and len(selected) < count:
            selected.append(combo)
    if not any(row[1] == baseline for row in ranked):
        raise ValueError("baseline 20/10/3 lacks eligible train/tune samples")
    return selected


def _normalize_train(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = train_x.mean(axis=(0, 1), dtype=np.float64)
    scale = np.maximum(train_x.std(axis=(0, 1), dtype=np.float64), 1e-6)
    return center.astype(np.float32), scale.astype(np.float32)


def _apply_normalization(values: np.ndarray, center: np.ndarray,
                         scale: np.ndarray) -> np.ndarray:
    result = np.clip((values - center) / scale, -8.0, 8.0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite standardized sequence")
    return result


@dataclass
class FittedPredictor:
    family: str
    model: object
    model_return: object | None
    center: np.ndarray | None
    scale: np.ndarray | None
    device: str
    lookback: int

    def predict(self, features: np.ndarray, *, batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        if features.ndim != 3 or features.shape[1:] != (self.lookback, 6):
            raise ValueError("feature window differs from fitted model")
        logits, returns = [], []
        if self.family == "catboost":
            flat = features.reshape(len(features), -1)
            for start in range(0, len(flat), batch_size):
                batch = flat[start:start + batch_size]
                logits.append(np.asarray(self.model.predict(
                    batch, prediction_type="RawFormulaVal"), dtype=np.float64).reshape(-1))
                returns.append(np.asarray(self.model_return.predict(batch),
                                          dtype=np.float64).reshape(-1))
        else:
            if self.center is None or self.scale is None:
                raise ValueError("neural feature normalization is missing")
            self.model.eval()
            with torch.inference_mode():
                for start in range(0, len(features), batch_size):
                    batch = _apply_normalization(features[start:start + batch_size],
                                                 self.center, self.scale)
                    prediction = self.model(torch.from_numpy(batch).to(self.device))
                    outputs = prediction.detach().cpu().numpy().astype(np.float64)
                    logits.append(outputs[:, 0])
                    returns.append(outputs[:, 1])
        if not logits:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        hit_logits, net_returns = np.concatenate(logits), np.concatenate(returns)
        if not np.isfinite(hit_logits).all() or not np.isfinite(net_returns).all():
            raise ValueError("model prediction is nonfinite")
        return hit_logits, net_returns


def _fit_torch(family: str, lookback: int, train_x: np.ndarray,
               hits: np.ndarray, net: np.ndarray, *, seed: int, epochs: int,
               batch_size: int, device: str) -> FittedPredictor:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_model(family, lookback).to(device)
    center, scale = _normalize_train(train_x)
    values = _apply_normalization(train_x, center, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        for positions in np.array_split(rng.permutation(len(values)),
                                        max(1, math.ceil(len(values) / batch_size))):
            x = torch.from_numpy(values[positions]).to(device)
            y_hit = torch.from_numpy(hits[positions].astype(np.float32)).to(device)
            y_net = torch.from_numpy(net[positions].astype(np.float32)).to(device)
            prediction = model(x)
            loss = (F.binary_cross_entropy_with_logits(prediction[:, 0], y_hit)
                    + 20.0 * F.smooth_l1_loss(prediction[:, 1], y_net, beta=0.05))
            if not bool(torch.isfinite(loss)):
                raise ValueError("training loss became nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return FittedPredictor(family, model, None, center, scale, device, lookback)


def _fit_catboost(lookback: int, train_x: np.ndarray, hits: np.ndarray,
                  net: np.ndarray, *, seed: int, iterations: int) -> FittedPredictor:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("CatBoost is required for the catboost research family") from exc
    if len(np.unique(hits)) != 2:
        raise ValueError("CatBoost hit classifier requires both training classes")
    flat = train_x.reshape(len(train_x), -1)
    classifier = CatBoostClassifier(iterations=iterations, depth=4,
                                    learning_rate=0.05, loss_function="Logloss",
                                    random_seed=seed, thread_count=1, verbose=False,
                                    allow_writing_files=False)
    regressor = CatBoostRegressor(iterations=iterations, depth=4,
                                  learning_rate=0.05, loss_function="RMSE",
                                  random_seed=seed, thread_count=1, verbose=False,
                                  allow_writing_files=False)
    classifier.fit(flat, hits.astype(np.int8))
    regressor.fit(flat, net)
    return FittedPredictor("catboost", classifier, regressor,
                           None, None, "cpu", lookback)


def _save_predictor(destination: Path, stem: str, fitted: FittedPredictor,
                    *, lookback: int, horizon: int, target_pct: float,
                    seed: int, train_count: int, cost_bps: float,
                    slippage_bps: float, source_sha256: str) -> dict:
    if fitted.family == "catboost":
        hit_path = destination / f"{stem}-hit.cbm"
        return_path = destination / f"{stem}-net.cbm"
        fitted.model.save_model(str(hit_path))
        fitted.model_return.save_model(str(return_path))
        checkpoint = {"hit": {"file": hit_path.name, "sha256": sha256_file(hit_path)},
                      "net": {"file": return_path.name, "sha256": sha256_file(return_path)}}
    else:
        weights = destination / f"{stem}.pt"
        with weights.open("xb") as stream:
            torch.save({"state_dict": fitted.model.state_dict(),
                        "family": fitted.family, "lookback": lookback}, stream)
        checkpoint = {"joint": {"file": weights.name, "sha256": sha256_file(weights)}}
    metadata = {
        "research_only": True, "deployment_allowed": False,
        "family": fitted.family, "lookback": lookback,
        "horizon": horizon, "target_pct": target_pct,
        "output_columns": ["target_hit_logit", "expected_net_return_fraction"],
        "entry": "hypothetical_observed_next_session_open_fill",
        "entry_executability_warning": (
            "The next-session OPEN cannot both be observed and used for a decision "
            "that fills at that identical OPEN. This is a retrospective proxy, "
            "not an executable opening-order simulation."),
        "exit": "first_daily_high_target_touch_else_Hth_close",
        "no_stop_loss": True,
        "training_split": "train", "training_eligible_rows": train_count,
        "cost_bps_round_trip": cost_bps, "slippage_bps_per_side": slippage_bps,
        "seed": seed, "source_sha256": source_sha256,
        "feature_contract": "completed_ohlcv_plus_query_log_gap_v2_6_channels",
        "query_price_training": "observed_next_session_open",
        "query_price_runtime_proposed": "intraday_current_price_only",
        "runtime_mismatch_warning": (
            "Training labels use observed next-session OPEN. During-session current-price "
            "queries face a different price/time distribution; today's unfinished "
            "high, low, close and volume must never enter features, and intraday "
            "performance or limit fills are not established by this research."),
        "feature_module_sha256": sha256_file(ROOT / "dockdack" / "mark1_target_horizon_models.py"),
        "normalization_train_only": None if fitted.center is None else {
            "center": fitted.center.astype(float).tolist(),
            "scale": fitted.scale.astype(float).tolist()},
        "checkpoint": checkpoint,
    }
    info = destination / f"{stem}-model.json"
    write_new_json(info, metadata)
    return {"metadata_file": info.name, "metadata_sha256": sha256_file(info),
            "checkpoint": checkpoint}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _temperature(logits: np.ndarray, hits: np.ndarray) -> tuple[float, float]:
    if len(logits) != len(hits) or not len(logits):
        raise ValueError("calibration predictions must align with nonempty labels")
    choices = np.exp(np.linspace(np.log(0.25), np.log(4.0), 33))
    scored = [(float(np.mean((_sigmoid(logits / temp) - hits) ** 2)), float(temp))
              for temp in choices]
    brier, temp = min(scored)
    return temp, brier


def _thresholds(scores: np.ndarray) -> list[float]:
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("nonempty finite calibration scores required")
    return sorted({-math.inf, 0.0, *(float(np.quantile(scores, q))
                                   for q in (0.5, 0.75, 0.9))})


def _calibrate_floor(bank, outcomes, positions: np.ndarray, scores: np.ndarray,
                     *, horizon: int, top_k: int, allocation: float,
                     min_trades: int) -> tuple[float, list[dict]]:
    attempts = []
    for floor in _thresholds(scores):
        summary = _evaluate(bank, outcomes, positions, scores, horizon=horizon,
                            top_k=top_k, allocation=allocation,
                            anchor=_anchor(bank, "calibration"), score_floor=floor)
        attempts.append({"floor": None if floor == -math.inf else floor,
                         "metrics": _compact(summary), "_raw_floor": floor})
    best = max(attempts, key=lambda row: _quality(row["metrics"], min_trades))
    public = [{key: value for key, value in row.items() if key != "_raw_floor"}
              for row in attempts]
    return float(best["_raw_floor"]), public


def _candidate_stem(combo: tuple[int, int, float], family: str) -> str:
    lookback, horizon, target_pct = combo
    return f"L{lookback}-H{horizon}-P{target_pct:g}-{family}"


def _period_comparison(bank, outcomes, positions: np.ndarray,
                       features: np.ndarray, model_scores: np.ndarray, *,
                       split: str, horizon: int, top_k: int, allocation: float,
                       score_floor: float) -> dict:
    common = {"horizon": horizon, "top_k": top_k,
              "allocation": allocation, "anchor": _anchor(bank, split)}
    model = _evaluate(bank, outcomes, positions, model_scores,
                      score_floor=score_floor, **common)
    unconditional = _evaluate(bank, outcomes, positions,
                              np.zeros(len(positions), dtype=np.float64), **common)
    momentum = _evaluate(bank, outcomes, positions, _momentum_rule(features), **common)
    return {"model": _compact(model), "unconditional": _compact(unconditional),
            "fixed_momentum": _compact(momentum)}


def _qualification(periods: dict[str, dict], *, min_trades: int) -> dict:
    """Predeclared development gate; never references the final test period."""
    if min_trades < 30:
        raise ValueError("predeclared minimum cannot be relaxed below 30")
    positive = {
        name: (values["model"]["trades"] > 0
               and values["model"]["mean_net_return"] is not None
               and values["model"]["mean_net_return"] > 0
               and values["model"]["total_return"] > 0)
        for name, values in periods.items()
    }
    selection = periods["selection"]
    model = selection["model"]
    selection_trades = model["trades"] >= min_trades
    selection_profit = positive["selection"]
    improves_both = (
        model["total_return"] > selection["unconditional"]["total_return"]
        and model["total_return"] > selection["fixed_momentum"]["total_return"]
    )
    two_periods = sum(positive.values()) >= 2
    return {
        "passed": bool(selection_trades and selection_profit and improves_both
                       and two_periods),
        "selection_minimum_nonoverlap_trades": min_trades,
        "selection_has_minimum_nonoverlap_trades": bool(selection_trades),
        "selection_mean_and_block_return_positive_after_cost": bool(selection_profit),
        "selection_beats_unconditional_and_fixed_momentum_total_return": bool(improves_both),
        "at_least_two_positive_development_periods": bool(two_periods),
        "positive_periods": positive,
    }


def _compare_models(bank, cache: FeatureCache, destination: Path,
                    shortlist: list[tuple[int, int, float]], *, families: tuple[str, ...],
                    cost_bps: float, slippage_bps: float, seed: int, epochs: int,
                    batch_size: int, catboost_iterations: int, device: str,
                    top_k: int, allocation: float, min_trades: int) -> tuple[list[dict], list[dict]]:
    records = []
    qualified = []
    for combo in shortlist:
        lookback, horizon, target_pct = combo
        outcomes = {name: bank.outcomes(
            target_pct=target_pct, horizon=horizon, cost_bps=cost_bps,
            slippage_bps=slippage_bps, split_name=name, lookback=lookback)
            for name in ("train", "tune", "calibration", "selection")}
        rows = {name: _eligible_features(cache, lookback, name, outcomes[name])
                for name in outcomes}
        if any(not len(rows[name][0]) for name in rows):
            records.append({"candidate": list(combo), "status": "insufficient_events"})
            continue
        train_positions, train_x = rows["train"]
        for family_index, family in enumerate(families):
            trial_seed = seed + family_index
            started = time.perf_counter()
            if family == "catboost":
                if len(np.unique(outcomes["train"].hit[train_positions])) != 2:
                    records.append({"candidate": list(combo), "family": family,
                                    "seed": trial_seed,
                                    "status": "train_hit_class_degenerate"})
                    continue
                fitted = _fit_catboost(lookback, train_x,
                                       outcomes["train"].hit[train_positions],
                                       outcomes["train"].net_return[train_positions],
                                       seed=trial_seed, iterations=catboost_iterations)
            else:
                fitted = _fit_torch(family, lookback, train_x,
                                    outcomes["train"].hit[train_positions],
                                    outcomes["train"].net_return[train_positions],
                                    seed=trial_seed, epochs=epochs,
                                    batch_size=batch_size, device=device)
            stem = _candidate_stem(combo, family)
            artifact = _save_predictor(
                destination, stem, fitted, lookback=lookback, horizon=horizon,
                target_pct=target_pct, seed=trial_seed,
                train_count=len(train_positions), cost_bps=cost_bps,
                slippage_bps=slippage_bps, source_sha256=bank.source_sha256)
            calibration_positions, calibration_x = rows["calibration"]
            calibration_logits, calibration_scores = fitted.predict(calibration_x)
            temperature, brier = _temperature(
                calibration_logits, outcomes["calibration"].hit[calibration_positions])
            floor, floor_trials = _calibrate_floor(
                bank, outcomes["calibration"], calibration_positions, calibration_scores,
                horizon=horizon, top_k=top_k, allocation=allocation,
                min_trades=min_trades)
            tune_positions, tune_x = rows["tune"]
            tune_logits, tune_scores = fitted.predict(tune_x)
            selection_positions, selection_x = rows["selection"]
            selection_logits, selection_scores = fitted.predict(selection_x)
            period_comparison = {
                name: _period_comparison(
                    bank, outcomes[name], rows[name][0], rows[name][1], scores,
                    split=name, horizon=horizon, top_k=top_k,
                    allocation=allocation, score_floor=floor)
                for name, scores in (("tune", tune_scores),
                                     ("calibration", calibration_scores),
                                     ("selection", selection_scores))
            }
            qualification = _qualification(period_comparison, min_trades=min_trades)
            selection_summary = period_comparison["selection"]["model"]
            selection_prob = _sigmoid(selection_logits / temperature)
            selection_brier = float(np.mean((
                selection_prob - outcomes["selection"].hit[selection_positions]) ** 2))
            predictions = destination / f"{stem}-development-predictions.npz"
            prediction_sha = _write_npz(
                predictions, calibration_bank_positions=calibration_positions,
                calibration_hit_logit=calibration_logits,
                calibration_expected_net_return=calibration_scores,
                tune_bank_positions=tune_positions,
                tune_hit_logit=tune_logits,
                tune_expected_net_return=tune_scores,
                selection_bank_positions=selection_positions,
                selection_hit_logit=selection_logits,
                selection_expected_net_return=selection_scores)
            record = {
                "candidate": list(combo), "family": family,
                "seed": trial_seed, "status": "completed",
                "train_eligible": len(train_positions),
                "calibration_eligible": len(calibration_positions),
                "selection_eligible": len(selection_positions),
                "calibration_temperature": temperature,
                "calibration_brier": brier,
                "calibration_score_floor": None if floor == -math.inf else floor,
                "calibration_floor_trials": floor_trials,
                "selection_brier": selection_brier,
                "selection_metrics": selection_summary,
                "development_period_comparison": period_comparison,
                "eligibility": {
                    name: _eligibility_counts(bank, outcomes[name], name,
                                              lookback, horizon)
                    for name in ("train", "tune", "calibration", "selection")},
                "predeclared_qualification": qualification,
                "artifact": artifact,
                "development_predictions": {"file": predictions.name, "sha256": prediction_sha},
                "fit_seconds": round(time.perf_counter() - started, 3),
            }
            records.append(record)
            if qualification["passed"]:
                qualified.append({"record": record, "fitted": fitted,
                                  "floor": floor, "temperature": temperature})
    qualified.sort(key=lambda item: _quality(item["record"]["selection_metrics"], min_trades),
                   reverse=True)
    return records, qualified[:3]


def _one_test(bank, cache: FeatureCache, selected: dict, *,
              cost_bps: float, slippage_bps: float, top_k: int,
              allocation: float, min_trades: int, destination: Path) -> dict:
    """Read test labels only after the model and score floor are frozen."""
    record = selected["record"]
    lookback, horizon, target_pct = record["candidate"]
    test = bank.outcomes(target_pct=target_pct, horizon=horizon,
                         cost_bps=cost_bps, slippage_bps=slippage_bps,
                         split_name="test", lookback=lookback)
    positions, features = _eligible_features(cache, lookback, "test", test)
    if not len(positions):
        raise ValueError("frozen policy has no eligible test events")
    logits, scores = selected["fitted"].predict(features)
    comparison = _period_comparison(
        bank, test, positions, features, scores, split="test",
        horizon=horizon, top_k=top_k, allocation=allocation,
        score_floor=selected["floor"])
    model_summary = comparison["model"]
    test_gate = {
        "minimum_nonoverlap_trades": min_trades,
        "has_minimum_nonoverlap_trades": model_summary["trades"] >= min_trades,
        "positive_mean_and_block_total_after_cost": (
            model_summary["mean_net_return"] is not None
            and model_summary["mean_net_return"] > 0
            and model_summary["total_return"] > 0),
        "beats_both_frozen_benchmarks": (
            model_summary["total_return"] > comparison["unconditional"]["total_return"]
            and model_summary["total_return"] > comparison["fixed_momentum"]["total_return"]),
    }
    test_gate["passed"] = all(value for key, value in test_gate.items()
                              if key != "minimum_nonoverlap_trades")
    probabilities = _sigmoid(logits / selected["temperature"])
    brier = float(np.mean((probabilities - test.hit[positions]) ** 2))
    stem = _candidate_stem((lookback, horizon, target_pct), record["family"])
    path = destination / f"{stem}-frozen-test-predictions.npz"
    digest = _write_npz(path, bank_positions=positions,
                        sample_indices=bank.sample_indices[positions],
                        hit_logit=logits, calibrated_hit_probability=probabilities,
                        expected_net_return=scores,
                        target_hit=test.hit[positions], net_return=test.net_return[positions],
                        selected_bank_positions=np.asarray(
                            nonoverlap_block_summary(
                                _score_array(bank, positions, scores), test.net_return,
                                bank.entry_ordinals, bank.symbol_ids, test.hit,
                                np.isin(np.arange(len(bank.sample_indices)), positions),
                                horizon=horizon, top_k=top_k, allocation=allocation,
                                score_floor=selected["floor"],
                                anchor_ordinal=_anchor(bank, "test"),
                            )["selected_indices"], dtype=np.int64))
    return {"candidate": record["candidate"], "family": record["family"],
            "eligible": len(positions), "brier": brier,
            "comparison": comparison,
            "frozen_test_gate": test_gate,
            "predictions": {"file": path.name, "sha256": digest}}


def run_market(market: str, destination: Path, args) -> dict:
    started = time.perf_counter()
    dataset, source, receipt = load_dataset(market, ROOT)
    selected_samples = _split_samples(dataset, train_cap=args.max_train,
                                      eval_cap=args.max_eval, seed=args.seed)
    bank = load_horizon_bank(dataset, source, ROOT, market,
                             sample_indices=selected_samples,
                             max_horizon=max(GRID_HORIZONS))
    cache = FeatureCache(bank, dataset)
    report = {
        "market": market, "research_only": True, "deployment_allowed": False,
        "source_sha256": bank.source_sha256,
        "sample_indices_sha256": _sha_array(bank.sample_indices),
        "sample_cap_train": args.max_train,
        "sample_cap_per_evaluation_split": args.max_eval,
        "split_counts": {name: int(np.count_nonzero(bank.split_names == name))
                         for name in SPLITS},
        "entry_date_ranges": {
            name: [str(np.datetime64(int(bank.entry_dates[bank.split_names == name].min()), "D")),
                   str(np.datetime64(int(bank.entry_dates[bank.split_names == name].max()), "D"))]
            for name in SPLITS},
        "entry": "next_session_observed_open",
        "entry_executability_warning": (
            "The observed next-session OPEN is used as both query and hypothetical "
            "fill price. A strategy cannot inspect that OPEN then guarantee a fill "
            "at the same OPEN; later intraday current-price queries differ from training."),
        "exit": "first_daily_high_target_touch_else_Hth_close",
        "feature_contract": "completed_ohlcv_plus_query_log_gap_v2_6_channels",
        "query_price_training": "observed_next_session_open",
        "query_price_runtime_proposed": "intraday_current_price_only",
        "runtime_mismatch_warning": (
            "A later intraday quote is not the next-session OPEN used in labels. "
            "The current-day partial OHLCV and post-query highs may not be inferred "
            "from daily bars; intraday success and fills remain unverified."),
        "no_stop_loss": True,
        "cost_bps_round_trip": args.cost_bps,
        "slippage_bps_per_side": args.slippage_bps,
        "top_k_per_H_spaced_block": args.top_k,
        "allocation_per_name": args.allocation,
        "block_anchor_ordinals": {name: _anchor(bank, name) for name in SPLITS},
        "receipt": receipt,
        "screen_grid": {"lookbacks": list(GRID_LOOKBACKS),
                        "horizons": list(GRID_HORIZONS),
                        "targets_pct": list(GRID_TARGETS)},
        "predeclared_qualification": {
            "selection_nonoverlap_trades_at_least": args.min_trades,
            "selection_mean_net_return_positive": True,
            "selection_block_total_return_positive": True,
            "selection_total_return_above_unconditional_and_fixed_momentum": True,
            "positive_direction_in_at_least_two_of":
                ["tune", "calibration", "selection"],
            "final_test_is_not_used_for_selection_or_threshold_adjustment": True,
            "up_to_three_policies_frozen_before_test_each_evaluated_once": True,
            "failed_gate_is_not_relaxed_or_deployed": True,
            "horizon_comparison": "per_session_log_growth_over_full_split_span",
        },
    }
    report["screening"] = _screen(
        bank, cache, destination, cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps, top_k=args.top_k,
        allocation=args.allocation, min_trades=args.min_trades)
    report["shortlist"] = [list(row) for row in _shortlist(
        report["screening"], args.candidate_count, args.min_trades)]
    if args.screen_only:
        report["status"] = "screen_only_test_not_read"
    else:
        report["model_trials"], frozen = _compare_models(
            bank, cache, destination,
            [tuple(row) for row in report["shortlist"]],
            families=tuple(args.families), cost_bps=args.cost_bps,
            slippage_bps=args.slippage_bps, seed=args.seed, epochs=args.epochs,
            batch_size=args.batch_size, catboost_iterations=args.catboost_iterations,
            device=args.device, top_k=args.top_k, allocation=args.allocation,
            min_trades=args.min_trades)
        report["frozen_policies"] = [{
            "candidate": chosen["record"]["candidate"],
            "family": chosen["record"]["family"],
            "seed": chosen["record"]["seed"],
            "score_floor": chosen["record"]["calibration_score_floor"],
            "temperature": chosen["record"]["calibration_temperature"],
            "selection_metrics": chosen["record"]["selection_metrics"],
            "artifact": chosen["record"]["artifact"],
        } for chosen in frozen]
        if frozen and args.development_only:
            report["test_once_each_frozen_policy"] = []
            report["test_gate_pass_count"] = 0
            report["status"] = "development_only_test_not_read"
        elif frozen:
            report["test_once_each_frozen_policy"] = [
                _one_test(bank, cache, chosen, cost_bps=args.cost_bps,
                          slippage_bps=args.slippage_bps, top_k=args.top_k,
                          allocation=args.allocation, min_trades=args.min_trades,
                          destination=destination)
                for chosen in frozen
            ]
            report["test_gate_pass_count"] = sum(
                result["frozen_test_gate"]["passed"]
                for result in report["test_once_each_frozen_policy"])
            report["status"] = (
                "research_complete_requires_independent_review"
                if report["test_gate_pass_count"] else
                "research_complete_no_test_robust_candidate")
        else:
            report["test_once_each_frozen_policy"] = []
            report["test_gate_pass_count"] = 0
            report["status"] = "no_qualified_candidate_test_not_read"
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["limitations"] = [
        "Current symbol inventory is not certified point-in-time; survivorship bias may remain.",
        "Missing, zero-volume and cross-segment future paths are excluded, which selects the sample; subcauses are not separable from the present bank.",
        "The next-session OPEN cannot be observed for a decision and filled at that identical OPEN; this is a retrospective entry proxy.",
        "Daily HIGH touching a sell limit is not evidence that the order would fill.",
        "A later intraday quote is a distribution shift from observed next-session OPEN training.",
        "The H-spaced top-K block proxy is not a cash/latency-aware brokerage backtest.",
        "Different H values have different block counts; ranking normalizes log growth by the full split session span.",
        "Up to three policies are frozen before test, each evaluated once; no post-test winner is selected.",
        "Real intraday path, interim drawdown and live profitability are unverified.",
        "This research runner performs no prototype, GUI, broker, or operating-ledger integration.",
    ]
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", nargs="+", choices=("domestic", "us"),
                        default=("domestic", "us"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-train", type=int, default=5000,
                        help="0 uses all approved train events; default is a bounded first pass")
    parser.add_argument("--max-eval", type=int, default=5000,
                        help="0 uses all approved rows in each later split")
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--families", nargs="+", choices=FAMILIES,
                        default=FAMILIES)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--catboost-iterations", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cost-bps", type=float, default=20.0,
                        help="round-trip fees and taxes in basis points")
    parser.add_argument("--slippage-bps", type=float, default=5.0,
                        help="additional adverse basis points on each side")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--allocation", type=float, default=0.1)
    parser.add_argument("--min-trades", type=int, default=30,
                        help="screening and threshold gate; cannot be below 30")
    parser.add_argument("--screen-only", action="store_true",
                        help="development check: no model comparison or test labels")
    parser.add_argument("--development-only", action="store_true",
                        help="train and select using development splits, without computing test outcomes")
    args = parser.parse_args(argv)
    if (args.screen_only and args.development_only
            or len(set(args.markets)) != len(args.markets)
            or len(set(args.families)) != len(args.families)
            or args.seed < 0 or args.max_train < 0 or args.max_eval < 0
            or args.candidate_count < 1 or args.epochs < 1 or args.batch_size < 1
            or args.catboost_iterations < 1 or args.top_k < 1
            or args.min_trades < 30 or not 0 < args.allocation <= 1 / args.top_k
            or not math.isfinite(args.cost_bps) or args.cost_bps < 0
            or not math.isfinite(args.slippage_bps) or args.slippage_bps < 0
            or (args.device == "cuda" and not torch.cuda.is_available())):
        parser.error("invalid research configuration or unavailable CUDA device")
    requested_device = args.device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if "catboost" in args.families:
        try:
            import catboost  # noqa: F401 - dependency preflight before creating output
        except ImportError:
            parser.error("CatBoost unavailable; use an environment with the research extra "
                         "or choose --families without catboost")
    try:
        destination = _safe_output(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = {
        "experiment": "Mark1_target_horizon", "research_only": True,
        "deployment_allowed": False, "not_paper_reproduction": True,
        "configuration": vars(args) | {"output_dir": str(destination),
                                        "requested_device": requested_device},
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "torch": torch.__version__},
        "code_sha256": {name: sha256_file(ROOT / name) for name in (
            "examples/research_mark1_target_horizon.py",
            "dockdack/mark1_target_horizon_models.py",
            "dockdack/mark1_target_horizon_data.py",
            "dockdack/mark1_target_horizon_eval.py")},
        "reports": {},
    }
    for market in args.markets:
        market_dir = destination / market
        market_dir.mkdir(exist_ok=False)
        report = run_market(market, market_dir, args)
        path = market_dir / "report.json"
        write_new_json(path, report)
        manifest["reports"][market] = {"file": str(path.relative_to(destination)),
                                      "sha256": sha256_file(path),
                                      "status": report["status"]}
        print(f"{market}: {report['status']} -> {path}", flush=True)
    write_new_json(destination / "manifest.json", manifest)


if __name__ == "__main__":
    main()
