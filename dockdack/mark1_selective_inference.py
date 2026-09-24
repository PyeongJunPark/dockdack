"""Read-only CPU inference for a complete selective Mark_1 research run.

This module exposes probabilities and a *research* policy mask, never BUY or
order permission. The training target describes the complete session, not the
remaining path after an arbitrary intraday entry. Both-market completion,
frozen feature/training code, and native model checksums are verified at load.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .mark1_data import TARGET
from .mark1_metrics import calibrated_probability
from .mark1_selective_features import FEATURE_NAMES, features_from_history
from .mark1_selective_models import (
    CLASS_NAMES, MODEL_NAMES, OWNER, SCHEMA_VERSION, load_model, model_path, predict_raw,
)
from .mark1_selective_policy import apply_policy, qualification


FOLD = "walk_2024"
SEEDS = (42, 43, 44)
MARKETS = ("domestic", "us")
ROOT = Path(__file__).resolve().parents[1]
PATH_LIMITATION = (
    "Whole-session conservative daily-bar event: high >= entry*1.01 and "
    "low > entry*0.991. Both-touch is failure. This is not a verified "
    "first-touch probability after an arbitrary intraday entry."
)


def _canonical(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Research artifacts must contain finite JSON-compatible metadata") from error


def _read(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError(f"Linked research artifact is not allowed: {path.name}")
    try:
        result = _canonical(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, UnicodeError) as error:
        raise ValueError(f"Complete research run requires readable {path.name}") from error
    if not isinstance(result, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return result


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or linked research artifact: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _positive_integer(value) -> bool:
    return type(value) is int and value > 0


def _same(left, right) -> bool:
    # Unlike Python dict equality, canonical JSON also distinguishes True/1.
    return json.dumps(_canonical(left), sort_keys=True) == json.dumps(_canonical(right), sort_keys=True)


def _metadata_without_reused(value):
    if not isinstance(value, dict) or type(value.get("reused")) is not bool:
        raise ValueError("Model metadata requires a boolean cache-reuse flag")
    return {key: item for key, item in value.items() if key != "reused"}


def _calibration(value, count):
    if (not isinstance(value, dict) or value.get("method") != "platt_monotone"
            or not _positive_integer(value.get("fit_samples")) or value["fit_samples"] != count
            or value.get("weighted") is not False):
        raise ValueError("A separate, unweighted, count-matched Platt calibration is required")
    calibrated_probability([0.], value)


def _split_dates(splits, fold):
    bounds = {
        "train": (fold["train_start"], fold["train_end"]),
        "tune": (f"{fold['tune_year']}-01-01", f"{fold['tune_year']}-12-31"),
        "probability_calibration": (f"{fold['calibration_year']}-01-01", f"{fold['calibration_year']}-06-30"),
        "policy_calibration": (f"{fold['calibration_year']}-07-01", f"{fold['calibration_year']}-12-31"),
        "audit": (f"{fold['selection_year']}-01-01", f"{fold['selection_year']}-12-31"),
    }
    for part, (minimum, maximum) in bounds.items():
        value = splits[part]
        first, last = value.get("first"), value.get("last")
        if (not isinstance(first, str) or not isinstance(last, str)
                or re.fullmatch(r"\d{4}-\d{2}-\d{2}", first) is None
                or re.fullmatch(r"\d{4}-\d{2}-\d{2}", last) is None
                or not minimum <= first <= last <= maximum
                or not _positive_integer(value.get("symbols")) or value["symbols"] > value["count"]):
            raise ValueError("Split dates/populations disagree with frozen chronological folds")
        try:
            np.datetime64(first, "D"), np.datetime64(last, "D")
        except ValueError as error:
            raise ValueError("Invalid calendar date in frozen splits") from error


def _source(value, market, frozen):
    if (not isinstance(value, dict) or value.get("market") != market or value.get("target") != TARGET
            or type(value.get("version")) is not int or value["version"] != 2
            or type(value.get("purge_sessions")) is not int or value["purge_sessions"] != 30
            or not _digest(value.get("database_sha256"))):
        raise ValueError("Source provenance/target/purge contract mismatch")
    expected = Path(frozen["db_dir"]) / f"{market}_daily_clean.sqlite3"
    if not isinstance(value.get("database_path"), str) or Path(value["database_path"]).resolve() != expected.resolve():
        raise ValueError("Source database path disagrees with frozen run")


def _validate_qualification(ensemble):
    if not isinstance(ensemble, dict) or not isinstance(ensemble.get("policy_selection"), dict):
        raise ValueError("Missing ensemble policy selection")
    calculated = qualification(ensemble.get("audit"))
    if not _same(calculated, ensemble.get("qualification")):
        raise ValueError("Ensemble qualification disagrees with frozen audit metrics")
    policy = ensemble["policy_selection"]
    flag = policy.get("calibration_qualified")
    if (type(flag) is not bool or policy.get("research_only") is not True
            or policy.get("deployment_allowed") is not False
            or not _same(qualification(policy.get("chosen_metrics")), policy.get("qualification"))
            or (flag and policy["qualification"]["qualified"] is not True)):
        raise ValueError("Inconsistent research-only calibration qualification")
    return calculated["qualified"] and flag


class SelectivePredictor:
    """Load the selected final-fold three-seed ensemble, with no order wiring."""

    def __init__(self, run_market_directory):
        # Importing the frozen runner is safe: main() is guarded; it neither
        # trains nor loads optional backends, databases, or trading components.
        from examples.train_mark1_selective import CODE_FILES, PROTOCOL

        original_folder = Path(run_market_directory).absolute()
        if original_folder.is_symlink():
            raise ValueError("Research market folder must not be linked")
        folder = original_folder.resolve()
        market = folder.name
        if market not in MARKETS:
            raise ValueError("Research market directory must be domestic or us")
        frozen = _read(folder.parent / "protocol.json")
        if not _same(frozen.get("protocol"), PROTOCOL):
            raise ValueError("Incompatible current/frozen selective protocol")
        if (PROTOCOL["target"] != TARGET or PROTOCOL["ensemble_seeds"] != list(SEEDS)
                or PROTOCOL["feature_names"] != list(FEATURE_NAMES) or len(FEATURE_NAMES) != 184):
            raise ValueError("Incompatible target, feature names or ensemble seeds")
        if (frozen.get("task_type") not in ("CPU", "GPU") or not _positive_integer(frozen.get("threads"))
                or not isinstance(frozen.get("db_dir"), str) or not isinstance(frozen.get("versions"), dict)):
            raise ValueError("Malformed frozen run context")
        hashes = frozen.get("code_sha256")
        if (not isinstance(hashes, dict) or set(hashes) != set(CODE_FILES)
                or any(not _digest(hashes[name]) or _sha(ROOT / name) != hashes[name] for name in CODE_FILES)):
            raise ValueError("Frozen research code hash mismatch")
        global_summary = _read(folder.parent / "summary.json")
        if set(global_summary) != set(MARKETS):
            raise ValueError("Both markets must finish before research inference")
        # Completion is cross-checked against both files, not a single flag in
        # the requested market. The other market's native weights are not read.
        for item_market in MARKETS:
            summary = _read(folder.parent / item_market / "summary.json")
            if (not _same(summary, global_summary[item_market]) or summary.get("market") != item_market
                    or summary.get("completed") is not True or summary.get("research_only") is not True
                    or summary.get("deployment_allowed") is not False
                    or summary.get("selected") not in MODEL_NAMES
                    or type(summary.get("research_qualified")) is not bool):
                raise ValueError("Global/per-market summaries are incomplete or inconsistent")
            _source(summary.get("source"), item_market, frozen)
        summary = global_summary[market]
        source = _read(folder / "source.json")
        if not _same(source, summary["source"]):
            raise ValueError("Source artifact disagrees with completed market summary")
        selected = summary["selected"]
        locked = _read(folder / "selection_locked.json")
        if (locked.get("market") != market or locked.get("selected") != selected
                or not _same(locked.get("source"), source)
                or not _same(locked.get("ranking"), summary.get("ranking"))
                or locked.get("selection_rule") != PROTOCOL["selection"]):
            raise ValueError("Architecture selection lock mismatch")
        ensembles = summary.get("ensembles")
        if not isinstance(ensembles, dict) or set(ensembles) != set(PROTOCOL["folds"]):
            raise ValueError("Both frozen development ensembles are required")
        gates = []
        for fold, ensemble in ensembles.items():
            if not _same(_read(folder / fold / "ensemble.json"), ensemble):
                raise ValueError("Ensemble artifact disagrees with completed summary")
            gates.append(_validate_qualification(ensemble))
        if summary["research_qualified"] is not all(gates):
            raise ValueError("Summary research qualification is inconsistent")
        ensemble = ensembles[FOLD]
        splits = _read(folder / FOLD / "splits.json")
        features = _read(folder / FOLD / "features" / "features.json")
        parts = {"train", "tune", "probability_calibration", "policy_calibration", "audit"}
        if (set(splits) != parts or not _same(features.get("splits"), splits)
                or features.get("features") != list(FEATURE_NAMES)
                or not isinstance(features.get("index_sha256"), dict)
                or set(features["index_sha256"]) != parts
                or any(not _digest(value) for value in features["index_sha256"].values())
                or any(not isinstance(value, dict) or not _positive_integer(value.get("count")) for value in splits.values())):
            raise ValueError("Frozen feature/split contract mismatch")
        _split_dates(splits, PROTOCOL["folds"][FOLD])
        if (ensemble["audit"].get("count") != splits["audit"]["count"]
                or ensemble["policy_selection"]["chosen_metrics"].get("count") != splits["policy_calibration"]["count"]):
            raise ValueError("Ensemble assessment counts disagree with frozen splits")
        calibrators = ensemble.get("calibration")
        if not isinstance(calibrators, dict) or set(calibrators) != {"success", "stop"}:
            raise ValueError("Ensemble success/stop calibration schema mismatch")
        count = splits["probability_calibration"]["count"]
        _calibration(calibrators["success"], count)
        if selected == "cat_joint6":
            _calibration(calibrators["stop"], count)
        elif calibrators["stop"] is not None:
            raise ValueError("Binary architecture must not invent a stop-risk calibration")
        policy = ensemble["policy_selection"].get("chosen_policy")
        if (not isinstance(policy, dict) or set(policy) != {"threshold", "stop_probability_cap"}
                or type(policy["threshold"]) not in (int, float)
                or policy["threshold"] not in PROTOCOL["thresholds"]
                or type(policy["stop_probability_cap"]) not in (int, float)
                or policy["stop_probability_cap"] not in PROTOCOL["joint_stop_caps"]
                or (selected != "cat_joint6" and policy["stop_probability_cap"] != 1.)):
            raise ValueError("Policy is outside the frozen grid/architecture contract")
        apply_policy([.5], policy, [0.] if selected == "cat_joint6" else None)
        members, seed_results = ensemble.get("member_sha256"), ensemble.get("seed_results")
        if (not isinstance(members, dict) or set(members) != {str(seed) for seed in SEEDS}
                or any(not _digest(value) for value in members.values())
                or not isinstance(seed_results, list) or len(seed_results) != len(SEEDS)):
            raise ValueError("Exactly three checksummed ensemble seeds are required")
        models, common_data = [], None
        for seed, result in zip(SEEDS, seed_results):
            if (not isinstance(result, dict) or type(result.get("seed")) is not int
                    or result["seed"] != seed or result.get("architecture") != selected):
                raise ValueError("Ensemble seed/architecture order mismatch")
            trial = folder / FOLD / f"{selected}-{seed}"
            native_path = model_path(trial, selected)
            metadata = _read(trial / "metadata.json")
            request = _read(trial / "request.json")
            trial_result = _read(trial / "result.json")
            # result.json is the exact member result embedded in ensemble.json.
            if not _same(trial_result, result):
                raise ValueError("Trial result disagrees with ensemble member")
            if not _same(_metadata_without_reused(metadata), _metadata_without_reused(result.get("model"))):
                raise ValueError("Model metadata disagrees with ensemble member")
            if (metadata.get("owner") != OWNER or metadata.get("schema_version") != SCHEMA_VERSION
                    or metadata.get("model_name") != selected or metadata.get("research_only") is not True
                    or metadata.get("deployment_allowed") is not False
                    or metadata.get("feature_count") != len(FEATURE_NAMES)
                    or metadata.get("model_sha256") != members[str(seed)]
                    or _sha(native_path) != members[str(seed)] or not _same(metadata.get("request"), request)
                    or not _same(metadata.get("params"), request.get("params"))
                    or not _same(metadata.get("versions"), request.get("versions"))
                    or not _positive_integer(metadata.get("best_iteration"))):
                raise ValueError("Native model checksum/metadata mismatch")
            version_key = "lightgbm" if selected == "lgbm_binary" else "catboost"
            expected_task = "CPU" if selected == "lgbm_binary" else frozen["task_type"]
            versions, params, data_hashes = request.get("versions"), request.get("params"), request.get("data_sha256")
            if (request.get("owner") != OWNER or request.get("schema_version") != SCHEMA_VERSION
                    or request.get("model_name") != selected or type(request.get("seed")) is not int or request["seed"] != seed
                    or request.get("class_names") != list(CLASS_NAMES)
                    or request.get("max_iterations") != PROTOCOL["maximum_iterations"]
                    or request.get("early_stopping") != PROTOCOL["early_stopping"]
                    or request.get("requested_task_type") != frozen["task_type"]
                    or request.get("effective_task_type") != expected_task
                    or metadata.get("effective_task_type") != expected_task
                    or request.get("wrapper_sha256") != hashes["dockdack/mark1_selective_models.py"]
                    or request.get("train_shape") != [splits["train"]["count"], len(FEATURE_NAMES)]
                    or request.get("tune_shape") != [splits["tune"]["count"], len(FEATURE_NAMES)]
                    or not isinstance(versions, dict) or versions.get("backend") != frozen["versions"].get(version_key)
                    or versions.get("numpy") != frozen["versions"].get("numpy")
                    or not isinstance(params, dict)
                    or params.get("num_threads" if selected == "lgbm_binary" else "thread_count") != frozen["threads"]
                    or not isinstance(data_hashes, dict) or set(data_hashes) != {"x_train", "y_train", "x_tune", "y_tune"}
                    or any(not _digest(value) for value in data_hashes.values())):
                raise ValueError("Native trial configuration/source/features mismatch")
            if common_data is not None and not _same(data_hashes, common_data):
                raise ValueError("Ensemble members were fitted on different input data")
            common_data = data_hashes
            models.append(load_model(selected, native_path))
        self.models = tuple(models)
        self.market, self.model_name = market, selected
        self.research_qualified = summary["research_qualified"]
        self._calibration, self._policy = copy.deepcopy(calibrators), copy.deepcopy(policy)
        self.metadata = {
            "market": market, "model_name": selected, "fold": FOLD, "seeds": list(SEEDS),
            "feature_names": list(FEATURE_NAMES), "target": TARGET, "class_names": list(CLASS_NAMES),
            "source": copy.deepcopy(source), "member_sha256": copy.deepcopy(members),
            "policy": copy.deepcopy(policy), "research_only": True, "deployment_allowed": False,
            "research_qualified": self.research_qualified, "path_limitation": PATH_LIMITATION,
        }

    def predict(self, history, entry_price, *, entry_is_session_open=False):
        """Return a diagnostic mask, never permission to trade or claim a path."""
        if type(entry_is_session_open) is not bool:
            raise ValueError("entry_is_session_open must be explicitly boolean")
        historical = np.asarray(history)
        entry = np.asarray(entry_price)
        if historical.shape != (30, 5) or entry.ndim != 0 or entry.dtype.kind not in "fiu":
            raise ValueError("Exactly 30 completed OHLCV bars and one numeric scalar entry are required")
        features = features_from_history(historical[None], entry.reshape(1), validate=True)
        quote = float(entry)
        if not math.isfinite(quote * 1.01):
            raise ValueError("Entry price is too large for finite barrier prices")
        raw = [predict_raw(model, self.model_name, features) for model in self.models]
        has_stop = self.model_name == "cat_joint6"
        for member in raw:
            if (not isinstance(member, dict) or set(member) != {"success_logits", "stop_logits"}
                    or np.asarray(member["success_logits"]).shape != (1,)
                    or not np.isfinite(member["success_logits"]).all()
                    or (has_stop and (np.asarray(member["stop_logits"]).shape != (1,)
                                     or not np.isfinite(member["stop_logits"]).all()))
                    or (not has_stop and member["stop_logits"] is not None)):
                raise ValueError("Ensemble returned invalid raw log odds")
        success_logit = math.fsum(float(member["success_logits"][0]) for member in raw) / len(raw)
        success = float(calibrated_probability([success_logit], self._calibration["success"])[0])
        stop = None
        if has_stop:
            stop_logit = math.fsum(float(member["stop_logits"][0]) for member in raw) / len(raw)
            stop = float(calibrated_probability([stop_logit], self._calibration["stop"])[0])
        selected = bool(apply_policy([success], self._policy, [stop] if has_stop else None)[0])
        reasons = []
        if not selected:
            reasons.append("frozen_policy_abstains")
        if not self.research_qualified:
            reasons.append("research_qualification_failed")
        if not entry_is_session_open:
            reasons.append("entry_is_not_declared_session_open_unvalidated")
        return {
            "probability_success": success, "probability_stop": stop,
            "policy_threshold": float(self._policy["threshold"]),
            "stop_probability_cap": float(self._policy["stop_probability_cap"]),
            "selected_research": selected, "research_qualified": self.research_qualified,
            "research_only": True, "deployment_allowed": False,
            "entry_is_session_open": entry_is_session_open,
            "entry_matches_evaluated_type": entry_is_session_open,
            "intraday_path_verified": False, "path_limitation": PATH_LIMITATION,
            "target": TARGET, "market": self.market, "model_name": self.model_name,
            "inference_device": "cpu", "ensemble_method": "mean_raw_logits_then_platt",
            "take_profit_pct": 1.0, "stop_loss_pct": .9,
            "candidate_entry_price": quote, "candidate_take_price": quote * 1.01,
            "candidate_stop_price": quote * .991, "notes": reasons,
        }
