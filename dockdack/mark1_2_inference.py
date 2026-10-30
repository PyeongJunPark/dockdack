"""Offline, CPU-FP32 Mark1.2 probabilities; never broker/order permission."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from . import mark1_2_models, mark1_deep_models, mark1_metrics
from .mark1_2_models import MODEL_NAMES, build_model
from .mark1_deep_models import CLASS_NAMES, FEATURE_NAMES, TARGET, features_from_history, success_logit
from .mark1_metrics import calibrated_probability

TITLE = "mark1.2 prototype"
OWNER = "dockdack.mark1_2_prototype"
MARKETS = ("domestic", "us")
SEEDS = (42, 43, 44)
FLAGS = {"research_only": True, "deployment_allowed": False, "intraday_path_verified": False}
SEMANTICS = {
    "lookback": 30, "feature_count": 18, "bar_columns": ["open", "high", "low", "close", "volume"],
    "target": TARGET, "class_names": list(CLASS_NAMES), "take_profit_pct": 1., "stop_loss_pct": .9,
    "both_touch": "stop_first_failure", "threshold": .5, "comparison": "strict_greater_than",
    "ensemble": "mean_raw_success_logits_then_monotone_platt",
    "history_dtype": "float32", "query_dtype": "float32", "feature_dtype": "float32",
    "inference_dtype": "float32", "ensemble_accumulation_dtype": "float32",
    "platt_dtype": "float64", "output_dtype": "float64",
    "precision": "history and candidate prices quantized to float32; features, inference and mean logits CPU FP32; calibration/output float64",
}
WARNINGS = [
    "Research probabilities only; no order permission or profitability guarantee.",
    "Whole-session daily OHLC labels do not verify the remaining path after an arbitrary intraday entry.",
    "Candidate-price augmentation is counterfactual data, not independently observed trades.",
    "Calibration uses observed session opens; calibration over arbitrary intraday prices is unverified.",
    "Known US symbols were quarantined; other corporate actions and survivorship issues may remain.",
    "Historical 2025+ results have already been reused in research, not a pristine independent test.",
]


def linked(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def unlinked_path(path):
    path = Path(path).absolute()
    if any(linked(item) for item in (path, *path.parents)):
        raise ValueError("Linked artifact paths are forbidden")
    return path


def sha256_file(path):
    path = unlinked_path(path)
    if not path.is_file():
        raise ValueError("Missing artifact file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    path = unlinked_path(path)
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Missing or oversized JSON artifact")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must be an object")
    return value


def checked_file(root, relative, digest):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or Path(relative).is_absolute() or any(part in ("", ".", "..") for part in relative.split("/"))
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("Unsafe artifact reference")
    root = unlinked_path(root)
    candidate = unlinked_path(root / relative)
    if not candidate.resolve().is_relative_to(root.resolve()) or sha256_file(candidate) != digest:
        raise ValueError("Artifact checksum mismatch")
    return candidate


def runtime_code_hashes():
    return {"inference": sha256_file(Path(__file__)),
            "models": sha256_file(Path(mark1_2_models.__file__)),
            "features": sha256_file(Path(mark1_deep_models.__file__)),
            "calibration": sha256_file(Path(mark1_metrics.__file__))}


def validate_calibration(value):
    if (not isinstance(value, dict) or value.get("method") != "platt_monotone"
            or value.get("weighted") is not False or type(value.get("fit_samples")) is not int
            or value["fit_samples"] <= 0 or type(value.get("slope")) not in (int, float)
            or type(value.get("bias")) not in (int, float)):
        raise ValueError("Invalid observed-open monotone calibration")
    calibrated_probability([0.], value)


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {"input_size", "sequence_length", "width", "dropout"}:
        raise ValueError("Invalid model configuration")
    if (type(config["input_size"]) is not int or config["input_size"] != 18
            or type(config["sequence_length"]) is not int or config["sequence_length"] != 31
            or type(config["width"]) is not int or not 4 <= config["width"] <= 256
            or type(config["dropout"]) not in (int, float)
            or not np.isfinite(config["dropout"]) or not 0 <= config["dropout"] < 1):
        raise ValueError("Invalid common 31x18 model contract")


def validate_state(state):
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing model state")
    for key, value in state.items():
        if (not isinstance(key, str) or not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32 or value.device.type != "cpu"
                or not bool(torch.isfinite(value).all())):
            raise ValueError("Model state must contain finite CPU float32 tensors only")


def validate_flags(value):
    if not isinstance(value, dict) or set(value) != set(FLAGS) or any(value[key] is not flag for key, flag in FLAGS.items()):
        raise ValueError("Research-only flags must remain explicit booleans")


def validate_source(value):
    if not isinstance(value, dict) or set(value) != {"source_files_sha256", "protocol_sha256", "summary_sha256"}:
        raise ValueError("Missing source provenance")
    files = value["source_files_sha256"]
    if not isinstance(files, dict) or not {"protocol.json", "summary.json", "status.json"}.issubset(files):
        raise ValueError("Incomplete source provenance")
    for relative, digest in files.items():
        if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
                or Path(relative).is_absolute() or any(part in ("", ".", "..") for part in relative.split("/"))
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Unsafe source provenance reference")
    if value["protocol_sha256"] != files["protocol.json"] or value["summary_sha256"] != files["summary.json"]:
        raise ValueError("Source provenance checksum mismatch")


def load_weights(path):
    path = unlinked_path(path)
    if not path.is_file() or path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Missing or oversized model checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a dictionary")
    validate_state(payload.get("state_dict"))
    return payload


class Predictor:
    """Three cached neural members, with fixed CPU-FP32 probability semantics."""

    def __init__(self, bundle, market, device="cpu"):
        if market not in MARKETS or str(device) != "cpu":
            raise ValueError("Choose domestic/us and CPU; this calibration is CPU-FP32 only")
        root = unlinked_path(bundle)
        if not root.is_dir():
            raise ValueError("Bundle directory is missing")
        digest_path = unlinked_path(root / "manifest.sha256")
        if not digest_path.is_file() or digest_path.stat().st_size > 128:
            raise ValueError("Missing or oversized manifest seal")
        digest = digest_path.read_text(encoding="ascii").strip()
        manifest = read_json(checked_file(root, "manifest.json", digest))
        validate_flags(manifest.get("risk_flags"))
        validate_source(manifest.get("source_run"))
        if (manifest.get("owner") != OWNER or manifest.get("title") != TITLE
                or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
                or manifest.get("completed") is not True or manifest.get("semantics") != SEMANTICS
                or manifest.get("risk_flags") != FLAGS or manifest.get("feature_names") != list(FEATURE_NAMES)
                or manifest.get("warnings") != WARNINGS or manifest.get("runtime_code_sha256") != runtime_code_hashes()
                or not isinstance(manifest.get("markets"), dict) or set(manifest["markets"]) != set(MARKETS)):
            raise ValueError("Bundle semantics/code/risk contract mismatch")
        expected = {"manifest.json", "manifest.sha256", "export-validation.json"}
        verified = {}
        for item_market in MARKETS:
            item = manifest["markets"][item_market]
            if (not isinstance(item, dict) or item.get("market") != item_market or item.get("fold") != "walk_2024"
                    or item.get("architecture") not in MODEL_NAMES or type(item.get("research_qualified")) is not bool
                    or item.get("risk_flags") != FLAGS):
                raise ValueError("Market/model contract mismatch")
            validate_config(item.get("model_config"))
            validate_calibration(item.get("calibration"))
            validate_flags(item.get("risk_flags"))
            members = item.get("members")
            if not isinstance(members, list) or len(members) != 3:
                raise ValueError("Exactly three ensemble members required")
            verified[item_market] = []
            for seed, member in zip(SEEDS, members):
                relative = f"{item_market}/seed{seed}.pt"
                if not isinstance(member, dict) or type(member.get("seed")) is not int or member["seed"] != seed or member.get("path") != relative:
                    raise ValueError("Member seed/path mismatch")
                original = f"{item_market}/walk_2024/{item['architecture']}-{seed}/model.pt"
                source_digest = manifest["source_run"]["source_files_sha256"].get(original)
                if not isinstance(source_digest, str) or member.get("source_checkpoint_sha256") != source_digest:
                    raise ValueError("Member source provenance mismatch")
                verified[item_market].append(checked_file(root, relative, member.get("sha256")))
                expected.add(relative)
        validation = read_json(checked_file(root, "export-validation.json", manifest.get("validation_sha256")))
        if (validation.get("completed") is not True or type(validation.get("cases")) is not int
                or validation["cases"] != 24 or validation.get("scope") != "synthetic_cpu_equivalence_not_profitability"):
            raise ValueError("Missing synthetic export equivalence evidence")
        actual = set()
        for path in root.rglob("*"):
            unlinked_path(path)
            relative = path.relative_to(root).as_posix()
            if path.is_file():
                actual.add(relative)
            elif relative not in MARKETS:
                raise ValueError("Unexpected bundle directory")
        if actual != expected:
            raise ValueError("Unexpected or missing bundle files")
        item = manifest["markets"][market]
        self._models = []
        for seed, path in zip(SEEDS, verified[market]):
            payload = load_weights(path)
            validate_flags(payload.get("risk_flags"))
            if (set(payload) != {"state_dict", "architecture", "model_config", "market", "seed", "target", "risk_flags"}
                    or payload.get("architecture") != item["architecture"] or payload.get("model_config") != item["model_config"]
                    or payload.get("market") != market or type(payload.get("seed")) is not int or payload["seed"] != seed
                    or payload.get("target") != TARGET or payload.get("risk_flags") != FLAGS):
                raise ValueError("Checkpoint identity/risk mismatch")
            model = build_model(item["architecture"], **item["model_config"]).cpu().eval()
            model.load_state_dict(payload["state_dict"], strict=True)
            self._models.append(model)
        self._calibration = copy.deepcopy(item["calibration"])
        self._metadata = {"title": TITLE, "market": market, "architecture": item["architecture"],
                          "model_config": copy.deepcopy(item["model_config"]), "bundle_manifest_sha256": digest,
                          "source_run": copy.deepcopy(manifest.get("source_run")),
                          "research_qualified": item["research_qualified"], "semantics": copy.deepcopy(SEMANTICS),
                          "warnings": list(WARNINGS), **FLAGS}

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    @torch.inference_mode()
    def predict_proba(self, history, entries):
        bars, prices = np.asarray(history), np.asarray(entries)
        if bars.ndim == 2:
            bars = bars[None]
        if (bars.ndim != 3 or bars.shape[1:] != (30, 5) or not len(bars)
                or bars.dtype.kind not in "iuf" or prices.dtype.kind not in "iuf"):
            raise ValueError("Numeric [N,30,5] completed history and candidate prices required")
        if prices.ndim == 0:
            prices = np.full(len(bars), prices.item(), dtype=np.float64)
        if prices.shape != (len(bars),) or not np.isfinite(bars).all() or not np.isfinite(prices).all():
            raise ValueError("Prices must be scalar or aligned finite [N]")
        if (np.any(prices <= 0) or np.any(bars[..., :4] <= 0) or np.any(bars[..., 4] < 0)
                or np.any(bars[..., 1] < bars[..., :4].max(axis=-1))
                or np.any(bars[..., 2] > bars[..., :4].min(axis=-1))):
            raise ValueError("Invalid positive OHLC/candidate prices or volume")
        with np.errstate(over="ignore", under="ignore"):
            bars, prices = bars.astype(np.float32), prices.astype(np.float32)
        if not np.isfinite(bars).all() or not np.isfinite(prices).all() or np.any(prices <= 0) or np.any(bars[..., :4] <= 0):
            raise ValueError("Inputs must remain finite positive prices in FP32")
        with torch.autocast("cpu", enabled=False):
            features = features_from_history(torch.from_numpy(bars), torch.from_numpy(prices), validate=True)
            logits = np.mean([success_logit(model(features)).numpy() for model in self._models], axis=0)
        probabilities = calibrated_probability(logits, self._calibration)
        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite ensemble probability")
        return np.asarray(probabilities, dtype=np.float64)
