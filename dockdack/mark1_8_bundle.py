"""Hash-checked research checkpoint contract for Mark1.8; no order interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dockdack.mark1_4_evolution import normalize_windows
from dockdack.mark1_8_allocation import AllocationNet
from dockdack.research_artifacts import read_json, sha256_file, write_new_json


_TRACKED_MANIFEST_SHA256 = "3bb277dd466d98ca381084410f7ac22be7fbc51966a88fa7ff71dbe438e3d41f"
_TRACKED_EXPECTED = {"domestic": (42, 0.1103103756904602),
                     "us": (41, 0.07466766238212585)}


@dataclass(frozen=True)
class Mark18Bundle:
    market: str
    model: AllocationNet | None
    threshold: float | None
    metadata: dict

    def predict(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return rank scores and equity fractions, never order instructions."""
        if self.model is None:
            raise RuntimeError("2021 calibration selected cash; no active model")
        features = normalize_windows(windows).reshape(-1, 150)
        scores, sizes = [], []
        with torch.inference_mode():
            for start in range(0, len(features), 4096):
                batch = torch.as_tensor(features[start:start + 4096])
                score, size = self.model(batch)
                scores.append(score.numpy())
                sizes.append(size.numpy())
        if not scores:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        return np.concatenate(scores), np.concatenate(sizes)

    def select(self, windows: np.ndarray, symbol_ids: np.ndarray) -> list[dict]:
        """Rank one pre-open cross-section; no broker/order object is created."""
        if self.model is None or self.threshold is None:
            return []
        identifiers = np.asarray(symbol_ids)
        if (identifiers.ndim != 1 or len(identifiers) != len(windows) or
                len(identifiers) > 100 or len(set(identifiers.tolist())) != len(identifiers)):
            raise ValueError("Expected unique <=100-symbol cross-section")
        scores, sizes = self.predict(windows)
        eligible = np.flatnonzero(scores > self.threshold)
        # Deterministic numerical symbol-id tie-break.
        order = np.lexsort((identifiers[eligible], -scores[eligible]))
        return [{"symbol": str(identifiers[row]),
                 "rank_score": float(scores[row]),
                 "equity_fraction": float(sizes[row])}
                for row in eligible[order[:10]]]


def write_bundle_manifest(directory: Path) -> Path:
    """Finalize an already completed report using immutable SHA-256 references."""
    root = Path(directory).resolve(strict=True)
    report_file = root / "report.json"
    report = read_json(report_file)
    if (report.get("version") != "Mark1.8" or report.get("market") not in
            {"domestic", "us"} or report.get("research_only") is not True or
            report.get("deployment_allowed") is not False):
        raise ValueError("Not a completed Mark1.8 research report")
    winner = report["calibration"]["winner"]
    file_hashes = {"report.json": sha256_file(report_file)}
    for seed in (41, 42, 43):
        for extension in ("model.pt", "scores.npz"):
            name = f"seed{seed}-{extension}"
            file_hashes[name] = sha256_file(root / name)
    manifest = {
        "format": "dockdack.mark1_8.research.v1", "market": report["market"],
        "research_only": True, "deployment_allowed": False,
        "lookback": 30, "channels": ["open", "high", "low", "close", "volume"],
        "input": "30 completed t-bars normalized within each window",
        "target": "hypothetical t+1 open to same-day close, net of roundtrip costs",
        "score": "uncalibrated rank score in [0,1], not a probability",
        "size": "fraction of starting session equity in [0,0.1]",
        "max_positions": 10, "roundtrip_cost_bps": 20.0,
        "selected_seed": winner["seed"] if winner else None,
        "frozen_score_threshold": winner["threshold"] if winner else None,
        "cash_only": winner is None, "files_sha256": file_hashes,
    }
    output = root / "manifest.json"
    write_new_json(output, manifest)
    return output


def load_bundle(directory: Path, *, market: str) -> Mark18Bundle:
    """Reject tampered, mismatched or partially written checkpoint folders."""
    supplied = Path(directory).absolute()
    if supplied.is_symlink():
        raise ValueError("Mark1.8 bundle path must not be linked")
    root = supplied.resolve(strict=True)
    if (root / "manifest.sha256").is_file():
        return _load_tracked_bundle(root, market=market)
    manifest = read_json(root / "manifest.json")
    if (manifest.get("format") != "dockdack.mark1_8.research.v1" or
            manifest.get("market") != market or market not in {"domestic", "us"} or
            manifest.get("research_only") is not True or
            manifest.get("deployment_allowed") is not False or
            manifest.get("lookback") != 30 or
            manifest.get("channels") != ["open", "high", "low", "close", "volume"] or
            manifest.get("max_positions") != 10):
        raise ValueError("Mark1.8 bundle contract mismatch")
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or set(files) != {
            "report.json", *(f"seed{seed}-{ext}" for seed in (41, 42, 43)
                             for ext in ("model.pt", "scores.npz"))}:
        raise ValueError("Incomplete Mark1.8 file manifest")
    for name, digest in files.items():
        if (not isinstance(digest, str) or len(digest) != 64 or
                sha256_file(root / name) != digest):
            raise ValueError(f"Changed Mark1.8 artifact: {name}")
    report = read_json(root / "report.json")
    winner = report["calibration"]["winner"]
    seed = manifest["selected_seed"]
    threshold = manifest["frozen_score_threshold"]
    if winner is None:
        if seed is not None or threshold is not None or manifest.get("cash_only") is not True:
            raise ValueError("Cash-only metadata mismatch")
        return Mark18Bundle(market, None, None, manifest)
    if (seed not in (41, 42, 43) or seed != winner["seed"] or
            not isinstance(threshold, (int, float)) or
            not np.isfinite(threshold) or threshold != winner["threshold"] or
            manifest.get("cash_only") is not False):
        raise ValueError("Frozen selection metadata mismatch")
    payload = torch.load(root / f"seed{seed}-model.pt", map_location="cpu",
                         weights_only=True)
    if (not isinstance(payload, dict) or payload.get("seed") != seed or
            payload.get("research_only") is not True or
            payload.get("deployment_allowed") is not False):
        raise ValueError("Checkpoint metadata mismatch")
    model = AllocationNet()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return Mark18Bundle(market, model, float(threshold), manifest)


def _load_tracked_bundle(root: Path, *, market: str) -> Mark18Bundle:
    """Load the compact two-market, SHA-pinned tracked research bundle."""
    if market not in _TRACKED_EXPECTED or root.is_symlink() or any(
            item.is_symlink() for item in root.iterdir()):
        raise ValueError("Invalid tracked Mark1.8 market or linked artifact")
    expected_files = {"manifest.json", "manifest.sha256"}
    for name, (seed, _) in _TRACKED_EXPECTED.items():
        expected_files.update((f"{name}-seed{seed}-model.pt", f"{name}-audit.npz"))
    if {item.name for item in root.iterdir()} != expected_files:
        raise ValueError("Tracked Mark1.8 file set mismatch")
    seal = (root / "manifest.sha256").read_text(encoding="ascii").strip()
    if (seal != _TRACKED_MANIFEST_SHA256 or
            sha256_file(root / "manifest.json") != seal):
        raise ValueError("Tracked Mark1.8 manifest seal mismatch")
    manifest = read_json(root / "manifest.json")
    if (manifest.get("schema_version") != 1 or
            manifest.get("title") != "mark1.8" or
            manifest.get("research_only") is not True or
            manifest.get("deployment_allowed") is not False or
            manifest.get("lookback") != 30 or
            manifest.get("bar_columns") != ["open", "high", "low", "close", "volume"] or
            manifest.get("entry") != "next_session_open" or
            manifest.get("exit") != "same_session_close" or
            manifest.get("cost_bps") != 20.0 or
            manifest.get("max_positions") != 10 or
            manifest.get("max_fraction_per_position") != .1 or
            manifest.get("threshold_comparison") != "strict_greater_than" or
            set(manifest.get("markets", {})) != set(_TRACKED_EXPECTED)):
        raise ValueError("Tracked Mark1.8 contract mismatch")
    for name, (seed, threshold) in _TRACKED_EXPECTED.items():
        entry = manifest["markets"][name]
        model_name = f"{name}-seed{seed}-model.pt"
        audit_name = f"{name}-audit.npz"
        symbols = [(item.get("symbol"), item.get("exchange")) for item in
                   entry.get("selected_symbols", []) if isinstance(item, dict)]
        if (entry.get("seed") != seed or entry.get("threshold") != threshold or
                entry.get("model_file") != model_name or
                entry.get("audit_file") != audit_name or
                entry.get("research_only") is not True or
                entry.get("deployment_allowed") is not False or
                entry.get("cash_outperformed_selected_in_2021") is not True or
                entry.get("catalog_point_in_time") is not False or
                len(symbols) != 100 or len(set(symbols)) != 100 or
                sha256_file(root / model_name) != entry.get("model_sha256") or
                sha256_file(root / audit_name) != entry.get("audit_sha256")):
            raise ValueError("Tracked Mark1.8 frozen market policy mismatch")
    seed, threshold = _TRACKED_EXPECTED[market]
    payload = torch.load(root / f"{market}-seed{seed}-model.pt",
                         map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("seed") != seed or
            payload.get("research_only") is not True or
            payload.get("deployment_allowed") is not False):
        raise ValueError("Tracked Mark1.8 checkpoint metadata mismatch")
    model = AllocationNet()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    metadata = {"title": "mark1.8", "version": "1.8", "market": market,
                "method": "direct risk/turnover-penalized daily allocation",
                "seed": seed, "score_metric": "uncalibrated_rank_score",
                "score_unit": "arbitrary_rank_score",
                "allocation_unit": "fraction_of_start_day_equity",
                "frozen_numeric_score_threshold": threshold,
                "threshold_comparison": "strict_greater_than",
                "lookback": 30,
                "target": "hypothetical next-session open to same-session close",
                "cost_bps": 20.0, "max_positions": 10,
                "selected_universe_size": 100,
                "selected_symbols": manifest["markets"][market]["selected_symbols"],
                "catalog_point_in_time": False,
                "cash_outperformed_selected_in_2021": True,
                "research_only": True, "research_qualified": False,
                "deployment_allowed": False,
                "bundle_manifest_sha256": seal,
                "source_report_sha256": manifest["markets"][market]["source_report_sha256"]}
    bundle = Mark18Bundle(market, model, threshold, metadata)
    with np.load(root / f"{market}-audit.npz", allow_pickle=False) as audit:
        windows = np.asarray(audit["windows"], dtype=np.float32)
        scores = np.asarray(audit["scores"], dtype=np.float32)
        sizes = np.asarray(audit["sizes"], dtype=np.float32)
    if (windows.ndim != 3 or windows.shape[1:] != (30, 5) or
            scores.shape != (len(windows),) or sizes.shape != (len(windows),) or
            not np.isfinite(scores).all() or not np.isfinite(sizes).all()):
        raise ValueError("Tracked Mark1.8 audit sample malformed")
    observed_score, observed_size = bundle.predict(windows)
    if (not np.allclose(observed_score, scores, atol=1e-6, rtol=1e-6) or
            not np.allclose(observed_size, sizes, atol=1e-6, rtol=1e-6)):
        raise ValueError("Tracked Mark1.8 checkpoint failed score/size audit")
    return bundle
