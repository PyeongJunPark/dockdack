"""Save a new mark1.1 prototype alias without changing the trained 0.5/0.4 bundle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dockdack.mark1_0504_inference import HalfPercentPredictor, RISK_FLAGS, sha256_file
from dockdack.mark1_1_prototype_inference import (
    BUNDLE_VERSION, DEFAULT_BUNDLE, TITLE, STRATEGY_ID, Mark11PrototypePredictor,
    alias_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def snapshot(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Source bundle must not contain linked paths")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    if not result:
        raise ValueError("Source bundle is empty")
    return result


def representative_queries():
    """Four deterministic, label-free histories with three virtual entry prices.

    Includes a float32 historical-barrier boundary. These are numerical checks,
    not new backtest samples or evidence of profitable trading.
    """
    day = np.arange(30, dtype=np.float64)
    centers = [np.full(30, 100.), 100. + day * .25, 120. - day * .3]
    histories = [np.column_stack((center, center * 1.008, center * .993,
                                 center * 1.001, 100000. + day * 200)) for center in centers]
    histories.append(np.tile([84700., 84700., 84361.2, 84700., 1000.], (30, 1)))
    for index, history in enumerate(histories):
        for factor in (.9975, 1., 1.0025):
            yield index, factor, history, float(history[-1, 3]) * factor


def _same_prediction(source, aliased):
    identity_fields = {
        "title", "version", "bundle_version", "bundle_manifest_sha256", "strategy_id",
        "source_title", "source_model_version", "source_bundle_manifest_sha256", "identity_only_alias",
    }
    source_values = {key: value for key, value in source.items() if key not in identity_fields}
    alias_values = {key: value for key, value in aliased.items() if key not in identity_fields}
    if source_values != alias_values:
        raise ValueError("Alias changed model probabilities, barriers, or risk flags")


def export_alias(source, destination):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Source bundle must be an existing, unlinked directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace existing bundle: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError("Destination parent must be an existing, unlinked directory")
    source, destination = source.resolve(), destination.resolve()
    if source == destination or destination.is_relative_to(source):
        raise ValueError("Alias destination must be independent of the source")
    before = snapshot(source)
    if any(name in before for name in ("alias.json", "alias.sha256", "alias-validation.json")):
        raise ValueError("Source must be the original 0.5/0.4 bundle, not an alias")
    originals = {market: HalfPercentPredictor(source, market) for market in ("domestic", "us")}
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-export-", dir=destination.parent)).resolve()
    if stage.parent != destination.parent or not stage.name.startswith(f".{destination.name}-export-"):
        raise ValueError("Export staging path escaped the intended parent")
    # Generated model bundles are copied artifacts; original source files stay byte-identical.
    for relative in before:
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    if snapshot(stage) != before:
        raise ValueError("Copied bundle differs from the source")
    rows = []
    for market, original in originals.items():
        copied = HalfPercentPredictor(stage, market)
        for index, factor, history, price in representative_queries():
            untouched = history.copy()
            expected, actual = original.predict(history, price), copied.predict(history, price)
            _same_prediction(expected, actual)
            if not np.array_equal(history, untouched):
                raise ValueError("Prediction modified query input")
            rows.append({"market": market, "history": index, "candidate_factor": factor,
                         "probability_success": actual["probability_success"],
                         "probability_stop": actual["probability_stop"], "absolute_error": 0.})
    if snapshot(source) != before:
        raise ValueError("Source bundle changed during alias export")
    write_json(stage / "alias-validation.json", {
        "passed": True, "identity_only_alias": True, "source_files_unchanged": True,
        "source_bundle_manifest_sha256": before["manifest.json"], "risk_flags": RISK_FLAGS,
        "prediction_comparisons": len(rows), "rows": rows,
        "scope": "CPU numerical equality on four deterministic synthetic histories and three prices per market; not a profitability or deployment test",
    })
    write_json(stage / "alias.json", {
        "contract": alias_contract(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bundle_manifest_sha256": before["manifest.json"], "source_files_sha256": before,
        "validation_sha256": sha256_file(stage / "alias-validation.json"),
        "exporter_sha256": sha256_file(Path(__file__)),
    })
    (stage / "alias.sha256").write_text(sha256_file(stage / "alias.json") + "\n", encoding="ascii")
    for market, original in originals.items():
        aliased = Mark11PrototypePredictor(stage, market)
        for _, _, history, price in representative_queries():
            result = aliased.predict(history, price)
            _same_prediction(original.predict(history, price), result)
            if result["title"] != TITLE or result["strategy_id"] != STRATEGY_ID:
                raise ValueError("Alias model attribution failed")
    if snapshot(source) != before:
        raise ValueError("Source bundle changed during final alias verification")
    if stage.parent != destination.parent or destination.exists() or destination.is_symlink():
        raise ValueError("Destination changed during alias verification")
    stage.rename(destination)
    return {"title": TITLE, "strategy_id": STRATEGY_ID, "version": BUNDLE_VERSION,
            "bundle": str(destination), "prediction_comparisons": len(rows),
            "bundle_manifest_sha256": sha256_file(destination / "alias.json"),
            "source_files_unchanged": True, **RISK_FLAGS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "models/mark1_0504")
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    print(json.dumps(export_alias(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
