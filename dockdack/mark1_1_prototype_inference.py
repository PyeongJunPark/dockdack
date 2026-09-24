"""Named, independently checksummed alias for the immutable 0.5/0.4 model.

The wrapper changes identification only. It does not retrain, alter probabilities,
relax the underlying research flags, or grant authority to submit any order.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .mark1_0504_inference import (
    BUNDLE_VERSION as SOURCE_VERSION, RISK_FLAGS, SEMANTICS, TARGET,
    TITLE as SOURCE_TITLE, HalfPercentPredictor, _checked_file, _read, _same,
    sha256_file,
)


TITLE = "mark1.1 prototype"
STRATEGY_ID = "mark1-1-prototype"
BUNDLE_VERSION = "20260924-v1"
OWNER = "dockdack.mark1_1_prototype"
SCHEMA_VERSION = 1
DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "models/mark1_1_prototype"


def alias_contract():
    """The immutable semantic and identity contract sealed by the exporter."""
    return {
        "owner": OWNER, "schema_version": SCHEMA_VERSION, "title": TITLE,
        "strategy_id": STRATEGY_ID, "version": BUNDLE_VERSION,
        "source_title": SOURCE_TITLE, "source_model_version": SOURCE_VERSION,
        "target": TARGET, "semantics": copy.deepcopy(SEMANTICS),
        "risk_flags": copy.deepcopy(RISK_FLAGS),
        "wrapper_sha256": sha256_file(Path(__file__)),
        "identity_only_alias": True,
    }


class Mark11PrototypePredictor:
    """0.5/0.4 CPU inference with stable attribution to mark1.1 prototype."""

    buy_threshold = .5

    def __init__(self, bundle_root=DEFAULT_BUNDLE, market="domestic"):
        root = Path(bundle_root).absolute()
        if root.is_symlink() or not root.is_dir():
            raise ValueError("mark1.1 bundle must be an existing, unlinked directory")
        checksum = root / "alias.sha256"
        sha256_file(checksum)
        digest = checksum.read_text(encoding="ascii").strip()
        alias = _read(_checked_file(root, "alias.json", digest))
        if not _same(alias.get("contract"), alias_contract()):
            raise ValueError("mark1.1 identity, code or research contract mismatch")
        source_files = alias.get("source_files_sha256")
        if (not isinstance(source_files, dict) or not source_files
                or "manifest.json" not in source_files or "manifest.sha256" not in source_files
                or any(name in source_files for name in ("alias.json", "alias.sha256", "alias-validation.json"))):
            raise ValueError("Invalid mark1.1 original artifact inventory")
        for name, source_digest in source_files.items():
            _checked_file(root, name, source_digest)
        if alias.get("source_bundle_manifest_sha256") != source_files["manifest.json"]:
            raise ValueError("mark1.1 source manifest provenance mismatch")
        validation = _read(_checked_file(root, "alias-validation.json", alias.get("validation_sha256")))
        if (validation.get("passed") is not True or validation.get("identity_only_alias") is not True
                or validation.get("source_bundle_manifest_sha256") != source_files["manifest.json"]
                or not _same(validation.get("risk_flags"), RISK_FLAGS)
                or validation.get("source_files_unchanged") is not True):
            raise ValueError("mark1.1 numerical alias validation mismatch")
        expected = set(source_files) | {"alias.json", "alias.sha256", "alias-validation.json"}
        actual = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("Linked mark1.1 bundle paths are not supported")
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
        if actual != expected:
            raise ValueError("mark1.1 bundle inventory differs from its sealed alias")
        self._source = HalfPercentPredictor(root, market)
        self.market, self.model_name = self._source.market, self._source.model_name
        self._identity = {
            "title": TITLE, "strategy_id": STRATEGY_ID, "version": BUNDLE_VERSION,
            "bundle_version": BUNDLE_VERSION, "bundle_manifest_sha256": digest,
            "source_title": SOURCE_TITLE, "source_model_version": SOURCE_VERSION,
            "source_bundle_manifest_sha256": source_files["manifest.json"],
            "identity_only_alias": True,
        }
        self._metadata = {**self._source.metadata, **self._identity}

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def predict(self, bars, current_price=None):
        return {**self._source.predict(bars, current_price), **self._identity}
