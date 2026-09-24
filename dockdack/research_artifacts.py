"""New research I/O core; logical frozen identities never become physical paths.

This module does not import a broker, fit models or alter frozen contracts.
Every relocation is explicit and content checked. Hashes are deliberately not
cached by mtime: SQLite WAL and changes during a read remain fail-closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import sqlite3
from typing import Mapping


HISTORICAL_ROOTS = (
    "C:/Users/user/Desktop/dockdack-data-collection",
    "C:/Users/user/Desktop/dockdack-mark_1",
    "C:/Users/user/Desktop/dockdack",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError("Expected a finite JSON object")
    return value


def write_new_json(path: Path, value: object) -> None:
    """Exclusive creation only; callers publish their complete directory later."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


@dataclass(frozen=True)
class ResolvedArtifact:
    logical_path: str
    physical_path: Path
    sha256: str

    def receipt(self) -> dict:
        return {"logical_path": self.logical_path, "physical_path": str(self.physical_path),
                "sha256": self.sha256, "logical_identity_unchanged": True}


class ArtifactResolver:
    def __init__(self, workspace: Path, *, relocations: Mapping[str, Path] | None = None) -> None:
        self.workspace = Path(workspace).resolve(strict=True)
        self.relocations = dict(relocations if relocations is not None else
                                {name: self.workspace for name in HISTORICAL_ROOTS})

    def location(self, logical_path: str) -> Path:
        """Map only named historical roots, or a file already under workspace."""
        logical = PureWindowsPath(logical_path)
        if ".." in logical.parts or (logical.drive and not logical.is_absolute()):
            raise ValueError("Parent traversal or drive-relative paths are not artifact identities")
        actual = Path(logical_path)
        if actual.is_absolute() and actual.resolve().is_relative_to(self.workspace):
            root, path = self.workspace, actual
        elif not logical.is_absolute() and not actual.is_absolute():
            root, path = self.workspace, self.workspace.joinpath(*logical.parts)
        else:
            matches = []
            for old, new in self.relocations.items():
                try:
                    relative = logical.relative_to(PureWindowsPath(old))
                except ValueError:
                    continue
                matches.append((Path(new).resolve(), Path(new).joinpath(*relative.parts)))
            if len(matches) != 1:
                raise ValueError("Artifact path has no unambiguous explicit relocation")
            root, path = matches[0]
        absolute = path.absolute()
        resolved = absolute.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Resolved artifact escaped its approved root")
        for item in (absolute, *absolute.parents):
            if item == root:
                break
            if item.is_symlink():
                raise ValueError("Linked artifact paths are not accepted")
        return resolved

    def file(self, logical_path: str, expected_sha256: str) -> ResolvedArtifact:
        if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise ValueError("An exact lowercase SHA-256 is required")
        path = self.location(logical_path)
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"Missing or changed artifact: {logical_path}")
        return ResolvedArtifact(logical_path, path, expected_sha256)

    def database(self, contract: Mapping[str, object], market: str) -> ResolvedArtifact:
        if market not in {"domestic", "us"} or contract.get("market") != market or contract.get("version") != 2:
            raise ValueError("Invalid frozen market/source contract")
        logical, digest = contract.get("database_path"), contract.get("database_sha256")
        if not isinstance(logical, str) or not isinstance(digest, str):
            raise ValueError("Frozen source identity is missing")
        path = self.location(logical)
        assert_no_wal(path)
        artifact = self.file(logical, digest)
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            metadata = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM metadata")}
            if metadata.get("market") != market or metadata.get("build_status") != "complete":
                raise ValueError("Relocated database is not a completed cleaned source for this market")
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"daily_bars", "training_samples", "sessions", "instruments"} <= tables:
                raise ValueError("Relocated database has the wrong cleaned schema")
        assert_no_wal(path)
        return artifact

    def recheck(self, artifact: ResolvedArtifact, *, database: bool = False) -> None:
        if database:
            assert_no_wal(artifact.physical_path)
        if sha256_file(artifact.physical_path) != artifact.sha256:
            raise RuntimeError("Artifact changed during the compatibility operation")
        if database:
            assert_no_wal(artifact.physical_path)


def assert_no_wal(database: Path) -> None:
    wal = Path(str(database) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Frozen source has a nonempty SQLite WAL")


def cache_key(contract: Mapping[str, object]) -> str:
    """Exact historical key algorithm, including the unchanged logical path."""
    return hashlib.sha256(json.dumps(dict(contract), sort_keys=True).encode()).hexdigest()[:16]


def verify_protocol_sources(protocol: Path, source_root: Path) -> dict[str, str]:
    """Verify all recorded code, accepting the older unambiguous basename schema."""
    hashes = read_json(protocol).get("code_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing frozen source hashes")
    resolver = ArtifactResolver(source_root, relocations={})
    result = {}
    for name, digest in hashes.items():
        if not isinstance(name, str):
            raise ValueError("Invalid frozen source name")
        relative = name if "/" in name else ("examples/" if name.startswith(("train_", "backtest_")) else "dockdack/") + name
        artifact = resolver.file(relative, digest)
        result[relative] = artifact.sha256
    return result


def verify_portable_bundle(bundle: Path, source_root: Path) -> dict:
    """Checksum-only shared audit: no model libraries, inference or fitting."""
    bundle = Path(bundle).resolve(strict=True)
    resolver = ArtifactResolver(bundle, relocations={})
    manifest_path = bundle / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    if (bundle / "manifest.sha256").read_text(encoding="ascii").strip() != manifest_hash:
        raise ValueError("Bundle manifest seal mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("completed") is not True or set(manifest.get("markets", {})) != {"domestic", "us"}:
        raise ValueError("Incomplete portable bundle")
    files = {"manifest.json": manifest_hash}
    for market, reference in manifest["markets"].items():
        artifact = resolver.file(reference["path"], reference["sha256"])
        files[reference["path"]] = artifact.sha256
        market_manifest = read_json(artifact.physical_path)
        if market_manifest.get("market") != market or not market_manifest.get("members"):
            raise ValueError("Bundle market identity or members are missing")
        for member in market_manifest["members"]:
            for path_key, hash_key in (("path", "sha256"), ("sidecar_path", "sidecar_sha256")):
                member_file = resolver.file(member[path_key], member[hash_key])
                files[member[path_key]] = member_file.sha256
    code = manifest.get("runtime_code_sha256")
    if not isinstance(code, dict) or not code:
        raise ValueError("Missing frozen runtime source seals")
    source_resolver = ArtifactResolver(source_root, relocations={})
    runtime_names = {"calibration": "mark1_metrics.py", "features": "mark1_selective_features.py",
                     "native_backend": "mark1_selective_models.py", "barrier_features": "mark1_0504_data.py",
                     "base_features": "mark1_selective_features.py", "inference": "mark1_0504_inference.py"}
    for name, digest in code.items():
        if name not in runtime_names:
            raise ValueError("Unknown runtime seal identity")
        source_resolver.file("dockdack/" + runtime_names[name], digest)
    alias_result = None
    if (bundle / "alias.json").exists():
        alias_digest = (bundle / "alias.sha256").read_text(encoding="ascii").strip()
        alias = read_json(resolver.file("alias.json", alias_digest).physical_path)
        contract = alias.get("contract", {})
        if contract.get("identity_only_alias") is not True or contract.get("risk_flags") != manifest.get("risk_flags"):
            raise ValueError("Alias changes the original research contract")
        source_resolver.file("dockdack/mark1_1_prototype_inference.py", contract.get("wrapper_sha256"))
        source_files = alias.get("source_files_sha256")
        if not isinstance(source_files, dict) or source_files.get("manifest.json") != manifest_hash:
            raise ValueError("Alias source inventory mismatch")
        for name, digest in source_files.items():
            resolver.file(name, digest)
        validation = read_json(resolver.file("alias-validation.json", alias.get("validation_sha256")).physical_path)
        if (validation.get("passed") is not True or validation.get("source_files_unchanged") is not True
                or validation.get("identity_only_alias") is not True):
            raise ValueError("Alias completion receipt mismatch")
        actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
        if actual != set(source_files) | {"alias.json", "alias.sha256", "alias-validation.json"}:
            raise ValueError("Alias bundle inventory differs from the sealed record")
        alias_result = {"verified": True, "sha256": alias_digest, "contract": contract}
    return {"bundle": str(bundle), "verified": True, "files": files,
            "runtime_code_sha256": code, "risk_flags": manifest.get("risk_flags"),
            "identity_alias": alias_result, "no_model_inference": True, "no_orders": True}
