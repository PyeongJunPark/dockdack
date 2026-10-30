"""Supported compatibility gateway for frozen research, never a training CLI.

Historical modules retain their byte seals. Only path construction and
checksum-map lookup views are adapted in this single-purpose process; source
contracts, cache keys, weights, features, logits and portfolio math are intact.
"""
from __future__ import annotations

from contextlib import contextmanager
import ast
import importlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sys

from dockdack.research_artifacts import (ArtifactResolver, cache_key, read_json,
                                       sha256_file, verify_portable_bundle, write_new_json)
from dockdack.research_compat import load_frozen_cache_compatible


RUNNERS = {
    "backtest-deep": "examples.backtest_mark1_deep",
    "backtest-selective": "examples.backtest_mark1_selective",
    "backtest-half": "examples.backtest_mark1_0504",
    "export-prototype": "examples.export_mark1_prototype",
    "export-half": "examples.export_mark1_0504",
}
COMPAT_MODULES = (
    "examples.train_mark1", "examples.train_mark1_deep", "examples.train_mark1_selective",
    "examples.train_mark1_0504", "examples.backtest_mark1", "examples.backtest_mark1_deep",
    "examples.backtest_mark1_selective", "examples.backtest_mark1_0504",
    "examples.export_mark1_prototype", "examples.export_mark1_0504",
    "dockdack.mark1_selective_inference", "dockdack.mark1_prototype_inference",
    "dockdack.mark1_0504_inference", "dockdack.mark1_selective_models",
    "dockdack.mark1_backtest_data",
)


def _absolute_identity(value: str) -> bool:
    return PureWindowsPath(value).is_absolute() or Path(value).is_absolute()


def project_hash_maps(value, resolver: ArtifactResolver, receipts: list):
    """Bind whole checksum dictionaries by exact bytes, never change source strings."""
    if isinstance(value, list):
        return [project_hash_maps(item, resolver, receipts) for item in value]
    if not isinstance(value, dict):
        return value
    if value and all(isinstance(key, str) and _absolute_identity(key) and isinstance(digest, str)
                     and re.fullmatch(r"[0-9a-f]{64}", digest) for key, digest in value.items()):
        result = {}
        for logical, digest in value.items():
            artifact = resolver.file(logical, digest)
            physical = str(artifact.physical_path)
            if physical in result and result[physical] != digest:
                raise ValueError("Conflicting logical artifacts mapped to the same physical file")
            result[physical] = digest
            receipts.append(artifact.receipt())
        return result
    return {key: project_hash_maps(item, resolver, receipts) for key, item in value.items()}


def cache_receipts(workspace: Path) -> dict[str, str]:
    """Use recorded export cache checksums, not an invented current checksum."""
    resolver = ArtifactResolver(workspace)
    result = {}
    for name in ("mark1_prototype", "mark1_0504"):
        path = Path(workspace) / "models" / name / "export-validation.json"
        receipt = read_json(path)
        if (receipt.get("passed") is not True or receipt.get("protected_files_unchanged") is not True
                or receipt.get("protected_sha256_before") != receipt.get("protected_sha256_after")):
            raise ValueError("Missing matching completed export receipt")
        for logical, digest in receipt["protected_sha256_before"].items():
            if not logical.lower().endswith(".npz"):
                continue
            physical = str(resolver.location(logical))
            if physical in result and result[physical] != digest:
                raise ValueError("Conflicting frozen cache receipts")
            result[physical] = digest
    if not result:
        raise ValueError("No recorded frozen cache checksums")
    return result


@contextmanager
def compatibility_context(workspace: Path, *, expected_caches: dict[str, str] | None = None,
                          mmap_root: Path | None = None):
    """Single-thread/process-only adapter; all imported globals are restored.

    Do not use this context inside the GUI or a concurrent inference process.
    The CLI runs in its own process and exposes no broker or fitting command.
    """
    resolver = ArtifactResolver(workspace)
    receipts = []
    compatibility_sources = {
        str(Path(__file__).with_name(name)): sha256_file(Path(__file__).with_name(name))
        for name in ("research_tools.py", "research_compat.py", "research_artifacts.py", "research_arrays.py")
    }
    expected_caches = cache_receipts(workspace) if expected_caches is None else expected_caches
    mmap_root = resolver.location(str(mmap_root)) if mmap_root is not None else None
    if mmap_root is not None and not mmap_root.is_dir():
        raise ValueError("--mmap-root must be an existing completed-store parent")
    saved = []

    class PathMeta(type):
        def __instancecheck__(cls, value):
            return isinstance(value, Path)

        def __getattr__(cls, name):
            return getattr(Path, name)

    class CompatiblePath(metaclass=PathMeta):
        def __new__(cls, *parts):
            path = Path(*parts)
            if _absolute_identity(str(path)):
                # Every absolute research path must belong to a declared root.
                path = resolver.location(str(path))
            else:
                # Validate relative CLI/input paths too, but preserve relativity
                # for the frozen portable-bundle path-safety predicates.
                resolver.location(str(path))
            return path

    class CompatibleJson:
        def __getattr__(self, name):
            return getattr(json, name)

        def loads(self, *args, **kwargs):
            return project_hash_maps(json.loads(*args, **kwargs), resolver, receipts)

    def compatible_cache(source, market, directory):
        directory = CompatiblePath(directory)
        path = str((directory / f"{market}-{cache_key(source)}.npz").resolve())
        expected = expected_caches.get(path)
        if expected is None:
            raise ValueError("No recorded immutable cache receipt for this requested cache")
        mmap_directory = (mmap_root / f"{market}-{cache_key(source)}") if mmap_root is not None else None
        dataset, actual, receipt = load_frozen_cache_compatible(source, market, directory,
            resolver=resolver, expected_cache_sha256=expected, mmap_directory=mmap_directory)
        if actual != source:
            raise ValueError("Logical frozen cache identity changed")
        receipts.append({**receipt, "mmap_directory": str(mmap_directory) if mmap_directory is not None else None})
        return dataset

    old_cwd, old_import_path = Path.cwd(), list(sys.path)
    try:
        # The runtime wheel intentionally omits historical examples. Their
        # explicitly selected source checkout is a temporary import location.
        sys.path.insert(0, str(resolver.workspace))
        modules = []
        for name in COMPAT_MODULES:
            relative = Path(*name.split(".")).with_suffix(".py")
            expected_source = resolver.workspace / relative
            if not expected_source.is_file() and name.startswith("dockdack."):
                expected_source = resolver.workspace / "src" / relative
            if not expected_source.is_file() or expected_source.is_symlink():
                raise FileNotFoundError(f"Required research source checkout file missing: {relative}")
            module = importlib.import_module(name)
            imported_source = Path(module.__file__).resolve()
            if name.startswith("examples.") and imported_source != expected_source.resolve():
                raise ValueError("A different examples checkout is already imported; use a fresh CLI process")
            if sha256_file(imported_source) != sha256_file(expected_source):
                raise ValueError(f"Installed/source research implementation differs: {name}")
            modules.append(module)
        for module in modules:
            for name, replacement in (("Path", CompatiblePath), ("json", CompatibleJson()),
                                      ("load_frozen_cache", compatible_cache)):
                if hasattr(module, name):
                    saved.append((module, name, getattr(module, name)))
                    setattr(module, name, replacement)
        os.chdir(resolver.workspace)
        yield {"resolver": resolver, "receipts": receipts, "modules": {module.__name__: module for module in modules},
               "compatibility_code_sha256": compatibility_sources}
    finally:
        # CWD can disappear while a long isolated research task runs. That
        # failure must not prevent restoring the import path or module globals.
        try:
            os.chdir(old_cwd)
        finally:
            try:
                sys.path[:] = old_import_path
            finally:
                for module, name, original in reversed(saved):
                    setattr(module, name, original)


def _exact_long_options(arguments: list[str], command: str, resolver: ArtifactResolver) -> None:
    """Do not let argparse abbreviations evade wrapper-level path validation."""
    source = resolver.workspace.joinpath(*RUNNERS[command].split(".")).with_suffix(".py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    allowed = {"--help"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            allowed.update(argument.value for argument in node.args
                           if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                           and argument.value.startswith("--"))
    for argument in arguments:
        if argument.startswith("--") and argument.split("=", 1)[0] not in allowed:
            raise ValueError("Use an exact documented long option; option abbreviations are not accepted")


def _fresh_output(arguments: list[str], command: str, resolver: ArtifactResolver) -> Path:
    option = "--output" if command.startswith("export-") else "--output-dir"
    values = []
    for index, argument in enumerate(arguments):
        name = argument.split("=", 1)[0]
        if name.startswith("--") and name != option and option.startswith(name):
            raise ValueError("Output option abbreviations are not accepted")
        if argument == option and index + 1 < len(arguments):
            values.append(arguments[index + 1])
        elif argument.startswith(option + "="):
            values.append(argument.split("=", 1)[1])
    if len(values) != 1:
        raise ValueError(f"Specify exactly one NEW {option}; historical defaults are never used")
    output = resolver.location(values[0])
    if output.exists() or output.is_symlink():
        raise FileExistsError("Historical outputs/bundles cannot be overwritten")
    return output


def run_legacy(command: str, arguments: list[str], *, workspace: Path,
               mmap_root: Path | None = None) -> int:
    if command not in RUNNERS:
        raise ValueError("Only the documented frozen backtest/export runners are allowed")
    resolver = ArtifactResolver(workspace)
    _exact_long_options(arguments, command, resolver)
    help_only = arguments == ["--help"]
    output = None if help_only else _fresh_output(arguments, command, resolver)
    # No receipt/file validation is needed for pure parser help.
    with compatibility_context(workspace, expected_caches={} if help_only else None,
                               mmap_root=None if help_only else mmap_root) as state:
        module = state["modules"][RUNNERS[command]]
        previous_argv = sys.argv
        try:
            sys.argv = [RUNNERS[command], *arguments]
            result = module.main() if command.startswith("export-") else module.main(arguments)
        finally:
            sys.argv = previous_argv
        if output is not None:
            if any(sha256_file(Path(path)) != digest for path, digest in state["compatibility_code_sha256"].items()):
                raise RuntimeError("Compatibility implementation changed during the research run")
            write_new_json(output / "relocation-compatibility.json", {
                "format": "dockdack-research-runner-v1", "runner": RUNNERS[command],
                "original_files_rewritten": False, "model_math_changed": False,
                "training_started": False, "orders_started": False,
                "compatibility_code_sha256": state["compatibility_code_sha256"],
                "logical_artifact_receipts": state["receipts"]})
        return result or 0


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--mmap-root", type=Path, help="Optional completed mmap parent containing market-cachekey directories")
    parser.add_argument("command", choices=(*RUNNERS, "preflight", "cache-info", "report-half", "bundle-check", "pack-cache"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if args.command in RUNNERS:
        return run_legacy(args.command, arguments, workspace=args.workspace, mmap_root=args.mmap_root)
    if args.mmap_root is not None:
        parser.error("--mmap-root applies only to frozen backtest/export runner commands")
    if args.command in {"preflight", "cache-info", "report-half"}:
        from dockdack.research_compat import main as compat_main
        return compat_main(["--workspace", str(args.workspace), args.command, *arguments])
    options = argparse.ArgumentParser(prog=args.command)
    if args.command == "bundle-check":
        options.add_argument("--bundle", type=Path, required=True)
        parsed = options.parse_args(arguments)
        result = verify_portable_bundle(parsed.bundle, args.workspace)
    else:
        from dockdack.research_arrays import unpack_frozen_cache
        options.add_argument("--source", type=Path, required=True)
        options.add_argument("--cache-dir", type=Path, required=True)
        options.add_argument("--cache-sha256", required=True)
        options.add_argument("--destination", type=Path, required=True)
        parsed = options.parse_args(arguments)
        source = read_json(parsed.source)
        result = unpack_frozen_cache(source, source["market"], parsed.cache_dir, parsed.destination,
            resolver=ArtifactResolver(args.workspace), expected_cache_sha256=parsed.cache_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
