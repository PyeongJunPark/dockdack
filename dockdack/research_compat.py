"""Read-only compatibility entrypoints for relocated frozen research artifacts.

No sealed JSON/source file is rewritten. This is not an automatic adapter for
every historical training/export CLI, and it never starts fitting or orders.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import zipfile

from dockdack.research_artifacts import (ArtifactResolver, cache_key, read_json,
                                       sha256_file, verify_protocol_sources, write_new_json)


def inspect_frozen_cache(source_contract: dict, market: str, cache_dir: Path, *,
                         resolver: ArtifactResolver, expected_cache_sha256: str) -> dict:
    """Validate relocated DB/schema and original NPZ without loading large arrays."""
    import numpy as np
    from dockdack.research_arrays import MEMBERS
    source = resolver.database(source_contract, market)
    path = Path(cache_dir) / f"{market}-{cache_key(source_contract)}.npz"
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_cache_sha256:
        raise ValueError("Frozen NPZ must match an explicit content receipt")
    shapes = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(MEMBERS) or set(names) != MEMBERS:
            raise ValueError("Invalid frozen NPZ members")
        for name in names:
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                reader = { (1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0 }.get(version)
                if reader is None:
                    raise ValueError("Unsupported NPY header version")
                shape, fortran, dtype = reader(stream)
                if dtype.hasobject:
                    raise ValueError("Object cache arrays are forbidden")
                shapes[name[:-4]] = {"shape": list(shape), "dtype": str(dtype), "fortran_order": fortran}
        if archive.getinfo("cache_config.npy").file_size > 1024 * 1024:
            raise ValueError("Oversized source contract")
        with archive.open("cache_config.npy") as stream:
            actual_contract = json.loads(str(np.load(stream, allow_pickle=False).item()))
        if actual_contract != source_contract:
            raise ValueError("Cache logical contract differs from frozen source")
    resolver.recheck(source, database=True)
    if sha256_file(path) != expected_cache_sha256:
        raise RuntimeError("Cache changed during validation")
    return {"source": source.receipt(), "source_contract": source_contract,
            "cache_path": str(path.resolve()), "cache_sha256": expected_cache_sha256,
            "cache_key": cache_key(source_contract), "arrays": shapes,
            "logical_identity_unchanged": True, "read_only": True}


def load_frozen_cache_compatible(source_contract: dict, market: str, cache_dir: Path, *,
                                 resolver: ArtifactResolver, expected_cache_sha256: str,
                                 mmap_directory: Path | None = None):
    """New explicit API; returns historical dataset type and unchanged contract.

    Without mmap_directory the historical NPZ still materializes full arrays.
    Passing a separately completed mmap store avoids that full-bank allocation.
    """
    import numpy as np
    from dockdack.mark1_data import Mark1Dataset
    from dockdack.research_arrays import ARRAY_NAMES, load_mmap_cache
    receipt = inspect_frozen_cache(source_contract, market, cache_dir, resolver=resolver,
                                   expected_cache_sha256=expected_cache_sha256)
    if mmap_directory is not None:
        manifest = read_json(Path(mmap_directory) / "completed.json")
        if manifest.get("source_cache_sha256") != expected_cache_sha256:
            raise ValueError("Mmap store came from a different frozen NPZ")
        mapped = load_mmap_cache(mmap_directory, source_contract)
        dataset = Mark1Dataset(**vars(mapped))
    else:
        with np.load(receipt["cache_path"], allow_pickle=False) as cached:
            dataset = Mark1Dataset(**{name: cached[name] for name in ARRAY_NAMES[:5]},
                splits={name[6:]: cached[name] for name in ARRAY_NAMES[5:]},
                manifest=json.loads(str(cached["manifest"].item())))
        if sha256_file(Path(receipt["cache_path"])) != expected_cache_sha256:
            raise RuntimeError("Cache changed while loading")
    source = resolver.file(source_contract["database_path"], source_contract["database_sha256"])
    resolver.recheck(source, database=True)
    return dataset, dict(source_contract), receipt


def bind_recorded_artifact(hashes: dict, target: Path, *, resolver: ArtifactResolver) -> tuple[dict, dict]:
    """Add a physical lookup alias only after the recorded logical file verifies.

    The original logical keys remain in this in-memory compatibility view.
    Neither this mapping nor the historical artifact is persisted over input.
    """
    if not isinstance(hashes, dict):
        raise ValueError("Missing artifact provenance map")
    target = Path(target).resolve()
    digest = sha256_file(target)
    physical_key = str(target)
    if physical_key in hashes and hashes[physical_key] != digest:
        raise ValueError("Conflicting physical provenance entry")
    candidates = []
    for logical, expected in hashes.items():
        if not isinstance(logical, str) or expected != digest:
            continue
        try:
            physical = resolver.location(logical)
        except ValueError:
            continue
        if physical == target:
            candidates.append(resolver.file(logical, expected))
    if not candidates:
        raise ValueError("No verified logical provenance entry for this physical artifact")
    result = dict(hashes)
    result[physical_key] = digest
    return result, {"physical_alias": physical_key, "verified_original_entries": [item.receipt() for item in candidates]}


def half_report_inputs(training: Path, backtest: Path, bundle: Path, *, resolver: ArtifactResolver):
    """Run EVERY original half-report check with one verified path-key alias.

    A private module instance avoids process-wide monkeypatching. Only its JSON
    reader projects old summary keys to current keys in memory; values and all
    original checksum/target/deployment/completion checks remain unchanged.
    """
    source_path = resolver.workspace / "examples/report_mark1_0504.py"
    spec = importlib.util.spec_from_file_location("_dockdack_half_report_compat", source_path)
    if spec is None or spec.loader is None:
        raise ValueError("Historical report source is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_reader = module.read_json
    backtest = Path(backtest).resolve()
    projected_paths = {backtest / "summary.json", *(backtest / market / "summary.json" for market in ("domestic", "us"))}
    aliases = []

    def compatibility_reader(path):
        value = original_reader(path)
        if Path(path).resolve() in projected_paths:
            rows = value.values() if set(value) == {"domestic", "us"} else [value]
            for row in rows:
                projected, receipt = bind_recorded_artifact(row.get("artifact_sha256"),
                    Path(training) / "summary.json", resolver=resolver)
                row["artifact_sha256"] = projected
                aliases.append(receipt)
        return value

    module.read_json = compatibility_reader
    data = module.load_inputs(training, backtest, bundle)
    data["compatibility"] = {"format": "dockdack-relocation-v1", "historical_inputs_unchanged": True,
        "physical_aliases": aliases, "original_reader_sha256": sha256_file(source_path),
        "compatibility_reader_sha256": sha256_file(Path(__file__)), "no_model_inference": True}
    return module, data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Canonical source/artifact workspace")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Verify frozen source seals; no training/inference")
    preflight.add_argument("--protocol", type=Path, action="append", required=True)
    cache = commands.add_parser("cache-info", help="Read-only source/NPZ metadata validation")
    cache.add_argument("--source", type=Path, required=True)
    cache.add_argument("--cache-dir", type=Path, required=True)
    cache.add_argument("--cache-sha256", required=True, help="Previously recorded complete NPZ checksum")
    report = commands.add_parser("report-half", help="Verify historical 0.5/0.4 report inputs; optionally render NEW output")
    report.add_argument("--training", type=Path, required=True)
    report.add_argument("--backtest", type=Path, required=True)
    report.add_argument("--bundle", type=Path, required=True)
    report.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    resolver = ArtifactResolver(args.workspace)
    if args.command == "preflight":
        result = {str(path): verify_protocol_sources(path, resolver.workspace) for path in args.protocol}
    elif args.command == "cache-info":
        source = read_json(args.source)
        result = inspect_frozen_cache(source, source["market"], args.cache_dir,
                                      resolver=resolver, expected_cache_sha256=args.cache_sha256)
    else:
        reader, data = half_report_inputs(args.training, args.backtest, args.bundle, resolver=resolver)
        result = {"verified": True, "compatibility": data["compatibility"], "orders_started": False}
        if args.output is not None:
            from dockdack.research_arrays import _new_stage
            output, stage = _new_stage(args.output)
            reader.comparison_figure(data, stage / "comparison.png")
            with (stage / "REPORT.md").open("x", encoding="utf-8") as stream:
                stream.write(reader.report_text(data))
            if any(sha256_file(Path(path)) != digest for path, digest in data["input_sha256"].items()):
                raise RuntimeError("Historical report inputs changed during rendering")
            write_new_json(stage / "report-inputs.json", {**result, "input_sha256": data["input_sha256"]})
            stage.rename(output)
            result["output"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
