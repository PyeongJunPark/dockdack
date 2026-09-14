"""Back up the collected SQLite databases and build verified ZIP release assets.

Only database snapshots are packaged; credentials and runtime logs are excluded.
Run from the repository root with Python 3.11 or newer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path


CHUNK_BYTES = 8 * 1024 * 1024
RELEASE_ASSET_LIMIT = 2 * 1024**3


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(connection: sqlite3.Connection) -> dict:
    check = [row[0] for row in connection.execute("PRAGMA quick_check")]
    if check != ["ok"]:
        raise ValueError(f"SQLite quick_check failed: {check}")
    instruments = connection.execute("SELECT COUNT(*) FROM instruments").fetchone()[0]
    statuses = dict(connection.execute(
        "SELECT status, COUNT(*) FROM collection_progress GROUP BY status"
    ))
    bars = connection.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]
    date_range = connection.execute(
        "SELECT (SELECT MIN(trade_date) FROM daily_bars), "
        "(SELECT MAX(trade_date) FROM daily_bars)"
    ).fetchone()
    with_bars = connection.execute(
        "SELECT COUNT(*) FROM (SELECT symbol, exchange FROM daily_bars "
        "GROUP BY symbol, exchange)"
    ).fetchone()[0]
    last_update = connection.execute(
        "SELECT MAX(updated_at) FROM collection_progress"
    ).fetchone()[0]
    errors = [
        dict(zip(("symbol", "exchange", "error", "updated_at"), row))
        for row in connection.execute(
            "SELECT symbol, exchange, error, updated_at FROM collection_progress "
            "WHERE status = 'error' ORDER BY symbol, exchange"
        )
    ]
    return {
        "instruments": instruments,
        "instruments_with_bars": with_bars,
        "progress_statuses": statuses,
        "daily_bars": bars,
        "earliest_trade_date": date_range[0],
        "latest_trade_date": date_range[1],
        "last_progress_update": last_update,
        "sqlite_quick_check": "ok",
        "errors": errors,
    }


def package_market(market: str, source_dir: Path, output_dir: Path, tag: str) -> dict:
    source = source_dir / f"{market}_daily.sqlite3"
    archive = output_dir / f"{market}_daily.{tag}.zip"
    if not source.is_file():
        raise FileNotFoundError(source)
    if archive.exists():
        raise FileExistsError(archive)
    with tempfile.TemporaryDirectory(prefix=f"{market}-snapshot-", dir=output_dir) as tmp:
        snapshot = Path(tmp) / source.name
        print(f"{market}: SQLite backup starting", flush=True)
        with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as src:
            with sqlite3.connect(snapshot) as dst:
                src.backup(dst, pages=4096)
                dst.execute("PRAGMA journal_mode=DELETE")
                print(f"{market}: validating snapshot and counting rows", flush=True)
                stats = summarize(dst)
            dst.close()
        src.close()
        snapshot_hash = hashlib.sha256()
        copied = 0
        total = snapshot.stat().st_size
        print(f"{market}: compressing {total:,} bytes", flush=True)
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as zipped:
            with snapshot.open("rb") as inp, zipped.open(source.name, "w", force_zip64=True) as out:
                while block := inp.read(CHUNK_BYTES):
                    snapshot_hash.update(block)
                    out.write(block)
                    copied += len(block)
                    if copied % (512 * 1024**2) == 0:
                        print(f"{market}: compressed input {copied / total:.0%}", flush=True)
        if archive.stat().st_size >= RELEASE_ASSET_LIMIT:
            raise ValueError(f"Release asset exceeds 2 GiB: {archive}")
        print(f"{market}: verifying decompressed ZIP checksum", flush=True)
        with zipfile.ZipFile(archive) as zipped:
            with zipped.open(source.name) as restored:
                restored_hash = hashlib.file_digest(restored, "sha256").hexdigest()
        if restored_hash != snapshot_hash.hexdigest():
            raise ValueError(f"ZIP round-trip checksum mismatch: {archive}")
        result = {
            "market": market,
            "database_filename": source.name,
            "database_bytes": total,
            "database_sha256": restored_hash,
            "asset_filename": archive.name,
            "asset_bytes": archive.stat().st_size,
            "asset_sha256": digest_file(archive),
            "download_url": f"https://github.com/PyeongJunPark/dockdack/releases/download/{tag}/{archive.name}",
            **stats,
        }
        print(f"{market}: verified {archive.name} ({archive.stat().st_size:,} bytes)", flush=True)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/kiwoom_daily"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in args.tag):
        parser.error("tag may contain only letters, digits, dots, underscores and hyphens")
    # Exclusive directory creation prevents overwriting existing artifacts.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "snapshot_tag": args.tag,
        "snapshot_created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "Kiwoom REST API (demo)",
        "release_url": f"https://github.com/PyeongJunPark/dockdack/releases/tag/{args.tag}",
        "coverage_note": "Per-symbol API history collected at different times; not a synchronized latest-day refresh. Status complete may include zero-row responses.",
        "markets": [],
    }
    for market in ("domestic", "us"):
        manifest["markets"].append(package_market(market, args.source_dir, args.output_dir, args.tag))
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums = [f"{m['asset_sha256']}  {m['asset_filename']}" for m in manifest["markets"]]
    sums.append(f"{digest_file(manifest_path)}  manifest.json")
    (args.output_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"Snapshot ready: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
