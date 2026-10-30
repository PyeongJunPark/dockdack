"""Explicit, offline ledger maintenance. No broker object or order access."""
from contextlib import closing, contextmanager
import argparse
import json
from pathlib import Path
import sqlite3


def _store(path, *, writable=False):
    """No schema migrations or environment relabeling in a maintenance dry run."""
    path = Path(path).resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        if db.execute("SELECT mode,scope FROM app_environment WHERE singleton=1").fetchone() is None:
            raise ValueError("계정 범위가 있는 장부가 아닙니다.")
    class ExistingLedger:
        @contextmanager
        def connection(self):
            with closing(sqlite3.connect(path.as_uri() + ("?mode=rw" if writable else "?mode=ro"), uri=True)) as db:
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA foreign_keys=ON")
                with db:
                    yield db
    return ExistingLedger()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    migrate = commands.add_parser("migrate-demo", help="Copy a confirmed legacy paper-account ledger; never overwrite")
    migrate.add_argument("--source", required=True, type=Path)
    migrate.add_argument("--destination", required=True, type=Path)
    migrate.add_argument("--scope", required=True)
    migrate.add_argument("--confirm", required=True)
    retain = commands.add_parser("archive-monitor", help="Dry run by default; lossless monitor text archive only")
    retain.add_argument("--db", required=True, type=Path)
    retain.add_argument("--days", type=int, default=90)
    retain.add_argument("--limit", type=int, default=2000)
    retain.add_argument("--apply", action="store_true")
    read = commands.add_parser("read-archive", help="Print a verified archived monitor-text batch")
    read.add_argument("--db", required=True, type=Path)
    read.add_argument("--id", required=True, type=int)
    args = parser.parse_args(argv)
    if args.command == "migrate-demo":
        from dockdack.persistence.account_migration import migrate_legacy_demo
        result = migrate_legacy_demo(args.source, args.destination, args.scope, confirmation=args.confirm)
        payload = {"copied_to": str(result.path), "source_preserved": True}
    else:
        from dockdack.persistence.retention import archive_monitor_events, read_monitor_archive
        from dockdack.lstm30_runtime import SessionLock
        # A running GUI owns this same lock. No hot retention while orders run.
        lock = SessionLock(args.db.resolve(strict=True).parent / "session.lock")
        lock.acquire()
        try:
            store = _store(args.db, writable=args.command == "archive-monitor" and args.apply)
            payload = (archive_monitor_events(store, days=args.days, limit=args.limit, apply=args.apply)
                       if args.command == "archive-monitor" else read_monitor_archive(store, args.id))
        finally:
            lock.release()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
