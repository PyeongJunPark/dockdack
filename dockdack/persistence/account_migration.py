"""Explicit copy-only migration of a user-confirmed legacy DEMO ledger."""
from contextlib import closing
from pathlib import Path
import os
import re
import sqlite3
from uuid import uuid4

from dockdack.models import TradingMode
from dockdack.watchlist import WatchStore


CONFIRM_LEGACY_DEMO = "CONFIRM_LEGACY_DEMO_OWNERSHIP"


def migrate_legacy_demo(source, destination, scope, *, confirmation):
    """Never overwrite/delete the source or an existing target; retain pending orders.

    The user must establish that the historical ledger belongs to the selected
    paper account and reset generation. API credentials are never written here.
    """
    if confirmation != CONFIRM_LEGACY_DEMO:
        raise ValueError("기존 장부가 현재 모의계정/리셋 세대의 기록임을 먼저 확인하세요.")
    if not re.fullmatch(r"[0-9a-f]{64}", scope):
        raise ValueError("설정된 모의계정 범위가 필요합니다.")
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError("기존 파일을 덮어쓰지 않습니다. 새 계정 장부 경로가 필요합니다.")
    from dockdack.lstm30_runtime import SessionLock
    lock = SessionLock(source.parent / "session.lock")
    lock.acquire()
    temporary = None
    try:
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as old:
            tables = {row[0] for row in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "watchlist" not in tables or old.execute("PRAGMA user_version").fetchone()[0] not in (0, 1, 2, 3):
                raise ValueError("지원하는 기존 매매 장부가 아닙니다.")
            if "app_environment" in tables:
                columns = {row[1] for row in old.execute("PRAGMA table_info(app_environment)")}
                binding = old.execute("SELECT mode" + (",scope" if "scope" in columns else "")
                                      + " FROM app_environment WHERE singleton=1").fetchone()
                if not binding or binding[0] != "demo" or (len(binding) > 1 and binding[1] != "demo"):
                    raise ValueError("이미 계정에 귀속되었거나 실전인 장부는 레거시 모의 이관 대상이 아닙니다.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(".migration-" + uuid4().hex + ".sqlite3")
            with closing(sqlite3.connect(temporary)) as new:
                old.backup(new)
                if "app_environment" not in tables:
                    new.execute("CREATE TABLE app_environment(singleton INTEGER PRIMARY KEY,mode TEXT NOT NULL,scope TEXT NOT NULL)")
                    new.execute("INSERT INTO app_environment VALUES(1,'demo',?)", (scope,))
                else:
                    if "scope" not in columns:
                        new.execute("ALTER TABLE app_environment ADD COLUMN scope TEXT NOT NULL DEFAULT 'demo'")
                    new.execute("UPDATE app_environment SET scope=? WHERE singleton=1", (scope,))
                new.commit()
                if new.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("복사된 장부 무결성 검사 실패")
        # Atomic create-if-absent publication; unlike replace this cannot clobber
        # a ledger another process created after the initial existence check.
        os.link(temporary, destination)
        return WatchStore(destination, mode=TradingMode.DEMO, storage_scope=scope)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        lock.release()
