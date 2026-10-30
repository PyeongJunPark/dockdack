"""Recoverable retention for high-frequency monitoring text, not trading facts.

Orders, fills, signals/HOLD identities, chart memberships and replay tombstones
remain in the hot database. Only old `monitor` event text is losslessly archived.
"""
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json


DEFAULT_MONITOR_DAYS = 90


def archive_monitor_events(store, *, now=None, days=DEFAULT_MONITOR_DAYS, limit=2000, apply=False):
    if type(days) is not int or days < 1 or type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError("보존 기간/묶음 크기가 올바르지 않습니다.")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("보존 정책 기준 시각에 시간대가 필요합니다.")
    cutoff = now.astimezone(timezone.utc) - timedelta(days=days)
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE" if apply else "BEGIN")
        # Event IDs are durable; do not compare differently offset ISO strings.
        rows = []
        for record in db.execute("""SELECT e.id,e.time,e.symbol,e.message FROM event_category_index c
                                JOIN events e ON e.id=c.event_id WHERE c.category='monitor'
                                AND julianday(e.time)<julianday(?)
                                ORDER BY c.event_id LIMIT ?""", (cutoff.isoformat(), limit)):
            try:
                at = datetime.fromisoformat(record["time"])
                if at.tzinfo is not None and at < cutoff:
                    rows.append(dict(record))
            except (ValueError, TypeError):
                continue
        result = {"policy": "monitor-text-only", "days": days, "eligible": len(rows), "applied": False}
        if not apply or not rows:
            return result
        raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        blob = gzip.compress(raw, mtime=0)
        digest = hashlib.sha256(raw).hexdigest()
        if gzip.decompress(blob) != raw:
            raise ValueError("압축 검증 실패; 원문을 보존합니다.")
        db.execute("""CREATE TABLE IF NOT EXISTS monitoring_archives(
                       id INTEGER PRIMARY KEY,created_at TEXT NOT NULL,first_event INTEGER NOT NULL,
                       last_event INTEGER NOT NULL,event_count INTEGER NOT NULL,digest TEXT NOT NULL,payload BLOB NOT NULL)""")
        cursor = db.execute("INSERT INTO monitoring_archives VALUES(NULL,?,?,?,?,?,?)",
                            (now.isoformat(), rows[0]["id"], rows[-1]["id"], len(rows), digest, blob))
        ids = [(row["id"],) for row in rows]
        db.executemany("DELETE FROM event_categories WHERE event_id=?", ids)
        db.executemany("DELETE FROM event_category_index WHERE event_id=?", ids)
        db.executemany("DELETE FROM events WHERE id=?", ids)
        return {**result, "applied": True, "archive_id": cursor.lastrowid,
                "uncompressed_bytes": len(raw), "compressed_bytes": len(blob)}


def read_monitor_archive(store, archive_id):
    with store.connection() as db:
        row = db.execute("SELECT digest,payload FROM monitoring_archives WHERE id=?", (archive_id,)).fetchone()
    if row is None:
        raise ValueError("보관 묶음이 없습니다.")
    raw = gzip.decompress(row["payload"])
    if hashlib.sha256(raw).hexdigest() != row["digest"]:
        raise ValueError("보관 묶음 무결성 검사 실패")
    return tuple(json.loads(raw))
