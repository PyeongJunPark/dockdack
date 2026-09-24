"""Fenced, resumable staging for a complete adjusted-price history.

Only ``publish`` changes the reader-facing daily_bars table. Partial/error
generations and the exact previous published rows are retained for inspection.
The broker has no snapshot/version API: the caller must revalidate its first
page on resume and immediately before publication.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from typing import Callable, Sequence


class GenerationConflict(RuntimeError):
    """A concurrent writer or source revision makes publication unsafe."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class DailyGenerationStore:
    def __init__(self, connection: sqlite3.Connection, *,
                 clock: Callable[[], float] = time.time, lease_seconds: float = 300) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.db, self.clock, self.lease_seconds = connection, clock, lease_seconds
        self.owner = uuid.uuid4().hex
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS daily_generations (
                id TEXT PRIMARY KEY, symbol TEXT NOT NULL, exchange TEXT NOT NULL,
                status TEXT NOT NULL, active INTEGER NOT NULL,
                owner TEXT, lease_until REAL NOT NULL DEFAULT 0,
                base_sha256 TEXT NOT NULL, anchor TEXT, request_json TEXT,
                next_key TEXT, pages INTEGER NOT NULL DEFAULT 0,
                final_page INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL, error TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_active_generation
                ON daily_generations(symbol, exchange) WHERE active=1;
            CREATE TABLE IF NOT EXISTS daily_generation_bars (
                generation_id TEXT NOT NULL, trade_date TEXT NOT NULL,
                row_json TEXT NOT NULL, PRIMARY KEY(generation_id, trade_date)
            );
            CREATE TABLE IF NOT EXISTS daily_generation_tokens (
                generation_id TEXT NOT NULL, token TEXT NOT NULL,
                PRIMARY KEY(generation_id, token)
            );
            CREATE TABLE IF NOT EXISTS daily_generation_previous (
                replacement_id TEXT NOT NULL, trade_date TEXT NOT NULL,
                row_json TEXT NOT NULL, PRIMARY KEY(replacement_id, trade_date)
            );
        """)
        self.db.commit()

    def pending(self, symbol: str, exchange: str) -> bool:
        return self.db.execute("SELECT 1 FROM daily_generations WHERE symbol=? AND exchange=? AND active=1",
                               (symbol, exchange)).fetchone() is not None

    def _fingerprint(self, symbol: str, exchange: str) -> str:
        digest = hashlib.sha256()
        for row in self.db.execute("SELECT * FROM daily_bars WHERE symbol=? AND exchange=? ORDER BY trade_date",
                                   (symbol, exchange)):
            digest.update((canonical_json(tuple(row)) + "\n").encode("utf-8"))
        return digest.hexdigest()

    def get(self, generation_id: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM daily_generations WHERE id=?", (generation_id,)).fetchone()
        if row is None:
            raise GenerationConflict("Unknown collection generation")
        return row

    def claim(self, symbol: str, exchange: str) -> sqlite3.Row:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            now = self.clock()
            row = self.db.execute("SELECT * FROM daily_generations WHERE symbol=? AND exchange=? AND active=1",
                                  (symbol, exchange)).fetchone()
            if row is not None and row["owner"] and row["lease_until"] > now:
                raise GenerationConflict("Another collector owns this instrument")
            if row is not None and row["base_sha256"] != self._fingerprint(symbol, exchange):
                self.db.execute("UPDATE daily_generations SET active=0,status='conflicted',owner=NULL WHERE id=?",
                                (row["id"],))
                row = None
            if row is None:
                generation_id = uuid.uuid4().hex
                self.db.execute("""INSERT INTO daily_generations
                    (id,symbol,exchange,status,active,owner,lease_until,base_sha256,updated_at)
                    VALUES (?,?,?,'running',1,?,?,?,?)""",
                    (generation_id, symbol, exchange, self.owner, now + self.lease_seconds,
                     self._fingerprint(symbol, exchange), now))
            else:
                generation_id = row["id"]
                self.db.execute("""UPDATE daily_generations SET status='running',owner=?,lease_until=?,
                                updated_at=?,error=NULL WHERE id=?""",
                                (self.owner, now + self.lease_seconds, now, generation_id))
            self.db.commit()
            return self.get(generation_id)
        except BaseException:
            self.db.rollback()
            raise

    def _owned(self, generation_id: str) -> sqlite3.Row:
        row = self.get(generation_id)
        if not row["active"] or row["owner"] != self.owner or row["lease_until"] <= self.clock():
            raise GenerationConflict("Collection lease expired or was replaced; old writer is fenced")
        return row

    def renew(self, generation_id: str) -> None:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self._owned(generation_id)
            self.db.execute("UPDATE daily_generations SET lease_until=?,updated_at=? WHERE id=?",
                            (self.clock() + self.lease_seconds, self.clock(), generation_id))

    def release(self, generation_id: str, *, error: str | None = None,
                superseded: bool = False) -> None:
        # A fenced old worker must never release the new owner's lease.
        with self.db:
            self.db.execute("""UPDATE daily_generations SET owner=NULL,lease_until=0,status=?,
                active=?,updated_at=?,error=? WHERE id=? AND owner=?""",
                ("superseded" if superseded else "error" if error else "partial",
                 0 if superseded else 1, self.clock(), (error or "")[:2000] or None,
                 generation_id, self.owner))

    def append(self, generation_id: str, rows: Sequence[tuple], *, anchor: str | None,
               request_json: str, next_key: str | None) -> None:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            generation = self._owned(generation_id)
            if generation["final_page"]:
                raise GenerationConflict("Cannot append after the final page")
            if generation["pages"] == 0 and anchor is None:
                raise ValueError("First page must have an anchor")
            if next_key:
                try:
                    self.db.execute("INSERT INTO daily_generation_tokens VALUES (?,?)", (generation_id, next_key))
                except sqlite3.IntegrityError as exc:
                    raise GenerationConflict("Broker repeated a continuation key") from exc
            prior_oldest = self.db.execute("SELECT MIN(trade_date) FROM daily_generation_bars WHERE generation_id=?",
                                          (generation_id,)).fetchone()[0]
            for row in rows:
                if len(row) != 15 or tuple(row[:2]) != (generation["symbol"], generation["exchange"]):
                    raise ValueError("Staged bar identity/schema mismatch")
                existing = self.db.execute("SELECT row_json FROM daily_generation_bars WHERE generation_id=? AND trade_date=?",
                                           (generation_id, row[2])).fetchone()
                if existing:
                    if json.loads(existing[0])[:-1] != list(row[:-1]):
                        raise GenerationConflict("Conflicting bars within one adjusted-history generation")
                else:
                    if prior_oldest is not None and row[2] >= prior_oldest:
                        raise GenerationConflict("Continuation page is not strictly older history")
                    self.db.execute("INSERT INTO daily_generation_bars VALUES (?,?,?)",
                                    (generation_id, row[2], canonical_json(row)))
            self.db.execute("""UPDATE daily_generations SET anchor=COALESCE(anchor,?),
                request_json=COALESCE(request_json,?),next_key=?,pages=pages+1,final_page=?,
                updated_at=?,lease_until=? WHERE id=?""",
                (anchor, request_json, next_key, int(next_key is None), self.clock(),
                 self.clock() + self.lease_seconds, generation_id))

    def publish(self, generation_id: str, *, verified_anchor: str, timestamp: str) -> None:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            generation = self._owned(generation_id)
            if not generation["final_page"] or verified_anchor != generation["anchor"]:
                raise GenerationConflict("Source changed or traversal is incomplete; publication refused")
            identity = (generation["symbol"], generation["exchange"])
            if self._fingerprint(*identity) != generation["base_sha256"]:
                raise GenerationConflict("Published history was modified by another writer")
            count, earliest, latest = self.db.execute("""SELECT COUNT(*),MIN(trade_date),MAX(trade_date)
                FROM daily_generation_bars WHERE generation_id=?""", (generation_id,)).fetchone()
            old_rows = self.db.execute("SELECT * FROM daily_bars WHERE symbol=? AND exchange=?", identity).fetchall()
            if old_rows and not count:
                raise GenerationConflict("An empty response cannot erase existing published history")
            self.db.executemany("INSERT INTO daily_generation_previous VALUES (?,?,?)",
                                ((generation_id, row[2], canonical_json(tuple(row))) for row in old_rows))
            self.db.execute("DELETE FROM daily_bars WHERE symbol=? AND exchange=?", identity)
            self.db.executemany("INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (json.loads(row[0]) for row in self.db.execute(
                    "SELECT row_json FROM daily_generation_bars WHERE generation_id=? ORDER BY trade_date", (generation_id,))))
            self.db.execute("""INSERT INTO collection_progress
                (symbol,exchange,status,pages_fetched,rows_seen,earliest_date,latest_date,updated_at)
                VALUES (?,?,'complete',?,?,?,?,?) ON CONFLICT(symbol,exchange) DO UPDATE SET
                status='complete',pages_fetched=excluded.pages_fetched,rows_seen=excluded.rows_seen,
                earliest_date=excluded.earliest_date,latest_date=excluded.latest_date,
                cont_yn=NULL,next_key=NULL,error=NULL,updated_at=excluded.updated_at""",
                (*identity, generation["pages"], count, earliest, latest, timestamp))
            self.db.execute("""UPDATE daily_generations SET active=0,status='published',owner=NULL,
                lease_until=0,updated_at=? WHERE id=?""", (self.clock(), generation_id))
