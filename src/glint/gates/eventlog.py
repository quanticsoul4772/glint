"""A hash-chained, append-only event log in SQLite with one writer.

Written for this event; the design follows the author's earlier agent-harness-v2 (private), and no code was copied
from it. The one writer serializes emits with an RLock and writes each event inside BEGIN IMMEDIATE; each row's hash
is sha256 over the previous hash and the row's fields with null-byte separators, starting from "genesis"; triggers
make UPDATE and DELETE abort. One log file per
run; `verify()` walks the chain by rowid and names the first row that does not match.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

GENESIS_HASH = "genesis"
SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    ts INTEGER NOT NULL,
    type TEXT NOT NULL,
    actor TEXT NOT NULL,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash_chain TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events table is append-only; updates are forbidden'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events table is append-only; deletes are forbidden'); END;
"""


class ChainBroken(Exception):
    pass


def chain_hash(prev_hash: str, event_id: str, ts: int, event_type: str, actor: str, payload_text: str) -> str:
    h = hashlib.sha256()
    for i, part in enumerate((prev_hash, event_id, str(ts), event_type, actor, payload_text)):
        if i:
            h.update(b"\x00")
        h.update(part.encode("utf-8"))
    return h.hexdigest()


class EventLog:
    """The single writer of one log file. Safe to share across threads (τ² runs simulations concurrently)."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.executescript(SCHEMA)

    def emit(self, event_type: str, payload: dict, actor: str = "glint.gates") -> str:
        """Append one event; returns its chain hash. The whole read-tail-then-insert runs in one BEGIN IMMEDIATE."""
        payload_text = json.dumps(payload, sort_keys=True, default=str)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT hash_chain FROM events ORDER BY rowid DESC LIMIT 1").fetchone()
                prev = row[0] if row else GENESIS_HASH
                event_id, ts = uuid.uuid4().hex, time.time_ns() // 1000
                digest = chain_hash(prev, event_id, ts, event_type, actor, payload_text)
                self._conn.execute("INSERT INTO events (id, ts, type, actor, payload, prev_hash, hash_chain) "
                                   "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                   (event_id, ts, event_type, actor, payload_text, prev, digest))
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return digest

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT count(*) FROM events").fetchone()[0]

    def events(self, event_type: str | None = None, after: int = 0) -> list[dict]:
        """The events in order, skipping the first `after` (rowids are 1, 2, ...: the log is append-only)."""
        with self._lock:
            rows = self._conn.execute("SELECT type, actor, ts, payload FROM events WHERE rowid > ? ORDER BY rowid",
                                      (after,)).fetchall()
        return [{"type": t, "actor": a, "ts": ts, "payload": json.loads(p)} for t, a, ts, p in rows
                if event_type is None or t == event_type]

    def verify(self) -> int:
        """Recompute the chain from genesis; raise ChainBroken naming the first bad row. Returns the event count."""
        with self._lock:
            rows = self._conn.execute("SELECT rowid, id, ts, type, actor, payload, prev_hash, hash_chain FROM events "
                                      "ORDER BY rowid").fetchall()
        prev = GENESIS_HASH
        for rowid, event_id, ts, event_type, actor, payload, stored_prev, stored in rows:
            if stored_prev != prev:
                raise ChainBroken(f"row {rowid}: prev_hash does not link to the row before it")
            if chain_hash(prev, event_id, ts, event_type, actor, payload) != stored:
                raise ChainBroken(f"row {rowid}: hash does not match its contents")
            prev = stored
        return len(rows)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
