"""A hash-chained, append-only event log in SQLite with one writer.

All code was written during the event; no earlier code was copied. The design follows the author's earlier private
projects (agent-harness-v2 and pgharness). The one writer serializes emits with an RLock and writes each event inside
BEGIN IMMEDIATE; each row's hash is computed over the previous hash and the row's fields with null-byte separators, starting from "genesis"; triggers
make UPDATE and DELETE abort. One log file per run; `verify()` walks the chain by rowid and names the first row that
does not match.

What the chain protects against depends on the key:
- **Unkeyed (sha256):** accidental damage and edits that do not recompute the chain. Anyone who can write the file can
  drop the triggers, edit a row and recompute every hash after it.
- **Keyed (HMAC-SHA256, GLINT_LOG_KEY):** a rewrite by anyone without the key. The key is in the gate server's
  environment, so keep it out of the agent's prompt and of any file the agent can read. The scheme is recorded in the file's meta table, so a keyed log is never verified as an unkeyed one.
- **Truncation:** cutting rows off the end leaves a valid shorter chain. `head()` returns (count, last hash); kept
  somewhere the agent cannot write, `verify(expected_head=...)` fails when the log no longer contains that head.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

GENESIS_HASH = "genesis"
KEY_ENV = "GLINT_LOG_KEY"
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
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS meta_no_update BEFORE UPDATE ON meta
BEGIN SELECT RAISE(ABORT, 'meta table is append-only; updates are forbidden'); END;
CREATE TRIGGER IF NOT EXISTS meta_no_delete BEFORE DELETE ON meta
BEGIN SELECT RAISE(ABORT, 'meta table is append-only; deletes are forbidden'); END;
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events table is append-only; updates are forbidden'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events table is append-only; deletes are forbidden'); END;
"""


class ChainBroken(Exception):
    pass


KEY_CHECK = b"glint-log-key-check"


def key_check(key: bytes) -> str:
    """A value that shows which key a keyed log was written with, without revealing the key."""
    return hmac.new(key, KEY_CHECK, hashlib.sha256).hexdigest()


def log_key() -> bytes | None:
    """The operator's chain key from GLINT_LOG_KEY, or None for an unkeyed log."""
    value = os.environ.get(KEY_ENV, "")
    return value.encode("utf-8") if value else None


def chain_hash(prev_hash: str, event_id: str, ts: int, event_type: str, actor: str, payload_text: str,
               key: bytes | None = None) -> str:
    h = hmac.new(key, digestmod=hashlib.sha256) if key else hashlib.sha256()
    for i, part in enumerate((prev_hash, event_id, str(ts), event_type, actor, payload_text)):
        if i:
            h.update(b"\x00")
        h.update(part.encode("utf-8"))
    return h.hexdigest()


class EventLog:
    """The single writer of one log file. Safe to share across threads (τ² runs simulations concurrently)."""

    def __init__(self, path: str | Path, key: bytes | None = None):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.key = key
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.executescript(SCHEMA)
        self.scheme = self._scheme()

    def _scheme(self) -> str:
        """The file's chain scheme, recorded on first open. A file from before the meta table, with events, is sha256."""
        wanted = "hmac-sha256" if self.key else "sha256"
        # one statement, so two processes opening a new file at once cannot both insert; whichever wins, both read it
        self._conn.execute("INSERT OR IGNORE INTO meta (k, v) SELECT 'chain', CASE WHEN EXISTS (SELECT 1 FROM events) "
                           "THEN 'sha256' ELSE ? END", (wanted,))
        recorded = self._conn.execute("SELECT v FROM meta WHERE k = 'chain'").fetchone()[0]
        if recorded == "hmac-sha256" and not self.key:
            raise ChainBroken(f"{self.path} is a keyed log: set {KEY_ENV} to open it")
        if recorded == "sha256" and self.key:
            # never advise dropping the key: a writer without the key can turn a keyed log into a valid unkeyed one
            raise ChainBroken(f"{self.path} records an unkeyed chain but {KEY_ENV} is set: if this log was written "
                              "with the key, the file was changed; do not verify it without the key")
        if self.key:
            self._check_key()
        return recorded

    def _check_key(self) -> None:
        """The key must be the one the log was written with: a different key would chain new events under it and
        leave the log unverifiable under either. A keyed file from before the check row is checked on its first row."""
        mine = key_check(self.key)
        row = self._conn.execute("SELECT v FROM meta WHERE k = 'key_check'").fetchone()
        if row is None:
            first = self._conn.execute("SELECT id, ts, type, actor, payload, hash_chain FROM events ORDER BY rowid "
                                       "LIMIT 1").fetchone()
            if first and not hmac.compare_digest(chain_hash(GENESIS_HASH, *first[:5], self.key), first[5]):
                raise ChainBroken(f"{self.path} was written with a different {KEY_ENV}")
            self._conn.execute("INSERT OR IGNORE INTO meta (k, v) VALUES ('key_check', ?)", (mine,))
            row = self._conn.execute("SELECT v FROM meta WHERE k = 'key_check'").fetchone()
        if not hmac.compare_digest(row[0], mine):
            raise ChainBroken(f"{self.path} was written with a different {KEY_ENV}")

    def emit(self, event_type: str, payload: dict, actor: str = "glint.gates") -> str:
        """Append one event; returns its chain hash. The whole read-tail-then-insert runs in one BEGIN IMMEDIATE."""
        payload_text = json.dumps(payload, sort_keys=True, default=str)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT hash_chain FROM events ORDER BY rowid DESC LIMIT 1").fetchone()
                prev = row[0] if row else GENESIS_HASH
                event_id, ts = uuid.uuid4().hex, time.time_ns() // 1000
                digest = chain_hash(prev, event_id, ts, event_type, actor, payload_text, self.key)
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

    def head(self) -> tuple[int, str]:
        """(event count, last chain hash): store it outside the agent's reach to detect truncation later."""
        with self._lock:
            row = self._conn.execute("SELECT count(*), (SELECT hash_chain FROM events ORDER BY rowid DESC LIMIT 1) "
                                     "FROM events").fetchone()
        return row[0], row[1] or GENESIS_HASH

    def events(self, event_type: str | None = None, after: int = 0) -> list[dict]:
        """The events in order, skipping the first `after` (rowids are 1, 2, ...: the log is append-only)."""
        with self._lock:
            rows = self._conn.execute("SELECT type, actor, ts, payload FROM events WHERE rowid > ? ORDER BY rowid",
                                      (after,)).fetchall()
        return [{"type": t, "actor": a, "ts": ts, "payload": json.loads(p)} for t, a, ts, p in rows
                if event_type is None or t == event_type]

    def verify(self, expected_head: tuple[int, str] | None = None) -> int:
        """Recompute the chain from genesis; raise ChainBroken naming the first bad row. With `expected_head` (from an
        earlier `head()`), also require that the log still holds that head. Returns the event count."""
        with self._lock:
            rows = self._conn.execute("SELECT rowid, id, ts, type, actor, payload, prev_hash, hash_chain FROM events "
                                      "ORDER BY rowid").fetchall()
        prev, hashes = GENESIS_HASH, [GENESIS_HASH]
        for rowid, event_id, ts, event_type, actor, payload, stored_prev, stored in rows:
            if stored_prev != prev:
                raise ChainBroken(f"row {rowid}: prev_hash does not link to the row before it")
            if not hmac.compare_digest(chain_hash(prev, event_id, ts, event_type, actor, payload, self.key), stored):
                raise ChainBroken(f"row {rowid}: hash does not match its contents")
            prev = stored
            hashes.append(stored)
        if expected_head is not None:
            n, digest = expected_head
            if n < 0:
                raise ValueError(f"a head's event count cannot be negative: {n}")
            if n > len(rows):
                raise ChainBroken(f"the log has {len(rows)} events but the recorded head is at event {n}: truncated")
            if hashes[n] != digest:
                raise ChainBroken(f"event {n} does not match the recorded head: the log was rewritten")
        return len(rows)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
