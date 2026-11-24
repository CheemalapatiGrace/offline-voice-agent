"""Thread-safe SQLite wrapper shared by the appointment backend and the trace store."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS appointments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_name TEXT NOT NULL,
  patient_key TEXT NOT NULL,
  date TEXT NOT NULL,
  time TEXT NOT NULL,
  reason TEXT,
  status TEXT NOT NULL DEFAULT 'booked',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_slot_booked ON appointments(date, time) WHERE status = 'booked';
CREATE INDEX IF NOT EXISTS ix_appt_patient ON appointments(patient_key);

CREATE TABLE IF NOT EXISTS idempotency (
  key TEXT PRIMARY KEY,
  tool TEXT NOT NULL,
  result_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS handoffs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  reason TEXT NOT NULL,
  urgency TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS turns (
  pk INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  turn_id INTEGER NOT NULL,
  ts REAL NOT NULL,
  user_text TEXT,
  reply_text TEXT,
  ttfa_ms REAL,
  total_ms REAL,
  outcome TEXT,
  trace_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_turns_session ON turns(session_id, pk);

CREATE TABLE IF NOT EXISTS spans (
  pk INTEGER PRIMARY KEY AUTOINCREMENT,
  turn_pk INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  name TEXT NOT NULL,
  start_ms REAL NOT NULL,
  dur_ms REAL NOT NULL,
  status TEXT NOT NULL,
  attrs_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_spans_name ON spans(name);

CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  operator TEXT NOT NULL,
  action TEXT NOT NULL,
  session_id TEXT,
  details_json TEXT
);

CREATE TABLE IF NOT EXISTS approvals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  session_id TEXT NOT NULL,
  tool TEXT NOT NULL,
  args_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  decided_by TEXT,
  decided_ts REAL,
  note TEXT
);
"""


class Database:
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def ping(self) -> bool:
        try:
            self.query("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def close(self) -> None:
        with self._lock:
            self._conn.close()
