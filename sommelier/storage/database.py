"""SQLite connection and schema ownership for runtime state."""

from __future__ import annotations

import sqlite3
import os
from pathlib import Path

DEFAULT_DB_PATH = Path(
    os.environ.get("SOMMELIER_DB_PATH", "data/sommelier.sqlite3")
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    memory_json TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    UNIQUE (session_id, turn_id, role)
);

CREATE INDEX IF NOT EXISTS messages_session_order
ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS traces_session_order
ON traces(session_id, id);

CREATE TABLE IF NOT EXISTS feedback_events (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_request TEXT NOT NULL,
    follow_up INTEGER NOT NULL CHECK (follow_up IN (0, 1)),
    feedback TEXT NOT NULL CHECK (
        feedback IN ('neutral', 'purchase_intent', 'negative_feedback')
    ),
    turn_success INTEGER NOT NULL CHECK (turn_success IN (0, 1)),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS feedback_session_order
ON feedback_events(session_id, created_at);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a configured SQLite connection."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def initialize_database(db_path: Path) -> None:
    """Create the runtime schema idempotently."""

    with connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(SCHEMA_SQL)
