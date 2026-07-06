"""Typed SQLite repository for complete successful agent turns."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from sommelier.agent.memory import SessionMemory, enforce_memory_limits
from sommelier.agent.profile import UserProfile
from sommelier.agent.tracer import ToolTrace
from sommelier.storage.database import (
    DEFAULT_DB_PATH,
    connect,
    initialize_database,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionRepository:
    """Persist compact agent state and full UI transcript separately."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        initialize_database(self.db_path)

    def load_session_memory(self, session_id: str) -> SessionMemory:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT memory_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return SessionMemory(session_id=session_id)
        return enforce_memory_limits(
            SessionMemory.model_validate_json(row["memory_json"])
        )

    def load_user_profile(self, session_id: str) -> UserProfile:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT profile_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return UserProfile(session_id=session_id)
        return UserProfile.model_validate_json(row["profile_json"])

    def load_messages(self, session_id: str) -> list[dict[str, str]]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        return [
            {"role": str(row["role"]), "content": str(row["content"])}
            for row in rows
        ]

    def load_recent_messages(
        self,
        session_id: str,
        limit: int = 4,
    ) -> list[dict[str, str]]:
        """Return the newest messages in chronological order."""

        if limit <= 0:
            return []
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            {"role": str(row["role"]), "content": str(row["content"])}
            for row in reversed(rows)
        ]

    def load_trace_events(self, session_id: str) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT trace_json
                FROM traces
                WHERE session_id = ?
                ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        return [json.loads(row["trace_json"]) for row in rows]

    def load_last_assistant_message(self, session_id: str) -> str | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT content
                FROM messages
                WHERE session_id = ? AND role = 'assistant'
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return None if row is None else str(row["content"])

    def save_feedback_event(
        self,
        *,
        turn_id: str,
        session_id: str,
        user_request: str,
        follow_up: bool,
        feedback: str,
        turn_success: bool,
    ) -> None:
        """Persist one analytics event independently and idempotently."""

        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO feedback_events (
                    turn_id, session_id, user_request, follow_up,
                    feedback, turn_success, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_id) DO NOTHING
                """,
                (
                    turn_id,
                    session_id,
                    user_request,
                    int(follow_up),
                    feedback,
                    int(turn_success),
                    _utc_now(),
                ),
            )

    def load_feedback_stats(
        self,
        session_id: str | None = None,
    ) -> dict[str, int]:
        where = ""
        parameters: tuple[str, ...] = ()
        if session_id is not None:
            where = "WHERE session_id = ?"
            parameters = (session_id,)
        with connect(self.db_path) as connection:
            row = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS total,
                    COALESCE(SUM(feedback = 'neutral'), 0) AS neutral,
                    COALESCE(SUM(feedback = 'purchase_intent'), 0)
                        AS purchase_intent,
                    COALESCE(SUM(feedback = 'negative_feedback'), 0)
                        AS negative_feedback,
                    COALESCE(SUM(turn_success = 1), 0) AS successful_turns,
                    COALESCE(SUM(turn_success = 0), 0) AS failed_turns
                FROM feedback_events
                {where}
                """,
                parameters,
            ).fetchone()
        return {key: int(row[key]) for key in row.keys()}

    def _insert_traces(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        turn_id: str,
        traces: list[ToolTrace],
        created_at: str,
    ) -> None:
        connection.executemany(
            """
            INSERT INTO traces (
                session_id, turn_id, trace_json, created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    turn_id,
                    json.dumps(
                        {
                            "session_id": session_id,
                            "turn_id": turn_id,
                            **trace.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                    ),
                    created_at,
                )
                for trace in traces
            ],
        )

    def persist_successful_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        memory: SessionMemory,
        profile: UserProfile,
        user_message: str,
        assistant_message: str,
        traces: list[ToolTrace],
    ) -> None:
        """Atomically persist one validated turn and its UI transcript."""

        if memory.session_id != session_id or profile.session_id != session_id:
            raise ValueError("persisted state session_id mismatch")
        now = _utc_now()
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            already_saved = connection.execute(
                """
                SELECT 1
                FROM messages
                WHERE session_id = ? AND turn_id = ?
                LIMIT 1
                """,
                (session_id, turn_id),
            ).fetchone()
            if already_saved is not None:
                return
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, memory_json, profile_json,
                    version, created_at, updated_at
                )
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    memory_json = excluded.memory_json,
                    profile_json = excluded.profile_json,
                    version = sessions.version + 1,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    enforce_memory_limits(memory).model_dump_json(),
                    profile.model_dump_json(),
                    now,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO messages (
                    session_id, turn_id, role, content, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (session_id, turn_id, "user", user_message, now),
                    (session_id, turn_id, "assistant", assistant_message, now),
                ],
            )
            self._insert_traces(
                connection,
                session_id=session_id,
                turn_id=turn_id,
                traces=traces,
                created_at=now,
            )

    def delete_session(self, session_id: str) -> None:
        """Delete one session and cascading transcript/traces."""

        with connect(self.db_path) as connection:
            connection.execute(
                "DELETE FROM feedback_events WHERE session_id = ?",
                (session_id,),
            )
            connection.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )


@lru_cache(maxsize=1)
def get_default_repository() -> SessionRepository:
    """Return the process-local repository for the default SQLite file."""

    return SessionRepository(DEFAULT_DB_PATH)
