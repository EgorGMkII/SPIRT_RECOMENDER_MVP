"""Durable JSONL storage for tool traces."""

from __future__ import annotations

import json
from pathlib import Path

from sommelier.agent.profile_store import profile_slug
from sommelier.agent.tracer import ToolTrace

DEFAULT_TRACE_DIR = Path("data/traces")


def trace_path(session_id: str, trace_dir: Path = DEFAULT_TRACE_DIR) -> Path:
    """Return the JSONL trace path for a session."""

    return trace_dir / f"{profile_slug(session_id)}.jsonl"


def append_trace_events(
    session_id: str,
    turn_id: str,
    traces: list[ToolTrace],
    trace_dir: Path = DEFAULT_TRACE_DIR,
) -> Path:
    """Append all traces from a turn to durable JSONL storage."""

    path = trace_path(session_id, trace_dir=trace_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for trace in traces:
            payload = {
                "session_id": session_id,
                "turn_id": turn_id,
                **trace.model_dump(mode="json"),
            }
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def load_trace_events(
    session_id: str,
    trace_dir: Path = DEFAULT_TRACE_DIR,
) -> list[dict]:
    """Load durable trace events for a session."""

    path = trace_path(session_id, trace_dir=trace_dir)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
