"""JSON persistence for session memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from sommelier.agent.memory import (
    CandidateMemory,
    ConversationMessage,
    LastTurnMemory,
    SessionMemory,
)
from sommelier.agent.profile_store import profile_slug

if TYPE_CHECKING:
    from sommelier.agent.state import AgentState

DEFAULT_SESSION_DIR = Path("data/sessions")
MAX_SESSION_MESSAGES = 20


def session_path(session_id: str, session_dir: Path = DEFAULT_SESSION_DIR) -> Path:
    """Return the durable memory JSON path for a session."""

    return session_dir / f"{profile_slug(session_id)}.json"


def load_session_memory(
    session_id: str,
    session_dir: Path = DEFAULT_SESSION_DIR,
) -> SessionMemory:
    """Load session memory or return an empty memory object."""

    path = session_path(session_id, session_dir=session_dir)
    if not path.exists():
        return SessionMemory(session_id=session_id)
    return SessionMemory.model_validate_json(path.read_text(encoding="utf-8"))


def save_session_memory(
    memory: SessionMemory,
    session_dir: Path = DEFAULT_SESSION_DIR,
) -> Path:
    """Save session memory as JSON."""

    path = session_path(memory.session_id, session_dir=session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(memory.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def build_last_turn_memory(state: "AgentState") -> LastTurnMemory:
    """Build a compact memory summary from the completed agent state."""

    candidates: list[CandidateMemory] = []
    for result in state.retrieval_results[:4]:
        candidates.append(
            CandidateMemory(
                item_id=result.product_id,
                name=result.profile.name,
                kind="rum",
                score=result.score,
            )
        )
    for result in state.cocktail_results[:5]:
        candidates.append(
            CandidateMemory(
                item_id=result.cocktail_id,
                name=result.profile.name,
                kind="cocktail",
                score=result.score,
            )
        )

    return LastTurnMemory(
        user_message=state.user_message,
        effective_user_message=state.effective_user_message or state.user_message,
        intent=str(state.parsed_intent.intent) if state.parsed_intent else None,
        search_query=state.search_query,
        expanded_query=state.expanded_query,
        cocktail_query=state.cocktail_query,
        final_answer=state.final_answer,
        candidates=candidates,
    )


def update_session_memory_from_state(state: "AgentState") -> SessionMemory:
    """Append the completed turn to session memory."""

    memory = state.session_memory or SessionMemory(session_id=state.session_id)
    memory.messages.append(ConversationMessage(role="user", content=state.user_message))
    if state.final_answer:
        memory.messages.append(
            ConversationMessage(role="assistant", content=state.final_answer)
        )
    memory.messages = memory.messages[-MAX_SESSION_MESSAGES:]
    memory.last_turn = build_last_turn_memory(state)
    return memory
