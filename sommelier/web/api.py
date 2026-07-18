"""FastAPI route definitions."""

from pathlib import Path

from fastapi import APIRouter, Request, Response, status
from sommelier.agent.graph import run_agent_turn
from sommelier.agent.state import AgentState
from sommelier.storage.session_repository import (
    SessionRepository,
    get_default_repository,
)
from sommelier.web.schemas import (
    CatalogStatus,
    ChatHistoryResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FeedbackStatsResponse,
    ProfileDebugResponse,
    SessionDebugResponse,
    TraceDebugResponse,
)

router = APIRouter()

PRODUCT_PROFILES_DIR = Path("data/catalog/search_profiles")
COCKTAIL_PROFILES_DIR = Path("data/catalog/cocktail_search_profiles")
INDEX_DIR = Path("data/indexes")


def _build_candidates(state: AgentState) -> list[dict]:
    """Build compact candidate payloads for the web client."""

    candidates: list[dict] = []
    shown = {
        (ref.kind, ref.id)
        for ref in (
            state.final_answer_result.shown_refs
            if state.final_answer_result
            else []
        )
    }
    for card in state.cards:
        if (card.kind, card.id) in shown:
            candidates.append(card.model_dump(mode="json"))
    return candidates


def _repository(request: Request) -> SessionRepository:
    return request.app.state.repository or get_default_repository()


@router.get(
    "/api/analytics/feedback",
    response_model=FeedbackStatsResponse,
)
def feedback_analytics(
    request: Request,
    session_id: str | None = None,
) -> FeedbackStatsResponse:
    """Return global or per-session feedback totals."""

    return FeedbackStatsResponse(
        session_id=session_id,
        **_repository(request).load_feedback_stats(session_id),
    )


@router.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Handle a chat request through the agent layer."""

    state = run_agent_turn(
        AgentState(
            session_id=payload.session_id,
            user_request=payload.message,
        ),
        config={"configurable": {"repository": _repository(request)}},
    )
    return ChatResponse(
        session_id=payload.session_id,
        answer=state.final_answer_result.answer if state.final_answer_result else "",
        follow_up=state.turn_resolution.follow_up if state.turn_resolution else False,
        request_scope=(
            state.turn_resolution.request_scope
            if state.turn_resolution
            else "conversation"
        ),
        effective_request=state.turn_resolution.effective_request if state.turn_resolution else None,
        profile=state.user_profile.model_dump(mode="json") if state.user_profile else None,
        candidates=_build_candidates(state),
        traces=[trace.model_dump(mode="json") for trace in state.tool_traces],
    )


@router.get("/api/catalog/status", response_model=CatalogStatus)
def catalog_status() -> CatalogStatus:
    """Return status for local catalog and retrieval artifacts."""

    product_profiles = list(PRODUCT_PROFILES_DIR.glob("*.json")) if PRODUCT_PROFILES_DIR.exists() else []
    cocktail_profiles = (
        list(COCKTAIL_PROFILES_DIR.glob("*.json"))
        if COCKTAIL_PROFILES_DIR.exists()
        else []
    )
    faiss_metadata_exists = (INDEX_DIR / "product_faiss_metadata.json").exists()
    faiss_index_exists = (INDEX_DIR / "product.faiss").exists()
    return CatalogStatus(
        product_profiles_count=len(product_profiles),
        cocktail_profiles_count=len(cocktail_profiles),
        faiss_metadata_exists=faiss_metadata_exists,
        faiss_index_exists=faiss_index_exists,
        bm25_ready=bool(product_profiles),
        cocktail_bm25_ready=bool(cocktail_profiles),
    )


@router.get("/api/sessions/{session_id}", response_model=SessionDebugResponse)
def session_debug(session_id: str, request: Request) -> SessionDebugResponse:
    """Return durable session memory for debugging."""

    memory = _repository(request).load_session_memory(session_id)
    return SessionDebugResponse(
        session_id=session_id,
        memory=memory.model_dump(mode="json"),
    )


@router.get("/api/traces/{session_id}", response_model=TraceDebugResponse)
def trace_debug(session_id: str, request: Request) -> TraceDebugResponse:
    """Return durable tool traces for debugging."""

    return TraceDebugResponse(
        session_id=session_id,
        traces=_repository(request).load_trace_events(session_id),
    )


@router.get("/api/profiles/{session_id}", response_model=ProfileDebugResponse)
def profile_debug(session_id: str, request: Request) -> ProfileDebugResponse:
    """Return durable user profile for debugging."""

    profile = _repository(request).load_user_profile(session_id)
    return ProfileDebugResponse(
        session_id=session_id,
        profile=profile.model_dump(mode="json"),
    )


@router.get(
    "/api/sessions/{session_id}/messages",
    response_model=ChatHistoryResponse,
)
def chat_history(session_id: str, request: Request) -> ChatHistoryResponse:
    """Return the full UI transcript without exposing it to the agent."""

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            ChatMessage.model_validate(message)
            for message in _repository(request).load_messages(session_id)
        ],
    )


@router.delete(
    "/api/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def reset_session(session_id: str, request: Request) -> Response:
    """Delete agent memory, profile, cart, transcript, traces and feedback."""

    _repository(request).delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
