"""FastAPI route definitions."""

from pathlib import Path

from fastapi import APIRouter
from sommelier.agent.graph import run_agent_turn
from sommelier.agent.memory_store import load_session_memory
from sommelier.agent.profile_store import load_user_profile
from sommelier.agent.state import AgentState
from sommelier.agent.trace_store import load_trace_events
from sommelier.web.schemas import (
    CatalogStatus,
    ChatRequest,
    ChatResponse,
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
    for result in state.retrieval_results:
        profile = result.profile
        candidates.append(
            {
                "kind": "rum",
                "id": result.product_id,
                "name": profile.name,
                "score": round(result.score, 4),
                "sources": state.retrieval_sources.get(result.product_id, []),
                "description": profile.display_description,
                "tasting_summary": profile.tasting_summary,
                "flavor_tags": profile.flavor_tags,
                "usage_tags": profile.usage_tags,
            }
        )
    for result in state.cocktail_results:
        profile = result.profile
        candidates.append(
            {
                "kind": "cocktail",
                "id": result.cocktail_id,
                "name": profile.name,
                "score": round(result.score, 4),
                "main_rum": profile.main_rum,
                "description": profile.description,
                "ingredients": profile.ingredients,
                "recipe_steps": profile.recipe_steps,
                "matched_tokens": result.matched_tokens,
            }
        )
    return candidates


@router.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Handle a chat request through the agent layer."""

    state = run_agent_turn(
        AgentState(
            session_id=request.session_id,
            user_message=request.message,
            use_llm_followup=True,
            use_llm_intent=True,
            use_llm_query=True,
            use_llm_food_query=True,
            use_llm_profile_update=True,
            use_llm_answer=True,
        )
    )
    return ChatResponse(
        session_id=request.session_id,
        answer=state.final_answer or "",
        intent=str(state.parsed_intent.intent) if state.parsed_intent else None,
        effective_user_message=state.effective_user_message,
        is_followup=state.is_followup,
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
def session_debug(session_id: str) -> SessionDebugResponse:
    """Return durable session memory for debugging."""

    memory = load_session_memory(session_id)
    return SessionDebugResponse(
        session_id=session_id,
        memory=memory.model_dump(mode="json"),
    )


@router.get("/api/traces/{session_id}", response_model=TraceDebugResponse)
def trace_debug(session_id: str) -> TraceDebugResponse:
    """Return durable tool traces for debugging."""

    return TraceDebugResponse(
        session_id=session_id,
        traces=load_trace_events(session_id),
    )


@router.get("/api/profiles/{session_id}", response_model=ProfileDebugResponse)
def profile_debug(session_id: str) -> ProfileDebugResponse:
    """Return durable user profile for debugging."""

    profile = load_user_profile(session_id)
    return ProfileDebugResponse(
        session_id=session_id,
        profile=profile.model_dump(mode="json"),
    )
