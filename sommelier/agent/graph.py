"""Minimal controlled LangGraph workflow over the retrieval layer."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import json

from sommelier.agent.memory import resolve_followup_context
from sommelier.agent.memory_store import (
    load_session_memory,
    save_session_memory,
    update_session_memory_from_state,
)
from sommelier.agent.nlu import parse_intent
from sommelier.agent.prompts import COCKTAIL_RESPONSE_PROMPT, SOMMELIER_RESPONSE_PROMPT
from sommelier.agent.profile_store import load_user_profile, save_user_profile
from sommelier.agent.schemas import IntentType, ParsedIntent
from sommelier.agent.state import AgentState
from sommelier.agent.trace_store import append_trace_events
from sommelier.agent.tracer import ToolTrace
from sommelier.agent.tools.profile_update import profile_update
from sommelier.retrieval.bm25_index import Bm25Index
from sommelier.retrieval.cocktail_search import search_cocktails
from sommelier.retrieval.faiss_index import FaissIndex, OpenAIEmbeddingProvider
from sommelier.retrieval.food_pairing_query import (
    DEFAULT_PAIRING_CAVEAT,
    normalize_food_pairing_query,
)
from sommelier.retrieval.query_normalizer import normalize_query
from sommelier.retrieval.schemas import SearchRequest

DEFAULT_INDEX_DIR = Path("data/indexes")
DEFAULT_PROFILES_DIR = Path("data/catalog/search_profiles")
DEFAULT_FAISS_TOP_K = 2
DEFAULT_BM25_TOP_K = 2
DEFAULT_COCKTAIL_TOP_K = 5


def load_agent_index(index_dir: Path = DEFAULT_INDEX_DIR) -> FaissIndex:
    """Load the product vector index used by the MVP agent."""

    return FaissIndex.load(index_dir, embedding_provider=OpenAIEmbeddingProvider())


def load_agent_bm25_index(profiles_dir: Path = DEFAULT_PROFILES_DIR) -> Bm25Index:
    """Load the lexical BM25 index used by the MVP agent."""

    return Bm25Index.load(profiles_dir)


def hybrid_search(
    query: str,
    normalize: bool = True,
    use_llm_query: bool = False,
) -> tuple[list, dict[str, list[str]], dict]:
    """Return FAISS top-1 plus BM25 top-1 with duplicate products merged."""

    normalized_query = (
        normalize_query(query, use_llm=use_llm_query)
        if normalize
        else query
    )
    debug: dict = {"normalized_query": normalized_query}
    try:
        faiss_results = load_agent_index().search(
            normalized_query,
            top_k=DEFAULT_FAISS_TOP_K,
            normalize=False,
        )
        debug["faiss_error"] = None
    except Exception as exc:
        faiss_results = []
        debug["faiss_error"] = str(exc)

    try:
        bm25_results = load_agent_bm25_index().search(
            normalized_query,
            top_k=DEFAULT_BM25_TOP_K,
            normalize=False,
        )
        debug["bm25_error"] = None
    except Exception as exc:
        bm25_results = []
        debug["bm25_error"] = str(exc)

    if not faiss_results and not bm25_results:
        errors = [
            message
            for message in (debug.get("faiss_error"), debug.get("bm25_error"))
            if message
        ]
        raise RuntimeError("; ".join(errors) or "hybrid retrieval returned no results")

    merged = []
    sources: dict[str, list[str]] = {}
    debug.update(
        {
            "faiss_top": [result.product_id for result in faiss_results],
            "bm25_top": [item.result.product_id for item in bm25_results],
            "bm25_matched_tokens": {
                item.result.product_id: item.matched_tokens for item in bm25_results
            },
        }
    )

    for result in faiss_results:
        sources.setdefault(result.product_id, []).append("faiss")
        merged.append(result)

    seen = {result.product_id for result in merged}
    for item in bm25_results:
        result = item.result
        sources.setdefault(result.product_id, []).append("bm25")
        if result.product_id not in seen:
            merged.append(result)
            seen.add(result.product_id)

    return merged, sources, debug


def _trace(
    state: AgentState,
    tool_name: str,
    tool_input: dict,
    output_summary: str,
    status: str = "success",
) -> None:
    """Append a trace event to agent state."""

    state.tool_traces.append(
        ToolTrace(
            tool_name=tool_name,
            input=tool_input,
            output_summary=output_summary,
            status=status,
        )
    )


def parse_intent_node(state: AgentState) -> AgentState:
    """Parse user text into a controlled MVP intent."""

    message = state.effective_user_message or state.user_message
    if state.followup_intent is not None:
        parsed_intent = ParsedIntent(
            intent=state.followup_intent,
            search=SearchRequest(query=message),
            confidence=1.0,
        )
    else:
        parsed_intent = parse_intent(message, use_llm=state.use_llm_intent)
    state.parsed_intent = parsed_intent
    _trace(
        state,
        "parse_intent",
        {
            "message": state.user_message,
            "effective_message": message,
            "is_followup": state.is_followup,
            "followup_intent": state.followup_intent,
            "followup_reason": state.followup_reason,
        },
        f"intent={state.parsed_intent.intent}",
    )
    return state



def memory_load_node(state: AgentState) -> AgentState:
    """Load durable session memory and resolve follow-up context."""

    try:
        state.session_memory = load_session_memory(state.session_id)
        resolution = resolve_followup_context(
            state.user_message,
            state.session_memory,
            use_llm=state.use_llm_followup,
        )
        state.is_followup = resolution.is_followup
        state.followup_intent = resolution.intent
        state.followup_reason = resolution.reason
        state.effective_user_message = resolution.effective_user_message
        state.avoid_previous_candidates = getattr(
            resolution,
            "avoid_previous_candidates",
            False,
        )
        has_last_turn = state.session_memory.last_turn is not None
        _trace(
            state,
            "memory_load",
            {
                "message": state.user_message,
                "is_followup": state.is_followup,
                "effective_message": state.effective_user_message,
                "followup_intent": state.followup_intent,
                "followup_reason": state.followup_reason,
                "avoid_previous_candidates": state.avoid_previous_candidates,
                "has_last_turn": has_last_turn,
            },
            (
                "loaded session memory; "
                f"followup={state.is_followup}; has_last_turn={has_last_turn}"
            ),
        )
    except Exception as exc:
        state.effective_user_message = state.user_message
        message = f"memory_load failed: {exc}"
        state.errors.append(message)
        _trace(state, "memory_load", {"message": state.user_message}, message, status="error")
    return state


def profile_update_node(state: AgentState) -> AgentState:
    """Load, update, and persist the simple user preference profile."""

    try:
        profile = state.user_profile or load_user_profile(state.session_id)
        if not (
            state.parsed_intent
            and state.parsed_intent.intent == IntentType.PROFILE_UPDATE
        ):
            state.user_profile = profile
            state.profile_updated = False
            _trace(
                state,
                "profile_update",
                {
                    "message": state.user_message,
                    "skipped": True,
                    "intent": state.parsed_intent.intent if state.parsed_intent else None,
                },
                "profile update skipped for non-profile intent",
            )
            return state

        updated_profile, result = profile_update(
            profile,
            state.user_message,
            use_llm=state.use_llm_profile_update,
        )
        state.user_profile = updated_profile
        changed = result.metadata.get("changed", {})
        state.profile_updated = any(changed.values())
        if state.profile_updated and state.persist_artifacts:
            save_user_profile(updated_profile)
        _trace(
            state,
            "profile_update",
            {"message": state.user_message},
            result.summary,
        )
    except Exception as exc:
        message = f"profile_update failed: {exc}"
        state.errors.append(message)
        _trace(state, "profile_update", {"message": state.user_message}, message, status="error")
    return state


def profile_ack_node(state: AgentState) -> AgentState:
    """Acknowledge a pure profile update without running retrieval."""

    state.final_answer = "Учту ваше предпочтение."
    _trace(
        state,
        "profile_ack",
        {"profile_updated": state.profile_updated},
        "profile update acknowledged",
    )
    return state


def direct_search_node(state: AgentState) -> AgentState:
    """Run direct product retrieval for a natural-language rum query."""

    query = (
        state.parsed_intent.search.query
        if state.parsed_intent and state.parsed_intent.search
        else state.effective_user_message or state.user_message
    )
    state.search_query = query
    try:
        results, sources, debug = hybrid_search(
            query,
            normalize=True,
            use_llm_query=state.use_llm_query,
        )
        state.search_query = debug.get("normalized_query", query)
        state.retrieval_results = results
        state.retrieval_sources = sources
        _trace(
            state,
            "direct_search",
            {
                "query": query,
                "faiss_top_k": DEFAULT_FAISS_TOP_K,
                "bm25_top_k": DEFAULT_BM25_TOP_K,
                **debug,
            },
            f"found={len(state.retrieval_results)}",
        )
    except Exception as exc:
        message = f"direct_search failed: {exc}"
        state.errors.append(message)
        _trace(state, "direct_search", {"query": query}, message, status="error")
    return state


def food_pairing_search_node(state: AgentState) -> AgentState:
    """Expand a food query and retrieve product candidates."""

    food_text = (
        state.parsed_intent.search.query
        if state.parsed_intent and state.parsed_intent.search
        else state.effective_user_message or state.user_message
    )
    try:
        expanded_query = normalize_food_pairing_query(
            food_text,
            use_llm=state.use_llm_food_query,
        )
        results, sources, debug = hybrid_search(expanded_query, normalize=False)
        state.search_query = expanded_query
        state.expanded_query = expanded_query
        state.retrieval_results = results
        state.retrieval_sources = sources
        state.retrieval_caveat = DEFAULT_PAIRING_CAVEAT
        _trace(
            state,
            "food_pairing_search",
            {
                "food_text": food_text,
                "expanded_query": expanded_query,
                "faiss_top_k": DEFAULT_FAISS_TOP_K,
                "bm25_top_k": DEFAULT_BM25_TOP_K,
                **debug,
            },
            f"found={len(state.retrieval_results)}",
        )
    except Exception as exc:
        message = f"food_pairing_search failed: {exc}"
        state.errors.append(message)
        _trace(state, "food_pairing_search", {"food_text": food_text}, message, status="error")
    return state


def cocktail_search_node(state: AgentState) -> AgentState:
    """Normalize a cocktail request and retrieve cocktail candidates."""

    query = (
        state.parsed_intent.search.query
        if state.parsed_intent and state.parsed_intent.search
        else state.effective_user_message or state.user_message
    )
    try:
        results = search_cocktails(query, top_k=DEFAULT_COCKTAIL_TOP_K)
        if state.avoid_previous_candidates and state.session_memory and state.session_memory.last_turn:
            previous_main_candidates = [
                candidate
                for candidate in state.session_memory.last_turn.candidates
                if candidate.kind == "cocktail"
            ][:1]
            previous_ids = {candidate.item_id for candidate in previous_main_candidates}
            state.avoid_candidate_ids = [candidate.item_id for candidate in previous_main_candidates]
            state.avoid_candidate_names = [candidate.name for candidate in previous_main_candidates]
            filtered_results = [
                result for result in results if result.cocktail_id not in previous_ids
            ]
            if filtered_results:
                results = filtered_results
        state.cocktail_results = results
        state.cocktail_query = results[0].normalized_query if results else query
        state.search_query = state.cocktail_query
        _trace(
            state,
            "cocktail_search",
            {
                "query": query,
                "normalized_query": state.cocktail_query,
                "top_k": DEFAULT_COCKTAIL_TOP_K,
                "avoid_previous_candidates": state.avoid_previous_candidates,
                "avoid_candidate_ids": state.avoid_candidate_ids,
                "avoid_candidate_names": state.avoid_candidate_names,
                "cocktail_top": [result.cocktail_id for result in results],
                "matched_tokens": {
                    result.cocktail_id: result.matched_tokens for result in results
                },
            },
            f"found={len(state.cocktail_results)}",
        )
    except Exception as exc:
        message = f"cocktail_search failed: {exc}"
        state.errors.append(message)
        _trace(state, "cocktail_search", {"query": query}, message, status="error")
    return state


def retrieval_node(state: AgentState) -> AgentState:
    """Validate and summarize retrieval output before answer generation."""

    if not state.retrieval_results and not state.cocktail_results and not state.errors:
        state.errors.append("retrieval returned no candidates")
    _trace(
        state,
        "retrieval",
        {
            "candidate_count": len(state.retrieval_results),
            "cocktail_candidate_count": len(state.cocktail_results),
        },
        (
            f"candidate_count={len(state.retrieval_results)}, "
            f"cocktail_candidate_count={len(state.cocktail_results)}"
        ),
    )
    return state


def _message_content(message) -> str:
    """Extract text content from a LangChain-style LLM response."""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(message, str):
        return message
    return str(message)


def build_answer_context(state: AgentState) -> dict:
    """Build a compact evidence payload for the final answer LLM."""

    candidates = []
    for result in state.retrieval_results:
        profile = result.profile
        candidates.append(
            {
                "product_id": result.product_id,
                "name": profile.name,
                "brand": profile.brand,
                "category": profile.category,
                "display_description": profile.display_description,
                "tasting_summary": profile.tasting_summary,
                "flavor_tags": profile.flavor_tags,
                "usage_tags": profile.usage_tags,
                "cocktail_names": profile.cocktail_names[:8],
                "evidence_fields": profile.evidence_fields,
                "score": round(result.score, 4),
                "retrieval_sources": state.retrieval_sources.get(result.product_id, []),
            }
        )
    return {
        "user_message": state.user_message,
        "effective_user_message": state.effective_user_message,
        "is_followup": state.is_followup,
        "intent": state.parsed_intent.intent if state.parsed_intent else None,
        "direct_or_expanded_query": state.expanded_query or state.search_query,
        "food_pairing_caveat": state.retrieval_caveat,
        "user_profile": state.user_profile.model_dump(mode="json") if state.user_profile else None,
        "candidates": candidates,
    }


def build_cocktail_answer_context(state: AgentState) -> dict:
    """Build a compact evidence payload for cocktail answer generation."""

    candidates = []
    for result in state.cocktail_results:
        profile = result.profile
        candidates.append(
            {
                "cocktail_id": result.cocktail_id,
                "name": profile.name,
                "main_rum": profile.main_rum,
                "description": profile.description,
                "ingredients": profile.ingredients,
                "recipe_steps": profile.recipe_steps,
                "matched_tokens": result.matched_tokens,
                "score": round(result.score, 4),
            }
        )
    return {
        "user_message": state.user_message,
        "effective_user_message": state.effective_user_message,
        "is_followup": state.is_followup,
        "intent": state.parsed_intent.intent if state.parsed_intent else None,
        "avoid_previous_candidates": state.avoid_previous_candidates,
        "avoid_candidate_ids": state.avoid_candidate_ids,
        "avoid_candidate_names": state.avoid_candidate_names,
        "normalized_cocktail_query": state.cocktail_query or state.search_query,
        "user_profile": state.user_profile.model_dump(mode="json") if state.user_profile else None,
        "cocktail_candidates": candidates,
    }


def build_sommelier_answer_prompt(state: AgentState) -> str:
    """Build the final response prompt from validated retrieval context."""

    context_json = json.dumps(
        build_answer_context(state),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    return SOMMELIER_RESPONSE_PROMPT.format(context_json=context_json)


def build_cocktail_answer_prompt(state: AgentState) -> str:
    """Build the final cocktail response prompt from validated context."""

    context_json = json.dumps(
        build_cocktail_answer_context(state),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    return COCKTAIL_RESPONSE_PROMPT.format(context_json=context_json)


def generate_sommelier_answer(state: AgentState, llm=None) -> str:
    """Generate a sommelier-style answer from validated retrieval evidence."""

    active_llm = llm
    if active_llm is None:
        from llm_module import get_langchain_openai_chat_model

        active_llm = get_langchain_openai_chat_model()
    return _message_content(active_llm.invoke(build_sommelier_answer_prompt(state))).strip()


def generate_cocktail_answer(state: AgentState, llm=None) -> str:
    """Generate a cocktail answer from validated cocktail evidence."""

    active_llm = llm
    if active_llm is None:
        from llm_module import get_langchain_openai_chat_model

        active_llm = get_langchain_openai_chat_model()
    return _message_content(active_llm.invoke(build_cocktail_answer_prompt(state))).strip()


def _deterministic_answer(state: AgentState) -> str:
    """Generate a deterministic answer when LLM answer writing is disabled."""

    lines = ["Я бы начал с этих вариантов:"]
    for index, result in enumerate(state.retrieval_results[:2], start=1):
        profile = result.profile
        description = profile.display_description or profile.searchable_text
        lines.append(
            f"{index}. {profile.name} — {description} "
            f"(score={result.score:.3f})"
        )

    if state.expanded_query:
        lines.append(f"Поисковая формулировка: {state.expanded_query}")
    elif state.retrieval_results:
        lines.append(f"Нормализованный запрос: {state.retrieval_results[0].normalized_query}")

    if state.retrieval_caveat:
        lines.append(f"Важно: {state.retrieval_caveat}")

    return "\n".join(lines)


def _deterministic_cocktail_answer(state: AgentState) -> str:
    """Generate a deterministic cocktail answer when LLM writing is disabled."""

    lines = ["Я бы начал с этих коктейлей:"]
    for index, result in enumerate(state.cocktail_results[:2], start=1):
        profile = result.profile
        lines.append(f"{index}. {profile.name} — {profile.description} (score={result.score:.3f})")
        lines.append(f"   Ром: {profile.main_rum}")
        if profile.ingredients:
            lines.append("   Ингредиенты: " + "; ".join(profile.ingredients))
        if profile.recipe_steps:
            lines.append("   Шаги: " + " / ".join(profile.recipe_steps))
    if state.cocktail_query:
        lines.append(f"Нормализованный коктейльный запрос: {state.cocktail_query}")
    return "\n".join(lines)


def answer_generation_node(state: AgentState) -> AgentState:
    """Generate the final MVP answer from retrieved candidates."""

    if state.errors:
        state.final_answer = (
            "Не смог выполнить поиск по текущему индексу. "
            "Проверь, что ProductSearchProfile и FAISS index уже собраны. "
            f"Техническая деталь: {state.errors[-1]}"
        )
        _trace(state, "answer_generation", {}, "error answer generated", status="error")
        return state

    if not state.retrieval_results and not state.cocktail_results:
        state.final_answer = "Не нашёл подходящих кандидатов в текущем каталоге."
        _trace(state, "answer_generation", {}, "empty answer generated")
        return state

    if state.cocktail_results:
        if state.use_llm_answer:
            try:
                state.final_answer = generate_cocktail_answer(state)
                _trace(
                    state,
                    "answer_generation",
                    {
                        "candidate_count": len(state.cocktail_results),
                        "mode": "llm",
                        "answer_type": "cocktail",
                    },
                    "cocktail answer generated",
                )
                return state
            except Exception as exc:
                state.errors.append(f"llm cocktail answer generation failed: {exc}")
                _trace(
                    state,
                    "answer_generation",
                    {
                        "candidate_count": len(state.cocktail_results),
                        "mode": "llm",
                        "answer_type": "cocktail",
                    },
                    f"llm cocktail answer failed, used deterministic fallback: {exc}",
                    status="error",
                )
        state.final_answer = _deterministic_cocktail_answer(state)
        _trace(
            state,
            "answer_generation",
            {
                "candidate_count": len(state.cocktail_results),
                "mode": "deterministic",
                "answer_type": "cocktail",
            },
            "cocktail answer generated",
        )
        return state

    if state.use_llm_answer:
        try:
            state.final_answer = generate_sommelier_answer(state)
            _trace(
                state,
                "answer_generation",
                {"candidate_count": len(state.retrieval_results), "mode": "llm"},
                "sommelier answer generated",
            )
            return state
        except Exception as exc:
            state.errors.append(f"llm answer generation failed: {exc}")
            _trace(
                state,
                "answer_generation",
                {"candidate_count": len(state.retrieval_results), "mode": "llm"},
                f"llm answer failed, used deterministic fallback: {exc}",
                status="error",
            )

    state.final_answer = _deterministic_answer(state)
    _trace(
        state,
        "answer_generation",
        {"candidate_count": len(state.retrieval_results), "mode": "deterministic"},
        "answer generated",
    )
    return state


def persistence_node(state: AgentState) -> AgentState:
    """Persist session memory and durable tool traces for this turn."""

    if not state.persist_artifacts:
        return state

    try:
        state.session_memory = update_session_memory_from_state(state)
        memory_path = save_session_memory(state.session_memory)
        _trace(
            state,
            "memory_save",
            {"path": str(memory_path)},
            "session memory saved",
        )
    except Exception as exc:
        message = f"memory_save failed: {exc}"
        state.errors.append(message)
        _trace(state, "memory_save", {}, message, status="error")

    try:
        trace_path = append_trace_events(
            session_id=state.session_id,
            turn_id=state.turn_id,
            traces=state.tool_traces,
        )
        _trace(
            state,
            "trace_save",
            {"path": str(trace_path), "event_count": len(state.tool_traces)},
            "tool traces saved",
        )
    except Exception as exc:
        message = f"trace_save failed: {exc}"
        state.errors.append(message)
        _trace(state, "trace_save", {}, message, status="error")
    return state


def route_after_intent(
    state: AgentState,
) -> Literal["profile_ack", "cocktail_search", "food_pairing_search", "direct_search"]:
    """Choose the next graph node after intent parsing."""

    if state.profile_updated or (
        state.parsed_intent and state.parsed_intent.intent == IntentType.PROFILE_UPDATE
    ):
        return "profile_ack"
    if state.parsed_intent and state.parsed_intent.intent == IntentType.COCKTAIL_EXPANSION:
        return "cocktail_search"
    if state.parsed_intent and state.parsed_intent.intent == IntentType.FOOD_PAIRING:
        return "food_pairing_search"
    return "direct_search"


def _run_sequential(state: AgentState) -> AgentState:
    """Fallback runner with the same node order as the LangGraph workflow."""

    state = memory_load_node(state)
    state = parse_intent_node(state)
    state = profile_update_node(state)
    route = route_after_intent(state)
    if route == "profile_ack":
        state = profile_ack_node(state)
        return persistence_node(state)
    if route == "cocktail_search":
        state = cocktail_search_node(state)
    elif route == "food_pairing_search":
        state = food_pairing_search_node(state)
    else:
        state = direct_search_node(state)
    state = retrieval_node(state)
    state = answer_generation_node(state)
    return persistence_node(state)


def run_agent_turn(state: AgentState) -> AgentState:
    """Run one controlled MVP agent turn."""

    graph = build_graph()
    if graph is _run_sequential:
        return _run_sequential(state)
    result = graph.invoke(state)
    if isinstance(result, AgentState):
        return result
    return AgentState.model_validate(result)


def build_graph():
    """Build the minimal LangGraph workflow when LangGraph is installed."""

    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return _run_sequential

    graph = StateGraph(AgentState)
    graph.add_node("memory_load", memory_load_node)
    graph.add_node("parse_intent", parse_intent_node)
    graph.add_node("profile_update", profile_update_node)
    graph.add_node("profile_ack", profile_ack_node)
    graph.add_node("direct_search", direct_search_node)
    graph.add_node("food_pairing_search", food_pairing_search_node)
    graph.add_node("cocktail_search", cocktail_search_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("answer_generation", answer_generation_node)
    graph.add_node("persistence", persistence_node)

    graph.set_entry_point("memory_load")
    graph.add_edge("memory_load", "parse_intent")
    graph.add_edge("parse_intent", "profile_update")
    graph.add_conditional_edges(
        "profile_update",
        route_after_intent,
        {
            "profile_ack": "profile_ack",
            "cocktail_search": "cocktail_search",
            "direct_search": "direct_search",
            "food_pairing_search": "food_pairing_search",
        },
    )
    graph.add_edge("profile_ack", "persistence")
    graph.add_edge("direct_search", "retrieval")
    graph.add_edge("food_pairing_search", "retrieval")
    graph.add_edge("cocktail_search", "retrieval")
    graph.add_edge("retrieval", "answer_generation")
    graph.add_edge("answer_generation", "persistence")
    graph.add_edge("persistence", END)
    return graph.compile()
