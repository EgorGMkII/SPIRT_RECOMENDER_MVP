"""Controlled agent workflow nodes."""

from __future__ import annotations

from sommelier.agent.memory import resolve_followup_context
from sommelier.agent.memory_store import (
    load_session_memory,
    save_session_memory,
    update_session_memory_from_state,
)
from sommelier.agent.nlu import parse_intent
from sommelier.agent.node_utils import trace
from sommelier.agent.profile_store import load_user_profile, save_user_profile
from sommelier.agent.schemas import IntentType, ParsedIntent
from sommelier.agent.search_runtime import (
    DEFAULT_BM25_TOP_K,
    DEFAULT_COCKTAIL_TOP_K,
    DEFAULT_FAISS_TOP_K,
    hybrid_search,
)
from sommelier.agent.state import AgentState
from sommelier.agent.tools.profile_update import profile_update
from sommelier.agent.trace_store import append_trace_events
from sommelier.retrieval.cocktail_search import search_cocktails
from sommelier.retrieval.food_pairing_query import (
    DEFAULT_PAIRING_CAVEAT,
    normalize_food_pairing_query,
)
from sommelier.retrieval.schemas import SearchRequest


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
    trace(
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
        trace(
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
        trace(state, "memory_load", {"message": state.user_message}, message, status="error")
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
            trace(
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
        trace(
            state,
            "profile_update",
            {"message": state.user_message},
            result.summary,
        )
    except Exception as exc:
        message = f"profile_update failed: {exc}"
        state.errors.append(message)
        trace(state, "profile_update", {"message": state.user_message}, message, status="error")
    return state


def profile_ack_node(state: AgentState) -> AgentState:
    """Acknowledge a pure profile update without running retrieval."""

    state.final_answer = "\u0423\u0447\u0442\u0443 \u0432\u0430\u0448\u0435 \u043f\u0440\u0435\u0434\u043f\u043e\u0447\u0442\u0435\u043d\u0438\u0435."
    trace(
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
        trace(
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
        trace(state, "direct_search", {"query": query}, message, status="error")
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
        trace(
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
        trace(state, "food_pairing_search", {"food_text": food_text}, message, status="error")
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
        trace(
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
        trace(state, "cocktail_search", {"query": query}, message, status="error")
    return state


def retrieval_node(state: AgentState) -> AgentState:
    """Validate and summarize retrieval output before answer generation."""

    if not state.retrieval_results and not state.cocktail_results and not state.errors:
        state.errors.append("retrieval returned no candidates")
    trace(
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


def persistence_node(state: AgentState) -> AgentState:
    """Persist session memory and durable tool traces for this turn."""

    if not state.persist_artifacts:
        return state

    try:
        state.session_memory = update_session_memory_from_state(state)
        memory_path = save_session_memory(state.session_memory)
        trace(
            state,
            "memory_save",
            {"path": str(memory_path)},
            "session memory saved",
        )
    except Exception as exc:
        message = f"memory_save failed: {exc}"
        state.errors.append(message)
        trace(state, "memory_save", {}, message, status="error")

    try:
        trace_path = append_trace_events(
            session_id=state.session_id,
            turn_id=state.turn_id,
            traces=state.tool_traces,
        )
        trace(
            state,
            "trace_save",
            {"path": str(trace_path), "event_count": len(state.tool_traces)},
            "tool traces saved",
        )
    except Exception as exc:
        message = f"trace_save failed: {exc}"
        state.errors.append(message)
        trace(state, "trace_save", {}, message, status="error")
    return state
