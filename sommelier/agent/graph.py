"""Controlled LangGraph workflow assembly for the sommelier agent."""

from __future__ import annotations

from typing import Literal

from sommelier.agent.answering import (
    answer_generation_node,
    build_answer_context,
    build_cocktail_answer_context,
    build_cocktail_answer_prompt,
    build_sommelier_answer_prompt,
    generate_cocktail_answer,
    generate_sommelier_answer,
)
from sommelier.agent.nodes import (
    cocktail_search_node,
    direct_search_node,
    food_pairing_search_node,
    memory_load_node,
    parse_intent_node,
    persistence_node,
    profile_ack_node,
    profile_update_node,
    retrieval_node,
)
from sommelier.agent.schemas import IntentType
from sommelier.agent.search_runtime import (
    DEFAULT_BM25_TOP_K,
    DEFAULT_COCKTAIL_TOP_K,
    DEFAULT_FAISS_TOP_K,
    DEFAULT_INDEX_DIR,
    DEFAULT_PROFILES_DIR,
    hybrid_search,
    load_agent_bm25_index,
    load_agent_index,
)
from sommelier.agent.state import AgentState


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
