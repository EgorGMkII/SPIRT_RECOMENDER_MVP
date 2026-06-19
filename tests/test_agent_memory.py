import shutil
from pathlib import Path
from uuid import uuid4

from sommelier.agent.graph import memory_load_node, parse_intent_node
from sommelier.agent.memory import (
    CandidateMemory,
    LastTurnMemory,
    SessionMemory,
    build_followup_resolution_prompt,
    resolve_followup_context,
)
from sommelier.agent.memory_store import load_session_memory, save_session_memory
from sommelier.agent.schemas import IntentType, ParsedIntent
from sommelier.agent.state import AgentState
from sommelier.agent.trace_store import append_trace_events, load_trace_events
from sommelier.agent.tracer import ToolTrace
from sommelier.retrieval.schemas import SearchRequest


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeFollowupLLM:
    def invoke(self, prompt: str) -> FakeMessage:
        assert "Previous turn" in prompt
        assert "Mojito" in prompt
        return FakeMessage(
            '{"is_followup":true,'
            '"intent":"cocktail_expansion",'
            '"effective_user_message":"Find a simpler cocktail similar to Mojito with lime and mint.",'
            '"reason":"The user asks for a simpler similar option after a Mojito recommendation."}'
        )


def _cocktail_memory() -> SessionMemory:
    return SessionMemory(
        session_id="s1",
        last_turn=LastTurnMemory(
            user_message="какой коктейль сделать с лаймом и мятой?",
            effective_user_message="какой коктейль сделать с лаймом и мятой?",
            intent="cocktail_expansion",
            cocktail_query="lime mint rum cocktail",
            candidates=[
                CandidateMemory(
                    item_id="mojito",
                    name="Mojito",
                    kind="cocktail",
                    score=3.2,
                )
            ],
        ),
    )


def test_followup_resolution_prompt_contains_previous_turn_context() -> None:
    prompt = build_followup_resolution_prompt(
        "а есть что-то похожее, но проще?",
        _cocktail_memory(),
    )

    assert "Previous turn" in prompt
    assert "какой коктейль сделать с лаймом и мятой?" in prompt
    assert "Mojito" in prompt
    assert "а есть что-то похожее, но проще?" in prompt


def test_followup_resolution_prompt_warns_against_sticky_topic_reuse() -> None:
    prompt = build_followup_resolution_prompt(
        "хочу ром с дубом, ванилью и пряностями",
        _cocktail_memory(),
    )

    assert "complete standalone request" in prompt
    assert "хочу ром с дубом, ванилью и пряностями" in prompt
    assert "not a follow-up, search_products" in prompt


def test_resolve_followup_context_uses_llm_structured_output() -> None:
    resolution = resolve_followup_context(
        "а есть что-то похожее, но проще?",
        _cocktail_memory(),
        llm=FakeFollowupLLM(),
        use_llm=True,
    )

    assert resolution.is_followup is True
    assert resolution.intent == IntentType.COCKTAIL_EXPANSION
    assert "simpler cocktail" in resolution.effective_user_message
    assert "Mojito" in resolution.reason


def test_memory_load_node_uses_followup_resolver(monkeypatch) -> None:
    monkeypatch.setattr(
        "sommelier.agent.graph.load_session_memory",
        lambda session_id: _cocktail_memory(),
    )
    monkeypatch.setattr(
        "sommelier.agent.graph.resolve_followup_context",
        lambda message, memory, use_llm=False: type(
            "Resolution",
            (),
            {
                "is_followup": True,
                "intent": IntentType.COCKTAIL_EXPANSION,
                "effective_user_message": "Find a simpler cocktail similar to Mojito.",
                "reason": "Follow-up to cocktail recommendation.",
            },
        )(),
    )
    state = memory_load_node(
        AgentState(
            session_id="s1",
            user_message="а есть что-то похожее, но проще?",
            use_llm_followup=True,
        )
    )

    assert state.is_followup is True
    assert state.followup_intent == IntentType.COCKTAIL_EXPANSION
    assert state.effective_user_message == "Find a simpler cocktail similar to Mojito."


def test_parse_intent_node_uses_resolved_followup_intent(monkeypatch) -> None:
    monkeypatch.setattr(
        "sommelier.agent.graph.parse_intent",
        lambda message, use_llm=False: ParsedIntent(
            intent=IntentType.SEARCH_PRODUCTS,
            search=SearchRequest(query=message),
            confidence=0.95,
        ),
    )
    state = AgentState(
        session_id="s1",
        user_message="а есть что-то похожее, но проще?",
        effective_user_message="Find a simpler cocktail similar to Mojito.",
        is_followup=True,
        followup_intent=IntentType.COCKTAIL_EXPANSION,
        persist_artifacts=False,
    )

    state = parse_intent_node(state)

    assert state.parsed_intent is not None
    assert state.parsed_intent.intent == IntentType.COCKTAIL_EXPANSION
    assert state.parsed_intent.search.query == "Find a simpler cocktail similar to Mojito."


def test_session_memory_store_roundtrip() -> None:
    session_dir = Path(".test_tmp") / f"sessions-{uuid4().hex}"
    memory = SessionMemory(
        session_id="Session 1",
        last_turn=LastTurnMemory(
            user_message="I want vanilla rum",
            effective_user_message="I want vanilla rum",
            intent="search_products",
            search_query="Rum with vanilla flavors.",
        ),
    )

    try:
        save_session_memory(memory, session_dir=session_dir)
        loaded = load_session_memory("Session 1", session_dir=session_dir)

        assert loaded == memory
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


def test_trace_store_writes_jsonl_events() -> None:
    trace_dir = Path(".test_tmp") / f"traces-{uuid4().hex}"
    traces = [
        ToolTrace(
            tool_name="parse_intent",
            input={"message": "hello"},
            output_summary="intent=search_products",
        )
    ]

    try:
        path = append_trace_events(
            session_id="Session 1",
            turn_id="turn-1",
            traces=traces,
            trace_dir=trace_dir,
        )
        events = load_trace_events("Session 1", trace_dir=trace_dir)

        assert path.exists()
        assert len(events) == 1
        assert events[0]["session_id"] == "Session 1"
        assert events[0]["turn_id"] == "turn-1"
        assert events[0]["tool_name"] == "parse_intent"
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)
