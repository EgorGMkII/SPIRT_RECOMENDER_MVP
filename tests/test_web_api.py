from pathlib import Path

from fastapi.testclient import TestClient

from sommelier.agent.profile import UserProfile
from sommelier.agent.schemas import IntentType, ParsedIntent
from sommelier.agent.state import AgentState
from sommelier.agent.tracer import ToolTrace
from sommelier.retrieval.schemas import SearchRequest
from sommelier.web.app import create_app


def test_chat_endpoint_returns_agent_payload(monkeypatch) -> None:
    def fake_run_agent_turn(state: AgentState) -> AgentState:
        assert state.use_llm_followup is True
        assert state.use_llm_intent is True
        assert state.use_llm_query is True
        assert state.use_llm_food_query is True
        assert state.use_llm_profile_update is True
        assert state.use_llm_answer is True
        state.final_answer = "Try BACARDI Anejo Cuatro."
        state.effective_user_message = state.user_message
        state.parsed_intent = ParsedIntent(
            intent=IntentType.SEARCH_PRODUCTS,
            search=SearchRequest(query=state.user_message),
            confidence=0.9,
        )
        state.user_profile = UserProfile(session_id=state.session_id, liked_flavors=["vanilla"])
        state.tool_traces.append(
            ToolTrace(
                tool_name="parse_intent",
                input={"message": state.user_message},
                output_summary="intent=search_products",
            )
        )
        return state

    monkeypatch.setattr("sommelier.web.api.run_agent_turn", fake_run_agent_turn)
    client = TestClient(create_app())

    response = client.post(
        "/api/chat",
        json={"session_id": "web-test", "message": "I want vanilla rum"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "web-test"
    assert payload["answer"] == "Try BACARDI Anejo Cuatro."
    assert payload["intent"] == "search_products"
    assert payload["profile"]["liked_flavors"] == ["vanilla"]
    assert payload["traces"][0]["tool_name"] == "parse_intent"


def test_catalog_status_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("sommelier.web.api.PRODUCT_PROFILES_DIR", Path("data/catalog/search_profiles"))
    client = TestClient(create_app())

    response = client.get("/api/catalog/status")

    assert response.status_code == 200
    payload = response.json()
    assert "product_profiles_count" in payload
    assert "cocktail_profiles_count" in payload
    assert "faiss_metadata_exists" in payload


def test_debug_endpoints_return_empty_defaults_for_new_session() -> None:
    client = TestClient(create_app())

    session_response = client.get("/api/sessions/web-debug-session")
    trace_response = client.get("/api/traces/web-debug-session")
    profile_response = client.get("/api/profiles/web-debug-session")

    assert session_response.status_code == 200
    assert session_response.json()["memory"]["session_id"] == "web-debug-session"
    assert trace_response.status_code == 200
    assert trace_response.json()["traces"] == []
    assert profile_response.status_code == 200
    assert profile_response.json()["profile"]["session_id"] == "web-debug-session"
