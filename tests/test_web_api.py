from pathlib import Path
import shutil
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from sommelier.agent.contracts import FinalAnswerResult, TurnResolution
from sommelier.agent.memory import CartItem, SessionMemory
from sommelier.agent.profile import UserProfile
from sommelier.agent.state import AgentState
from sommelier.storage.session_repository import SessionRepository
from sommelier.web.app import create_app


@pytest.fixture
def repository() -> SessionRepository:
    directory = Path(".test_tmp") / f"web-{uuid4().hex}"
    try:
        yield SessionRepository(directory / "sommelier.sqlite3")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_chat_endpoint_returns_follow_up_contract(
    monkeypatch, repository: SessionRepository
) -> None:
    def fake_run_agent_turn(state: AgentState, config=None) -> AgentState:
        assert config["configurable"]["repository"] is repository
        state.final_answer_result = FinalAnswerResult(
            answer="Try a Mojito.",
            assistant_summary="Suggested Mojito.",
        )
        state.turn_resolution = TurnResolution(
            follow_up=True,
            request_scope="recipe",
            initial_request="fresh cocktail",
            effective_request="Mojito recipe",
        )
        state.user_profile = UserProfile(session_id=state.session_id)
        return state

    monkeypatch.setattr("sommelier.web.api.run_agent_turn", fake_run_agent_turn)
    client = TestClient(create_app(repository))
    response = client.post(
        "/api/chat",
        json={"session_id": "web-test", "message": "How do I make the first?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["follow_up"] is True
    assert payload["request_scope"] == "recipe"
    assert payload["effective_request"] == "Mojito recipe"
    assert "relation" not in payload


def test_catalog_status_endpoint(repository: SessionRepository) -> None:
    client = TestClient(create_app(repository))
    response = client.get("/api/catalog/status")
    assert response.status_code == 200
    assert "product_profiles_count" in response.json()


def test_debug_endpoints_return_empty_v4_memory(
    repository: SessionRepository,
) -> None:
    client = TestClient(create_app(repository))
    response = client.get("/api/sessions/web-debug-v4")
    assert response.status_code == 200
    assert response.json()["memory"]["schema_version"] == 4
    assert response.json()["memory"]["turns"] == []


def test_chat_history_returns_empty_or_ordered_messages(
    repository: SessionRepository,
) -> None:
    client = TestClient(create_app(repository))
    empty = client.get("/api/sessions/unknown/messages")
    assert empty.status_code == 200
    assert empty.json() == {"session_id": "unknown", "messages": []}

    repository.persist_successful_turn(
        session_id="web-history",
        turn_id="turn-1",
        memory=SessionMemory(session_id="web-history"),
        profile=UserProfile(session_id="web-history"),
        user_message="Question",
        assistant_message="Answer",
        traces=[],
    )
    response = client.get("/api/sessions/web-history/messages")
    assert response.status_code == 200
    assert response.json()["messages"] == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]


def test_reset_session_deletes_all_durable_user_state(
    repository: SessionRepository,
) -> None:
    session_id = "web-reset"
    repository.persist_successful_turn(
        session_id=session_id,
        turn_id="turn-reset",
        memory=SessionMemory(
            session_id=session_id,
            cart=[CartItem(id="bacardi-reserva-ocho-rum", amount=2)],
        ),
        profile=UserProfile(session_id=session_id, liked_flavors=["oak"]),
        user_message="Question",
        assistant_message="Answer",
        traces=[],
    )
    repository.save_feedback_event(
        turn_id="feedback-reset",
        session_id=session_id,
        user_request="Not useful",
        follow_up=True,
        feedback="negative_feedback",
        turn_success=True,
    )
    client = TestClient(create_app(repository))

    response = client.delete(f"/api/sessions/{session_id}")

    assert response.status_code == 204
    assert repository.load_messages(session_id) == []
    assert repository.load_session_memory(session_id).cart == []
    assert repository.load_user_profile(session_id).liked_flavors == []
    assert repository.load_trace_events(session_id) == []
    assert repository.load_feedback_stats(session_id)["total"] == 0


def test_frontend_restores_history_before_enabling_form() -> None:
    script = Path("sommelier/web/static/chat.js").read_text(encoding="utf-8")
    assert "localStorage.getItem(sessionIdKey)" in script
    assert "async function loadChatHistory()" in script
    assert "/messages`" in script
    assert "messages.replaceChildren()" in script
    assert "setFormEnabled(false)" in script
    assert "setFormEnabled(true)" in script
    assert "Меня зовут Бакард-ИИ" in script
    assert "Чем я могу быть полезен" in script
    assert "Как тебя зовут? Что хотелось бы сегодня?" in script
    assert 'method: "DELETE"' in script
    assert "Очистить весь диалог, память, профиль и корзину?" in script


def test_feedback_analytics_endpoint(
    repository: SessionRepository,
) -> None:
    repository.save_feedback_event(
        turn_id="feedback-1",
        session_id="analytics",
        user_request="Ответ не тот",
        follow_up=True,
        feedback="negative_feedback",
        turn_success=False,
    )
    client = TestClient(create_app(repository))

    response = client.get(
        "/api/analytics/feedback",
        params={"session_id": "analytics"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "analytics",
        "total": 1,
        "neutral": 0,
        "purchase_intent": 0,
        "negative_feedback": 1,
        "successful_turns": 0,
        "failed_turns": 1,
    }


def test_app_warms_llm_once_when_keepalive_is_enabled(
    repository: SessionRepository,
) -> None:
    calls: list[str] = []

    class FakeModel:
        def invoke(self, prompt: str):
            calls.append(prompt)
            return "OK"

    app = create_app(
        repository,
        keepalive_seconds=3600,
        keepalive_model_factory=FakeModel,
    )
    with TestClient(app) as client:
        assert client.get("/api/catalog/status").status_code == 200
        assert app.state.llm_warmup_succeeded is True

    assert calls == ["Reply with exactly: OK"]
