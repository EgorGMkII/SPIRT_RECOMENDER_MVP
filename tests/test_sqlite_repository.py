from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from sommelier.agent.memory import CartItem, SessionMemory
from sommelier.agent.profile import UserProfile
from sommelier.agent.tracer import ToolTrace
from sommelier.storage.database import connect
from sommelier.storage.session_repository import SessionRepository


@pytest.fixture
def repository() -> SessionRepository:
    directory = Path(".test_tmp") / f"sqlite-{uuid4().hex}"
    try:
        yield SessionRepository(directory / "sommelier.sqlite3")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_repository_creates_schema_and_returns_empty_state(
    repository: SessionRepository,
) -> None:
    assert repository.load_session_memory("new") == SessionMemory(session_id="new")
    assert repository.load_user_profile("new") == UserProfile(session_id="new")
    assert repository.load_messages("new") == []
    assert repository.load_trace_events("new") == []
    assert repository.load_last_assistant_message("new") is None
    assert repository.load_feedback_stats() == {
        "total": 0,
        "neutral": 0,
        "purchase_intent": 0,
        "negative_feedback": 0,
        "successful_turns": 0,
        "failed_turns": 0,
    }

    with connect(repository.db_path) as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"sessions", "messages", "traces", "feedback_events"} <= tables


def test_database_path_can_be_configured_for_deployment(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "SOMMELIER_DB_PATH", "/app/runtime/sommelier.sqlite3"
    )
    import importlib
    import sommelier.storage.database as database

    reloaded = importlib.reload(database)
    assert reloaded.DEFAULT_DB_PATH == Path("/app/runtime/sommelier.sqlite3")
    monkeypatch.delenv("SOMMELIER_DB_PATH")
    importlib.reload(database)


def test_successful_turn_round_trips_state_messages_and_traces(
    repository: SessionRepository,
) -> None:
    memory = SessionMemory(
        session_id="s",
        cart=[CartItem(id="bacardi-reserva-ocho-rum", amount=2)],
    )
    profile = UserProfile(session_id="s", liked_flavors=["oak"])
    trace = ToolTrace(tool_name="show_cart", output_summary="cart_items=1")

    repository.persist_successful_turn(
        session_id="s",
        turn_id="turn-1",
        memory=memory,
        profile=profile,
        user_message="Покажи корзину",
        assistant_message="В корзине две бутылки.",
        traces=[trace],
    )

    assert repository.load_session_memory("s") == memory
    assert repository.load_user_profile("s") == profile
    assert repository.load_messages("s") == [
        {"role": "user", "content": "Покажи корзину"},
        {"role": "assistant", "content": "В корзине две бутылки."},
    ]
    loaded_traces = repository.load_trace_events("s")
    assert loaded_traces[0]["turn_id"] == "turn-1"
    assert loaded_traces[0]["tool_name"] == "show_cart"
    assert (
        repository.load_last_assistant_message("s")
        == "В корзине две бутылки."
    )
    assert repository.load_recent_messages("s", limit=1) == [
        {"role": "assistant", "content": "В корзине две бутылки."}
    ]


def test_repeated_turn_id_is_idempotent(
    repository: SessionRepository,
) -> None:
    kwargs = dict(
        session_id="s",
        turn_id="same-turn",
        memory=SessionMemory(session_id="s"),
        profile=UserProfile(session_id="s"),
        user_message="hello",
        assistant_message="hi",
        traces=[],
    )
    repository.persist_successful_turn(**kwargs)
    repository.persist_successful_turn(**kwargs)

    assert len(repository.load_messages("s")) == 2
    with connect(repository.db_path) as connection:
        version = connection.execute(
            "SELECT version FROM sessions WHERE session_id = 's'"
        ).fetchone()["version"]
    assert version == 1


def test_transaction_rolls_back_everything_on_trace_failure(
    repository: SessionRepository, monkeypatch
) -> None:

    def fail(*args, **kwargs):
        raise RuntimeError("trace insert failed")

    monkeypatch.setattr(repository, "_insert_traces", fail)
    with pytest.raises(RuntimeError, match="trace insert failed"):
        repository.persist_successful_turn(
            session_id="s",
            turn_id="turn-1",
            memory=SessionMemory(
                session_id="s",
                cart=[CartItem(id="bacardi-reserva-ocho-rum", amount=1)],
            ),
            profile=UserProfile(session_id="s", liked_flavors=["oak"]),
            user_message="add",
            assistant_message="added",
            traces=[],
        )

    assert repository.load_session_memory("s") == SessionMemory(session_id="s")
    assert repository.load_messages("s") == []


def test_feedback_events_are_independent_idempotent_and_aggregated(
    repository: SessionRepository,
) -> None:
    repository.save_feedback_event(
        turn_id="success",
        session_id="s",
        user_request="Хочу купить этот ром",
        follow_up=False,
        feedback="purchase_intent",
        turn_success=True,
    )
    repository.save_feedback_event(
        turn_id="failed",
        session_id="s",
        user_request="Ответ не тот",
        follow_up=True,
        feedback="negative_feedback",
        turn_success=False,
    )
    repository.save_feedback_event(
        turn_id="failed",
        session_id="s",
        user_request="duplicate",
        follow_up=False,
        feedback="neutral",
        turn_success=True,
    )
    repository.save_feedback_event(
        turn_id="other",
        session_id="other",
        user_request="Мне не нравится сладкий ром",
        follow_up=False,
        feedback="neutral",
        turn_success=True,
    )

    assert repository.load_feedback_stats("s") == {
        "total": 2,
        "neutral": 0,
        "purchase_intent": 1,
        "negative_feedback": 1,
        "successful_turns": 1,
        "failed_turns": 1,
    }
    assert repository.load_feedback_stats()["total"] == 3

    repository.delete_session("s")
    assert repository.load_feedback_stats("s")["total"] == 0
