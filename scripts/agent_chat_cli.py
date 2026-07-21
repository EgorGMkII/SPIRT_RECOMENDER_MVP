"""CLI harness for adaptive real-LLM conversations with the sommelier agent."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from pprint import pprint

from langchain_core.messages import AIMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sommelier.agent.graph import run_agent_turn
from sommelier.agent.state import AgentState
from sommelier.storage.session_repository import SessionRepository


DEFAULT_SESSION_ID = "cli-agent-chat"
DEFAULT_DB_PATH = PROJECT_ROOT / ".test_tmp" / "agent_chat_cli.sqlite3"
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY",
)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def clear_dead_local_proxy() -> None:
    for key in PROXY_ENV_KEYS:
        value = os.environ.get(key, "")
        if value.startswith("http://127.0.0.1:9"):
            os.environ.pop(key, None)


def require_env() -> bool:
    missing = [
        key
        for key in ("OPENAI_API_KEY", "HYDRA_BASE_URL", "OPENAI_MODEL")
        if not os.environ.get(key)
    ]
    if missing:
        print("Missing required env keys:", ", ".join(missing))
        return False
    return True


def tool_events(state: AgentState) -> list[dict[str, object]]:
    return [
        {
            "tool": trace.input.get("tool"),
            "args": trace.input.get("args", {}),
            "status": trace.status,
            "summary": trace.output_summary,
        }
        for trace in state.tool_traces
        if trace.tool_name == "tool_call"
    ]


def non_tool_events(state: AgentState) -> list[str]:
    return [
        f"{trace.tool_name}: {trace.status} — {trace.output_summary}"
        for trace in state.tool_traces
        if trace.tool_name != "tool_call"
    ]


def tool_agent_text_messages(state: AgentState) -> list[str]:
    texts: list[str] = []
    for message in state.messages:
        if not isinstance(message, AIMessage):
            continue
        if getattr(message, "tool_calls", None):
            continue
        content = str(message.content or "").strip()
        if content:
            texts.append(content[:500])
    return texts


def print_turn_debug(state: AgentState, *, answer_chars: int) -> None:
    resolution = state.turn_resolution
    answer = state.final_answer_result
    print("\nASSISTANT:")
    print((answer.answer if answer else "")[:answer_chars])
    print("\nDEBUG:")
    print("  answer_mode:", state.answer_mode)
    print("  follow_up:", None if resolution is None else resolution.follow_up)
    print("  scope:", None if resolution is None else resolution.request_scope)
    print(
        "  initial_request:",
        None if resolution is None else resolution.initial_request,
    )
    print(
        "  effective_request:",
        None if resolution is None else resolution.effective_request,
    )
    print(
        "  negative_request:",
        None if resolution is None else resolution.negative_request,
    )
    print("  tools:")
    events = tool_events(state)
    if not events:
        print("    (none)")
    for event in events:
        print(
            "   ",
            f"{event['tool']}({event['args']}) -> "
            f"{event['status']}: {event['summary']}",
        )
    print("  trace:")
    traces = non_tool_events(state)
    if not traces:
        print("    (none)")
    for trace in traces:
        print("   ", trace)
    no_tool_messages = tool_agent_text_messages(state)
    print("  tool_agent_text:")
    if not no_tool_messages:
        print("    (none)")
    for message in no_tool_messages:
        print("   ", message)
    shown_refs = (
        [ref.model_dump(mode="json") for ref in answer.shown_refs]
        if answer
        else []
    )
    print("  shown_refs:", shown_refs)
    print("  full_cards:", [(card.kind, card.id) for card in state.cards])
    print("  errors:", state.errors)


def ask(
    repository: SessionRepository,
    *,
    session_id: str,
    message: str,
    answer_chars: int,
) -> AgentState:
    print("\nUSER:")
    print(message)
    state = run_agent_turn(
        AgentState(session_id=session_id, user_request=message),
        config={"configurable": {"repository": repository}},
    )
    print_turn_debug(state, answer_chars=answer_chars)
    return state


def show_memory(repository: SessionRepository, session_id: str) -> None:
    memory = repository.load_session_memory(session_id)
    profile = repository.load_user_profile(session_id)
    print("\nSESSION MEMORY:")
    pprint(memory.model_dump(mode="json"))
    print("\nUSER PROFILE:")
    pprint(profile.model_dump(mode="json"))


def show_messages(repository: SessionRepository, session_id: str) -> None:
    print("\nMESSAGES:")
    pprint(repository.load_messages(session_id))


def interactive(
    repository: SessionRepository,
    *,
    session_id: str,
    answer_chars: int,
) -> int:
    print(f"Interactive session: {session_id}")
    print("Commands: /exit, /reset, /memory, /messages")
    while True:
        try:
            message = input("\nYOU> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message in {"/exit", "/quit"}:
            return 0
        if message == "/reset":
            repository.delete_session(session_id)
            print(f"Cleared session: {session_id}")
            continue
        if message == "/memory":
            show_memory(repository, session_id)
            continue
        if message == "/messages":
            show_messages(repository, session_id)
            continue
        ask(
            repository,
            session_id=session_id,
            message=message,
            answer_chars=answer_chars,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=DEFAULT_SESSION_ID)
    parser.add_argument("--message", help="Send one message and exit.")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--show-memory", action="store_true")
    parser.add_argument("--show-messages", action="store_true")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--answer-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    clear_dead_local_proxy()
    if not require_env():
        return 2

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    repository = SessionRepository(args.db_path)
    print("DB:", args.db_path)
    print("SESSION:", args.session)
    print("MODEL:", os.environ["OPENAI_MODEL"])
    print("BASE_URL:", os.environ["HYDRA_BASE_URL"])

    if args.reset:
        repository.delete_session(args.session)
        print(f"Cleared session: {args.session}")
    if args.show_memory:
        show_memory(repository, args.session)
    if args.show_messages:
        show_messages(repository, args.session)
    if args.message:
        ask(
            repository,
            session_id=args.session,
            message=args.message,
            answer_chars=args.answer_chars,
        )
    if args.interactive or not any(
        (args.message, args.reset, args.show_memory, args.show_messages)
    ):
        return interactive(
            repository,
            session_id=args.session,
            answer_chars=args.answer_chars,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
