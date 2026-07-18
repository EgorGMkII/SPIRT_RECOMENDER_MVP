"""Linear turn-based LangGraph runtime."""

import json
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig

from sommelier.agent.contracts import CatalogListOutput, FinalAnswerResult
from sommelier.agent.feedback import classify_feedback
from sommelier.agent.memory import ShownResult, TurnMemory, enforce_memory_limits
from sommelier.agent.profile import apply_profile_patch
from sommelier.agent.resolver import (
    resolve_turn,
    truncate_recent_messages,
    validate_turn_resolution,
)
from sommelier.agent.state import AgentState
from sommelier.agent.tool_agent import (
    execute_tool,
    required_cart_tool_completed,
    route_after_tool_agent,
    tool_calling_agent,
)
from sommelier.agent.tracer import ToolTrace
from sommelier.storage.session_repository import (
    SessionRepository,
    get_default_repository,
)

ANSWER_PROMPT = """You are an experienced rum sommelier and cocktail expert.
Write the actual user-facing answer, not a summary or a description of supplied
data. Answer naturally in the user's language with confident, practical expert
judgment. Explain why the recommendation fits the request using concrete
flavor, style, serve or recipe evidence.

Keep the voice warm, relaxed and conversational. For recommendations, make the
description vivid with two to four relevant sensory details grounded in the
current cards. Restrained marketing language may make verified qualities feel
appealing, but must never create unsupported properties, promises or occasions.

FIRST identify the action requested by the current user_message, then style the
answer for that action. Do not reuse recommendation phrasing for every turn:

1. Direct recommendation:
   recommend the best-fitting object and present it with appealing, restrained
   marketing language, for example "Конечно, думаю, вам подойдёт...". Explain
   its relevant taste and character. Do not sound like a database or ranking
   report.
2. Recipe / how-to / ingredients:
   the object has already been selected. Start directly with a phrase such as
   "Вот как приготовить Old Cuban:" and give the requested ingredients and
   preparation steps available in the current card. Do NOT say that the drink
   "вам подойдёт", do not recommend it again, and do not repeat the earlier
   sales pitch or full tasting description.
3. Explanation or clarification about a shown object:
   answer the specific question directly from its looked-up card. Do not
   re-introduce it as a new recommendation.
4. Food-pairing recommendation:
   name the pairing, explain the flavor interaction, and preserve any evidence
   caveat supplied by the food-search tool. Do not turn it into a generic
   product recommendation.
5. Profile update or smalltalk:
   for a profile-only update, briefly confirm exactly what was remembered and
   then offer to make a recommendation if the user wants one. Do not name,
   suggest or describe any catalog product until the user explicitly asks.
   Example: "Запомнил: вам нравятся мята и сладкие вкусы. Если хотите, могу
   подобрать ром или коктейль с таким профилем."
6. Cart action:
   after add_cart, briefly confirm the product id and resulting amount;
   after dellete_cart, briefly confirm removal;
   after show_cart, list exactly the id/amount pairs returned by the tool or
   clearly say that the cart is empty. Do not turn a cart confirmation into a
   recommendation and do not invent product names, prices or availability.
   Describe a cart change ONLY when the corresponding successful cart tool
   output is present. User intent, memory summaries and lookup_by_ids do not
   prove that the cart changed.
7. Insufficient recommendation context:
   when the user requests a recommendation but there are no current cards, no
   search tool result, and neither the request, profile nor saved context gives
   a meaningful selection criterion, ask exactly one concise clarification
   question covering at most two useful parameters, such as preferred flavor
   and occasion. Do not recommend anything, do not imply that a search ran,
   and return shown_refs=[].
   This rule does not apply when the user explicitly delegates the choice with
   wording such as "на свой вкус", "удиви меня" or "выбери сам".

8. Complete catalog listing:
   when request_scope="catalog_listing", reproduce every exact name returned
   by list_catalog, once and in the returned order. A short heading and a
   numbered or bulleted list are allowed. Do not rank, recommend, describe or
   add catalog facts. Return shown_refs=[] because browsing a complete
   name list is not a recommendation and does not load full cards. The normal
   five-object shown_refs limit does not apply to this listing mode.

After an actual recommendation or requested selection containing one or more
shown_refs, end with one brief, non-pushy request for feedback,
for example: "Подходит такое направление или сделать вариант суше/ярче?"
Do this only for a recommendation or selection. Do NOT request feedback after
a recipe, explanation, profile update, cart action, smalltalk, failed search or
clarification question.

Do not add secondary products merely to make the prose more colorful, and do
not expand shown_refs for objects not actually named in the answer.

Light humor is allowed only when it matches the user's tone or during
smalltalk. Never add jokes to recipes, error messages, cart actions or serious
constraints. Tell a bar story only when the user explicitly asks for one. If
the story is not supported by catalog evidence, clearly introduce it as a
fictional bar vignette; never present invented events involving BACARDÍ,
bartenders, venues or brand history as facts.

Do not begin with "Да", "Yes" or an equivalent unless the user actually asked
a yes/no question. Match the scope of the answer to the request. Give a recipe,
ingredient quantities or preparation steps ONLY when explicitly requested.
For a recommendation request, do not append an unsolicited recipe.

shown_refs is the ordered list of catalog objects explicitly named in the
answer. Include recommendations, requested alternatives and comparison objects
that the user can reasonably refer to later as "first", "second" or "them".
Exclude tool candidates that are not named in the final answer.

Use saved turn summaries for conversational references and only current cards
for catalog facts requiring full evidence. Treat negative_request as a hard
selection constraint whenever a card explicitly shows a conflict. For example,
do not recommend a cocktail containing sugar syrup as a clearly non-sweet
choice. If no supplied card safely satisfies the request, say so honestly
instead of forcing a recommendation.

recent_dialogue contains only the last three completed exchanges, with long
messages truncated. Use it solely for conversational continuity, corrections
and avoiding repetition. Priority is strict:
1. current turn_resolution;
2. current cards and tool_messages;
3. recent_dialogue.
Never recover the current target, catalog facts or shown_refs from
recent_dialogue when it conflicts with the current resolution or current
cards. An older assistant mistake is context to correct, not evidence to reuse.

Never say "the card says", "in the supplied data", "search result", "tool" or
other internal wording. Do not use Markdown bold or asterisks. Do not invent
facts. Keep every recipe attached to the correct cocktail. Except for
request_scope="catalog_listing", return at most five catalog objects.

Return FinalAnswerResult:
- answer: polished direct response to the user;
- shown_refs: catalog objects explicitly named in the answer, in first-mention
  order, up to five;
- assistant_summary: short factual memory summary, not the user-facing answer.
  No marketing language: state what the user asked, what was answered, which
  objects were named, and any important temporary constraints/profile/cart
  action."""

logger = logging.getLogger(__name__)

RECIPE_REQUEST_MARKERS = (
    "как приготовить",
    "приготовить",
    "рецепт",
    "ингредиент",
    "ingredients",
    "recipe",
    "how to make",
    "make it",
)

RECIPE_ANSWER_MARKERS = (
    "как приготовить",
    "ингредиенты",
    "приготовление",
    "вот как приготовить",
    "ingredients",
    "preparation",
)


def _configurable(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable", {})


def _llm(config: RunnableConfig | None, key: str) -> Any:
    configured = _configurable(config).get(key)
    if configured is not None:
        return configured
    from llm_module import get_langchain_openai_chat_model
    return get_langchain_openai_chat_model()


def _repository(config: RunnableConfig | None) -> SessionRepository:
    configured = _configurable(config).get("repository")
    return configured if configured is not None else get_default_repository()


def load_memory_and_profile(
    state: AgentState,
    config: RunnableConfig | None = None,
) -> dict:
    repository = _repository(config)
    return {
        "session_memory": repository.load_session_memory(state.session_id),
        "user_profile": repository.load_user_profile(state.session_id),
        "tool_traces": state.tool_traces + [
            ToolTrace(
                tool_name="load_memory_and_profile",
                output_summary="memory and profile loaded",
            )
        ],
    }


def resolve_turn_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    try:
        result = resolve_turn(
            user_request=state.user_request,
            memory=state.session_memory,
            profile=state.user_profile,
            llm=_llm(config, "resolver_llm"),
            recent_messages=_repository(config).load_recent_messages(
                state.session_id,
                limit=6,
            ),
        )
        return {
            "turn_resolution": result,
            "tool_traces": state.tool_traces + [
                ToolTrace(
                    tool_name="resolve_turn",
                    output_summary=f"follow_up={result.follow_up}",
                )
            ],
        }
    except Exception as exc:
        logger.exception(
            "Turn resolution failed for session=%s turn=%s",
            state.session_id,
            state.turn_id,
        )
        return {"errors": state.errors + [f"resolve_turn_failed:{type(exc).__name__}"]}


def validate_turn_resolution_node(state: AgentState) -> dict:
    if state.errors or state.turn_resolution is None:
        return {}
    try:
        validate_turn_resolution(
            state.turn_resolution, state.user_request, state.session_memory
        )
        return {}
    except Exception as exc:
        return {
            "errors": state.errors
            + [f"turn_resolution_validation_failed:{type(exc).__name__}"]
        }


def classify_feedback_node(
    state: AgentState,
    config: RunnableConfig | None = None,
) -> dict:
    """Classify analytics without affecting the main agent workflow."""

    try:
        follow_up = (
            state.turn_resolution.follow_up
            if state.turn_resolution is not None
            else False
        )
        result = classify_feedback(
            user_request=state.user_request,
            previous_assistant_answer=_repository(
                config
            ).load_last_assistant_message(state.session_id),
            follow_up=follow_up,
            llm=_llm(config, "feedback_llm"),
        )
        return {"feedback_result": result}
    except Exception:
        logger.exception(
            "Feedback classification failed for session=%s turn=%s",
            state.session_id,
            state.turn_id,
        )
        return {}


def apply_optional_profile_patch(state: AgentState) -> dict:
    if state.errors or state.turn_resolution is None:
        return {}
    patch = state.turn_resolution.profile_patch
    if patch is None:
        return {}
    return {
        "user_profile": apply_profile_patch(
            state.user_profile.model_copy(deep=True), patch
        )
    }


def _answer_llm(config: RunnableConfig | None) -> Any:
    return _llm(config, "answer_llm")


def _catalog_listing_from_messages(state: AgentState) -> CatalogListOutput | None:
    for message in reversed(state.messages):
        if (
            getattr(message, "type", "") != "tool"
            or getattr(message, "name", None) != "list_catalog"
        ):
            continue
        envelope = json.loads(getattr(message, "content", ""))
        if envelope.get("ok") is not True:
            return None
        return CatalogListOutput.model_validate(envelope.get("output", {}))
    return None


def _unsolicited_recipe_in_answer(state: AgentState, result: FinalAnswerResult) -> bool:
    if state.turn_resolution is None or state.turn_resolution.request_scope == "recipe":
        return False
    request = state.user_request.lower()
    if any(marker in request for marker in RECIPE_REQUEST_MARKERS):
        return False
    answer = result.answer.lower()
    return any(marker in answer for marker in RECIPE_ANSWER_MARKERS)


def generate_answer(state: AgentState, config: RunnableConfig | None = None) -> dict:
    if state.errors or state.turn_resolution is None:
        return {}
    if not required_cart_tool_completed(state):
        return {"errors": state.errors + ["required_cart_tool_missing"]}
    payload = {
        "user_request": state.user_request,
        "turn_resolution": state.turn_resolution.model_dump(mode="json"),
        "turns": [
            turn.model_dump(mode="json")
            for turn in state.session_memory.turns[-6:]
        ],
        "recent_dialogue": truncate_recent_messages(
            _repository(config).load_recent_messages(
                state.session_id,
                limit=6,
            )
        ),
        "cart": [
            item.model_dump(mode="json") for item in state.session_memory.cart
        ],
        "profile": state.user_profile.model_dump(mode="json"),
        "cards": [card.model_dump(mode="json") for card in state.cards],
        "tool_messages": [
            getattr(message, "content", "")
            for message in state.messages
            if getattr(message, "type", "") == "tool"
        ],
    }
    structured = _answer_llm(config).with_structured_output(
        FinalAnswerResult, method="function_calling"
    )
    known = {(card.kind, card.id) for card in state.cards}
    listing = _catalog_listing_from_messages(state)
    feedback = ""
    for _ in range(2):
        try:
            raw = structured.invoke(
                f"{ANSWER_PROMPT}\nINPUT:\n"
                f"{json.dumps(payload, ensure_ascii=False)}{feedback}"
            )
            result = FinalAnswerResult.model_validate(raw)
            if _unsolicited_recipe_in_answer(state, result):
                raise ValueError("do not include recipe unless explicitly requested")
            if state.turn_resolution.request_scope == "catalog_listing":
                if listing is None:
                    raise ValueError("catalog_listing requires successful list_catalog")
                if result.shown_refs:
                    raise ValueError("catalog_listing requires shown_refs=[]")
                missing = [
                    item.name for item in listing.items if item.name not in result.answer
                ]
                if missing:
                    raise ValueError(
                        "catalog listing omitted exact names: " + ", ".join(missing)
                    )
            if any((ref.kind, ref.id) not in known for ref in result.shown_refs):
                raise ValueError("shown_refs requires a current full card")
            expected_kind = {
                "product": "product",
                "food_pairing": "product",
                "cocktail": "cocktail",
                "recipe": "cocktail",
            }.get(state.turn_resolution.request_scope)
            if expected_kind and any(
                ref.kind != expected_kind for ref in result.shown_refs
            ):
                raise ValueError(
                    f"request_scope={state.turn_resolution.request_scope!r} "
                    f"requires shown_refs kind={expected_kind!r}"
                )
            return {
                "final_answer_result": result,
                "tool_traces": state.tool_traces + [
                    ToolTrace(
                        tool_name="generate_answer",
                        output_summary=f"shown_refs={len(result.shown_refs)}",
                    )
                ],
            }
        except Exception as exc:
            feedback = f"\nVALIDATION ERROR: {exc}. Correct the result."
    return {"errors": state.errors + ["final_answer_validation_failed"]}


def build_turn_memory(state: AgentState) -> dict:
    if state.errors or state.final_answer_result is None:
        return {}
    resolution = state.turn_resolution
    card_map = {(card.kind, card.id): card for card in state.cards}
    shown: list[ShownResult] = []
    for ref in state.final_answer_result.shown_refs:
        card = card_map[(ref.kind, ref.id)]
        summary = (
            getattr(card, "description", "")
            or getattr(card, "display_description", "")
            or card.name
        )[:800]
        shown.append(
            ShownResult(
                kind=ref.kind,
                id=ref.id,
                name=card.name,
                summary=summary,
            )
        )
    memory = state.session_memory.model_copy(deep=True)
    memory.turns.append(
        TurnMemory(
            follow_up=resolution.follow_up,
            request_scope=resolution.request_scope,
            user_request=state.user_request,
            initial_request=resolution.initial_request,
            effective_request=resolution.effective_request,
            negative_request=resolution.negative_request,
            assistant_summary=state.final_answer_result.assistant_summary,
            shown_results=shown,
        )
    )
    return {"session_memory": enforce_memory_limits(memory)}


def persist(state: AgentState, config: RunnableConfig | None = None) -> dict:
    if state.errors or state.final_answer_result is None:
        return {}
    if not _configurable(config).get("persist", True):
        return {}
    try:
        traces = state.tool_traces + [
            ToolTrace(
                tool_name="persist",
                output_summary="turn, transcript, profile and traces saved",
            )
        ]
        _repository(config).persist_successful_turn(
            session_id=state.session_id,
            turn_id=state.turn_id,
            memory=state.session_memory,
            profile=state.user_profile,
            user_message=state.user_request,
            assistant_message=state.final_answer_result.answer,
            traces=traces,
        )
        return {"tool_traces": traces}
    except Exception as exc:
        return {"errors": state.errors + [f"persist_failed:{type(exc).__name__}"]}


def persist_feedback(
    state: AgentState,
    config: RunnableConfig | None = None,
) -> dict:
    """Persist analytics independently without changing the main result."""

    if (
        state.feedback_result is None
        or not _configurable(config).get("persist", True)
    ):
        return {}
    main_persist_succeeded = not state.errors and any(
        trace.tool_name == "persist" and trace.status == "success"
        for trace in state.tool_traces
    )
    try:
        _repository(config).save_feedback_event(
            turn_id=state.turn_id,
            session_id=state.session_id,
            user_request=state.user_request,
            follow_up=(
                state.turn_resolution.follow_up
                if state.turn_resolution is not None
                else False
            ),
            feedback=state.feedback_result.feedback,
            turn_success=main_persist_succeeded,
        )
    except Exception:
        logger.exception(
            "Feedback persistence failed for session=%s turn=%s",
            state.session_id,
            state.turn_id,
        )
    return {}


def safe_error(state: AgentState) -> dict:
    return {
        "final_answer_result": FinalAnswerResult(
            answer="Не удалось надёжно обработать запрос. Попробуйте ещё раз.",
            shown_refs=[],
            assistant_summary="Ход завершён с ошибкой без сохранения.",
        )
    }


def route_after_resolution(state: AgentState) -> str:
    return "safe_error" if state.errors else "tool_calling_agent"


def route_after_answer(state: AgentState) -> str:
    return (
        "safe_error"
        if state.errors or state.final_answer_result is None
        else "build_turn_memory"
    )


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load_memory_and_profile", load_memory_and_profile)
    graph.add_node("resolve_turn", resolve_turn_node)
    graph.add_node("validate_turn_resolution", validate_turn_resolution_node)
    graph.add_node("classify_feedback", classify_feedback_node)
    graph.add_node("apply_optional_profile_patch", apply_optional_profile_patch)
    graph.add_node("tool_calling_agent", tool_calling_agent)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("build_turn_memory", build_turn_memory)
    graph.add_node("persist", persist)
    graph.add_node("persist_feedback", persist_feedback)
    graph.add_node("safe_error", safe_error)

    graph.add_edge(START, "load_memory_and_profile")
    graph.add_edge("load_memory_and_profile", "resolve_turn")
    graph.add_edge("resolve_turn", "validate_turn_resolution")
    graph.add_edge("validate_turn_resolution", "classify_feedback")
    graph.add_edge("classify_feedback", "apply_optional_profile_patch")
    graph.add_conditional_edges("apply_optional_profile_patch", route_after_resolution)
    graph.add_conditional_edges("tool_calling_agent", route_after_tool_agent)
    graph.add_edge("execute_tool", "tool_calling_agent")
    graph.add_conditional_edges("generate_answer", route_after_answer)
    graph.add_edge("build_turn_memory", "persist")
    graph.add_edge("persist", "persist_feedback")
    graph.add_edge("safe_error", "persist_feedback")
    graph.add_edge("persist_feedback", END)
    return graph.compile()


AGENT_GRAPH = build_graph()


def run_agent_turn(
    state: AgentState,
    config: RunnableConfig | None = None,
) -> AgentState:
    result = AGENT_GRAPH.invoke(
        state,
        config={
            "recursion_limit": 16,
            "configurable": _configurable(config),
        },
    )
    return result if isinstance(result, AgentState) else AgentState.model_validate(result)
