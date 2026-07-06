"""Bounded native tool-calling loop for compact turn memory."""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from sommelier.agent.contracts import (
    CatalogListOutput,
    CocktailCandidate,
    CocktailSearchOutput,
    ProductCandidate,
    ProductSearchOutput,
)
from sommelier.agent.cart_tools import (
    AddCartInput,
    CART_TOOL_MAP,
    CartOutput,
    DeleteCartInput,
    ShowCartInput,
)
from sommelier.agent.memory import CartItem
from sommelier.agent.search_tools import AGENT_TOOLS, TOOL_MAP
from sommelier.agent.state import AgentState
from sommelier.agent.tracer import ToolTrace

logger = logging.getLogger(__name__)

TOOL_PROMPT = """Decide whether a tool is needed.
Catalog tools: search_products, search_products_for_food, search_cocktails,
lookup_by_id, list_catalog.
Cart tools: add_cart, dellete_cart, show_cart.
Identify the requested action before choosing a tool:
- turn_resolution.request_scope is the authoritative description of the
  CURRENT action. Saved turns provide context but must not replace that scope;
- request_scope="cocktail" with an exact shown rum name in effective_request
  is already a complete request: call search_cocktails now. Do not answer that
  cocktails could be selected later and do not ask for an extra preference
  before performing the requested search;
- request_scope="catalog_listing" means the user explicitly wants the complete
  catalog of names. Call list_catalog exactly once with kind="product" for
  rums/products or kind="cocktail" for cocktails. Do not use a ranked search
  and do not load full cards;
- a new recommendation normally requires the matching search tool;
- search_products_for_food is ONLY for pairing rum with edible food: a dish,
  ingredient, meal or cuisine;
- a named cocktail is NOT food. Choosing a rum to make Old Fashioned, Daiquiri,
  Mojito or any other cocktail uses search_products, not
  search_products_for_food;
- a recipe, ingredients, preparation method or factual explanation about a
  previously shown object uses lookup_by_id;
- "add to cart" uses add_cart with the exact product id and desired amount;
- "remove/delete from cart" uses dellete_cart with the exact product id;
- "show/what is in cart" uses show_cart;
- cart tools are for product ids, never cocktail ids;
- if a product to add has not been shown and its exact id is unavailable, first
  call search_products, then call add_cart using a returned id;
- turn_resolution.cart_action is mandatory: add -> add_cart,
  delete -> dellete_cart, show -> show_cart. Do not finish the tool loop until
  that exact cart tool has returned success. lookup_by_id never changes cart;
- profile-only updates and smalltalk need no catalog tool.
  A statement like "мне нравится мята и сладкое" is NOT an implicit request
  for a recommendation. Do not search merely because flavor words are present.
- before a NEW recommendation search, check whether the current request,
  UserProfile and saved turn context together contain at least one meaningful
  selection criterion: flavor, ingredient, food, cocktail, occasion, mood,
  serving style or product type;
- if none of those criteria is available, do not search. Finish without a tool
  call so the final answer can ask one concise clarification question. Do not
  invent a search query or silently choose preferences for the user;
- explicit permission for free choice, such as "на свой вкус", "удиви меня",
  "выбери сам", "surprise me" or "you choose", is sufficient context and must
  proceed to the appropriate recommendation search;
- never require this clarification before a recipe, factual explanation,
  lookup of a shown object, profile update, smalltalk or cart action;
- when at least one meaningful criterion is present, preserve the normal tool
  selection rules above and search without asking unnecessary questions.
Use positive effective_request for search; never put negative_request into a
query. After a search, inspect returned cards against negative_request. If the
best cards explicitly conflict, use the remaining tool call for a better
positive search or finish without forcing a recommendation.
Resolve "first/second/last" from the ordered shown_results inside saved turns.
Use lookup_by_id to obtain full details for a previously shown object.
Emit at most one tool call per message and at most two calls per user request.
Do not write the final user-facing answer."""

SEARCH_DEDUP_TURNS = 2
ALL_TOOLS = [*AGENT_TOOLS, *CART_TOOL_MAP.values()]
ALL_TOOL_MAP = {**TOOL_MAP, **CART_TOOL_MAP}
CART_ACTION_TO_TOOL = {
    "add": "add_cart",
    "delete": "dellete_cart",
    "show": "show_cart",
}


def required_cart_tool_completed(state: AgentState) -> bool:
    action = state.turn_resolution.cart_action if state.turn_resolution else None
    if action is None:
        return True
    required = CART_ACTION_TO_TOOL[action]
    return any(
        trace.tool_name == "tool_call"
        and trace.status == "success"
        and trace.input.get("tool") == required
        for trace in state.tool_traces
    )


def _configurable(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable", {})


def _tool_llm(config: RunnableConfig | None) -> Any:
    configured = _configurable(config).get("tool_llm")
    if configured is not None:
        return configured
    from llm_module import get_langchain_openai_chat_model
    return get_langchain_openai_chat_model()


def _profile_patch_has_changes(state: AgentState) -> bool:
    patch = state.turn_resolution.profile_patch if state.turn_resolution else None
    if patch is None:
        return False
    return any(
        preference.add or preference.remove
        for preference in (
            patch.liked_flavors,
            patch.disliked_flavors,
            patch.liked_cocktails,
            patch.disliked_cocktails,
        )
    )


def _is_profile_only_turn(state: AgentState) -> bool:
    text = state.user_request.lower()
    preference_correction = any(
        marker in text
        for marker in (
            "это мое предпочтение",
            "это моё предпочтение",
            "я не просил советовать",
            "i meant this as my preference",
        )
    )
    preference_statement = _profile_patch_has_changes(state) or any(
        marker in text
        for marker in (
            "мне нравится",
            "я люблю",
            "я предпочитаю",
            "i like",
            "i love",
            "i prefer",
        )
    )
    explicit_action = any(
        marker in text
        for marker in (
            "посовет",
            "подбер",
            "порекоменд",
            "предлож",
            "найд",
            "покаж",
            "как приготов",
            "рецепт",
            "добав",
            "корзин",
            "удал",
            "убер",
            "recommend",
            "suggest",
            "find ",
            "show ",
            "recipe",
            "add ",
            "cart",
            "remove",
            "delete",
        )
    )
    return preference_correction or (preference_statement and not explicit_action)


def tool_calling_agent(
    state: AgentState,
    config: RunnableConfig | None = None,
) -> dict:
    if _is_profile_only_turn(state):
        return {
            "messages": [
                AIMessage(content="No tool needed for profile-only update.")
            ]
        }
    action = state.turn_resolution.cart_action if state.turn_resolution else None
    required_tool = CART_ACTION_TO_TOOL.get(action)
    cart_incomplete = not required_cart_tool_completed(state)
    if state.tool_call_count >= 2:
        result = {"messages": [AIMessage(content="Tool budget exhausted.")]}
        if cart_incomplete:
            result["errors"] = state.errors + ["required_cart_tool_missing"]
        return result
    context = {
        "user_request": state.user_request,
        "turn_resolution": state.turn_resolution.model_dump(mode="json"),
        "turns": [turn.model_dump(mode="json") for turn in state.session_memory.turns],
        "cart": [item.model_dump(mode="json") for item in state.session_memory.cart],
        "profile": state.user_profile.model_dump(mode="json"),
        "remaining_budget": 2 - state.tool_call_count,
    }
    history = state.messages or [
        SystemMessage(content=TOOL_PROMPT),
        HumanMessage(content=json.dumps(context, ensure_ascii=False)),
    ]
    bound = _tool_llm(config).bind_tools(ALL_TOOLS)
    response = bound.invoke(history)
    if not isinstance(response, AIMessage):
        response = AIMessage(content=getattr(response, "content", str(response)))
    if len(response.tool_calls) > 1:
        response = AIMessage(content="", tool_calls=[response.tool_calls[0]])
    if cart_incomplete:
        called = response.tool_calls[0]["name"] if response.tool_calls else None
        preparatory = action == "add" and called in {
            "search_products",
            "lookup_by_id",
        }
        if called != required_tool and not preparatory:
            correction = HumanMessage(
                content=(
                    f"REQUIRED ACTION: call {required_tool} now. "
                    "A natural-language claim does not change the cart."
                )
            )
            response = bound.invoke([*history, response, correction])
            if not isinstance(response, AIMessage):
                response = AIMessage(
                    content=getattr(response, "content", str(response))
                )
            if len(response.tool_calls) > 1:
                response = AIMessage(content="", tool_calls=[response.tool_calls[0]])
            called = response.tool_calls[0]["name"] if response.tool_calls else None
            preparatory = action == "add" and called in {
                "search_products",
                "lookup_by_id",
            }
            if called != required_tool and not preparatory:
                return {
                    "messages": [AIMessage(content="Required cart tool missing.")],
                    "errors": state.errors + ["required_cart_tool_missing"],
                }
    if _cocktail_search_is_required(state):
        called = response.tool_calls[0]["name"] if response.tool_calls else None
        if called != "search_cocktails":
            correction = HumanMessage(
                content=(
                    "REQUIRED ACTION: request_scope is cocktail and "
                    "effective_request names a shown product. Call "
                    "search_cocktails now; do not defer the requested search."
                )
            )
            response = bound.invoke([*history, response, correction])
            if not isinstance(response, AIMessage):
                response = AIMessage(
                    content=getattr(response, "content", str(response))
                )
            if len(response.tool_calls) > 1:
                response = AIMessage(
                    content="",
                    tool_calls=[response.tool_calls[0]],
                )
            called = response.tool_calls[0]["name"] if response.tool_calls else None
            if called != "search_cocktails":
                return {
                    "messages": [
                        AIMessage(content="Required cocktail search missing.")
                    ],
                    "errors": state.errors + ["required_cocktail_search_missing"],
                }
    listing_kind = _required_catalog_listing_kind(state)
    if listing_kind is not None:
        call = response.tool_calls[0] if response.tool_calls else None
        if (
            call is None
            or call["name"] != "list_catalog"
            or call.get("args", {}).get("kind") != listing_kind
        ):
            correction = HumanMessage(
                content=(
                    "REQUIRED ACTION: this is a complete catalog-list request. "
                    f'Call list_catalog now with kind="{listing_kind}".'
                )
            )
            response = bound.invoke([*history, response, correction])
            if not isinstance(response, AIMessage):
                response = AIMessage(
                    content=getattr(response, "content", str(response))
                )
            if len(response.tool_calls) > 1:
                response = AIMessage(content="", tool_calls=[response.tool_calls[0]])
            call = response.tool_calls[0] if response.tool_calls else None
            if (
                call is None
                or call["name"] != "list_catalog"
                or call.get("args", {}).get("kind") != listing_kind
            ):
                return {
                    "messages": [AIMessage(content="Required catalog listing missing.")],
                    "errors": state.errors + ["required_catalog_listing_missing"],
                }
    return {"messages": history + [response] if not state.messages else [response]}


def _all_shown_refs(state: AgentState) -> set[tuple[str, str]]:
    return {
        (item.kind, item.id)
        for turn in state.session_memory.turns
        for item in turn.shown_results
    }


def _recent_shown_refs(state: AgentState) -> set[tuple[str, str]]:
    return {
        (item.kind, item.id)
        for turn in state.session_memory.turns[-SEARCH_DEDUP_TURNS:]
        for item in turn.shown_results
    }


def _parse_cards(name: str, output: dict) -> list[ProductCandidate | CocktailCandidate]:
    if name == "list_catalog":
        CatalogListOutput.model_validate(output)
        return []
    if name == "lookup_by_id":
        raw = output["card"]
        model = CocktailCandidate if raw.get("kind") == "cocktail" else ProductCandidate
        return [model.model_validate(raw)]
    if name == "search_cocktails":
        return list(CocktailSearchOutput.model_validate(output).candidates)
    return list(ProductSearchOutput.model_validate(output).candidates)


def _product_refs_available_to_add(state: AgentState) -> set[str]:
    return {
        item.id
        for turn in state.session_memory.turns
        for item in turn.shown_results
        if item.kind == "product"
    } | {card.id for card in state.cards if card.kind == "product"}


def _cocktail_search_is_required(state: AgentState) -> bool:
    if (
        state.tool_call_count != 0
        or state.turn_resolution is None
        or state.turn_resolution.request_scope != "cocktail"
    ):
        return False
    effective = " ".join(
        state.turn_resolution.effective_request.lower().split()
    )
    return any(
        " ".join(item.name.lower().split()) in effective
        for turn in state.session_memory.turns
        for item in turn.shown_results
        if item.kind == "product"
    )


def _required_catalog_listing_kind(state: AgentState) -> str | None:
    if (
        state.tool_call_count != 0
        or state.turn_resolution is None
        or state.turn_resolution.request_scope != "catalog_listing"
    ):
        return None
    text = (
        f"{state.user_request} {state.turn_resolution.effective_request}"
    ).lower()
    if "коктейл" in text or "cocktail" in text:
        return "cocktail"
    if "ром" in text or "rum" in text or "product" in text:
        return "product"
    return None


def _execute_cart_tool(
    state: AgentState,
    name: str,
    args: dict,
) -> tuple[dict, object]:
    memory = state.session_memory.model_copy(deep=True)
    if name == "add_cart":
        parsed = AddCartInput.model_validate(args)
        if parsed.id not in _product_refs_available_to_add(state):
            raise ValueError("cart_product_not_available")
        existing = next((item for item in memory.cart if item.id == parsed.id), None)
        if existing is None:
            memory.cart.append(CartItem(id=parsed.id, amount=parsed.amount))
        else:
            if existing.amount + parsed.amount > 99:
                raise ValueError("cart_amount_limit_exceeded")
            existing.amount += parsed.amount
        action = "added"
    elif name == "dellete_cart":
        parsed = DeleteCartInput.model_validate(args)
        if not any(item.id == parsed.id for item in memory.cart):
            raise ValueError("cart_item_not_found")
        memory.cart = [item for item in memory.cart if item.id != parsed.id]
        action = "deleted"
    else:
        ShowCartInput.model_validate(args)
        action = "shown"
    output = CartOutput(action=action, items=memory.cart).model_dump(mode="json")
    return output, memory


def execute_tool(state: AgentState) -> dict:
    call = state.messages[-1].tool_calls[0]
    name, args, call_id = call["name"], call["args"], call["id"]
    signature = json.dumps({"name": name, "args": args}, sort_keys=True)
    error: str | None = None
    output: dict = {}
    returned_count = 0
    if state.tool_call_count >= 2:
        error = "tool_budget_exhausted"
    elif name not in ALL_TOOL_MAP:
        error = "unknown_tool"
    elif signature in state.canonical_tool_calls:
        error = "duplicate_tool_call"
    elif name == "lookup_by_id":
        ref = (str(args.get("kind")), str(args.get("id")))
        allowed = _all_shown_refs(state) | {
            (card.kind, card.id) for card in state.cards
        }
        if ref not in allowed:
            error = "lookup_ref_not_allowed"
    updated_memory = state.session_memory
    if error is None:
        try:
            if name in CART_TOOL_MAP:
                output, updated_memory = _execute_cart_tool(state, name, args)
            else:
                output = TOOL_MAP[name].invoke(args)
        except ValueError as exc:
            message = str(exc)
            error = (
                message
                if message
                in {
                    "cart_product_not_available",
                    "cart_item_not_found",
                    "cart_amount_limit_exceeded",
                }
                else "invalid_tool_input"
            )
        except Exception:
            logger.exception(
                "Tool execution failed: tool=%s session=%s turn=%s",
                name,
                state.session_id,
                state.turn_id,
            )
            error = "tool_execution_failed"

    cards = list(state.cards)
    if error is None and name not in CART_TOOL_MAP:
        returned = _parse_cards(name, output)
        if name not in {"lookup_by_id", "list_catalog"}:
            shown = _recent_shown_refs(state)
            returned = [card for card in returned if (card.kind, card.id) not in shown]
            if "candidates" in output:
                output["candidates"] = [card.model_dump(mode="json") for card in returned]
        returned_count = len(returned)
        for card in returned:
            if not any(old.kind == card.kind and old.id == card.id for old in cards):
                cards.append(card)

    content = {"ok": error is None, "error": error, "output": output}
    result = {
        "messages": [
            ToolMessage(
                content=json.dumps(content, ensure_ascii=False),
                tool_call_id=call_id,
                name=name,
            )
        ],
        "tool_call_count": state.tool_call_count + 1,
        "canonical_tool_calls": state.canonical_tool_calls + [signature],
        "cards": cards,
        "session_memory": updated_memory,
        "tool_traces": state.tool_traces + [
            ToolTrace(
                tool_name="tool_call",
                input={"tool": name, "args": args},
                output_summary=error
                or (
                    (
                        f"cart_items={len(updated_memory.cart)}"
                        if name in CART_TOOL_MAP
                        else (
                            (
                                f"catalog_items={output.get('total', 0)}"
                                if name == "list_catalog"
                                else (
                                    f"returned_after_filter={returned_count}; "
                                    f"cards_in_current_turn={len(cards)}"
                                )
                            )
                        )
                    )
                ),
                status="error" if error else "success",
            )
        ],
    }
    return result


def route_after_tool_agent(state: AgentState) -> str:
    last = state.messages[-1]
    return (
        "execute_tool"
        if getattr(last, "tool_calls", None) and state.tool_call_count < 2
        else "generate_answer"
    )
