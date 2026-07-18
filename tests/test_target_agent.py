import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from sommelier.agent.contracts import (
    CatalogListOutput,
    CocktailCandidate,
    FeedbackResult,
    FinalAnswerResult,
    ProductCandidate,
    TurnResolution,
)
from sommelier.agent.search_tools import _list_catalog
from sommelier.agent.graph import ANSWER_PROMPT, generate_answer, run_agent_turn
from sommelier.agent.memory import (
    CartItem,
    CatalogRef,
    SessionMemory,
    ShownResult,
    TurnMemory,
    enforce_memory_limits,
)
from sommelier.agent.profile import PreferencePatch, ProfilePatch, UserProfile
from sommelier.agent.resolver import (
    RESOLVER_PROMPT,
    resolve_turn,
    validate_turn_resolution,
)
from sommelier.agent.state import AgentState
from sommelier.agent.tool_agent import (
    ALL_TOOLS,
    TOOL_PROMPT,
    execute_tool,
    required_cart_tool_completed,
    tool_calling_agent,
)
from sommelier.storage.session_repository import SessionRepository


def _turn(initial: str = "fresh cocktail") -> TurnMemory:
    return TurnMemory(
        follow_up=False,
        user_request=initial,
        initial_request=initial,
        effective_request=initial,
        negative_request=None,
        assistant_summary="Suggested Mojito and Old Cuban.",
        shown_results=[
            ShownResult(
                kind="cocktail", id="mojito", name="Mojito", summary="Mint and lime."
            ),
            ShownResult(
                kind="cocktail",
                id="old-cuban",
                name="Old Cuban",
                summary="Mint, lime and bitters.",
            ),
        ],
    )


def test_memory_keeps_last_twelve_ordered_turns() -> None:
    memory = SessionMemory(
        session_id="s",
        turns=[_turn(str(index)) for index in range(15)],
    )
    enforce_memory_limits(memory)
    assert len(memory.turns) == 12
    assert memory.turns[0].user_request == "3"
    assert [item.id for item in memory.turns[-1].shown_results] == [
        "mojito",
        "old-cuban",
    ]


def test_turn_memory_keeps_up_to_five_ordered_shown_results() -> None:
    turn = _turn().model_copy(
        update={
            "shown_results": [
                ShownResult(
                    kind="cocktail",
                    id=f"cocktail-{index}",
                    name=f"Cocktail {index}",
                    summary="Shown.",
                )
                for index in range(5)
            ]
        }
    )
    assert [item.id for item in turn.shown_results] == [
        "cocktail-0",
        "cocktail-1",
        "cocktail-2",
        "cocktail-3",
        "cocktail-4",
    ]


def test_resolution_validation_for_new_and_follow_up() -> None:
    memory = SessionMemory(session_id="s", turns=[_turn()])
    new = TurnResolution(
        follow_up=False,
        initial_request="new request",
        effective_request="positive request",
    )
    validate_turn_resolution(new, "new request", memory)

    follow = TurnResolution(
        follow_up=True,
        initial_request="fresh cocktail",
        effective_request="more bitter cocktail",
        negative_request="not sweet",
    )
    validate_turn_resolution(follow, "more bitter", memory)

    with pytest.raises(ValueError):
        validate_turn_resolution(
            follow.model_copy(update={"initial_request": "unknown"}),
            "more bitter",
            memory,
        )

    with pytest.raises(ValueError, match="negative constraints"):
        validate_turn_resolution(
            follow.model_copy(
                update={
                    "effective_request": "Свежий, но не сладкий коктейль.",
                    "negative_request": "Не сладкий.",
                }
            ),
            "more bitter",
            memory,
        )


def test_resolver_prompt_forbids_softened_negative_in_effective_request() -> None:
    assert '"низкая сладость"' in RESOLVER_PROMPT
    assert "WRONG effective_request examples" in RESOLVER_PROMPT
    assert "keep that effective_request unchanged" in RESOLVER_PROMPT


def test_resolver_prompt_treats_standalone_target_switch_as_new_request() -> None:
    assert '"теперь", "а теперь", "now", "next"' in RESOLVER_PROMPT
    assert "topical similarity alone is not enough" in RESOLVER_PROMPT


def test_resolver_prompt_resolves_pronouns_by_recent_compatible_result() -> None:
    assert "PRONOUN AND REFERENT RESOLUTION IS RECENCY-FIRST" in RESOLVER_PROMPT
    assert "MOST RECENT compatible object in shown_results" in RESOLVER_PROMPT
    assert "Skip profile-only" in RESOLVER_PROMPT
    assert "recency has priority over topical similarity" in RESOLVER_PROMPT
    assert "exact object\n  name from shown_results" in RESOLVER_PROMPT
    assert "Подобрать коктейли на основе BACARDÍ Spiced" in RESOLVER_PROMPT
    assert "BACARDÍ Coconut rum is WRONG" in RESOLVER_PROMPT


def test_resolver_prompt_preserves_action_during_object_correction() -> None:
    assert "CORRECTIONS MUST PRESERVE THE PENDING ACTION" in RESOLVER_PROMPT
    assert "replace only\n  the target object" in RESOLVER_PROMPT
    assert "wrong assistant answer does not cancel" in RESOLVER_PROMPT
    assert "Do not return to food pairing" in RESOLVER_PROMPT
    assert "Теперь посоветуй выдержанный ром для коктейлей" in RESOLVER_PROMPT
    assert "=> follow_up=false" in RESOLVER_PROMPT


def test_resolver_requires_explicit_cart_action() -> None:
    request = "Покажи, что сейчас в корзине."
    result = TurnResolution(
        follow_up=False,
        request_scope="cart",
        initial_request=request,
        effective_request="Показать, что сейчас в корзине.",
    )
    with pytest.raises(ValueError, match="cart_action must be 'show'"):
        validate_turn_resolution(result, request, SessionMemory(session_id="cart"))

    validate_turn_resolution(
        result.model_copy(update={"cart_action": "show"}),
        request,
        SessionMemory(session_id="cart"),
    )


def test_resolver_requires_profile_patch_for_explicit_preference() -> None:
    request = "Привет, мне нравится мята и сладкое."
    result = TurnResolution(
        follow_up=False,
        request_scope="profile",
        initial_request=request,
        effective_request="Запомнить предпочтения: мята и сладкие вкусы.",
    )
    with pytest.raises(ValueError, match="requires a non-empty profile_patch"):
        validate_turn_resolution(result, request, SessionMemory(session_id="s"))

    validate_turn_resolution(
        result.model_copy(
            update={
                "profile_patch": ProfilePatch(
                    liked_flavors=PreferencePatch(
                        add=["мята", "сладкие вкусы"]
                    )
                )
            }
        ),
        request,
        SessionMemory(session_id="s"),
    )


def test_tool_prompt_does_not_treat_cocktail_as_food() -> None:
    assert "a named cocktail is NOT food" in TOOL_PROMPT
    assert "Old Fashioned" in TOOL_PROMPT
    assert "uses search_products" in TOOL_PROMPT


def test_tool_prompt_clarifies_only_underspecified_recommendations() -> None:
    assert "at least one meaningful\n  selection criterion" in TOOL_PROMPT
    assert "do not search" in TOOL_PROMPT
    assert '"на свой вкус", "удиви меня"' in TOOL_PROMPT
    assert "never require this clarification before a recipe" in TOOL_PROMPT
    assert "search without asking unnecessary questions" in TOOL_PROMPT


def _cart_tool_state(
    memory: SessionMemory,
    name: str,
    args: dict,
) -> AgentState:
    return AgentState(
        session_id=memory.session_id,
        user_request=name,
        session_memory=memory,
        user_profile=UserProfile(session_id=memory.session_id),
        turn_resolution=TurnResolution(
            follow_up=False,
            initial_request=name,
            effective_request=name,
        ),
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": name, "args": args, "id": f"call-{name}"}
                ],
            )
        ],
    )


def test_cart_tools_add_increment_show_and_delete_product() -> None:
    product_turn = _turn("aged rum")
    product_turn.shown_results = [
        ShownResult(
            kind="product",
            id="bacardi-anejo-cuatro-rum",
            name="BACARDÍ Añejo Cuatro",
            summary="Aged rum.",
        )
    ]
    memory = SessionMemory(session_id="cart", turns=[product_turn])

    added = execute_tool(
        _cart_tool_state(
            memory,
            "add_cart",
            {"id": "bacardi-anejo-cuatro-rum", "amount": 2},
        )
    )
    assert added["session_memory"].cart == [
        CartItem(id="bacardi-anejo-cuatro-rum", amount=2)
    ]
    assert memory.cart == []

    incremented = execute_tool(
        _cart_tool_state(
            added["session_memory"],
            "add_cart",
            {"id": "bacardi-anejo-cuatro-rum", "amount": 1},
        )
    )
    assert incremented["session_memory"].cart[0].amount == 3

    shown = execute_tool(
        _cart_tool_state(incremented["session_memory"], "show_cart", {})
    )
    content = json.loads(shown["messages"][0].content)
    assert content["output"]["items"] == [
        {"id": "bacardi-anejo-cuatro-rum", "amount": 3}
    ]

    deleted = execute_tool(
        _cart_tool_state(incremented["session_memory"], "dellete_cart", {
            "id": "bacardi-anejo-cuatro-rum"
        })
    )
    assert deleted["session_memory"].cart == []


def test_cart_rejects_cocktail_or_unshown_product_id() -> None:
    memory = SessionMemory(session_id="cart", turns=[_turn()])
    output = execute_tool(
        _cart_tool_state(memory, "add_cart", {"id": "mojito", "amount": 1})
    )
    assert output["session_memory"].cart == []
    assert '"cart_product_not_available"' in output["messages"][0].content


def test_cart_tools_are_bound_and_prompted_explicitly() -> None:
    names = {tool.name for tool in ALL_TOOLS}
    assert {"add_cart", "dellete_cart", "show_cart"} <= names
    assert "lookup_by_ids" in names
    assert "lookup_by_id" not in names
    assert "cart tools are for product ids, never cocktail ids" in TOOL_PROMPT
    assert "first\n  call search_products, then call add_cart" in TOOL_PROMPT


def test_profile_only_turn_deterministically_skips_tools() -> None:
    class ToolsMustNotBeBound:
        def bind_tools(self, tools):
            raise AssertionError("profile-only turn must not bind catalog tools")

    request = "Привет, мне нравится мята и сладкое."
    state = AgentState(
        session_id="profile",
        user_request=request,
        session_memory=SessionMemory(session_id="profile"),
        user_profile=UserProfile(session_id="profile"),
        turn_resolution=TurnResolution(
            follow_up=False,
            initial_request=request,
            effective_request="Запомнить предпочтения: мята и сладкие вкусы.",
            profile_patch=ProfilePatch(
                liked_flavors=PreferencePatch(
                    add=["мята", "сладкие вкусы"]
                )
            ),
        ),
    )
    output = tool_calling_agent(
        state,
        config={"configurable": {"tool_llm": ToolsMustNotBeBound()}},
    )
    assert output["messages"][0].tool_calls == []
    assert "profile-only" in output["messages"][0].content


def test_profile_correction_deterministically_skips_tools() -> None:
    request = "Нет, это моё предпочтение, я не просил советовать."
    state = AgentState(
        session_id="profile",
        user_request=request,
        session_memory=SessionMemory(session_id="profile"),
        user_profile=UserProfile(session_id="profile"),
        turn_resolution=TurnResolution(
            follow_up=True,
            initial_request="Привет, мне нравится мята и сладкое.",
            effective_request="Уточнить, что это предпочтение пользователя.",
        ),
    )
    output = tool_calling_agent(state)
    assert output["messages"][0].tool_calls == []
    assert "profile-only" in output["messages"][0].content


class SequenceToolFake:
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.responses.pop(0)


def test_required_cart_action_retries_no_tool_response() -> None:
    resolution = TurnResolution(
        follow_up=False,
        initial_request="Покажи корзину.",
        effective_request="Показать корзину.",
        cart_action="show",
    )
    state = AgentState(
        session_id="cart",
        user_request="Покажи корзину.",
        session_memory=SessionMemory(session_id="cart"),
        user_profile=UserProfile(session_id="cart"),
        turn_resolution=resolution,
    )
    fake = SequenceToolFake(
        [
            AIMessage(content="В корзине пусто."),
            AIMessage(
                content="",
                tool_calls=[{"name": "show_cart", "args": {}, "id": "show-1"}],
            ),
        ]
    )
    output = tool_calling_agent(
        state, config={"configurable": {"tool_llm": fake}}
    )
    assert output["messages"][-1].tool_calls[0]["name"] == "show_cart"

    executed = execute_tool(
        state.model_copy(update={"messages": output["messages"]})
    )
    completed = state.model_copy(
        update={
            "session_memory": executed["session_memory"],
            "tool_traces": executed["tool_traces"],
        }
    )
    assert required_cart_tool_completed(completed)


def test_final_answer_is_blocked_without_required_cart_tool() -> None:
    state = AgentState(
        session_id="cart",
        user_request="Убери ром из корзины.",
        session_memory=SessionMemory(
            session_id="cart",
            cart=[CartItem(id="bacardi-reserva-ocho-rum", amount=1)],
        ),
        user_profile=UserProfile(session_id="cart"),
        turn_resolution=TurnResolution(
            follow_up=False,
            initial_request="Убери ром из корзины.",
            effective_request="Убрать ром из корзины.",
            cart_action="delete",
        ),
    )
    output = generate_answer(
        state,
        config={
            "configurable": {
                "answer_llm": StructuredFake(
                    FinalAnswerResult(
                        answer="Удалил.",
                        assistant_summary="Удалён ром.",
                    )
                )
            }
        },
    )
    assert output["errors"] == ["required_cart_tool_missing"]


def test_answer_prompt_requires_marketing_tone_and_no_unsolicited_recipe() -> None:
    assert "appealing, restrained" in ANSWER_PROMPT
    assert "marketing language" in ANSWER_PROMPT
    assert 'Do not begin with "Да"' in ANSWER_PROMPT
    assert "ONLY when explicitly requested" in ANSWER_PROMPT
    assert "do not append an unsolicited recipe" in ANSWER_PROMPT
    assert "shown_refs is the ordered list" in ANSWER_PROMPT
    assert "explicitly named in the answer" in ANSWER_PROMPT
    assert '"Вот как приготовить Old Cuban:"' in ANSWER_PROMPT
    assert "do not recommend it again" in ANSWER_PROMPT
    assert "after show_cart" in ANSWER_PROMPT
    assert "do not invent product names, prices or availability" in ANSWER_PROMPT
    assert "lookup_by_ids do not\n   prove that the cart changed" in ANSWER_PROMPT
    assert "offer to make a recommendation" in ANSWER_PROMPT
    assert "Do not name,\n   suggest or describe any catalog product" in ANSWER_PROMPT


def test_answer_prompt_defines_clarification_feedback_and_safe_style() -> None:
    assert "two to four relevant sensory details" in ANSWER_PROMPT
    assert "ask exactly one concise clarification" in ANSWER_PROMPT
    assert "return shown_refs=[]" in ANSWER_PROMPT
    assert "After an actual recommendation" in ANSWER_PROMPT
    assert "Do NOT request feedback after" in ANSWER_PROMPT
    assert "Do not add secondary products" in ANSWER_PROMPT
    assert "Light humor is allowed only" in ANSWER_PROMPT
    assert "fictional bar vignette" in ANSWER_PROMPT
    assert "never present invented events" in ANSWER_PROMPT


def test_lookup_rejects_reference_not_present_in_memory() -> None:
    state = AgentState(
        session_id="s",
        user_request="recipe",
        session_memory=SessionMemory(session_id="s", turns=[_turn()]),
        user_profile=UserProfile(session_id="s"),
        turn_resolution=TurnResolution(
            follow_up=True,
            initial_request="fresh cocktail",
            effective_request="Mojito recipe",
        ),
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_by_ids",
                        "args": {"kind": "cocktail", "ids": ["unknown"]},
                        "id": "call-1",
                    }
                ],
            )
        ],
    )
    output = execute_tool(state)
    assert '"lookup_ref_not_allowed"' in output["messages"][0].content
    assert output["cards"] == []


def test_lookup_loads_full_shown_cocktail_cards() -> None:
    state = AgentState(
        session_id="s",
        user_request="How do I make the first?",
        session_memory=SessionMemory(session_id="s", turns=[_turn()]),
        user_profile=UserProfile(session_id="s"),
        turn_resolution=TurnResolution(
            follow_up=True,
            initial_request="fresh cocktail",
            effective_request="Mojito recipe",
        ),
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_by_ids",
                        "args": {"kind": "cocktail", "ids": ["mojito", "old-cuban"]},
                        "id": "call-1",
                    }
                ],
            )
        ],
    )
    output = execute_tool(state)
    assert [card.id for card in output["cards"]] == ["mojito", "old-cuban"]
    assert output["cards"][0].ingredients
    assert output["cards"][0].recipe_steps


def test_lookup_loads_full_shown_product_card() -> None:
    turn = _turn()
    turn.shown_results = [
        ShownResult(
            kind="product",
            id="bacardi-anejo-cuatro-rum",
            name="BACARDÍ Añejo Cuatro",
            summary="Aged rum.",
        )
    ]
    state = AgentState(
        session_id="s",
        user_request="Tell me more",
        session_memory=SessionMemory(session_id="s", turns=[turn]),
        user_profile=UserProfile(session_id="s"),
        turn_resolution=TurnResolution(
            follow_up=True,
            initial_request="fresh cocktail",
            effective_request="Details about Añejo Cuatro",
        ),
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_by_ids",
                        "args": {
                            "kind": "product",
                            "ids": ["bacardi-anejo-cuatro-rum"],
                        },
                        "id": "call-1",
                    }
                ],
            )
        ],
    )
    output = execute_tool(state)
    assert output["cards"][0].kind == "product"
    assert output["cards"][0].id == "bacardi-anejo-cuatro-rum"


def test_duplicate_and_third_tool_calls_are_blocked() -> None:
    call = {
        "name": "search_cocktails",
        "args": {"query": "fresh lime mint"},
        "id": "call-1",
    }
    signature = json.dumps(
        {"name": call["name"], "args": call["args"]}, sort_keys=True
    )
    base = dict(
        session_id="s",
        user_request="cocktail",
        session_memory=SessionMemory(session_id="s"),
        user_profile=UserProfile(session_id="s"),
        turn_resolution=TurnResolution(
            follow_up=False,
            initial_request="cocktail",
            effective_request="fresh cocktail",
        ),
        messages=[AIMessage(content="", tool_calls=[call])],
    )
    duplicate = execute_tool(
        AgentState(**base, tool_call_count=1, canonical_tool_calls=[signature])
    )
    assert '"duplicate_tool_call"' in duplicate["messages"][0].content

    third = execute_tool(AgentState(**base, tool_call_count=2))
    assert '"tool_budget_exhausted"' in third["messages"][0].content


def test_search_filters_already_shown_references() -> None:
    state = AgentState(
        session_id="s",
        user_request="another cocktail",
        session_memory=SessionMemory(session_id="s", turns=[_turn()]),
        user_profile=UserProfile(session_id="s"),
        turn_resolution=TurnResolution(
            follow_up=True,
            initial_request="fresh cocktail",
            effective_request="another fresh lime mint cocktail",
        ),
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_cocktails",
                        "args": {"query": "fresh lime mint rum cocktail"},
                        "id": "call-1",
                    }
                ],
            )
        ],
    )
    output = execute_tool(state)
    ids = {card.id for card in output["cards"]}
    assert "mojito" not in ids
    assert "old-cuban" not in ids


def test_search_dedup_uses_only_last_two_turns() -> None:
    oldest = _turn()
    recent_one = _turn("recent one")
    recent_one.shown_results = [
        ShownResult(
            kind="cocktail", id="not-in-search-1", name="Old 1", summary="Old."
        )
    ]
    recent_two = _turn("recent two")
    recent_two.shown_results = [
        ShownResult(
            kind="cocktail", id="not-in-search-2", name="Old 2", summary="Old."
        )
    ]
    state = AgentState(
        session_id="s",
        user_request="fresh cocktail",
        session_memory=SessionMemory(
            session_id="s", turns=[oldest, recent_one, recent_two]
        ),
        user_profile=UserProfile(session_id="s"),
        turn_resolution=TurnResolution(
            follow_up=False,
            initial_request="fresh cocktail",
            effective_request="fresh lime mint cocktail",
        ),
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_cocktails",
                        "args": {"query": "fresh lime mint rum cocktail"},
                        "id": "call-1",
                    }
                ],
            )
        ],
    )
    output = execute_tool(state)
    assert "mojito" in {card.id for card in output["cards"]}


class StructuredFake:
    def __init__(self, value):
        self.value = value

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, prompt):
        return self.value


class RecordingStructuredFake(StructuredFake):
    def __init__(self, value):
        super().__init__(value)
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.value


class NoToolFake:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content="No catalog tool needed.")


class FailingStructuredFake:
    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, prompt):
        raise RuntimeError("classifier unavailable")


class SequenceStructuredFake(StructuredFake):
    def __init__(self, values):
        self.values = list(values)

    def invoke(self, prompt):
        return self.values.pop(0)


def test_list_catalog_returns_complete_compact_catalogs() -> None:
    products = CatalogListOutput.model_validate(_list_catalog("product"))
    cocktails = CatalogListOutput.model_validate(_list_catalog("cocktail"))

    assert products.total == len(products.items) == 15
    assert cocktails.total == len(cocktails.items) == 42
    assert len({item.id for item in products.items}) == products.total
    assert set(products.items[0].model_dump()) == {"kind", "id", "name"}


def test_full_catalog_request_has_dedicated_scope() -> None:
    request = "Покажи полный список всех ромов."
    resolution = TurnResolution(
        follow_up=False,
        request_scope="catalog_listing",
        initial_request=request,
        effective_request="Перечислить все ромы в каталоге.",
    )
    validate_turn_resolution(
        resolution,
        request,
        SessionMemory(session_id="catalog-scope"),
    )


def test_catalog_listing_forces_list_tool_and_keeps_cards_empty() -> None:
    request = "Покажи полный список ромов."
    state = AgentState(
        session_id="catalog-list",
        user_request=request,
        session_memory=SessionMemory(session_id="catalog-list"),
        user_profile=UserProfile(session_id="catalog-list"),
        turn_resolution=TurnResolution(
            follow_up=False,
            request_scope="catalog_listing",
            initial_request=request,
            effective_request="Перечислить все ромы в каталоге.",
        ),
    )
    fake = SequenceToolFake(
        [
            AIMessage(content="Можно посмотреть несколько вариантов."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_catalog",
                        "args": {"kind": "product"},
                        "id": "list-products",
                    }
                ],
            ),
        ]
    )

    decision = tool_calling_agent(
        state,
        config={"configurable": {"tool_llm": fake}},
    )
    assert decision["messages"][-1].tool_calls[0]["name"] == "list_catalog"

    executed = execute_tool(
        state.model_copy(update={"messages": decision["messages"]})
    )
    assert executed["cards"] == []
    envelope = json.loads(executed["messages"][0].content)
    assert envelope["output"]["total"] == 15


def test_catalog_listing_answer_retries_omission_and_uses_no_refs() -> None:
    listing = CatalogListOutput.model_validate(_list_catalog("product"))
    request = "Покажи полный список ромов."
    state = AgentState(
        session_id="catalog-answer",
        user_request=request,
        session_memory=SessionMemory(session_id="catalog-answer"),
        user_profile=UserProfile(session_id="catalog-answer"),
        turn_resolution=TurnResolution(
            follow_up=False,
            request_scope="catalog_listing",
            initial_request=request,
            effective_request="Перечислить все ромы в каталоге.",
        ),
        messages=[
            ToolMessage(
                content=json.dumps(
                    {"ok": True, "error": None, "output": listing.model_dump()},
                    ensure_ascii=False,
                ),
                tool_call_id="list-products",
                name="list_catalog",
            )
        ],
    )
    complete_answer = "\n".join(item.name for item in listing.items)
    fake = SequenceStructuredFake(
        [
            FinalAnswerResult(
                answer=listing.items[0].name,
                assistant_summary="Неполный список.",
            ),
            FinalAnswerResult(
                answer=complete_answer,
                assistant_summary="Перечислены все ромы каталога.",
            ),
        ]
    )

    output = generate_answer(
        state,
        config={"configurable": {"answer_llm": fake}},
    )
    assert output["final_answer_result"].answer == complete_answer
    assert output["final_answer_result"].shown_refs == []


def test_profile_only_turn_builds_memory_without_tools() -> None:
    request = "Remember that I like oak."
    resolution = TurnResolution(
        follow_up=False,
        request_scope="profile",
        initial_request=request,
        effective_request=request,
        profile_patch=ProfilePatch(
            liked_flavors=PreferencePatch(add=["oak"])
        ),
    )
    answer = FinalAnswerResult(
        answer="Запомнил.",
        shown_refs=[],
        assistant_summary="Сохранено предпочтение дуба.",
    )
    directory = Path(".test_tmp") / f"graph-{uuid4().hex}"
    try:
        state = run_agent_turn(
            AgentState(session_id="turn-v4-test", user_request=request),
            config={
                "configurable": {
                    "resolver_llm": StructuredFake(resolution),
                    "feedback_llm": StructuredFake(
                        FeedbackResult(feedback="neutral")
                    ),
                    "tool_llm": NoToolFake(),
                    "answer_llm": StructuredFake(answer),
                    "repository": SessionRepository(
                        directory / "sommelier.sqlite3"
                    ),
                    "persist": False,
                }
            },
        )
        assert state.tool_call_count == 0
        assert state.user_profile.liked_flavors == ["oak"]
        assert (
            state.session_memory.turns[-1].assistant_summary
            == answer.assistant_summary
        )
        assert state.session_memory.turns[-1].shown_results == []
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_graph_persists_complete_successful_turn_to_sqlite() -> None:
    directory = Path(".test_tmp") / f"graph-persist-{uuid4().hex}"
    repository = SessionRepository(directory / "sommelier.sqlite3")
    request = "Remember that I like oak."
    resolution = TurnResolution(
        follow_up=False,
        request_scope="profile",
        initial_request=request,
        effective_request=request,
        profile_patch=ProfilePatch(
            liked_flavors=PreferencePatch(add=["oak"])
        ),
    )
    answer = FinalAnswerResult(
        answer="Запомнил.",
        assistant_summary="Сохранено предпочтение дуба.",
    )
    try:
        state = run_agent_turn(
            AgentState(session_id="persisted", user_request=request),
            config={
                "configurable": {
                    "resolver_llm": StructuredFake(resolution),
                    "feedback_llm": StructuredFake(
                        FeedbackResult(feedback="neutral")
                    ),
                    "tool_llm": NoToolFake(),
                    "answer_llm": StructuredFake(answer),
                    "repository": repository,
                }
            },
        )
        assert state.errors == []
        assert repository.load_messages("persisted") == [
            {"role": "user", "content": request},
            {"role": "assistant", "content": "Запомнил."},
        ]
        assert repository.load_user_profile("persisted").liked_flavors == ["oak"]
        assert len(repository.load_session_memory("persisted").turns) == 1
        assert repository.load_trace_events("persisted")[-1]["tool_name"] == "persist"
        assert repository.load_feedback_stats("persisted") == {
            "total": 1,
            "neutral": 1,
            "purchase_intent": 0,
            "negative_feedback": 0,
            "successful_turns": 1,
            "failed_turns": 0,
        }
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_only_recent_transcript_is_loaded_into_resolver_and_answer() -> None:
    directory = Path(".test_tmp") / f"graph-transcript-{uuid4().hex}"
    repository = SessionRepository(directory / "sommelier.sqlite3")
    secret_transcript = "FULL-TRANSCRIPT-MUST-NOT-ENTER-PROMPT"
    repository.persist_successful_turn(
        session_id="isolated",
        turn_id="old-turn",
        memory=SessionMemory(session_id="isolated"),
        profile=UserProfile(session_id="isolated"),
        user_message="old question",
        assistant_message=secret_transcript,
        traces=[],
    )
    for index in (1, 2, 3):
        repository.persist_successful_turn(
            session_id="isolated",
            turn_id=f"recent-turn-{index}",
            memory=SessionMemory(session_id="isolated"),
            profile=UserProfile(session_id="isolated"),
            user_message=f"RECENT USER {index}",
            assistant_message=f"RECENT ASSISTANT {index}",
            traces=[],
        )
    request = "Hello"
    resolver = RecordingStructuredFake(
        TurnResolution(
            follow_up=False,
            initial_request=request,
            effective_request=request,
        )
    )
    answer = FinalAnswerResult(
        answer="Здравствуйте.",
        assistant_summary="Короткое приветствие.",
    )
    answer_llm = RecordingStructuredFake(answer)
    try:
        state = run_agent_turn(
            AgentState(session_id="isolated", user_request=request),
            config={
                "configurable": {
                    "resolver_llm": resolver,
                    "feedback_llm": StructuredFake(
                        FeedbackResult(feedback="neutral")
                    ),
                    "tool_llm": NoToolFake(),
                    "answer_llm": answer_llm,
                    "repository": repository,
                    "persist": False,
                }
            },
        )
        assert state.errors == []
        assert secret_transcript not in resolver.prompts[0]
        assert secret_transcript not in answer_llm.prompts[0]
        assert "RECENT USER 1" in resolver.prompts[0]
        assert "RECENT ASSISTANT 2" in resolver.prompts[0]
        assert "RECENT USER 1" in answer_llm.prompts[0]
        assert "RECENT ASSISTANT 2" in answer_llm.prompts[0]
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_failed_graph_turn_does_not_write_sqlite() -> None:
    directory = Path(".test_tmp") / f"graph-failed-{uuid4().hex}"
    repository = SessionRepository(directory / "sommelier.sqlite3")
    request = "Hello"
    try:
        state = run_agent_turn(
            AgentState(session_id="failed", user_request=request),
            config={
                "configurable": {
                    "resolver_llm": StructuredFake(
                        TurnResolution(
                            follow_up=False,
                            initial_request=request,
                            effective_request=request,
                        )
                    ),
                    "feedback_llm": StructuredFake(
                        FeedbackResult(feedback="neutral")
                    ),
                    "tool_llm": NoToolFake(),
                    "answer_llm": StructuredFake(
                        {"answer": "", "assistant_summary": ""}
                    ),
                    "repository": repository,
                }
            },
        )
        assert "final_answer_validation_failed" in state.errors
        assert repository.load_messages("failed") == []
        assert repository.load_session_memory("failed").turns == []
        assert repository.load_feedback_stats("failed")["failed_turns"] == 1
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_feedback_classifier_failure_does_not_change_successful_turn() -> None:
    directory = Path(".test_tmp") / f"graph-feedback-failed-{uuid4().hex}"
    repository = SessionRepository(directory / "sommelier.sqlite3")
    request = "Hello"
    try:
        state = run_agent_turn(
            AgentState(session_id="feedback-failed", user_request=request),
            config={
                "configurable": {
                    "resolver_llm": StructuredFake(
                        TurnResolution(
                            follow_up=False,
                            initial_request=request,
                            effective_request=request,
                        )
                    ),
                    "feedback_llm": FailingStructuredFake(),
                    "tool_llm": NoToolFake(),
                    "answer_llm": StructuredFake(
                        FinalAnswerResult(
                            answer="Hello.",
                            assistant_summary="Greeting.",
                        )
                    ),
                    "repository": repository,
                }
            },
        )

        assert state.errors == []
        assert state.final_answer_result.answer == "Hello."
        assert repository.load_messages("feedback-failed")[-1] == {
            "role": "assistant",
            "content": "Hello.",
        }
        assert repository.load_feedback_stats("feedback-failed")["total"] == 0
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_legacy_turn_memory_defaults_request_scope_to_conversation() -> None:
    turn = TurnMemory.model_validate(
        {
            "follow_up": False,
            "user_request": "Hello",
            "initial_request": "Hello",
            "effective_request": "Hello",
            "assistant_summary": "Greeting",
        }
    )
    assert turn.request_scope == "conversation"


def test_resolver_payload_uses_turn_shown_results_without_recent_objects() -> None:
    memory = SessionMemory(
        session_id="recent",
        turns=[
            _turn("old").model_copy(
                update={
                    "shown_results": [
                        ShownResult(
                            kind="product",
                            id="coconut",
                            name="BACARDÍ Coconut rum",
                            summary="Coconut",
                        )
                    ]
                }
            ),
            _turn("profile").model_copy(
                update={"request_scope": "profile", "shown_results": []}
            ),
            _turn("new").model_copy(
                update={
                    "shown_results": [
                        ShownResult(
                            kind="product",
                            id="spiced",
                            name="BACARDÍ Spiced",
                            summary="Spiced",
                        ),
                        ShownResult(
                            kind="product",
                            id="coconut",
                            name="BACARDÍ Coconut rum",
                            summary="Coconut again",
                        ),
                    ]
                }
            ),
        ],
    )

    result = TurnResolution(
        follow_up=False,
        request_scope="conversation",
        initial_request="Hello",
        effective_request="Hello",
    )
    fake = RecordingStructuredFake(result)

    resolve_turn(
        user_request="Hello",
        memory=memory,
        profile=UserProfile(session_id="recent"),
        llm=fake,
    )

    payload = json.loads(fake.prompts[0].split("INPUT:\n", 1)[1])
    assert "recent_objects" not in payload
    assert payload["turns"][-1]["shown_results"][0]["name"] == "BACARDÍ Spiced"


def test_resolver_payload_limits_turns_and_recent_messages() -> None:
    memory = SessionMemory(
        session_id="limits",
        turns=[_turn(f"turn-{index}") for index in range(8)],
    )
    result = TurnResolution(
        follow_up=False,
        request_scope="conversation",
        initial_request="Hello",
        effective_request="Hello",
    )
    fake = RecordingStructuredFake(result)

    resolve_turn(
        user_request="Hello",
        memory=memory,
        profile=UserProfile(session_id="limits"),
        llm=fake,
        recent_messages=[
            {"role": "user", "content": f"message-{index}"}
            for index in range(6)
        ],
    )

    payload = json.loads(fake.prompts[0].split("INPUT:\n", 1)[1])
    assert len(payload["turns"]) == 6
    assert payload["turns"][0]["user_request"] == "turn-2"
    assert len(payload["recent_messages"]) == 6
    assert payload["recent_messages"][0]["content"] == "message-0"


def test_constraint_only_scope_inheritance_is_owned_by_resolver_prompt() -> None:
    memory = SessionMemory(
        session_id="scope",
        turns=[
            TurnMemory(
                follow_up=True,
                request_scope="cocktail",
                user_request="Какие коктейли из него?",
                initial_request="Ром к мясу",
                effective_request="Коктейли на основе BACARDÍ Spiced.",
                assistant_summary="Обсуждались коктейли.",
            )
        ],
    )
    valid = TurnResolution(
        follow_up=True,
        request_scope="cocktail",
        initial_request="Ром к мясу",
        effective_request=(
            "Подобрать сладкие коктейли на основе BACARDÍ Spiced."
        ),
    )
    validate_turn_resolution(valid, "Давай сладкие варианты", memory)
    assert "MUST inherit the previous scope" in RESOLVER_PROMPT


def test_explicit_scope_change_resolves_latest_product_pronoun() -> None:
    memory = SessionMemory(
        session_id="scope-change",
        turns=[
            TurnMemory(
                follow_up=False,
                request_scope="food_pairing",
                user_request="Подбери ром к мясу",
                initial_request="Подбери ром к мясу",
                effective_request="Подобрать ром к жареному мясу.",
                assistant_summary="Рекомендован BACARDÍ Spiced.",
                shown_results=[
                    ShownResult(
                        kind="product",
                        id="spiced",
                        name="BACARDÍ Spiced",
                        summary="Spiced rum",
                    )
                ],
            )
        ],
    )
    result = TurnResolution(
        follow_up=True,
        request_scope="cocktail",
        initial_request="Подбери ром к мясу",
        effective_request="Подобрать коктейли на основе BACARDÍ Spiced.",
    )
    validate_turn_resolution(result, "Какие коктейли из него?", memory)

    with pytest.raises(ValueError, match="BACARDÍ Spiced"):
        validate_turn_resolution(
            result.model_copy(
                update={
                    "effective_request": (
                        "Подобрать коктейли на основе BACARDÍ Coconut rum."
                    )
                }
            ),
            "Какие коктейли из него?",
            memory,
        )


def test_known_product_cocktail_scope_is_prompted_not_forced() -> None:
    memory = SessionMemory(
        session_id="forced-cocktail",
        turns=[
            _turn("rum").model_copy(
                update={
                    "request_scope": "food_pairing",
                    "shown_results": [
                        ShownResult(
                            kind="product",
                            id="spiced",
                            name="BACARDÍ Spiced",
                            summary="Spiced rum",
                        )
                    ],
                }
            )
        ],
    )
    state = AgentState(
        session_id="forced-cocktail",
        user_request="Какие коктейли из него?",
        session_memory=memory,
        user_profile=UserProfile(session_id="forced-cocktail"),
        turn_resolution=TurnResolution(
            follow_up=True,
            request_scope="cocktail",
            initial_request="rum",
            effective_request="Подобрать коктейли на основе BACARDÍ Spiced.",
        ),
    )
    fake = SequenceToolFake([AIMessage(content="I can select cocktails later.")])

    output = tool_calling_agent(
        state,
        config={"configurable": {"tool_llm": fake}},
    )
    assert output["messages"][-1].content == "I can select cocktails later."
    assert "call search_cocktails" in TOOL_PROMPT


def test_final_answer_retries_wrong_kind_for_scope() -> None:
    directory = Path(".test_tmp") / f"scope-kind-{uuid4().hex}"
    repository = SessionRepository(directory / "sommelier.sqlite3")
    product = ProductCandidate(
        id="spiced",
        name="BACARDÍ Spiced",
    )
    cocktail = CocktailCandidate(
        id="spiced-colada",
        name="Spiced Colada",
    )
    state = AgentState(
        session_id="scope-kind",
        user_request="Дай сладкий коктейль",
        session_memory=SessionMemory(session_id="scope-kind"),
        user_profile=UserProfile(session_id="scope-kind"),
        turn_resolution=TurnResolution(
            follow_up=True,
            request_scope="cocktail",
            initial_request="Дай коктейль",
            effective_request="Подобрать сладкий коктейль.",
        ),
        cards=[product, cocktail],
    )
    fake = SequenceStructuredFake(
        [
            FinalAnswerResult(
                answer="BACARDÍ Spiced.",
                shown_refs=[CatalogRef(kind="product", id="spiced")],
                assistant_summary="Wrong product.",
            ),
            FinalAnswerResult(
                answer="Spiced Colada.",
                shown_refs=[
                    CatalogRef(kind="cocktail", id="spiced-colada")
                ],
                assistant_summary="Recommended cocktail.",
            ),
        ]
    )
    try:
        output = generate_answer(
            state,
            config={
                "configurable": {
                    "answer_llm": fake,
                    "repository": repository,
                }
            },
        )
        assert output["final_answer_result"].shown_refs == [
            CatalogRef(kind="cocktail", id="spiced-colada")
        ]
    finally:
        shutil.rmtree(directory, ignore_errors=True)
