from pathlib import Path
from uuid import uuid4

from sommelier.agent.graph import (
    answer_generation_node,
    build_cocktail_answer_prompt,
    build_sommelier_answer_prompt,
    cocktail_search_node,
    direct_search_node,
    food_pairing_search_node,
    hybrid_search,
    parse_intent_node,
    retrieval_node,
    run_agent_turn,
)
from sommelier.agent.schemas import IntentType, ParsedIntent
from sommelier.agent.state import AgentState
from sommelier.agent.profile import UserProfile
from sommelier.agent.profile import ProfileUpdate
from sommelier.agent.memory import CandidateMemory, LastTurnMemory, SessionMemory
from sommelier.catalog.cocktail_profiles import CocktailSearchProfile
from sommelier.catalog.search_profiles import product_card_to_search_profile
from sommelier.retrieval.cocktail_search import CocktailSearchResult
from sommelier.retrieval.faiss_index import SearchResult
from sommelier.retrieval.schemas import SearchRequest


def _profile(product_id: str = "bacardi-anejo-cuatro-rum"):
    card = {
        "product_id": product_id,
        "source_url": f"https://example.com/{product_id}",
        "brand": "Bacardi",
        "name": "BACARDI Anejo Cuatro",
        "category": "gold rum",
        "short_description": "Smooth gold rum for cocktails.",
        "marketing_description": "Vanilla, oak, honey and clove rum for cocktails.",
        "tasting_notes": "Mild vanilla, toasted oak, clove and honey.",
        "nose": "Vanilla and cinnamon",
        "palate": "Dark honey and clove",
        "finish": "Toffee and oak",
        "process": "Aged rum blend.",
        "how_to_serve": "Use in sophisticated cocktails.",
        "cocktail_names": ["Cuatro Highball"],
        "extraction_warnings": [],
    }
    return product_card_to_search_profile(card, Path(f"{product_id}.json"))


class StubIndex:
    def __init__(self, product_id: str = "bacardi-anejo-cuatro-rum") -> None:
        self.queries: list[str] = []
        self.product_id = product_id

    def search(self, query_text: str, top_k: int = 5, normalize: bool = True):
        self.queries.append(query_text)
        return [
            SearchResult(
                product_id=self.product_id,
                score=0.91,
                normalized_query=query_text,
                profile=_profile(self.product_id),
            )
        ][:top_k]


class StubBm25Result:
    def __init__(self, result: SearchResult) -> None:
        self.result = result
        self.matched_tokens = ["vanilla", "oak"]


class StubBm25Index:
    def __init__(self, product_id: str = "bacardi-anejo-cuatro-rum") -> None:
        self.queries: list[str] = []
        self.product_id = product_id

    def search(self, query_text: str, top_k: int = 5, normalize: bool = True):
        self.queries.append(query_text)
        return [
            StubBm25Result(
                SearchResult(
                    product_id=self.product_id,
                    score=1.7,
                    normalized_query=query_text,
                    profile=_profile(self.product_id),
                )
            )
        ][:top_k]


class FailingIndex:
    def search(self, query_text: str, top_k: int = 5, normalize: bool = True):
        raise RuntimeError("Connection error.")


def _cocktail_result(cocktail_id: str = "mojito") -> CocktailSearchResult:
    names = {
        "mojito": "Mojito",
        "daiquiri": "Daiquiri",
    }
    name = names.get(cocktail_id, cocktail_id)
    profile = CocktailSearchProfile(
        cocktail_id=cocktail_id,
        source_url=f"https://example.com/{cocktail_id}",
        name=name,
        main_rum="BACARDI Carta Blanca rum",
        description="A refreshing mint and lime rum cocktail.",
        ingredients=["50 ml BACARDI Carta Blanca rum", "25 ml lime juice", "8 mint leaves"],
        recipe_steps=["Build over ice.", "Top with soda."],
        searchable_text="Mojito BACARDI Carta Blanca rum lime mint soda cocktail recipe",
    )
    return CocktailSearchResult(
        cocktail_id=cocktail_id,
        score=3.2,
        normalized_query="mojito cocktail recipe mint lime white rum",
        matched_tokens=["mojito", "mint", "lime", "rum"],
        profile=profile,
    )


def _cocktail_memory_for_prompt() -> SessionMemory:
    return SessionMemory(
        session_id="s1",
        last_turn=LastTurnMemory(
            user_message="какой коктейль сделать с лаймом и мятой?",
            effective_user_message="cocktail with lime and mint",
            intent="cocktail_expansion",
            cocktail_query="lime mint cocktail",
            final_answer="Лучший вариант — Mojito.",
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


def _parsed_intent(intent: IntentType, message: str) -> ParsedIntent:
    return ParsedIntent(
        intent=intent,
        search=SearchRequest(query=message),
        confidence=0.9,
    )


def test_parse_intent_node_routes_russian_food_query(monkeypatch) -> None:
    monkeypatch.setattr(
        "sommelier.agent.graph.parse_intent",
        lambda message, use_llm=False: _parsed_intent(IntentType.FOOD_PAIRING, message),
    )
    state = parse_intent_node(
        AgentState(
            session_id="s1",
            user_message="Ем шашлык из свинины, какой ром подойдёт?",
            use_llm_intent=True,
        )
    )

    assert state.parsed_intent is not None
    assert state.parsed_intent.intent == IntentType.FOOD_PAIRING


def test_parse_intent_node_routes_cocktail_query(monkeypatch) -> None:
    monkeypatch.setattr(
        "sommelier.agent.graph.parse_intent",
        lambda message, use_llm=False: _parsed_intent(
            IntentType.COCKTAIL_EXPANSION,
            message,
        ),
    )
    state = parse_intent_node(
        AgentState(session_id="s1", user_message="Дай рецепт мохито", use_llm_intent=True)
    )

    assert state.parsed_intent is not None
    assert state.parsed_intent.intent == IntentType.COCKTAIL_EXPANSION


def test_direct_search_node_uses_user_query(monkeypatch) -> None:
    index = StubIndex()
    bm25 = StubBm25Index()
    monkeypatch.setattr("sommelier.agent.graph.load_agent_index", lambda: index)
    monkeypatch.setattr("sommelier.agent.graph.load_agent_bm25_index", lambda: bm25)
    state = parse_intent_node(
        AgentState(session_id="s1", user_message="I want vanilla and oak rum for cocktails")
    )

    state = direct_search_node(state)

    assert index.queries == ["I want vanilla and oak rum for cocktails"]
    assert bm25.queries == ["I want vanilla and oak rum for cocktails"]
    assert state.retrieval_results[0].product_id == "bacardi-anejo-cuatro-rum"
    assert state.retrieval_sources["bacardi-anejo-cuatro-rum"] == ["faiss", "bm25"]


def test_hybrid_search_merges_faiss_top_one_and_bm25_top_one(monkeypatch) -> None:
    monkeypatch.setattr(
        "sommelier.agent.graph.load_agent_index",
        lambda: StubIndex(product_id="faiss-rum"),
    )
    monkeypatch.setattr(
        "sommelier.agent.graph.load_agent_bm25_index",
        lambda: StubBm25Index(product_id="bm25-rum"),
    )

    results, sources, debug = hybrid_search("vanilla oak rum")

    assert [result.product_id for result in results] == ["faiss-rum", "bm25-rum"]
    assert sources == {"faiss-rum": ["faiss"], "bm25-rum": ["bm25"]}
    assert debug["faiss_top"] == ["faiss-rum"]
    assert debug["bm25_top"] == ["bm25-rum"]


def test_hybrid_search_falls_back_to_bm25_when_faiss_fails(monkeypatch) -> None:
    monkeypatch.setattr("sommelier.agent.graph.load_agent_index", lambda: FailingIndex())
    monkeypatch.setattr(
        "sommelier.agent.graph.load_agent_bm25_index",
        lambda: StubBm25Index(product_id="bm25-rum"),
    )

    results, sources, debug = hybrid_search("нужен ром к мясу с грибами")

    assert [result.product_id for result in results] == ["bm25-rum"]
    assert sources == {"bm25-rum": ["bm25"]}
    assert debug["faiss_error"] == "Connection error."
    assert debug["bm25_top"] == ["bm25-rum"]


def test_parse_intent_routes_dinner_meat_mushroom_query(monkeypatch) -> None:
    monkeypatch.setattr(
        "sommelier.agent.graph.parse_intent",
        lambda message, use_llm=False: _parsed_intent(IntentType.FOOD_PAIRING, message),
    )
    state = parse_intent_node(
        AgentState(
            session_id="s1",
            user_message="нужен ром для ужина. буду есть мясо с грибами",
            use_llm_intent=True,
        )
    )

    assert state.parsed_intent is not None
    assert state.parsed_intent.intent == IntentType.FOOD_PAIRING


def test_food_pairing_search_node_expands_food_query(monkeypatch) -> None:
    index = StubIndex()
    bm25 = StubBm25Index()
    monkeypatch.setattr("sommelier.agent.graph.load_agent_index", lambda: index)
    monkeypatch.setattr("sommelier.agent.graph.load_agent_bm25_index", lambda: bm25)
    state = parse_intent_node(
        AgentState(session_id="s1", user_message="шашлык из свинины")
    )

    state = food_pairing_search_node(state)

    assert state.expanded_query is not None
    assert state.expanded_query.startswith("Rum for food pairing with this dish:")
    assert index.queries == [state.expanded_query]
    assert bm25.queries == [state.expanded_query]
    assert state.retrieval_caveat is not None


def test_minimal_agent_generates_answer_with_candidates(monkeypatch) -> None:
    monkeypatch.setattr("sommelier.agent.graph.load_agent_index", lambda: StubIndex())
    monkeypatch.setattr(
        "sommelier.agent.graph.load_agent_bm25_index",
        lambda: StubBm25Index(),
    )

    state = run_agent_turn(
        AgentState(
            session_id="s1",
            user_message="I want vanilla and oak rum for cocktails",
            persist_artifacts=False,
        )
    )

    assert state.final_answer is not None
    assert "BACARDI Anejo Cuatro" in state.final_answer
    assert "Нормализованный запрос" in state.final_answer
    assert [trace.tool_name for trace in state.tool_traces] == [
        "memory_load",
        "parse_intent",
        "profile_update",
        "direct_search",
        "retrieval",
        "answer_generation",
    ]


def test_cocktail_search_node_uses_cocktail_retrieval(monkeypatch) -> None:
    def fake_search(query, top_k=5):
        assert query == "Дай рецепт мохито"
        assert top_k == 5
        return [_cocktail_result()]

    monkeypatch.setattr("sommelier.agent.graph.search_cocktails", fake_search)
    state = parse_intent_node(
        AgentState(session_id="s1", user_message="Дай рецепт мохито")
    )

    state = cocktail_search_node(state)

    assert state.cocktail_results[0].cocktail_id == "mojito"
    assert state.cocktail_query == "mojito cocktail recipe mint lime white rum"
    assert state.tool_traces[-1].tool_name == "cocktail_search"


def test_minimal_agent_generates_cocktail_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        "sommelier.agent.graph.search_cocktails",
        lambda query, top_k=5: [_cocktail_result()],
    )
    monkeypatch.setattr(
        "sommelier.agent.graph.parse_intent",
        lambda message, use_llm=False: _parsed_intent(
            IntentType.COCKTAIL_EXPANSION,
            message,
        ),
    )

    state = run_agent_turn(
        AgentState(
            session_id=f"profile-update-test-{uuid4().hex}",
            user_message="Дай рецепт мохито",
            persist_artifacts=False,
            use_llm_intent=True,
        )
    )

    assert state.final_answer is not None
    assert "Mojito" in state.final_answer
    assert "50 ml BACARDI Carta Blanca rum" in state.final_answer
    assert [trace.tool_name for trace in state.tool_traces] == [
        "memory_load",
        "parse_intent",
        "profile_update",
        "cocktail_search",
        "retrieval",
        "answer_generation",
    ]


def test_profile_update_turn_acknowledges_without_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(
        "sommelier.agent.graph.parse_intent",
        lambda message, use_llm=False: _parsed_intent(IntentType.PROFILE_UPDATE, message),
    )
    monkeypatch.setattr(
        "sommelier.agent.tools.profile_update.extract_profile_update",
        lambda message, llm=None, use_llm=False: ProfileUpdate(
            disliked_flavors=["sweet", "coconut"],
        ),
    )

    state = run_agent_turn(
        AgentState(
            session_id=f"profile-update-test-{uuid4().hex}",
            user_message="мне не нравится слишком сладкое и кокос",
            persist_artifacts=False,
            use_llm_intent=True,
            use_llm_profile_update=True,
        )
    )

    assert state.final_answer == "Учту ваше предпочтение."
    assert state.profile_updated is True
    assert state.user_profile is not None
    assert state.user_profile.disliked_flavors == ["coconut", "sweet"]
    assert [trace.tool_name for trace in state.tool_traces] == [
        "memory_load",
        "parse_intent",
        "profile_update",
        "profile_ack",
    ]


def test_recommendation_with_preference_does_not_stop_at_profile_ack(monkeypatch) -> None:
    monkeypatch.setattr(
        "sommelier.agent.graph.parse_intent",
        lambda message, use_llm=False: _parsed_intent(IntentType.SEARCH_PRODUCTS, message),
    )
    monkeypatch.setattr("sommelier.agent.graph.load_agent_index", lambda: StubIndex())
    monkeypatch.setattr(
        "sommelier.agent.graph.load_agent_bm25_index",
        lambda: StubBm25Index(),
    )

    state = run_agent_turn(
        AgentState(
            session_id=f"search-with-preference-{uuid4().hex}",
            user_message="тогда посоветуй ром для коктейлей, но без сладкого профиля",
            persist_artifacts=False,
            use_llm_intent=True,
            use_llm_profile_update=True,
        )
    )

    assert state.profile_updated is False
    assert state.final_answer is not None
    assert state.final_answer != "Учту."
    assert [trace.tool_name for trace in state.tool_traces] == [
        "memory_load",
        "parse_intent",
        "profile_update",
        "direct_search",
        "retrieval",
        "answer_generation",
    ]


def test_answer_generation_includes_food_pairing_caveat() -> None:
    state = AgentState(session_id="s1", user_message="стейк")
    state.expanded_query = "Aged rum with oak, vanilla, spice and caramel for grilled beef."
    state.retrieval_caveat = "Pairing is inferred."
    state.retrieval_results = [
        SearchResult(
            product_id="bacardi-anejo-cuatro-rum",
            score=0.8,
            normalized_query=state.expanded_query,
            profile=_profile(),
        )
    ]

    state = retrieval_node(state)
    state = answer_generation_node(state)

    assert state.final_answer is not None
    assert "Pairing is inferred." in state.final_answer
    assert "Поисковая формулировка" in state.final_answer


def test_sommelier_answer_prompt_contains_guardrails() -> None:
    state = AgentState(
        session_id="s1",
        user_message="I want vanilla rum",
        user_profile=UserProfile(
            session_id="s1",
            liked_flavors=["vanilla"],
            disliked_flavors=["coconut"],
        ),
    )
    state.retrieval_results = [
        SearchResult(
            product_id="bacardi-anejo-cuatro-rum",
            score=0.8,
            normalized_query="Rum with vanilla flavors.",
            profile=_profile(),
        )
    ]

    prompt = build_sommelier_answer_prompt(state)

    assert "Use only the product candidates and evidence provided" in prompt
    assert "Do not invent products" in prompt
    assert "BACARDI Anejo Cuatro" in prompt
    assert "Vanilla and cinnamon" in prompt
    assert "disliked_flavors" in prompt
    assert "coconut" in prompt


def test_cocktail_answer_prompt_contains_recipe_guardrails() -> None:
    state = AgentState(
        session_id="s1",
        user_message="Дай рецепт мохито",
        user_profile=UserProfile(session_id="s1", liked_cocktails=["mojito"]),
    )
    state.parsed_intent = parse_intent_node(state).parsed_intent
    state.cocktail_results = [_cocktail_result()]
    state.cocktail_query = "mojito cocktail recipe mint lime white rum"

    prompt = build_cocktail_answer_prompt(state)

    assert "Use only the cocktail candidates and evidence provided" in prompt
    assert "Do not invent cocktail names" in prompt
    assert "translate the step prose into the user's language" in prompt
    assert "50 ml BACARDI Carta Blanca rum" in prompt
    assert "Build over ice." in prompt
    assert "liked_cocktails" in prompt
    assert "mojito" in prompt


def test_cocktail_answer_prompt_contains_previous_turn_for_alternatives() -> None:
    state = AgentState(
        session_id="s1",
        user_message="а есть что-то похожее, но проще?",
        effective_user_message="simpler alternative to Mojito",
        is_followup=True,
        followup_intent=IntentType.COCKTAIL_EXPANSION,
        avoid_previous_candidates=False,
    )
    state.session_memory = _cocktail_memory_for_prompt()
    state.parsed_intent = _parsed_intent(
        IntentType.COCKTAIL_EXPANSION,
        "simpler alternative to Mojito",
    )
    state.cocktail_results = [_cocktail_result("mojito"), _cocktail_result("daiquiri")]

    prompt = build_cocktail_answer_prompt(state)

    assert "previous_turn" in prompt
    assert "Mojito" in prompt
    assert "already offered" in prompt
    assert "choose a different cocktail candidate" in prompt


def test_answer_generation_can_use_llm_writer(monkeypatch) -> None:
    state = AgentState(
        session_id="s1",
        user_message="I want vanilla rum",
        use_llm_answer=True,
    )
    state.retrieval_results = [
        SearchResult(
            product_id="bacardi-anejo-cuatro-rum",
            score=0.8,
            normalized_query="Rum with vanilla flavors.",
            profile=_profile(),
        )
    ]
    monkeypatch.setattr(
        "sommelier.agent.graph.generate_sommelier_answer",
        lambda current_state: "Sommelier answer from validated evidence.",
    )

    state = answer_generation_node(state)

    assert state.final_answer == "Sommelier answer from validated evidence."
    assert state.tool_traces[-1].input["mode"] == "llm"


def test_answer_generation_can_use_cocktail_llm_writer(monkeypatch) -> None:
    state = AgentState(
        session_id="s1",
        user_message="Дай рецепт мохито",
        use_llm_answer=True,
    )
    state.cocktail_results = [_cocktail_result()]
    monkeypatch.setattr(
        "sommelier.agent.graph.generate_cocktail_answer",
        lambda current_state: "Cocktail answer from validated recipe.",
    )

    state = answer_generation_node(state)

    assert state.final_answer == "Cocktail answer from validated recipe."
    assert state.tool_traces[-1].input["mode"] == "llm"
    assert state.tool_traces[-1].input["answer_type"] == "cocktail"
