from pathlib import Path
from uuid import uuid4

from sommelier.agent.graph import run_agent_turn
from sommelier.agent.memory import FollowupResolution, SessionMemory
from sommelier.agent.profile import ProfileUpdate, UserProfile, apply_profile_update
from sommelier.agent.schemas import IntentType, ParsedIntent, ToolResult
from sommelier.agent.state import AgentState
from sommelier.catalog.cocktail_profiles import CocktailSearchProfile
from sommelier.catalog.search_profiles import product_card_to_search_profile
from sommelier.retrieval.cocktail_search import CocktailSearchResult
from sommelier.retrieval.faiss_index import SearchResult
from sommelier.retrieval.schemas import SearchRequest


def _product(product_id: str, name: str, description: str) -> SearchResult:
    card = {
        "product_id": product_id,
        "source_url": f"https://example.com/{product_id}",
        "brand": "BACARDI",
        "name": name,
        "category": "rum",
        "short_description": description,
        "marketing_description": description,
        "tasting_notes": description,
        "nose": "",
        "palate": "",
        "finish": "",
        "process": "",
        "how_to_serve": "",
        "cocktail_names": [],
        "extraction_warnings": [],
    }
    return SearchResult(
        product_id=product_id,
        score=0.8,
        normalized_query=description,
        profile=product_card_to_search_profile(card, Path(f"{product_id}.json")),
    )


def _cocktail(cocktail_id: str, name: str, description: str) -> CocktailSearchResult:
    profile = CocktailSearchProfile(
        cocktail_id=cocktail_id,
        source_url=f"https://example.com/{cocktail_id}",
        name=name,
        main_rum="BACARDI Carta Blanca rum",
        description=description,
        ingredients=["50 ml BACARDI Carta Blanca rum", "25 ml lime juice"],
        recipe_steps=["Build with ice.", "Serve."],
        searchable_text=f"{name} {description} lime mint rum cocktail",
    )
    return CocktailSearchResult(
        cocktail_id=cocktail_id,
        score=1.0,
        normalized_query="lime mint rum cocktail",
        matched_tokens=["lime", "mint", "rum"],
        profile=profile,
    )


def test_dialog_scenario_routes_memory_profile_and_answers(monkeypatch) -> None:
    session_id = f"dialog-scenario-{uuid4().hex}"
    memory_store: dict[str, SessionMemory] = {}
    profile_store: dict[str, UserProfile] = {}

    messages = [
        "нужен ром для ужина. буду есть мясо с грибами",
        "а какой  вариант попроще?",
        "мне не нравится слишком сладкое и кокос",
        "тогда посоветуй ром для коктейлей, но без сладкого профиля",
        "какой коктейль сделать с лаймом и мятой?",
        "а есть что-то похожее, но проще?",
        "мне нравится мохито, но не люблю пина коладу",
        "посоветуй освежающий коктейль",
        "хочу ром с дубом, ванилью и пряностями.",
    ]

    def fake_load_memory(current_session_id: str) -> SessionMemory:
        return memory_store.get(current_session_id, SessionMemory(session_id=current_session_id))

    def fake_save_memory(memory: SessionMemory) -> Path:
        memory_store[memory.session_id] = memory
        return Path("memory.json")

    def fake_load_profile(current_session_id: str) -> UserProfile:
        return profile_store.get(current_session_id, UserProfile(session_id=current_session_id))

    def fake_save_profile(profile: UserProfile) -> Path:
        profile_store[profile.session_id] = profile
        return Path("profile.json")

    def fake_followup(message: str, memory: SessionMemory, use_llm: bool = False) -> FollowupResolution:
        if message == "а какой  вариант попроще?":
            return FollowupResolution(
                is_followup=True,
                intent=IntentType.FOOD_PAIRING,
                effective_user_message="simpler rum option for dinner with meat and mushrooms",
                avoid_previous_candidates=False,
                reason="User asks to simplify the previous dinner recommendation.",
            )
        if message == "а есть что-то похожее, но проще?":
            return FollowupResolution(
                is_followup=True,
                intent=IntentType.COCKTAIL_EXPANSION,
                effective_user_message="simpler alternative to Mojito, not Mojito",
                avoid_previous_candidates=True,
                reason="User asks for a similar but simpler cocktail alternative.",
            )
        return FollowupResolution(
            is_followup=False,
            intent=None,
            effective_user_message=message,
            avoid_previous_candidates=False,
            reason="Standalone request.",
        )

    def fake_parse_intent(message: str, use_llm: bool = False) -> ParsedIntent:
        if message in {
            "мне не нравится слишком сладкое и кокос",
            "мне нравится мохито, но не люблю пина коладу",
        }:
            intent = IntentType.PROFILE_UPDATE
        elif "коктейль" in message:
            intent = IntentType.COCKTAIL_EXPANSION
        elif "ужин" in message or "мясо" in message or "гриб" in message:
            intent = IntentType.FOOD_PAIRING
        else:
            intent = IntentType.SEARCH_PRODUCTS
        return ParsedIntent(
            intent=intent,
            search=SearchRequest(query=message),
            confidence=0.95,
        )

    def fake_profile_update(profile: UserProfile, message: str, llm=None, use_llm: bool = False):
        if message == "мне не нравится слишком сладкое и кокос":
            update = ProfileUpdate(disliked_flavors=["sweet", "coconut"])
        elif message == "мне нравится мохито, но не люблю пина коладу":
            update = ProfileUpdate(
                liked_cocktails=["mojito"],
                disliked_cocktails=["pina colada"],
            )
        else:
            update = ProfileUpdate(ignored=[message])
        updated = apply_profile_update(profile, update)
        changed = {
            "liked_flavors": update.liked_flavors,
            "disliked_flavors": update.disliked_flavors,
            "liked_cocktails": update.liked_cocktails,
            "disliked_cocktails": update.disliked_cocktails,
        }
        return updated, ToolResult(
            tool_name="profile_update",
            summary="Profile update applied.",
            metadata={"changed": changed, "update": update.model_dump(mode="json")},
        )

    def fake_hybrid_search(query: str, normalize: bool = True, use_llm_query: bool = False):
        if "проще" in query or "simpler" in query or "cocktail" in query:
            results = [
                _product("bacardi-carta-blanca", "BACARDI Carta Blanca", "Dry light rum for cocktails."),
                _product("bacardi-anejo-cuatro", "BACARDI Anejo Cuatro", "Oak vanilla spice rum."),
            ]
        elif "дуб" in query or "ваниль" in query or "пряност" in query:
            results = [
                _product("bacardi-reserva-ocho", "BACARDI Reserva Ocho", "Oak vanilla spice aged rum."),
                _product("bacardi-anejo-cuatro", "BACARDI Anejo Cuatro", "Vanilla oak clove rum."),
            ]
        else:
            results = [
                _product("bacardi-reserva-ocho", "BACARDI Reserva Ocho", "Rich dinner rum."),
                _product("bacardi-spiced", "BACARDI Spiced", "Simpler spiced rum."),
            ]
        return results, {result.product_id: ["fake"] for result in results}, {"normalized_query": query}

    def fake_search_cocktails(query: str, top_k: int = 5):
        if "alternative" in query or "not Mojito" in query:
            return [
                _cocktail("mojito", "Mojito", "Fresh lime and mint cocktail."),
                _cocktail("daiquiri", "Daiquiri", "Simpler lime rum cocktail without mint."),
                _cocktail("mojito-jug", "Mojito jug", "Large-format Mojito."),
            ]
        if "освежающий" in query:
            return [
                _cocktail("mojito", "Mojito", "Fresh lime and mint cocktail."),
                _cocktail("daiquiri", "Daiquiri", "Simple bright lime cocktail."),
            ]
        return [
            _cocktail("mojito", "Mojito", "Fresh lime and mint cocktail."),
            _cocktail("old-cuban", "Old Cuban", "Mint lime sparkling cocktail."),
        ]

    def fake_sommelier_answer(state: AgentState) -> str:
        first = state.retrieval_results[0].profile.name
        return f"Рекомендую {first}: подходит под запрос."

    def fake_cocktail_answer(state: AgentState) -> str:
        first = state.cocktail_results[0].profile.name
        return f"Рекомендую {first}. Рецепт: {first} ingredients and steps."

    monkeypatch.setattr("sommelier.agent.graph.load_session_memory", fake_load_memory)
    monkeypatch.setattr("sommelier.agent.graph.save_session_memory", fake_save_memory)
    monkeypatch.setattr("sommelier.agent.graph.load_user_profile", fake_load_profile)
    monkeypatch.setattr("sommelier.agent.graph.save_user_profile", fake_save_profile)
    monkeypatch.setattr("sommelier.agent.graph.resolve_followup_context", fake_followup)
    monkeypatch.setattr("sommelier.agent.graph.parse_intent", fake_parse_intent)
    monkeypatch.setattr("sommelier.agent.graph.profile_update", fake_profile_update)
    monkeypatch.setattr("sommelier.agent.graph.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        "sommelier.agent.graph.normalize_food_pairing_query",
        lambda food_text, use_llm=False: food_text,
    )
    monkeypatch.setattr("sommelier.agent.graph.search_cocktails", fake_search_cocktails)
    monkeypatch.setattr("sommelier.agent.graph.generate_sommelier_answer", fake_sommelier_answer)
    monkeypatch.setattr("sommelier.agent.graph.generate_cocktail_answer", fake_cocktail_answer)
    monkeypatch.setattr(
        "sommelier.agent.graph.append_trace_events",
        lambda session_id, turn_id, traces: Path("trace.jsonl"),
    )

    transcript: list[tuple[str, str, str | None]] = []
    states: list[AgentState] = []
    for message in messages:
        state = run_agent_turn(
            AgentState(
                session_id=session_id,
                user_message=message,
                persist_artifacts=True,
                use_llm_followup=True,
                use_llm_intent=True,
                use_llm_query=True,
                use_llm_food_query=True,
                use_llm_profile_update=True,
                use_llm_answer=True,
            )
        )
        states.append(state)
        transcript.append((message, state.final_answer or "", state.parsed_intent.intent if state.parsed_intent else None))

    for user_message, answer, intent in transcript:
        print(f"\nUSER: {user_message}\nINTENT: {intent}\nASSISTANT: {answer}")

    assert states[2].final_answer == "Учту ваше предпочтение."
    assert states[2].profile_updated is True
    assert states[3].parsed_intent.intent == IntentType.SEARCH_PRODUCTS
    assert states[3].final_answer != "Учту ваше предпочтение."
    assert states[5].avoid_previous_candidates is True
    assert states[5].avoid_candidate_ids == ["mojito"]
    assert states[5].cocktail_results[0].cocktail_id == "daiquiri"
    assert "Daiquiri" in (states[5].final_answer or "")
    assert states[6].parsed_intent.intent == IntentType.PROFILE_UPDATE
    assert states[6].final_answer == "Учту ваше предпочтение."
    assert states[7].parsed_intent.intent == IntentType.COCKTAIL_EXPANSION
    assert "Mojito" in (states[7].final_answer or "")
    assert states[8].parsed_intent.intent == IntentType.SEARCH_PRODUCTS
    assert "BACARDI Reserva Ocho" in (states[8].final_answer or "")
