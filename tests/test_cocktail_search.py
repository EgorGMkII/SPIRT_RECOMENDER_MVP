import json
import shutil
from pathlib import Path
from uuid import uuid4

from sommelier.catalog.cocktail_profiles import build_cocktail_profiles
from sommelier.retrieval.cocktail_query_normalizer import normalize_cocktail_query
from sommelier.retrieval.cocktail_search import search_cocktails


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeMessage:
        self.prompts.append(prompt)
        return FakeMessage(self.response)


def _card(
    cocktail_id: str,
    name: str,
    main_rum: str,
    description: str,
    ingredients: list[dict],
) -> dict:
    return {
        "cocktail_id": cocktail_id,
        "source_url": f"https://www.bacardi.com/rum-cocktails/{cocktail_id}/",
        "brand": "Bacardi",
        "name": name,
        "title": name,
        "main_rum": main_rum,
        "short_description": description,
        "marketing_description": "",
        "recipe": {
            "servings": "1",
            "prep_time": None,
            "difficulty": None,
            "ingredients": ingredients,
            "steps": ["Mix ingredients and serve over ice."],
        },
        "glassware": None,
        "garnish": None,
        "method": None,
        "raw_text_excerpt": description,
        "source_metadata": {},
        "extraction_confidence": 0.9,
        "extraction_warnings": [],
    }


def _profiles_dir(work_dir: Path) -> Path:
    input_dir = work_dir / "cards"
    output_dir = work_dir / "profiles"
    input_dir.mkdir(parents=True)
    cards = [
        _card(
            "mojito",
            "Mojito",
            "BACARDÍ Carta Blanca rum",
            "A refreshing mint and lime white rum cocktail.",
            [
                {"name": "BACARDÍ Carta Blanca rum", "amount": "50 ml"},
                {"name": "lime juice", "amount": "25 ml"},
                {"name": "mint", "amount": "8 leaves"},
            ],
        ),
        _card(
            "pina-colada",
            "Piña Colada",
            "BACARDÍ Coconut rum",
            "A creamy pineapple and coconut rum cocktail.",
            [
                {"name": "BACARDÍ Coconut rum", "amount": "50 ml"},
                {"name": "pineapple juice", "amount": "35 ml"},
                {"name": "coconut cream", "amount": "25 ml"},
            ],
        ),
    ]
    for card in cards:
        (input_dir / f"{card['cocktail_id']}.json").write_text(
            json.dumps(card, ensure_ascii=False),
            encoding="utf-8",
        )
    build_cocktail_profiles(input_dir, output_dir, force=True)
    return output_dir


def test_normalize_cocktail_query_uses_llm() -> None:
    llm = FakeLLM("mojito cocktail recipe mint lime white rum")

    normalized = normalize_cocktail_query("give me a mojito recipe", llm=llm)

    assert normalized == "mojito cocktail recipe mint lime white rum"
    assert "give me a mojito recipe" in llm.prompts[0]


def test_search_cocktails_returns_mojito_for_llm_normalized_query() -> None:
    work_dir = Path(".test_tmp") / f"cocktail-search-{uuid4().hex}"
    try:
        results = search_cocktails(
            "give me a mojito recipe",
            top_k=1,
            profiles_dir=_profiles_dir(work_dir),
            llm=FakeLLM("mojito cocktail recipe mint lime white rum"),
        )

        assert results[0].cocktail_id == "mojito"
        assert "mojito" in results[0].matched_tokens
        assert results[0].normalized_query == "mojito cocktail recipe mint lime white rum"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_search_cocktails_returns_pina_colada_for_pineapple_coconut() -> None:
    work_dir = Path(".test_tmp") / f"cocktail-search-{uuid4().hex}"
    try:
        results = search_cocktails(
            "cocktail with pineapple and coconut",
            top_k=1,
            profiles_dir=_profiles_dir(work_dir),
            llm=FakeLLM("pineapple coconut rum cocktail"),
        )

        assert results[0].cocktail_id == "pina-colada"
        assert {"pineapple", "coconut"}.issubset(set(results[0].matched_tokens))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
