import json
import shutil
from pathlib import Path
from uuid import uuid4

from sommelier.catalog.cocktail_profiles import (
    build_cocktail_profiles,
    cocktail_card_to_search_profile,
    load_cocktail_profiles,
)


def _cocktail_card(cocktail_id: str = "mojito") -> dict:
    return {
        "cocktail_id": cocktail_id,
        "source_url": f"https://www.bacardi.com/rum-cocktails/{cocktail_id}/",
        "brand": "Bacardi",
        "name": "Mojito",
        "title": "Mojito Rum Cocktail",
        "main_rum": "BACARDÍ Carta Blanca rum",
        "short_description": "A refreshing mint and lime rum cocktail.",
        "marketing_description": "A classic highball for warm evenings.",
        "recipe": {
            "servings": "1",
            "prep_time": "5 minutes",
            "difficulty": "easy",
            "ingredients": [
                {"name": "BACARDÍ Carta Blanca rum", "amount": "50 ml"},
                {"name": "lime juice", "amount": "25 ml"},
                {"name": "mint", "amount": "8 leaves"},
            ],
            "steps": ["Build over ice.", "Top with soda."],
        },
        "glassware": "highball",
        "garnish": "mint sprig",
        "method": "build",
        "raw_text_excerpt": "Mojito BACARDÍ Carta Blanca lime mint soda",
        "source_metadata": {},
        "extraction_confidence": 0.9,
        "extraction_warnings": [],
    }


def test_cocktail_card_to_search_profile_is_compact() -> None:
    profile = cocktail_card_to_search_profile(_cocktail_card())

    assert profile.cocktail_id == "mojito"
    assert profile.main_rum == "BACARDÍ Carta Blanca rum"
    assert profile.ingredients == [
        "50 ml BACARDÍ Carta Blanca rum",
        "25 ml lime juice",
        "8 leaves mint",
    ]
    assert profile.recipe_steps == ["Build over ice.", "Top with soda."]
    assert "mint and lime" in profile.searchable_text
    assert "glassware" not in profile.model_dump()


def test_build_and_load_cocktail_profiles() -> None:
    work_dir = Path(".test_tmp") / f"cocktail-profiles-{uuid4().hex}"
    input_dir = work_dir / "cards"
    output_dir = work_dir / "profiles"
    try:
        input_dir.mkdir(parents=True)
        (input_dir / "mojito.json").write_text(
            json.dumps(_cocktail_card(), ensure_ascii=False),
            encoding="utf-8",
        )

        files = build_cocktail_profiles(input_dir, output_dir, force=True)
        profiles = load_cocktail_profiles(output_dir)

        assert len(files) == 1
        assert len(profiles) == 1
        assert profiles[0].name == "Mojito"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
