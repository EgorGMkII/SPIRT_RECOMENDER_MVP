"""Build compact searchable profiles from CocktailCard JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class CocktailSearchProfile(BaseModel):
    """Minimal cocktail representation optimized for BM25 retrieval."""

    cocktail_id: str
    source_url: HttpUrl
    name: str
    main_rum: str
    description: str
    ingredients: list[str] = Field(default_factory=list)
    recipe_steps: list[str] = Field(default_factory=list)
    searchable_text: str


def _clean_join(values: list[str | None]) -> str:
    """Join non-empty text values into compact plain text."""

    return " ".join(" ".join(value.split()) for value in values if value).strip()


def _ingredient_text(ingredient: dict[str, Any]) -> str:
    """Format one ingredient for display and search."""

    name = str(ingredient.get("name") or "").strip()
    amount = str(ingredient.get("amount") or "").strip()
    if amount and name:
        return f"{amount} {name}"
    return name or amount


def cocktail_card_to_search_profile(card: dict[str, Any]) -> CocktailSearchProfile:
    """Convert one CocktailCard dictionary into a compact search profile."""

    recipe = card.get("recipe") or {}
    ingredients = [
        text
        for text in (_ingredient_text(item) for item in recipe.get("ingredients", []))
        if text
    ]
    recipe_steps = [str(step) for step in recipe.get("steps", []) if step]
    description = _clean_join(
        [
            card.get("short_description"),
            card.get("marketing_description"),
        ]
    )
    searchable_text = _clean_join(
        [
            str(card.get("name") or ""),
            f"main rum {card.get('main_rum')}" if card.get("main_rum") else None,
            description,
            "ingredients " + ", ".join(ingredients) if ingredients else None,
            "recipe steps " + " ".join(recipe_steps) if recipe_steps else None,
        ]
    )
    return CocktailSearchProfile(
        cocktail_id=str(card["cocktail_id"]),
        source_url=card["source_url"],
        name=str(card["name"]),
        main_rum=str(card.get("main_rum") or ""),
        description=description,
        ingredients=ingredients,
        recipe_steps=recipe_steps,
        searchable_text=searchable_text,
    )


def build_cocktail_profile_file(
    input_file: Path,
    output_dir: Path,
    force: bool = False,
) -> Path:
    """Build and save one CocktailSearchProfile JSON file."""

    card = json.loads(input_file.read_text(encoding="utf-8"))
    profile = cocktail_card_to_search_profile(card)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_file.stem}.json"
    if output_file.exists() and not force:
        return output_file
    output_file.write_text(
        json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file


def build_cocktail_profiles(
    input_dir: Path,
    output_dir: Path,
    force: bool = False,
    limit: int | None = None,
) -> list[Path]:
    """Build CocktailSearchProfile files for CocktailCard JSON files."""

    cocktail_files = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if limit is not None:
        cocktail_files = cocktail_files[:limit]
    return [
        build_cocktail_profile_file(path, output_dir=output_dir, force=force)
        for path in cocktail_files
    ]


def load_cocktail_profiles(profiles_dir: Path) -> list[CocktailSearchProfile]:
    """Load CocktailSearchProfile JSON files from a directory."""

    profiles: list[CocktailSearchProfile] = []
    for path in sorted(profiles_dir.glob("*.json")):
        profiles.append(
            CocktailSearchProfile.model_validate_json(path.read_text(encoding="utf-8"))
        )
    return profiles
