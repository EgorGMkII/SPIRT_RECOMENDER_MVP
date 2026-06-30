"""The four read-only catalog model tools."""

from pathlib import Path
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from sommelier.agent.contracts import (
    CocktailCandidate,
    CocktailSearchOutput,
    ProductCandidate,
    ProductSearchOutput,
)
from sommelier.agent.search_runtime import hybrid_search
from sommelier.catalog.cocktail_profiles import load_cocktail_profiles
from sommelier.catalog.search_profiles import load_search_profiles
from sommelier.retrieval.cocktail_search import CocktailBm25Index
from sommelier.retrieval.food_pairing_query import DEFAULT_PAIRING_CAVEAT

PRODUCT_PROFILES_DIR = Path("data/catalog/search_profiles")
COCKTAIL_PROFILES_DIR = Path("data/catalog/cocktail_search_profiles")


class ProductSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=600)
    limit: int = Field(default=7, ge=1, le=7)


class FoodSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    food_description: str = Field(min_length=1, max_length=600)
    search_query: str = Field(min_length=1, max_length=600)
    limit: int = Field(default=7, ge=1, le=7)


class CocktailSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=600)
    limit: int = Field(default=7, ge=1, le=7)


class LookupByIdInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["product", "cocktail"]
    id: str = Field(min_length=1, max_length=300)


def _product_candidate(profile, sources: list[str] | None = None) -> ProductCandidate:
    return ProductCandidate(
        id=profile.product_id,
        name=profile.name,
        category=profile.category or "",
        display_description=profile.display_description or "",
        tasting_summary=profile.tasting_summary or "",
        how_to_serve=str(profile.evidence_fields.get("how_to_serve", "")),
        cocktail_names=profile.cocktail_names,
        evidence_fields={
            key: str(value) for key, value in profile.evidence_fields.items()
        },
        retrieval_sources=sources or [],
    )


def _cocktail_candidate(profile) -> CocktailCandidate:
    return CocktailCandidate(
        id=profile.cocktail_id,
        name=profile.name,
        main_rum=profile.main_rum,
        description=profile.description,
        ingredients=profile.ingredients,
        recipe_steps=profile.recipe_steps,
    )


def _product_search(query: str, limit: int = 7) -> dict:
    results, sources, _ = hybrid_search(
        query,
        normalize=False,
        faiss_top_k=limit,
        bm25_top_k=limit,
        limit=limit,
    )
    return ProductSearchOutput(
        query=query,
        candidates=[
            _product_candidate(item.profile, sources.get(item.product_id, []))
            for item in results
        ],
    ).model_dump(mode="json")


def _food_search(food_description: str, search_query: str, limit: int = 7) -> dict:
    output = ProductSearchOutput.model_validate(_product_search(search_query, limit))
    output.caveats = [DEFAULT_PAIRING_CAVEAT]
    return output.model_dump(mode="json")


def _cocktail_search(query: str, limit: int = 7) -> dict:
    results = CocktailBm25Index.load(COCKTAIL_PROFILES_DIR).search(
        query, top_k=limit
    )
    return CocktailSearchOutput(
        query=query,
        candidates=[_cocktail_candidate(item.profile) for item in results],
    ).model_dump(mode="json")


def _lookup_by_id(kind: str, id: str) -> dict:
    if kind == "product":
        profiles = {
            profile.product_id: profile
            for profile in load_search_profiles(PRODUCT_PROFILES_DIR)
        }
        if id not in profiles:
            raise ValueError("unknown product id")
        card = _product_candidate(profiles[id])
    else:
        profiles = {
            profile.cocktail_id: profile
            for profile in load_cocktail_profiles(COCKTAIL_PROFILES_DIR)
        }
        if id not in profiles:
            raise ValueError("unknown cocktail id")
        card = _cocktail_candidate(profiles[id])
    return {"card": card.model_dump(mode="json")}


search_products = StructuredTool.from_function(
    func=_product_search,
    name="search_products",
    description="Search Bacardi rum products using a positive English query.",
    args_schema=ProductSearchInput,
)
search_products_for_food = StructuredTool.from_function(
    func=_food_search,
    name="search_products_for_food",
    description=(
        "Search rum to pair with edible food (a dish, ingredient, meal, or "
        "cuisine). Never use for choosing rum for a cocktail or mixed drink."
    ),
    args_schema=FoodSearchInput,
)
search_cocktails = StructuredTool.from_function(
    func=_cocktail_search,
    name="search_cocktails",
    description="Search cocktail recipes using a positive English query.",
    args_schema=CocktailSearchInput,
)
lookup_by_id = StructuredTool.from_function(
    func=_lookup_by_id,
    name="lookup_by_id",
    description="Load one full card by a kind/id already shown to the user.",
    args_schema=LookupByIdInput,
)

AGENT_TOOLS = [
    search_products,
    search_products_for_food,
    search_cocktails,
    lookup_by_id,
]
TOOL_MAP = {tool.name: tool for tool in AGENT_TOOLS}
