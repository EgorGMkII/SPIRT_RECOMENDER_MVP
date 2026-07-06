"""Typed contracts shared by the target agent workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sommelier.agent.profile import ProfilePatch
from sommelier.agent.memory import CatalogRef, RequestScope, ShownResult


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TurnResolution(StrictModel):
    follow_up: bool
    request_scope: RequestScope = "conversation"
    initial_request: str = Field(min_length=1, max_length=1200)
    effective_request: str = Field(min_length=1, max_length=1200)
    negative_request: str | None = Field(default=None, max_length=1200)
    cart_action: Literal["add", "delete", "show"] | None = None
    profile_patch: ProfilePatch | None = None
    reasoning_note: str = Field(default="", max_length=500)


class FeedbackResult(StrictModel):
    feedback: Literal["neutral", "purchase_intent", "negative_feedback"]


class FinalAnswerResult(StrictModel):
    answer: str = Field(min_length=1)
    used_result_refs: list[CatalogRef] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Catalog objects that are central recommendations or direct subjects "
            "of the answer; exclude incidental alternatives and comparisons."
        ),
    )
    assistant_summary: str = Field(min_length=1, max_length=800)


class ProductCandidate(CatalogRef):
    kind: Literal["product"] = "product"
    name: str
    category: str = ""
    display_description: str = ""
    tasting_summary: str = ""
    how_to_serve: str = ""
    cocktail_names: list[str] = Field(default_factory=list)
    evidence_fields: dict[str, str] = Field(default_factory=dict)
    retrieval_sources: list[str] = Field(default_factory=list)


class CocktailCandidate(CatalogRef):
    kind: Literal["cocktail"] = "cocktail"
    name: str
    main_rum: str = ""
    description: str = ""
    ingredients: list[str] = Field(default_factory=list)
    recipe_steps: list[str] = Field(default_factory=list)


class ProductSearchOutput(StrictModel):
    query: str
    candidates: list[ProductCandidate] = Field(default_factory=list, max_length=7)
    caveats: list[str] = Field(default_factory=list)


class CocktailSearchOutput(StrictModel):
    query: str
    candidates: list[CocktailCandidate] = Field(default_factory=list, max_length=7)
    caveats: list[str] = Field(default_factory=list)


class CatalogListItem(CatalogRef):
    name: str = Field(min_length=1, max_length=300)


class CatalogListOutput(StrictModel):
    kind: Literal["product", "cocktail"]
    total: int = Field(ge=0)
    items: list[CatalogListItem] = Field(default_factory=list, max_length=100)
