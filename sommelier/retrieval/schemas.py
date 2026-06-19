"""Pydantic contracts for legacy retrieval and ranking scaffolding."""

from pydantic import BaseModel, Field
from sommelier.catalog.schemas import CatalogEntry


class SearchRequest(BaseModel):
    """Search request used by older agent scaffolding.

    New MVP retrieval normalizes the free-text `query` and searches
    ProductSearchProfile vectors. Tag fields remain compatibility metadata.
    """

    query: str
    product_tags: list[str] = Field(default_factory=list)
    food_tags: list[str] = Field(default_factory=list)
    user_preference_tags: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)


class RetrievalCandidate(BaseModel):
    """Candidate returned by legacy tag/vector retrieval scaffolding."""

    entry: CatalogEntry
    source: str
    vector_score: float | None = None
    matched_tags: list[str] = Field(default_factory=list)


class RankedRecommendation(BaseModel):
    """Deterministically ranked recommendation result."""

    entry: CatalogEntry
    score: float
    evidence: list[str] = Field(default_factory=list)
