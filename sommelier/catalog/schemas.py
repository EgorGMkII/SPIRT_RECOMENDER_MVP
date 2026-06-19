"""Pydantic models for product catalog data contracts."""

from datetime import datetime
from enum import StrEnum
from pydantic import BaseModel, Field, HttpUrl


class TagSpace(StrEnum):
    """Distinct namespaces for tags."""

    PRODUCT = "product"
    FOOD = "food"
    USER_PREFERENCE = "user_preference"


class Tag(BaseModel):
    """A controlled vocabulary tag."""

    value: str
    label: str
    space: TagSpace


class ProductTags(BaseModel):
    """Separated tag spaces attached to a catalog product."""

    product: list[str] = Field(default_factory=list)
    food: list[str] = Field(default_factory=list)
    cocktail: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    """Source metadata for catalog entries."""

    source_url: HttpUrl | None = None
    fetched_at: datetime | None = None
    parsed_at: datetime | None = None
    parser_version: str = "stub-v0"


class ProductCard(BaseModel):
    """Structured product data extracted from a source page."""

    product_id: str
    name: str
    brand: str = "Bacardi"
    raw_description: str = ""
    normalized_description: str = ""
    tags: ProductTags = Field(default_factory=ProductTags)
    cocktail_uses: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


class ProductProfile(BaseModel):
    """Normalized searchable product profile."""

    product_id: str
    profile_text: str
    tags: ProductTags = Field(default_factory=ProductTags)


class CatalogEntry(BaseModel):
    """Final catalog entry consumed by retrieval and ranking."""

    card: ProductCard
    profile: ProductProfile
