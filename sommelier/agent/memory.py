"""Compact turn-based durable memory."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_TURNS = 12
RequestScope = Literal[
    "product",
    "cocktail",
    "recipe",
    "food_pairing",
    "cart",
    "profile",
    "catalog_listing",
    "conversation",
]


class MemoryModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class CatalogRef(MemoryModel):
    kind: Literal["product", "cocktail"]
    id: str = Field(min_length=1, max_length=300)


class ShownResult(CatalogRef):
    name: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=800)


class CartItem(MemoryModel):
    id: str = Field(min_length=1, max_length=300)
    amount: int = Field(ge=1, le=99)


class TurnMemory(MemoryModel):
    follow_up: bool
    request_scope: RequestScope = "conversation"
    user_request: str = Field(min_length=1, max_length=1200)
    initial_request: str = Field(min_length=1, max_length=1200)
    effective_request: str = Field(min_length=1, max_length=1200)
    negative_request: str | None = Field(default=None, max_length=1200)
    assistant_summary: str = Field(min_length=1, max_length=800)
    shown_results: list[ShownResult] = Field(default_factory=list, max_length=3)


class SessionMemory(MemoryModel):
    schema_version: Literal[4] = 4
    session_id: str
    turns: list[TurnMemory] = Field(default_factory=list)
    cart: list[CartItem] = Field(default_factory=list)


def enforce_memory_limits(memory: SessionMemory) -> SessionMemory:
    """Keep only the bounded conversational context used by the agent."""

    memory.turns = memory.turns[-MAX_TURNS:]
    return memory
