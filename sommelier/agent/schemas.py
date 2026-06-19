"""Pydantic contracts for agent inputs, outputs, and tool calls."""

from enum import StrEnum
from pydantic import BaseModel, Field
from sommelier.retrieval.schemas import RankedRecommendation, SearchRequest


class IntentType(StrEnum):
    """Supported controlled user intents."""

    SEARCH_PRODUCTS = "search_products"
    FOOD_PAIRING = "food_pairing"
    FOOD_FOR_RUM = "food_for_rum"
    COCKTAIL_EXPANSION = "cocktail_expansion"
    PROFILE_UPDATE = "profile_update"
    UNKNOWN = "unknown"


class ParsedIntent(BaseModel):
    """Structured intent parsed from a user message."""

    intent: IntentType = IntentType.UNKNOWN
    search: SearchRequest | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ToolResult(BaseModel):
    """Generic structured tool result."""

    tool_name: str
    summary: str
    recommendations: list[RankedRecommendation] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
