"""Request and response models for the web API."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message."""

    session_id: str
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """Outgoing chat response."""

    session_id: str
    answer: str
    follow_up: bool
    request_scope: Literal[
        "product",
        "cocktail",
        "recipe",
        "food_pairing",
        "cart",
        "profile",
        "catalog_listing",
        "conversation",
    ] = "conversation"
    answer_mode: Literal["hard", "soft"] = "hard"
    effective_request: str | None = None
    profile: dict | None = None
    candidates: list[dict] = Field(default_factory=list)
    traces: list[dict] = Field(default_factory=list)


class ChatMessage(BaseModel):
    """One full transcript message used only by the web client."""

    role: Literal["user", "assistant"]
    content: str


class ChatHistoryResponse(BaseModel):
    """Complete transcript for one browser session."""

    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)


class CatalogStatus(BaseModel):
    """Current local catalog/index artifact status."""

    product_profiles_count: int = 0
    cocktail_profiles_count: int = 0
    faiss_metadata_exists: bool = False
    faiss_index_exists: bool = False
    bm25_ready: bool = False
    cocktail_bm25_ready: bool = False


class SessionDebugResponse(BaseModel):
    """Debug response for durable session memory."""

    session_id: str
    memory: dict | None = None


class TraceDebugResponse(BaseModel):
    """Debug response for durable tool traces."""

    session_id: str
    traces: list[dict] = Field(default_factory=list)


class ProfileDebugResponse(BaseModel):
    """Debug response for durable user profile."""

    session_id: str
    profile: dict | None = None


class FeedbackStatsResponse(BaseModel):
    """Aggregated independent feedback analytics."""

    session_id: str | None = None
    total: int = 0
    neutral: int = 0
    purchase_intent: int = 0
    negative_feedback: int = 0
    successful_turns: int = 0
    failed_turns: int = 0
