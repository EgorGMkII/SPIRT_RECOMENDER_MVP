"""Request and response models for the web API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message."""

    session_id: str
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """Outgoing chat response."""

    session_id: str
    answer: str
    intent: str | None = None
    effective_user_message: str | None = None
    is_followup: bool = False
    profile: dict | None = None
    candidates: list[dict] = Field(default_factory=list)
    traces: list[dict] = Field(default_factory=list)


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
