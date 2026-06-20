"""Pydantic contracts shared by lightweight retrieval-facing tools."""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Free-text search request.

    Runtime retrieval does not consume product/food tag filters. The current
    MVP normalizes this text and searches ProductSearchProfile records with
    FAISS and BM25.
    """

    query: str
    limit: int = Field(default=5, ge=1, le=20)
