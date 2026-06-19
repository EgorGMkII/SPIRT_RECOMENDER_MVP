"""FAISS-compatible vector index over ProductSearchProfile.searchable_text."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol
from pydantic import BaseModel, Field

from sommelier.catalog.search_profiles import ProductSearchProfile, load_search_profiles
from sommelier.retrieval.query_normalizer import normalize_query


class EmbeddingProvider(Protocol):
    """Embedding provider interface for real or deterministic fake embeddings."""

    def embed(self, text: str) -> list[float]:
        """Return a vector embedding for text."""


class FakeEmbeddingProvider:
    """Deterministic hashing embedding for tests and local MVP demos."""

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        """Embed text into a deterministic normalized bag-of-words vector."""

        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        return _normalize(vector)


class OpenAIEmbeddingProvider:
    """Real embedding provider backed by llm_module.py."""

    def embed(self, text: str) -> list[float]:
        """Embed text with the configured OpenAI-compatible embeddings API."""

        from llm_module import get_openai_embedding

        return _normalize(get_openai_embedding(text))


class SearchResult(BaseModel):
    """Vector search result with matched profile metadata."""

    product_id: str
    score: float
    normalized_query: str
    profile: ProductSearchProfile


class VectorIndexPayload(BaseModel):
    """Serializable fallback vector index payload."""

    dimensions: int
    profiles: list[ProductSearchProfile] = Field(default_factory=list)
    vectors: list[list[float]] = Field(default_factory=list)


def _tokens(text: str) -> list[str]:
    """Tokenize text for deterministic fake embeddings."""

    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in normalized.split() if len(token) > 1]


def _normalize(vector: list[float]) -> list[float]:
    """L2 normalize a vector."""

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    """Compute dot product."""

    return sum(a * b for a, b in zip(left, right))


class FaissIndex:
    """Vector index for normalized-query search over product profiles."""

    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.embedding_provider = embedding_provider or FakeEmbeddingProvider()
        self.profiles: list[ProductSearchProfile] = []
        self.vectors: list[list[float]] = []
        self._faiss_index = None

    @property
    def dimensions(self) -> int:
        """Return vector dimension."""

        if self.vectors:
            return len(self.vectors[0])
        probe = self.embedding_provider.embed("probe")
        return len(probe)

    def build(self, profiles: list[ProductSearchProfile]) -> None:
        """Build an in-memory vector index from search profiles."""

        self.profiles = profiles
        self.vectors = [
            self.embedding_provider.embed(profile.searchable_text)
            for profile in profiles
        ]
        try:
            import faiss  # type: ignore
            import numpy as np

            if self.vectors:
                matrix = np.array(self.vectors, dtype="float32")
                index = faiss.IndexFlatIP(matrix.shape[1])
                index.add(matrix)
                self._faiss_index = index
        except Exception:
            self._faiss_index = None

    def save(self, index_dir: Path) -> None:
        """Save index metadata and fallback vectors to disk."""

        index_dir.mkdir(parents=True, exist_ok=True)
        payload = VectorIndexPayload(
            dimensions=self.dimensions,
            profiles=self.profiles,
            vectors=self.vectors,
        )
        (index_dir / "product_faiss_metadata.json").write_text(
            payload.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if self._faiss_index is not None:
            try:
                import faiss  # type: ignore

                faiss.write_index(self._faiss_index, str(index_dir / "product.faiss"))
            except Exception:
                pass

    @classmethod
    def load(
        cls,
        index_dir: Path,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> "FaissIndex":
        """Load index metadata and fallback vectors from disk."""

        payload = VectorIndexPayload.model_validate_json(
            (index_dir / "product_faiss_metadata.json").read_text(encoding="utf-8")
        )
        index = cls(embedding_provider=embedding_provider)
        index.profiles = payload.profiles
        index.vectors = payload.vectors
        faiss_path = index_dir / "product.faiss"
        if faiss_path.exists():
            try:
                import faiss  # type: ignore

                index._faiss_index = faiss.read_index(str(faiss_path))
            except Exception:
                index._faiss_index = None
        return index

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        normalize: bool = True,
    ) -> list[SearchResult]:
        """Normalize query text, embed it, and return top-k similar profiles."""

        normalized_query = normalize_query(query_text) if normalize else query_text
        query_vector = self.embedding_provider.embed(normalized_query)
        if not self.profiles:
            return []
        if self.vectors and len(query_vector) != len(self.vectors[0]):
            raise ValueError(
                "Query embedding dimension does not match index vector dimension. "
                "Rebuild the index with the same embedding provider used for search."
            )

        scored = [
            (index, _dot(query_vector, vector))
            for index, vector in enumerate(self.vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            SearchResult(
                product_id=self.profiles[index].product_id,
                score=score,
                normalized_query=normalized_query,
                profile=self.profiles[index],
            )
            for index, score in scored[:top_k]
        ]


def build_index_from_profiles(
    profiles_dir: Path,
    embedding_provider: EmbeddingProvider | None = None,
) -> FaissIndex:
    """Build a vector index from ProductSearchProfile files."""

    index = FaissIndex(embedding_provider=embedding_provider)
    index.build(load_search_profiles(profiles_dir))
    return index
