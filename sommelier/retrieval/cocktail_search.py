"""BM25 cocktail retrieval over compact CocktailSearchProfile records."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sommelier.catalog.cocktail_profiles import (
    CocktailSearchProfile,
    load_cocktail_profiles,
)
from sommelier.retrieval.bm25_index import tokenize
from sommelier.retrieval.cocktail_query_normalizer import normalize_cocktail_query

DEFAULT_COCKTAIL_PROFILES_DIR = Path("data/catalog/cocktail_search_profiles")


class CocktailSearchResult(BaseModel):
    """Cocktail BM25 search result."""

    cocktail_id: str
    score: float
    normalized_query: str
    matched_tokens: list[str] = Field(default_factory=list)
    profile: CocktailSearchProfile


class CocktailBm25Index:
    """In-memory BM25 index for CocktailSearchProfile records."""

    def __init__(self, profiles: list[CocktailSearchProfile]) -> None:
        self.profiles = profiles
        self.documents = [tokenize(profile.searchable_text) for profile in profiles]
        self.average_doc_length = (
            sum(len(document) for document in self.documents) / len(self.documents)
            if self.documents
            else 0.0
        )
        self.document_frequency: dict[str, int] = defaultdict(int)
        for document in self.documents:
            for token in set(document):
                self.document_frequency[token] += 1

    @classmethod
    def load(cls, profiles_dir: Path = DEFAULT_COCKTAIL_PROFILES_DIR) -> "CocktailBm25Index":
        """Load cocktail profiles and build an in-memory BM25 index."""

        return cls(load_cocktail_profiles(profiles_dir))

    def search(self, normalized_query: str, top_k: int = 5) -> list[CocktailSearchResult]:
        """Return top-k BM25 matches for an already normalized query."""

        query_tokens = tokenize(normalized_query)
        if not self.profiles or not query_tokens:
            return []

        total_docs = len(self.documents)
        scored: list[tuple[int, float, list[str]]] = []
        for index, document in enumerate(self.documents):
            term_frequency = Counter(document)
            score = 0.0
            matched_tokens: list[str] = []
            for token in query_tokens:
                if not term_frequency[token]:
                    continue
                matched_tokens.append(token)
                doc_frequency = self.document_frequency[token]
                idf = math.log(
                    1 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)
                )
                denominator = term_frequency[token] + 1.5 * (
                    1 - 0.75 + 0.75 * len(document) / max(self.average_doc_length, 1)
                )
                score += idf * (term_frequency[token] * 2.5) / denominator
            scored.append((index, score, sorted(set(matched_tokens))))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            CocktailSearchResult(
                cocktail_id=self.profiles[index].cocktail_id,
                score=score,
                normalized_query=normalized_query,
                matched_tokens=matched_tokens,
                profile=self.profiles[index],
            )
            for index, score, matched_tokens in scored[:top_k]
        ]


def search_cocktails(
    query: str,
    top_k: int = 5,
    profiles_dir: Path = DEFAULT_COCKTAIL_PROFILES_DIR,
    llm: Any | None = None,
) -> list[CocktailSearchResult]:
    """Normalize a cocktail query with an LLM and search cocktail profiles."""

    normalized_query = normalize_cocktail_query(query, llm=llm)
    index = CocktailBm25Index.load(profiles_dir)
    return index.search(normalized_query, top_k=top_k)
