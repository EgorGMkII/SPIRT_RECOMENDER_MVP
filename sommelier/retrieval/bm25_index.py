"""Lightweight BM25 retrieval over ProductSearchProfile.searchable_text."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path

from pydantic import BaseModel

from sommelier.catalog.search_profiles import ProductSearchProfile, load_search_profiles
from sommelier.retrieval.faiss_index import SearchResult
from sommelier.retrieval.query_normalizer import normalize_query


class Bm25Result(BaseModel):
    """BM25 search result with matched lexical tokens."""

    result: SearchResult
    matched_tokens: list[str]


def tokenize(text: str) -> list[str]:
    """Tokenize text for simple lexical retrieval."""

    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in normalized.split() if len(token) > 1]


class Bm25Index:
    """In-memory BM25 index for small ProductSearchProfile catalogs."""

    def __init__(self, profiles: list[ProductSearchProfile]) -> None:
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
    def load(cls, profiles_dir: Path) -> "Bm25Index":
        """Load search profiles and build an in-memory BM25 index."""

        return cls(load_search_profiles(profiles_dir))

    def search(self, query_text: str, top_k: int = 5, normalize: bool = True) -> list[Bm25Result]:
        """Return top-k BM25 matches for a query."""

        normalized_query = normalize_query(query_text) if normalize else query_text
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
            Bm25Result(
                result=SearchResult(
                    product_id=self.profiles[index].product_id,
                    score=score,
                    normalized_query=normalized_query,
                    profile=self.profiles[index],
                ),
                matched_tokens=matched_tokens,
            )
            for index, score, matched_tokens in scored[:top_k]
        ]
