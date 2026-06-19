"""Legacy hybrid retrieval scaffold.

The current MVP retrieval path is normalized-query vector search in
`sommelier.retrieval.faiss_index`. This module remains for compatibility with
earlier tests and should not be used as the primary product search architecture.
"""

from sommelier.catalog.schemas import CatalogEntry
from sommelier.retrieval.faiss_index import FaissIndex
from sommelier.retrieval.ranker import rank_candidates
from sommelier.retrieval.schemas import RankedRecommendation, RetrievalCandidate, SearchRequest
from sommelier.retrieval.tag_search import search_by_tags


def merge_candidates(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    """Merge candidates by product ID while preserving evidence."""

    merged: dict[str, RetrievalCandidate] = {}
    for candidate in candidates:
        product_id = candidate.entry.card.product_id
        if product_id not in merged:
            merged[product_id] = candidate
            continue
        existing = merged[product_id]
        existing.matched_tags = sorted(set(existing.matched_tags + candidate.matched_tags))
        existing.source = "+".join(sorted(set(existing.source.split("+") + [candidate.source])))
        if candidate.vector_score is not None:
            existing.vector_score = max(existing.vector_score or 0.0, candidate.vector_score)
    return list(merged.values())


def hybrid_search(
    entries: list[CatalogEntry],
    request: SearchRequest,
    index: FaissIndex | None = None,
) -> list[RankedRecommendation]:
    """Run tag search, vector search, merge results, then rank deterministically."""

    vector_index = index or FaissIndex()
    tag_candidates = search_by_tags(entries, request)
    vector_candidates = vector_index.search(request)
    return rank_candidates(merge_candidates(tag_candidates + vector_candidates), request)
