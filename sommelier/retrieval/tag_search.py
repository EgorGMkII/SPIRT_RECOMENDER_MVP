"""Legacy deterministic tag search.

Tags are now secondary metadata. Primary retrieval should use normalized-query
vector search over ProductSearchProfile.searchable_text.
"""

from sommelier.catalog.schemas import CatalogEntry
from sommelier.retrieval.schemas import RetrievalCandidate, SearchRequest


def search_by_tags(entries: list[CatalogEntry], request: SearchRequest) -> list[RetrievalCandidate]:
    """Return entries matching requested controlled tags."""

    requested = set(request.product_tags + request.food_tags)
    if not requested:
        return []

    candidates: list[RetrievalCandidate] = []
    for entry in entries:
        available = set(entry.card.tags.product + entry.card.tags.food)
        matched = sorted(requested & available)
        if matched:
            candidates.append(
                RetrievalCandidate(entry=entry, source="tag", matched_tags=matched)
            )
    return candidates
