"""Deterministic recommendation ranking."""

from sommelier.retrieval.anti_pairing import matching_anti_pairings
from sommelier.retrieval.pairing_rules import matching_pairings
from sommelier.retrieval.schemas import RankedRecommendation, RetrievalCandidate, SearchRequest


def rank_candidates(
    candidates: list[RetrievalCandidate],
    request: SearchRequest,
) -> list[RankedRecommendation]:
    """Rank candidates with deterministic scoring rules."""

    ranked: list[RankedRecommendation] = []
    for candidate in candidates:
        product_tags = candidate.entry.card.tags.product
        score = 0.0
        evidence: list[str] = []

        if candidate.matched_tags:
            score += len(candidate.matched_tags) * 10
            evidence.append(f"matched_tags={','.join(candidate.matched_tags)}")

        pairings = matching_pairings(product_tags, request.food_tags)
        if pairings:
            score += len(pairings) * 5
            evidence.append(f"pairings={','.join(pairings)}")

        anti_pairings = matching_anti_pairings(product_tags, request.food_tags)
        if anti_pairings:
            score -= len(anti_pairings) * 7
            evidence.append(f"anti_pairings={','.join(anti_pairings)}")

        if candidate.vector_score is not None:
            score += candidate.vector_score
            evidence.append(f"vector_score={candidate.vector_score:.3f}")

        ranked.append(RankedRecommendation(entry=candidate.entry, score=score, evidence=evidence))

    return sorted(ranked, key=lambda item: item.score, reverse=True)[: request.limit]
