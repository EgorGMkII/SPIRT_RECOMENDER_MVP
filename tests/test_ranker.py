from sommelier.catalog.product_profiles import build_product_profile
from sommelier.catalog.schemas import CatalogEntry, ProductCard, ProductTags
from sommelier.retrieval.ranker import rank_candidates
from sommelier.retrieval.schemas import RetrievalCandidate, SearchRequest


def test_ranker_rewards_matching_tags_and_pairings() -> None:
    card = ProductCard(
        product_id="aged-dark",
        name="Aged Dark",
        tags=ProductTags(product=["aged", "dark"], food=["bbq"]),
    )
    entry = CatalogEntry(card=card, profile=build_product_profile(card))
    request = SearchRequest(query="rum for bbq", product_tags=["dark"], food_tags=["bbq"])
    candidate = RetrievalCandidate(entry=entry, source="tag", matched_tags=["dark"])

    ranked = rank_candidates([candidate], request)

    assert ranked[0].score > 10
    assert any("pairings=" in evidence for evidence in ranked[0].evidence)
