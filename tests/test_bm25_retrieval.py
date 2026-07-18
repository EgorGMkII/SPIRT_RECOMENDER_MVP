from pathlib import Path

from sommelier.catalog.search_profiles import product_card_to_search_profile
from sommelier.retrieval.bm25_index import Bm25Index, tokenize


def _profile(product_id: str, name: str, text: str):
    card = {
        "product_id": product_id,
        "source_url": f"https://example.com/{product_id}",
        "brand": "Test",
        "name": name,
        "category": "rum",
        "short_description": text,
        "marketing_description": text,
        "tasting_notes": text,
        "nose": "",
        "palate": "",
        "finish": "",
        "process": "",
        "how_to_serve": text,
        "cocktail_names": [],
        "extraction_warnings": [],
    }
    return product_card_to_search_profile(card, Path(f"{product_id}.json"))


def test_tokenize_keeps_alphanumeric_terms() -> None:
    assert tokenize("Vanilla, oak & highball!") == ["vanilla", "oak", "highball"]


def test_bm25_search_returns_expected_product() -> None:
    index = Bm25Index(
        [
            _profile("vanilla-oak", "Vanilla Oak", "Vanilla oak rum for cocktails."),
            _profile("citrus-white", "Citrus White", "Light citrus white rum."),
        ]
    )

    results = index.search("vanilla and oak rum", top_k=1)

    assert results[0].result.product_id == "vanilla-oak"
    assert "vanilla" in results[0].matched_tokens
    assert "oak" in results[0].matched_tokens
