from pathlib import Path

from sommelier.catalog.search_profiles import product_card_to_search_profile
from sommelier.retrieval.faiss_index import FaissIndex, FakeEmbeddingProvider
from sommelier.retrieval.query_normalizer import normalize_query


def _profile(product_id: str, name: str, text: str):
    card = {
        "product_id": product_id,
        "source_url": f"https://example.com/{product_id}",
        "brand": "Test",
        "name": name,
        "category": "rum",
        "short_description": text,
        "marketing_description": text,
        "tasting_notes": "",
        "nose": "",
        "palate": "",
        "finish": "",
        "process": "",
        "how_to_serve": "",
        "cocktail_names": [],
        "extraction_warnings": [],
    }
    return product_card_to_search_profile(card, Path(f"{product_id}.json"))


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeQueryNormalizerLLM:
    def invoke(self, prompt: str) -> FakeMessage:
        assert "sweet rum with caramel" in prompt
        return FakeMessage(
            "Sweet rum with caramel notes. Suitable for cocktails and mixing drinks."
        )


def test_query_normalization_uses_llm() -> None:
    normalized = normalize_query(
        "I need a sweet rum with caramel notes for mixing cocktails.",
        llm=FakeQueryNormalizerLLM(),
        use_llm=True,
    )

    assert normalized == "Sweet rum with caramel notes. Suitable for cocktails and mixing drinks."


def test_query_normalization_fallback_is_compact_original_text() -> None:
    normalized = normalize_query("I need a sweet rum with caramel notes for mixing cocktails.")

    assert normalized == "I need a sweet rum with caramel notes for mixing cocktails."


def test_faiss_wrapper_with_fake_embeddings_returns_expected_product() -> None:
    profiles = [
        _profile("vanilla-oak", "Vanilla Oak", "Rum with vanilla oak smooth cocktail profile."),
        _profile("tropical", "Tropical", "Rum with pineapple coconut guava fruit profile."),
    ]
    index = FaissIndex(embedding_provider=FakeEmbeddingProvider())
    index.build(profiles)

    results = index.search("I want a vanilla and oak rum for cocktails.", top_k=1)

    assert results[0].product_id == "vanilla-oak"


def test_normalized_query_is_used_during_retrieval() -> None:
    profiles = [
        _profile("caramel-mixer", "Caramel Mixer", "Sweet rum with caramel flavors for cocktails and mixing drinks."),
        _profile("citrus", "Citrus", "Citrus rum for bright fruit serves."),
    ]
    index = FaissIndex(embedding_provider=FakeEmbeddingProvider())
    index.build(profiles)

    results = index.search("I need a sweet rum with caramel notes for mixing cocktails.", top_k=1)

    assert results[0].normalized_query == "I need a sweet rum with caramel notes for mixing cocktails."
    assert results[0].product_id == "caramel-mixer"
