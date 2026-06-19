from pathlib import Path

from sommelier.agent.tools.food_pairing import FoodPairingInput, food_pairing
from sommelier.catalog.search_profiles import product_card_to_search_profile
from sommelier.retrieval.faiss_index import FaissIndex, FakeEmbeddingProvider
from sommelier.retrieval.food_pairing_query import (
    DEFAULT_PAIRING_CAVEAT,
    normalize_food_pairing_query,
    search_for_food_pairing,
)


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeFoodPairingLLM:
    def invoke(self, prompt: str) -> FakeMessage:
        assert "мясо с грибами" in prompt
        return FakeMessage(
            "Rum for meat and mushrooms. Rich aged profile with oak, caramel, "
            "spice, vanilla, toasted or earthy notes."
        )


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


def _index() -> FaissIndex:
    profiles = [
        _profile(
            "aged-rum",
            "Aged Rum",
            "Rich aged rum with oak caramel spice vanilla toasted earthy notes for meat.",
        ),
        _profile(
            "light-rum",
            "Light Rum",
            "Light white rum with citrus fresh cocktails for fish seafood.",
        ),
        _profile(
            "dessert-rum",
            "Dessert Rum",
            "Sweet vanilla caramel coconut chocolate rum for dessert.",
        ),
    ]
    index = FaissIndex(embedding_provider=FakeEmbeddingProvider(dimensions=512))
    index.build(profiles)
    return index


def test_food_pairing_query_uses_llm_reformulation() -> None:
    expanded = normalize_food_pairing_query(
        "мясо с грибами",
        llm=FakeFoodPairingLLM(),
        use_llm=True,
    )

    assert expanded.startswith("Rum for meat and mushrooms.")
    assert "oak" in expanded
    assert "earthy" in expanded


def test_food_pairing_query_has_generic_fallback_without_llm() -> None:
    expanded = normalize_food_pairing_query("buckwheat", use_llm=False)

    assert expanded.startswith("Rum for food pairing with this dish: buckwheat.")


def test_search_for_food_pairing_uses_expanded_query() -> None:
    result = search_for_food_pairing(
        "мясо с грибами",
        top_k=1,
        index=_index(),
        llm=FakeFoodPairingLLM(),
        use_llm=True,
    )

    assert result.expanded_query.startswith("Rum for meat and mushrooms.")
    assert result.retrieval_results[0].product_id == "aged-rum"


def test_food_pairing_tool_contains_caveat_and_no_fake_source_claim() -> None:
    index = _index()
    import shutil
    from uuid import uuid4

    index_dir = Path(".test_tmp") / f"food-pairing-{uuid4().hex}"
    try:
        index.save(index_dir)
        output = food_pairing(
            FoodPairingInput(food_text="рыба", top_k=1, index_dir=index_dir)
        )
    finally:
        shutil.rmtree(index_dir, ignore_errors=True)

    assert DEFAULT_PAIRING_CAVEAT in output.summary
    assert output.metadata["caveat"] == DEFAULT_PAIRING_CAVEAT
    assert "Bacardi says" not in output.summary
    assert "source-backed Bacardi food-pairing claims" in output.summary
