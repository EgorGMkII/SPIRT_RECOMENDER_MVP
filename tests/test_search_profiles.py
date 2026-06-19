from pathlib import Path

from sommelier.catalog.search_profiles import (
    ProductSearchProfile,
    extract_lightweight_tags,
    product_card_to_search_profile,
)


def _card() -> dict:
    return {
        "product_id": "bacardi-anejo-cuatro-rum",
        "source_url": "https://www.bacardi.com/our-rums/anejo-cuatro-rum/",
        "brand": "Bacardi",
        "name": "BACARDI Anejo Cuatro",
        "category": "gold rum",
        "short_description": "A smooth rum for cocktails.",
        "marketing_description": "Mild vanilla, toasted oak and honey.",
        "tasting_notes": "Enjoy notes of vanilla and oak.",
        "nose": "Vanilla and cinnamon",
        "palate": "Dark honey and clove",
        "finish": "Toffee and oak",
        "process": "Aged under the Caribbean sun.",
        "how_to_serve": "Great in a highball.",
        "cocktail_names": ["Cuatro Highball"],
        "recommended_rums": ["Unrelated Related Rum"],
        "faq_items": [{"question": "How do you make rum?", "answer": "FAQ answer."}],
        "source_metadata": {"description": "metadata should not be searchable"},
        "extraction_warnings": ["warning should not be searchable"],
    }


def test_product_search_profile_schema() -> None:
    profile = product_card_to_search_profile(_card(), Path("card.json"))

    validated = ProductSearchProfile.model_validate(profile.model_dump())

    assert validated.product_id == "bacardi-anejo-cuatro-rum"
    assert "vanilla" in validated.flavor_tags
    assert "highball" in validated.usage_tags


def test_product_card_to_search_profile_excludes_non_product_fields() -> None:
    profile = product_card_to_search_profile(_card(), Path("card.json"))

    assert "Unrelated Related Rum" not in profile.searchable_text
    assert "FAQ answer" not in profile.searchable_text
    assert "metadata should not be searchable" not in profile.searchable_text
    assert "warning should not be searchable" not in profile.searchable_text
    assert "vanilla" in profile.searchable_text
    assert "oak" in profile.searchable_text


def test_lightweight_tag_extraction_uses_known_keywords_only() -> None:
    flavor_tags, usage_tags = extract_lightweight_tags(
        "Vanilla, toasted oak, banana, mystery descriptor, cocktail mixer, neat."
    )

    assert flavor_tags == ["vanilla", "oak", "banana"]
    assert usage_tags == ["cocktail", "mixer", "neat"]


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def invoke(self, prompt: str) -> FakeMessage:
        return FakeMessage(
            "BACARDI Anejo Cuatro is a gold aged rum for cocktails and highballs. "
            "It has vanilla, toasted oak, honey, clove, cinnamon, caramel and toffee notes "
            "with a smooth rich profile suited to oak-and-spice rum searches."
        )


def test_llm_searchable_text_replaces_field_concat() -> None:
    profile = product_card_to_search_profile(
        _card(),
        Path("card.json"),
        use_llm_searchable_text=True,
        llm=FakeLLM(),
    )

    assert profile.searchable_text.startswith("BACARDI Anejo Cuatro is a gold aged rum")
    assert "Unrelated Related Rum" not in profile.searchable_text
    assert "FAQ answer" not in profile.searchable_text
