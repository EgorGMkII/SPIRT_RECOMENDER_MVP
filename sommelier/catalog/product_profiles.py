"""Build normalized searchable product profiles."""

from sommelier.catalog.schemas import ProductCard, ProductProfile


def build_product_profile(card: ProductCard) -> ProductProfile:
    """Create normalized profile text from validated product fields."""

    parts = [
        card.name,
        card.brand,
        card.normalized_description or card.raw_description,
        " ".join(card.tags.product),
        " ".join(card.tags.food),
        " ".join(card.cocktail_uses),
    ]
    return ProductProfile(
        product_id=card.product_id,
        profile_text=" ".join(part for part in parts if part).strip(),
        tags=card.tags,
    )
