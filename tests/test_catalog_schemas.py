from sommelier.catalog.schemas import ProductCard, ProductTags


def test_product_card_validation() -> None:
    card = ProductCard(
        product_id="bacardi-dark",
        name="Bacardi Dark",
        tags=ProductTags(product=["dark"], food=["bbq"]),
    )

    assert card.product_id == "bacardi-dark"
    assert card.tags.product == ["dark"]
