from sommelier.catalog.schemas import TagSpace
from sommelier.catalog.tag_vocabulary import allowed_values


def test_tag_spaces_remain_separate() -> None:
    product_tags = allowed_values(TagSpace.PRODUCT)
    food_tags = allowed_values(TagSpace.FOOD)
    user_tags = allowed_values(TagSpace.USER_PREFERENCE)

    assert "dark" in product_tags
    assert "bbq" in food_tags
    assert "likes_sweet" in user_tags
    assert product_tags.isdisjoint(food_tags)
