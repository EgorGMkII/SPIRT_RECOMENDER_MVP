"""Controlled vocabulary definitions for tags."""

from sommelier.catalog.schemas import Tag, TagSpace


PRODUCT_TAGS: tuple[Tag, ...] = (
    Tag(value="vanilla", label="Vanilla", space=TagSpace.PRODUCT),
    Tag(value="caramel", label="Caramel", space=TagSpace.PRODUCT),
    Tag(value="oak", label="Oak", space=TagSpace.PRODUCT),
    Tag(value="smoky", label="Smoky", space=TagSpace.PRODUCT),
    Tag(value="aged", label="Aged", space=TagSpace.PRODUCT),
    Tag(value="dark", label="Dark", space=TagSpace.PRODUCT),
)

FOOD_TAGS: tuple[Tag, ...] = (
    Tag(value="pork", label="Pork", space=TagSpace.FOOD),
    Tag(value="beef", label="Beef", space=TagSpace.FOOD),
    Tag(value="seafood", label="Seafood", space=TagSpace.FOOD),
    Tag(value="dessert", label="Dessert", space=TagSpace.FOOD),
    Tag(value="bbq", label="BBQ", space=TagSpace.FOOD),
    Tag(value="grilled", label="Grilled", space=TagSpace.FOOD),
)

USER_PREFERENCE_TAGS: tuple[Tag, ...] = (
    Tag(value="likes_sweet", label="Likes sweet", space=TagSpace.USER_PREFERENCE),
    Tag(value="dislikes_smoky", label="Dislikes smoky", space=TagSpace.USER_PREFERENCE),
    Tag(value="prefers_cocktails", label="Prefers cocktails", space=TagSpace.USER_PREFERENCE),
)


def all_tags() -> tuple[Tag, ...]:
    """Return every controlled tag."""

    return PRODUCT_TAGS + FOOD_TAGS + USER_PREFERENCE_TAGS


def allowed_values(space: TagSpace) -> set[str]:
    """Return allowed tag values for a tag space."""

    return {tag.value for tag in all_tags() if tag.space == space}
