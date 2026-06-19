"""Food pairing rules for rum recommendations."""

PAIRING_RULES: dict[str, list[str]] = {
    "dark": ["bbq", "dessert"],
    "aged": ["beef", "grilled"],
    "vanilla": ["dessert"],
}


def matching_pairings(product_tags: list[str], food_tags: list[str]) -> list[str]:
    """Return deterministic product/food pairing matches."""

    requested_food = set(food_tags)
    matches: list[str] = []
    for tag in product_tags:
        for food_tag in PAIRING_RULES.get(tag, []):
            if food_tag in requested_food:
                matches.append(f"{tag}:{food_tag}")
    return matches
