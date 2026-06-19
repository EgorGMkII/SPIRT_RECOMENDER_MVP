"""Anti-pairing rules for combinations to avoid."""

ANTI_PAIRING_RULES: dict[str, list[str]] = {
    "smoky": ["seafood"],
}


def matching_anti_pairings(product_tags: list[str], food_tags: list[str]) -> list[str]:
    """Return deterministic anti-pairing matches."""

    requested_food = set(food_tags)
    matches: list[str] = []
    for tag in product_tags:
        for food_tag in ANTI_PAIRING_RULES.get(tag, []):
            if food_tag in requested_food:
                matches.append(f"{tag}:{food_tag}")
    return matches
