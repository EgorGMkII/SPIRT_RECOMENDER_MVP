"""Controlled routing logic for agent intents."""

from sommelier.agent.schemas import IntentType, ParsedIntent


def route_intent(intent: ParsedIntent) -> str:
    """Map parsed intent to an explicit tool route."""

    routes = {
        IntentType.SEARCH_PRODUCTS: "search_products",
        IntentType.FOOD_PAIRING: "food_pairing",
        IntentType.FOOD_FOR_RUM: "food_for_rum",
        IntentType.COCKTAIL_EXPANSION: "cocktail_expansion",
        IntentType.PROFILE_UPDATE: "profile_update",
    }
    return routes.get(intent.intent, "search_products")
