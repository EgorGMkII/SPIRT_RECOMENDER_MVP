"""LLM-based normalization of cocktail search queries."""

from __future__ import annotations

from typing import Any


COCKTAIL_QUERY_NORMALIZER_PROMPT = """
Rewrite the user's cocktail request as one concise English search query for BM25.

Keep important cocktail names, rum names, ingredients, flavors, and recipe intent.
Translate Russian terms to English when needed.
Do not include negative or avoided ingredients/flavors as BM25 search terms.
If the user says "без кокоса", "without coconut", "not sweet", or "avoid X",
omit the avoided term from the search query and express the desired positive style
when useful, such as "fresh", "dry", "citrus", "light", or "refreshing".
Do not answer the user.
Do not invent ingredients.
Return plain text only, no JSON and no bullets.

Examples:
User: дай рецепт мохито
Search query: mojito cocktail recipe mint lime white rum

User: коктейль с ананасом и кокосом
Search query: pineapple coconut rum cocktail

User: что сделать с carta blanca
Search query: BACARDÍ Carta Blanca rum cocktail recipe

User: простой коктейль с колой
Search query: easy simple rum cola cocktail

User: освежающий коктейль без кокоса
Search query: refreshing citrus light rum cocktail

User: коктейль похожий на мохито, но не сладкий
Search query: mojito style fresh lime mint light rum cocktail

User: {query}
Search query:
""".strip()


def _message_content(message: Any) -> str:
    """Extract string content from an LLM response."""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(message, str):
        return message
    return str(message)


def build_cocktail_query_normalization_prompt(text: str) -> str:
    """Build the LLM prompt for cocktail query normalization."""

    return COCKTAIL_QUERY_NORMALIZER_PROMPT.format(query=text)


def normalize_cocktail_query(text: str, llm: Any | None = None) -> str:
    """Normalize a user cocktail request into a short BM25-friendly query."""

    active_llm = llm
    if active_llm is None:
        from llm_module import get_langchain_openai_chat_model

        active_llm = get_langchain_openai_chat_model()
    response = active_llm.invoke(build_cocktail_query_normalization_prompt(text))
    normalized = _message_content(response).strip().strip("`")
    return " ".join(normalized.split()) or text
