"""LLM-based normalization of cocktail search queries."""

from __future__ import annotations

from typing import Any


COCKTAIL_QUERY_NORMALIZER_PROMPT = """
Rewrite the user's cocktail request as one concise English search query for BM25.

Keep important cocktail names, rum names, ingredients, flavors, and recipe intent.
Translate Russian terms to English when needed.
Do not include negative or avoided ingredients/flavors as BM25 search terms.
If the user says "\u0431\u0435\u0437 \u043a\u043e\u043a\u043e\u0441\u0430", "without coconut", "not sweet", or "avoid X",
omit the avoided term from the search query and express the desired positive style
when useful, such as "fresh", "dry", "citrus", "light", or "refreshing".
Do not answer the user.
Do not invent ingredients.
Return plain text only, no JSON and no bullets.

Examples:
User: \u0434\u0430\u0439 \u0440\u0435\u0446\u0435\u043f\u0442 \u043c\u043e\u0445\u0438\u0442\u043e
Search query: mojito cocktail recipe mint lime white rum

User: \u043a\u043e\u043a\u0442\u0435\u0439\u043b\u044c \u0441 \u0430\u043d\u0430\u043d\u0430\u0441\u043e\u043c \u0438 \u043a\u043e\u043a\u043e\u0441\u043e\u043c
Search query: pineapple coconut rum cocktail

User: \u0447\u0442\u043e \u0441\u0434\u0435\u043b\u0430\u0442\u044c \u0441 carta blanca
Search query: BACARD\u00cd Carta Blanca rum cocktail recipe

User: \u043f\u0440\u043e\u0441\u0442\u043e\u0439 \u043a\u043e\u043a\u0442\u0435\u0439\u043b\u044c \u0441 \u043a\u043e\u043b\u043e\u0439
Search query: easy simple rum cola cocktail

User: \u043e\u0441\u0432\u0435\u0436\u0430\u044e\u0449\u0438\u0439 \u043a\u043e\u043a\u0442\u0435\u0439\u043b\u044c \u0431\u0435\u0437 \u043a\u043e\u043a\u043e\u0441\u0430
Search query: refreshing citrus light rum cocktail

User: \u043a\u043e\u043a\u0442\u0435\u0439\u043b\u044c \u043f\u043e\u0445\u043e\u0436\u0438\u0439 \u043d\u0430 \u043c\u043e\u0445\u0438\u0442\u043e, \u043d\u043e \u043d\u0435 \u0441\u043b\u0430\u0434\u043a\u0438\u0439
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
