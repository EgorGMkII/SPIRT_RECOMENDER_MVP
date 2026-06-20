"""LLM-based normalization of natural-language product search queries."""

from __future__ import annotations

import re
from typing import Any


PRODUCT_QUERY_NORMALIZER_PROMPT = """
Rewrite the user's rum/product request as a concise English search description for retrieval.

Goal:
- Preserve the user's real intent and constraints.
- Use natural language, not tags or JSON.
- Include relevant taste, aroma, style, serve, cocktail use, food context, mood,
  sweetness/dryness, intensity, and age/style.
- Do not include negative or avoided descriptors as searchable terms.
  If the user says "without coconut", "not sweet", "avoid vanilla", or similar,
  omit those avoided words from the search query.
- Convert negative constraints into positive desired profile language only when useful:
  "not sweet" -> "dry, clean, balanced"; "without coconut" -> do not mention coconut.
- Do not limit yourself to a fixed vocabulary.
- Do not invent product names or product facts.
- Keep it embedding-friendly: 1-3 short sentences.

Examples:
User: I want a smooth rum with vanilla and oak notes that works well in cocktails.
Search query: Smooth rum with vanilla and oak flavors. Suitable for cocktails. Approachable profile with gentle barrel notes.

User: \u043d\u0443\u0436\u0435\u043d \u0440\u043e\u043c \u0434\u043b\u044f \u0443\u0436\u0438\u043d\u0430, \u0431\u0443\u0434\u0443 \u0435\u0441\u0442\u044c \u043c\u044f\u0441\u043e \u0441 \u0433\u0440\u0438\u0431\u0430\u043c\u0438
Search query: Rum for dinner with meat and mushrooms. Rich enough for savory food, with oak, spice, caramel, toasted or earthy depth.

User: \u0445\u043e\u0447\u0443 \u0447\u0442\u043e-\u0442\u043e \u0441\u0443\u0445\u043e\u0435, \u043d\u0435 \u0441\u043b\u0430\u0434\u043a\u043e\u0435, \u043c\u043e\u0436\u043d\u043e \u043f\u0438\u0442\u044c \u0447\u0438\u0441\u0442\u044b\u043c
Search query: Dry rum for sipping neat. Clean, mature or barrel-influenced profile with balanced complexity.

User: \u0440\u043e\u043c \u0434\u043b\u044f \u043c\u043e\u0445\u0438\u0442\u043e, \u0431\u0435\u0437 \u043a\u043e\u043a\u043e\u0441\u0430
Search query: Rum for mojito cocktails. Fresh light rum profile for mint and lime.

User: \u0442\u043e\u0433\u0434\u0430 \u043f\u043e\u0441\u043e\u0432\u0435\u0442\u0443\u0439 \u0440\u043e\u043c \u0434\u043b\u044f \u043a\u043e\u043a\u0442\u0435\u0439\u043b\u0435\u0439, \u043d\u043e \u0431\u0435\u0437 \u0441\u043b\u0430\u0434\u043a\u043e\u0433\u043e \u043f\u0440\u043e\u0444\u0438\u043b\u044f
Search query: Rum for cocktails with a dry, clean, balanced and fresh profile.

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


def build_query_normalization_prompt(text: str) -> str:
    """Build the LLM prompt for product query normalization."""

    return PRODUCT_QUERY_NORMALIZER_PROMPT.format(query=text)


def _fallback_normalize_query(text: str) -> str:
    """Emergency fallback when LLM query normalization is unavailable."""

    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:300]


def normalize_query(
    text: str,
    llm: Any | None = None,
    use_llm: bool = False,
) -> str:
    """Normalize user text into a concise embedding-friendly search description."""

    if not use_llm:
        return _fallback_normalize_query(text)

    active_llm = llm
    if active_llm is None:
        from llm_module import get_langchain_openai_chat_model

        active_llm = get_langchain_openai_chat_model()

    try:
        response = active_llm.invoke(build_query_normalization_prompt(text))
        normalized = _message_content(response).strip().strip("`").strip('"')
        return " ".join(normalized.split()) or _fallback_normalize_query(text)
    except Exception:
        return _fallback_normalize_query(text)
