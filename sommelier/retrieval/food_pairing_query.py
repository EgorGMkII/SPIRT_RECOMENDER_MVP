"""Inference-based food-pairing retrieval via query expansion.

Product pages do not provide a reliable direct rum-to-food pairing database.
This module reformulates a food description into a rum search query and then
uses the existing ProductSearchProfile retrieval layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from sommelier.retrieval.faiss_index import FaissIndex, SearchResult

DEFAULT_PAIRING_CAVEAT = (
    "Food pairing is inferred from product profile similarity and general pairing "
    "language, not from source-backed Bacardi food-pairing claims."
)

FOOD_PAIRING_QUERY_PROMPT = """
Rewrite the user's food/dinner description into an embedding-friendly rum search query.

Goal:
- Use general culinary knowledge to describe the kind of rum profile that could fit the food.
- Do not include negative or avoided descriptors as searchable terms.
  If the user says "not sweet", "without coconut", "avoid vanilla", or similar,
  omit those avoided words from the search query.
- Convert negative constraints into positive pairing language only when useful:
  "not sweet" -> "dry, savory-friendly, balanced"; "without coconut" -> do not mention coconut.
- Do not claim the pairing is source-backed.
- Do not invent a specific product.
- Keep it concise: 1-3 sentences.
- Write in English because product search profiles are mostly English.

Good output style:
"Rum for grilled beef and mushrooms. Rich aged profile with oak, caramel, spice, vanilla, toasted or earthy notes. Suitable for dinner pairing."

User food text:
\u043c\u044f\u0441\u043e \u0441 \u0433\u0440\u0438\u0431\u0430\u043c\u0438, \u043d\u043e \u043d\u0435 \u0441\u043b\u0430\u0434\u043a\u0438\u0439 \u0440\u043e\u043c

Search query:
"Rum for meat and mushrooms. Dry, savory-friendly aged profile with oak, spice, toasted or earthy notes. Suitable for dinner pairing."

User food text:
{food_text}
""".strip()


class FoodPairingSearchResult(BaseModel):
    """Result of food query expansion and vector retrieval."""

    original_food_text: str
    expanded_query: str
    retrieval_results: list[SearchResult] = Field(default_factory=list)
    caveat: str = DEFAULT_PAIRING_CAVEAT


def build_food_pairing_query_prompt(food_text: str) -> str:
    """Build the LLM prompt for food-pairing query reformulation."""

    return FOOD_PAIRING_QUERY_PROMPT.format(food_text=food_text)


def _message_content(message: Any) -> str:
    """Extract text content from an LLM response."""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(message, str):
        return message
    return str(message)


def _fallback_food_pairing_query(food_text: str) -> str:
    """Emergency fallback when LLM query reformulation is unavailable."""

    return (
        f"Rum for food pairing with this dish: {food_text}. "
        "Use a balanced rum profile with enough flavor intensity for the food. "
        "Consider aged, spiced, fruity, fresh, cocktail-friendly or sipping rum depending on context."
    )


def normalize_food_pairing_query(
    food_text: str,
    llm: Any | None = None,
    use_llm: bool = False,
) -> str:
    """Reformulate Russian or English food text into a rum search query."""

    if not use_llm:
        return _fallback_food_pairing_query(food_text)

    active_llm = llm
    if active_llm is None:
        from llm_module import get_langchain_openai_chat_model

        active_llm = get_langchain_openai_chat_model()

    try:
        response = active_llm.invoke(build_food_pairing_query_prompt(food_text))
        expanded = _message_content(response).strip().strip('"')
        if not expanded:
            return _fallback_food_pairing_query(food_text)
        return expanded
    except Exception:
        return _fallback_food_pairing_query(food_text)


def search_for_food_pairing(
    food_text: str,
    top_k: int = 5,
    index: FaissIndex | None = None,
    index_dir: Path = Path("data/indexes"),
    llm: Any | None = None,
    use_llm: bool = False,
) -> FoodPairingSearchResult:
    """Search product profiles for food-pairing candidates via query reformulation."""

    expanded_query = normalize_food_pairing_query(food_text, llm=llm, use_llm=use_llm)
    active_index = index or FaissIndex.load(index_dir)
    results = active_index.search(expanded_query, top_k=top_k)
    return FoodPairingSearchResult(
        original_food_text=food_text,
        expanded_query=expanded_query,
        retrieval_results=results,
    )
