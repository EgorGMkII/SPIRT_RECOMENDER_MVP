"""Structured intent parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from sommelier.agent.schemas import IntentType, ParsedIntent
from sommelier.retrieval.schemas import SearchRequest


class IntentParsePayload(BaseModel):
    """LLM intent parser payload."""

    intent: IntentType
    query: str = Field(min_length=1)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


INTENT_PARSE_PROMPT = """
Classify the user's message for a controlled rum sommelier assistant.

Return exactly one JSON object:
{{
  "intent": "search_products | food_pairing | cocktail_expansion | profile_update | unknown",
  "query": "concise query text to pass to the selected retrieval tool",
  "confidence": 0.0
}}

Intent meanings:
- search_products: user asks for a rum/product recommendation by flavor, style, occasion, or use.
- food_pairing: user asks what rum fits food, dinner, a dish, meat, fish, dessert, spice, etc.
- cocktail_expansion: user asks for a cocktail, drink, mixer, recipe, or what to make with ingredients.
- profile_update: user explicitly states stable likes/dislikes/preferences and is not asking for a recommendation.
- unknown: unclear.

Rules:
- Understand Russian and English.
- Do not use keyword-only matching; infer the user's goal.
- Recommendation requests win over profile_update. If the user asks to recommend,
  find, choose, suggest, make, or give a recipe, classify the request by the
  recommendation task and keep preferences inside query.
- Use profile_update only when the message is mainly a stable preference statement
  and does not ask for a recommendation, recipe, search, comparison, or pairing.
- Keep query short and faithful to the message.
- Do not invent product names or facts.
- Return JSON only.

Examples:
User: мне не нравится слишком сладкое и кокос
JSON: {{"intent":"profile_update","query":"мне не нравится слишком сладкое и кокос","confidence":0.95}}

User: мне нравится мохито, но не люблю пина коладу
JSON: {{"intent":"profile_update","query":"мне нравится мохито, но не люблю пина коладу","confidence":0.95}}

User: тогда посоветуй ром для коктейлей, но без сладкого профиля
JSON: {{"intent":"search_products","query":"ром для коктейлей без сладкого профиля","confidence":0.95}}

User: хочу ром с дубом, ванилью и пряностями
JSON: {{"intent":"search_products","query":"ром с дубом, ванилью и пряностями","confidence":0.95}}

User: какой коктейль сделать с лаймом и мятой?
JSON: {{"intent":"cocktail_expansion","query":"коктейль с лаймом и мятой","confidence":0.95}}

User: нужен ром к стейку
JSON: {{"intent":"food_pairing","query":"ром к стейку","confidence":0.95}}

User message:
{message}
""".strip()


def _message_content(message: Any) -> str:
    """Extract text content from an LLM response."""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(message, str):
        return message
    return str(message)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from an LLM response."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def build_intent_parse_prompt(message: str) -> str:
    """Build the LLM intent parsing prompt."""

    return INTENT_PARSE_PROMPT.format(message=message)


def _fallback_parse_intent(message: str) -> ParsedIntent:
    """Small emergency fallback when LLM intent parsing is unavailable."""

    return ParsedIntent(
        intent=IntentType.SEARCH_PRODUCTS,
        search=SearchRequest(query=message),
        confidence=0.2,
    )


def parse_intent(
    message: str,
    llm: Any | None = None,
    use_llm: bool = False,
) -> ParsedIntent:
    """Parse a user message into the minimal MVP intent schema."""

    if not use_llm:
        return _fallback_parse_intent(message)

    active_llm = llm
    if active_llm is None:
        from llm_module import get_langchain_openai_chat_model

        active_llm = get_langchain_openai_chat_model()

    try:
        response = active_llm.invoke(build_intent_parse_prompt(message))
        payload = IntentParsePayload.model_validate(
            _extract_json_object(_message_content(response))
        )
        return ParsedIntent(
            intent=payload.intent,
            search=SearchRequest(query=payload.query),
            confidence=payload.confidence,
        )
    except Exception:
        return _fallback_parse_intent(message)
