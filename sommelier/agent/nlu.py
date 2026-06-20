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
User: \u043c\u043d\u0435 \u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f \u0441\u043b\u0438\u0448\u043a\u043e\u043c \u0441\u043b\u0430\u0434\u043a\u043e\u0435 \u0438 \u043a\u043e\u043a\u043e\u0441
JSON: {{"intent":"profile_update","query":"\u043c\u043d\u0435 \u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f \u0441\u043b\u0438\u0448\u043a\u043e\u043c \u0441\u043b\u0430\u0434\u043a\u043e\u0435 \u0438 \u043a\u043e\u043a\u043e\u0441","confidence":0.95}}

User: \u043c\u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f \u043c\u043e\u0445\u0438\u0442\u043e, \u043d\u043e \u043d\u0435 \u043b\u044e\u0431\u043b\u044e \u043f\u0438\u043d\u0430 \u043a\u043e\u043b\u0430\u0434\u0443
JSON: {{"intent":"profile_update","query":"\u043c\u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f \u043c\u043e\u0445\u0438\u0442\u043e, \u043d\u043e \u043d\u0435 \u043b\u044e\u0431\u043b\u044e \u043f\u0438\u043d\u0430 \u043a\u043e\u043b\u0430\u0434\u0443","confidence":0.95}}

User: \u0442\u043e\u0433\u0434\u0430 \u043f\u043e\u0441\u043e\u0432\u0435\u0442\u0443\u0439 \u0440\u043e\u043c \u0434\u043b\u044f \u043a\u043e\u043a\u0442\u0435\u0439\u043b\u0435\u0439, \u043d\u043e \u0431\u0435\u0437 \u0441\u043b\u0430\u0434\u043a\u043e\u0433\u043e \u043f\u0440\u043e\u0444\u0438\u043b\u044f
JSON: {{"intent":"search_products","query":"\u0440\u043e\u043c \u0434\u043b\u044f \u043a\u043e\u043a\u0442\u0435\u0439\u043b\u0435\u0439 \u0431\u0435\u0437 \u0441\u043b\u0430\u0434\u043a\u043e\u0433\u043e \u043f\u0440\u043e\u0444\u0438\u043b\u044f","confidence":0.95}}

User: \u0445\u043e\u0447\u0443 \u0440\u043e\u043c \u0441 \u0434\u0443\u0431\u043e\u043c, \u0432\u0430\u043d\u0438\u043b\u044c\u044e \u0438 \u043f\u0440\u044f\u043d\u043e\u0441\u0442\u044f\u043c\u0438
JSON: {{"intent":"search_products","query":"\u0440\u043e\u043c \u0441 \u0434\u0443\u0431\u043e\u043c, \u0432\u0430\u043d\u0438\u043b\u044c\u044e \u0438 \u043f\u0440\u044f\u043d\u043e\u0441\u0442\u044f\u043c\u0438","confidence":0.95}}

User: \u043a\u0430\u043a\u043e\u0439 \u043a\u043e\u043a\u0442\u0435\u0439\u043b\u044c \u0441\u0434\u0435\u043b\u0430\u0442\u044c \u0441 \u043b\u0430\u0439\u043c\u043e\u043c \u0438 \u043c\u044f\u0442\u043e\u0439\u003f
JSON: {{"intent":"cocktail_expansion","query":"\u043a\u043e\u043a\u0442\u0435\u0439\u043b\u044c \u0441 \u043b\u0430\u0439\u043c\u043e\u043c \u0438 \u043c\u044f\u0442\u043e\u0439","confidence":0.95}}

User: \u043d\u0443\u0436\u0435\u043d \u0440\u043e\u043c \u043a \u0441\u0442\u0435\u0439\u043a\u0443
JSON: {{"intent":"food_pairing","query":"\u0440\u043e\u043c \u043a \u0441\u0442\u0435\u0439\u043a\u0443","confidence":0.95}}

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
