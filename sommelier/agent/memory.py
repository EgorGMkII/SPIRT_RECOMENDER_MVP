"""Session memory models and LLM follow-up resolution."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from sommelier.agent.schemas import IntentType


class ConversationMessage(BaseModel):
    """One durable conversation message."""

    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CandidateMemory(BaseModel):
    """Compact candidate memory saved from a previous turn."""

    item_id: str
    name: str
    kind: str
    score: float | None = None


class LastTurnMemory(BaseModel):
    """Compact summary of the last completed agent turn."""

    user_message: str
    effective_user_message: str
    intent: str | None = None
    search_query: str | None = None
    expanded_query: str | None = None
    cocktail_query: str | None = None
    final_answer: str | None = None
    candidates: list[CandidateMemory] = Field(default_factory=list)


class SessionMemory(BaseModel):
    """Durable session memory for one user session."""

    session_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    last_turn: LastTurnMemory | None = None


class FollowupResolution(BaseModel):
    """Structured output for resolving whether a message continues the last turn."""

    is_followup: bool = False
    intent: IntentType | None = None
    effective_user_message: str
    avoid_previous_candidates: bool = False
    reason: str = ""


FOLLOWUP_RESOLUTION_PROMPT = """
Resolve whether the current user message is a continuation of the previous assistant turn.

Return exactly one JSON object:
{{
  "is_followup": true,
  "intent": "search_products | food_pairing | cocktail_expansion | profile_update | unknown",
  "effective_user_message": "short standalone request for the next tool",
  "avoid_previous_candidates": false,
  "reason": "brief reason"
}}

Rules:
- Understand Russian and English.
- Use the previous turn context, not keyword rules.
- If the current message is a complete standalone request, set is_followup=false even if
  the previous turn was related.
- If the current message clearly asks for a different domain, switch topic:
  rum/product requests are search_products, cocktail/recipe requests are cocktail_expansion,
  food/dinner pairing requests are food_pairing.
- If the current message depends on previous candidates, recipe, food context, or wording,
  set is_followup=true.
- If it is a follow-up, preserve the previous topic unless the user clearly switches topics.
- For cocktail follow-ups like "something similar but simpler" or
  "\u0430 \u0435\u0441\u0442\u044c \u0447\u0442\u043e-\u0442\u043e \u043f\u043e\u0445\u043e\u0436\u0435\u0435, \u043d\u043e \u043f\u0440\u043e\u0449\u0435\u003f", keep intent=cocktail_expansion and MUST
  set avoid_previous_candidates=true.
- If the user asks for a similar, simpler, alternative, or another option after a
  recommendation, set avoid_previous_candidates=true and make effective_user_message
  ask for an alternative, not the same item.
- If the user asks for details, recipe, ingredients, or comparison of the same item,
  keep avoid_previous_candidates=false.
- effective_user_message must be concise and standalone enough for intent parsing/retrieval.
- Do not invent product facts. You may include previous candidate names as context.
- If there is no useful previous turn, set is_followup=false and effective_user_message=current message.
- Return JSON only.

Examples:
- Previous topic: cocktail. Current: "\u0430 \u0435\u0441\u0442\u044c \u0447\u0442\u043e-\u0442\u043e \u043f\u043e\u0445\u043e\u0436\u0435\u0435, \u043d\u043e \u043f\u0440\u043e\u0449\u0435\u003f"
  -> follow-up cocktail_expansion, avoid_previous_candidates=true.
- Previous topic: cocktail. Current: "\u0445\u043e\u0447\u0443 \u0440\u043e\u043c \u0441 \u0434\u0443\u0431\u043e\u043c, \u0432\u0430\u043d\u0438\u043b\u044c\u044e \u0438 \u043f\u0440\u044f\u043d\u043e\u0441\u0442\u044f\u043c\u0438"
  -> not a follow-up, search_products.
- Previous topic: rum recommendation. Current: "\u043c\u043d\u0435 \u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f \u0441\u043b\u0438\u0448\u043a\u043e\u043c \u0441\u043b\u0430\u0434\u043a\u043e\u0435 \u0438 \u043a\u043e\u043a\u043e\u0441"
  -> not a follow-up, profile_update.
- Previous topic: profile update. Current: "\u0442\u043e\u0433\u0434\u0430 \u043f\u043e\u0441\u043e\u0432\u0435\u0442\u0443\u0439 \u0440\u043e\u043c \u0434\u043b\u044f \u043a\u043e\u043a\u0442\u0435\u0439\u043b\u0435\u0439, \u043d\u043e \u0431\u0435\u0437 \u0441\u043b\u0430\u0434\u043a\u043e\u0433\u043e \u043f\u0440\u043e\u0444\u0438\u043b\u044f"
  -> not a follow-up, search_products.

Previous turn:
{last_turn_json}

Current user message:
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


def build_followup_resolution_prompt(message: str, memory: SessionMemory) -> str:
    """Build the LLM prompt for follow-up resolution."""

    last_turn_json = (
        memory.last_turn.model_dump_json(indent=2)
        if memory.last_turn
        else "{}"
    )
    return FOLLOWUP_RESOLUTION_PROMPT.format(
        last_turn_json=last_turn_json,
        message=message,
    )


def resolve_followup_context(
    message: str,
    memory: SessionMemory | None,
    llm: Any | None = None,
    use_llm: bool = False,
) -> FollowupResolution:
    """Resolve follow-up context with an LLM and return a standalone request."""

    if not memory or not memory.last_turn or not use_llm:
        return FollowupResolution(
            is_followup=False,
            intent=None,
            effective_user_message=message,
            avoid_previous_candidates=False,
            reason="No LLM follow-up resolution used.",
        )

    active_llm = llm
    if active_llm is None:
        from llm_module import get_langchain_openai_chat_model

        active_llm = get_langchain_openai_chat_model()

    try:
        response = active_llm.invoke(build_followup_resolution_prompt(message, memory))
        return FollowupResolution.model_validate(
            _extract_json_object(_message_content(response))
        )
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        return FollowupResolution(
            is_followup=False,
            intent=None,
            effective_user_message=message,
            avoid_previous_candidates=False,
            reason="Follow-up resolution failed; treated as a new request.",
        )
