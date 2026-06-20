"""Profile update extraction and application."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from sommelier.agent.profile import ProfileUpdate, UserProfile, apply_profile_update
from sommelier.agent.schemas import ToolResult

PROFILE_UPDATE_PROMPT = """
Extract stable user taste preferences from the message.

Return exactly one JSON object with:
{{
  "liked_flavors": [],
  "disliked_flavors": [],
  "liked_cocktails": [],
  "disliked_cocktails": [],
  "ignored": []
}}

Rules:
- Extract only explicit stable preferences.
- "I like/love..." or "\u043b\u044e\u0431\u043b\u044e/\u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f..." can create liked preferences.
- "I dislike/hate/avoid..." or "\u043d\u0435 \u043b\u044e\u0431\u043b\u044e/\u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f/\u0431\u0435\u0437/\u0438\u0437\u0431\u0435\u0433\u0430\u044e..." can create disliked preferences.
- If the user merely asks for something once, put it in ignored, not preferences.
- Normalize values to short English lowercase terms where possible.
- Examples: vanilla, oak, coconut, sweet, mojito, pina colada.
- Do not invent preferences.
- Return JSON only.

User message:
{message}
""".strip()


def has_profile_signal(message: str) -> bool:
    """Deprecated compatibility shim.

    Profile detection is intentionally handled by the LLM extractor now, not by
    brittle word triggers.
    """

    return False


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


def build_profile_update_prompt(message: str) -> str:
    """Build the preference extraction prompt."""

    return PROFILE_UPDATE_PROMPT.format(message=message)


def extract_profile_update(
    message: str,
    llm: Any | None = None,
    use_llm: bool = False,
) -> ProfileUpdate:
    """Extract a controlled profile update from a user message."""

    if not use_llm:
        return ProfileUpdate(ignored=[message])

    active_llm = llm
    if active_llm is None:
        from llm_module import get_langchain_openai_chat_model

        active_llm = get_langchain_openai_chat_model()
    response = active_llm.invoke(build_profile_update_prompt(message))
    try:
        payload = _extract_json_object(_message_content(response))
        return ProfileUpdate.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        return ProfileUpdate(ignored=[message])


def profile_update(
    profile: UserProfile,
    message: str,
    llm: Any | None = None,
    use_llm: bool = False,
) -> tuple[UserProfile, ToolResult]:
    """Extract and apply a profile update from one user message."""

    update = extract_profile_update(message, llm=llm, use_llm=use_llm)
    updated = apply_profile_update(profile, update)
    changed = {
        "liked_flavors": update.liked_flavors,
        "disliked_flavors": update.disliked_flavors,
        "liked_cocktails": update.liked_cocktails,
        "disliked_cocktails": update.disliked_cocktails,
    }
    changed_count = sum(len(values) for values in changed.values())
    return (
        updated,
        ToolResult(
            tool_name="profile_update",
            summary=f"Profile update applied: {changed_count} preference value(s).",
            metadata={
                "update": update.model_dump(mode="json"),
                "changed": changed,
            },
        ),
    )
