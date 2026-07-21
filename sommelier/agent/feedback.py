"""Independent analytics classification for user feedback."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from sommelier.agent.contracts import FeedbackResult


FEEDBACK_PROMPT = """Classify the current user message for analytics only.
Return exactly one FeedbackResult.

Labels:
- purchase_intent: the user explicitly intends to buy, order, take, or add a
  product to the cart.
- negative_feedback: the user criticizes the assistant's previous answer or
  behavior, for example "не то", "ты ошибся", or "я не просил советовать".
- neutral: every other message, including normal requests, preferences,
  restrictions, and negative opinions about a rum, taste, or cocktail.

If follow_up is true, interpret previous_assistant_answer and user_request
together to determine whether the user is criticizing the assistant. If
follow_up is false, ignore previous_assistant_answer completely.

For mixed signals use this strict priority:
negative_feedback > purchase_intent > neutral.

Examples:
- "Хочу купить этот ром" -> purchase_intent
- "Ответ не тот, я не просил советовать" -> negative_feedback
- "Мне не нравится сладкий ром" -> neutral
- "Этот ром невкусный" -> neutral
"""

NEGATIVE_FEEDBACK_PATTERNS = (
    r"\bне\s+то\b",
    r"\bне\s+тот\b",
    r"\bне\s+так\b",
    r"\bошиб",
    r"\bне\s+просил",
    r"\bне\s+просила",
    r"\bя\s+имел[аи]?\s+в\s+виду\b",
    r"\bя\s+говорил[а]?\s+про\b",
    r"\bнет,\s*про\b",
    r"\bwrong\b",
    r"\bnot\s+what\s+i\s+asked\b",
)

PURCHASE_PATTERNS = (
    r"\bкуп",
    r"\bзакаж",
    r"\bвозьм",
    r"\bдобав[ьи]",
    r"\bв\s+корзин",
    r"\bbuy\b",
    r"\border\b",
    r"\badd\s+to\s+cart\b",
)


def _fallback_feedback(
    *,
    user_request: str,
    previous_assistant_answer: str | None,
    follow_up: bool,
) -> FeedbackResult:
    """Deterministic safety net when structured analytics output is invalid."""

    text = user_request.casefold()
    if follow_up and previous_assistant_answer:
        text = f"{previous_assistant_answer.casefold()}\n{text}"
    if any(re.search(pattern, text) for pattern in NEGATIVE_FEEDBACK_PATTERNS):
        return FeedbackResult(feedback="negative_feedback")
    if any(re.search(pattern, user_request.casefold()) for pattern in PURCHASE_PATTERNS):
        return FeedbackResult(feedback="purchase_intent")
    return FeedbackResult(feedback="neutral")


def classify_feedback(
    *,
    user_request: str,
    previous_assistant_answer: str | None,
    follow_up: bool,
    llm: Any,
) -> FeedbackResult:
    """Make one structured classification call without retries or fallback."""

    payload = {
        "user_request": user_request,
        "follow_up": follow_up,
        "previous_assistant_answer": (
            previous_assistant_answer if follow_up else None
        ),
    }
    structured = llm.with_structured_output(
        FeedbackResult,
        method="function_calling",
    )
    raw = structured.invoke(
        f"{FEEDBACK_PROMPT}\nINPUT:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        return FeedbackResult.model_validate(raw)
    except (ValidationError, TypeError):
        return _fallback_feedback(
            user_request=user_request,
            previous_assistant_answer=previous_assistant_answer,
            follow_up=follow_up,
        )
