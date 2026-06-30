"""Independent analytics classification for user feedback."""

from __future__ import annotations

import json
from typing import Any

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
    return FeedbackResult.model_validate(raw)
