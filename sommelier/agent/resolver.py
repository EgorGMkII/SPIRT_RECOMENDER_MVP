"""Structured resolution of one user turn against compact turn memory."""

import json
import re
from typing import Any

from pydantic import ValidationError

from sommelier.agent.contracts import TurnResolution
from sommelier.agent.memory import SessionMemory
from sommelier.agent.profile import UserProfile

RESOLVER_PROMPT = """Resolve the current message for a rum sommelier.
Return TurnResolution only.

STRICT FIELD SEPARATION:
- effective_request describes only what the user WANTS to find;
- negative_request describes everything the user wants to EXCLUDE or AVOID;
- never translate an exclusion into a softened positive-looking property;
- if removing an exclusion leaves the previous effective_request unchanged,
  keep that effective_request unchanged.

follow_up=false:
- initial_request must exactly equal the current user_request;
- effective_request is a standalone formulation containing ONLY desired,
  positive properties and task context;
- negative_request contains only temporary exclusions, or null.

follow_up=true:
- use it only when the message modifies the SAME request or explicitly refers
  to a saved answer/object (for example "ещё", "первый из последних", "его",
  "этот ром", "как его приготовить");
- initial_request must exactly equal initial_request from one saved turn;
- effective_request is the complete updated request containing ONLY desired,
  positive properties and task context;
- negative_request is the complete updated temporary exclusions, or null.

FOLLOW-UP DECISION IS STRICT:
- topical similarity alone is not enough;
- words such as "теперь", "а теперь", "now", "next" are discourse markers and
  do NOT make a request a follow-up;
- a complete standalone request for a different target type or task is NEW;
- switching from cocktails to a rum product, from a product to food pairing,
  or to another independently answerable recommendation is NEW unless the user
  explicitly asks to use or clarify a previous result;
- durable preferences in UserProfile are always available and do not require
  marking the next request as follow_up.
- statements such as "мне нравится...", "я люблю...", "я предпочитаю...",
  "I like..." and "I prefer..." are profile-only updates unless the user also
  explicitly asks for a recommendation, search, recipe or other action;
- for a profile-only update, return the appropriate profile_patch and formulate
  effective_request as remembering/updating the preference. Do not transform
  the preference statement into a recommendation request;
- a correction such as "это моё предпочтение, я не просил советовать" is also
  profile-only and must not become a catalog request.
- cart commands are standalone actions and normally follow_up=false. Use
  follow_up=true only when the cart command explicitly identifies an object
  through a saved answer, for example "добавь первый из последних вариантов"
  or "добавь его в корзину".
- set cart_action="add" for adding/increasing a cart item;
- set cart_action="delete" for removing an item;
- set cart_action="show" for viewing cart contents;
- otherwise set cart_action=null. cart_action describes the requested action,
  not a tool choice.

Examples:
- Previous: "Посоветуй свежий коктейль с лаймом и мятой."
  Current: "А есть что-то ещё, но не сладкое?"
  => follow_up=true; it modifies the same cocktail request.
- Previous: "Посоветуй свежий коктейль с лаймом и мятой."
  Current: "Как приготовить первый из последних вариантов?"
  => follow_up=true; it explicitly refers to the previous shown results.
- Previous: "Посоветуй свежий коктейль с лаймом и мятой."
  Current: "Теперь посоветуй выдержанный ром для коктейлей, но не сладкий."
  => follow_up=false; this is a standalone product request with a new
     initial_request, despite "Теперь".

Negative meaning must not appear in effective_request in ANY wording. This
includes direct negatives ("not sweet", "не сладкий", "without sugar",
"без сахара") and softened or inverted paraphrases ("low sweetness",
"низкая сладость", "less sweet", "менее сладкий", "dry instead of sweet").
All of that meaning belongs exclusively in negative_request.

Example:
user: "А есть что-то ещё, но не сладкое?"
effective_request: "Свежий коктейль с лаймом и мятой."
negative_request: "Исключить сладкие коктейли."

WRONG effective_request examples for that user message:
- "Свежий коктейль с лаймом и мятой с низкой сладостью."
- "Менее сладкий свежий коктейль с лаймом и мятой."
- "Сухой свежий коктейль с лаймом и мятой."

Return profile_patch only for explicit durable preferences. It is independent
of follow_up. A pure preference statement MUST produce profile_patch but MUST
NOT imply a recommendation.

Example:
user: "Привет, мне нравится мята и сладкое."
effective_request: "Запомнить предпочтения: мята и сладкие вкусы."
profile_patch: add "мята" and "сладкие вкусы" to liked_flavors
Do not search and do not recommend a product.

Do not choose tools, queries or catalog IDs. Do not answer."""


def _normalized_clause(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s-]", " ", value.lower()).split())


def _validate_negative_separation(result: TurnResolution) -> None:
    if not result.negative_request:
        return
    effective = _normalized_clause(result.effective_request)
    clauses = [
        _normalized_clause(clause)
        for clause in re.split(r"[;,.\n]+", result.negative_request)
        if _normalized_clause(clause)
    ]
    leaked = [clause for clause in clauses if clause and clause in effective]
    if leaked:
        raise ValueError(
            "negative constraints must appear only in negative_request; "
            f"remove from effective_request: {leaked}"
        )


def _expected_cart_action(user_request: str, effective_request: str) -> str | None:
    text = f"{user_request} {effective_request}".lower()
    cart_context = any(token in text for token in ("корзин", "cart", "туда же"))
    if not cart_context:
        return None
    if any(token in text for token in ("добав", "полож", "add ")):
        return "add"
    if any(token in text for token in ("убер", "удал", "remove", "delete")):
        return "delete"
    if any(token in text for token in ("покаж", "что сейчас", "show", "what is")):
        return "show"
    return None


def _is_explicit_preference_statement(user_request: str) -> bool:
    text = user_request.lower()
    return any(
        marker in text
        for marker in (
            "мне нравится",
            "я люблю",
            "я предпочитаю",
            "мой любим",
            "запомни",
            "i like",
            "i love",
            "i prefer",
            "remember that",
        )
    )


def _profile_patch_has_changes(result: TurnResolution) -> bool:
    patch = result.profile_patch
    if patch is None:
        return False
    return any(
        preference.add or preference.remove
        for preference in (
            patch.liked_flavors,
            patch.disliked_flavors,
            patch.liked_cocktails,
            patch.disliked_cocktails,
        )
    )


def validate_turn_resolution(
    result: TurnResolution,
    user_request: str,
    memory: SessionMemory,
) -> TurnResolution:
    if result.follow_up:
        if not memory.turns:
            raise ValueError("follow_up requires saved turns")
        allowed = {turn.initial_request for turn in memory.turns}
        if result.initial_request not in allowed:
            raise ValueError("follow_up initial_request must match saved turn")
    elif result.initial_request != user_request:
        raise ValueError("new turn initial_request must equal user_request exactly")
    if result.negative_request is not None:
        result.negative_request = " ".join(result.negative_request.split()) or None
    expected_cart_action = _expected_cart_action(
        user_request, result.effective_request
    )
    if expected_cart_action and result.cart_action != expected_cart_action:
        raise ValueError(
            f"cart_action must be {expected_cart_action!r} for this request"
        )
    if (
        _is_explicit_preference_statement(user_request)
        and not _profile_patch_has_changes(result)
    ):
        raise ValueError(
            "explicit durable preference requires a non-empty profile_patch"
        )
    _validate_negative_separation(result)
    return result


def resolve_turn(
    *,
    user_request: str,
    memory: SessionMemory,
    profile: UserProfile,
    llm: Any,
) -> TurnResolution:
    structured = llm.with_structured_output(TurnResolution, method="function_calling")
    payload = {
        "user_request": user_request,
        "turns": [turn.model_dump(mode="json") for turn in memory.turns],
        "user_profile": profile.model_dump(mode="json"),
    }
    feedback = ""
    last_error: Exception | None = None
    for _ in range(2):
        try:
            raw = structured.invoke(
                f"{RESOLVER_PROMPT}\nINPUT:\n{json.dumps(payload, ensure_ascii=False)}{feedback}"
            )
            return validate_turn_resolution(
                TurnResolution.model_validate(raw), user_request, memory
            )
        except (ValidationError, ValueError, TypeError) as exc:
            last_error = exc
            feedback = f"\nVALIDATION ERROR: {exc}. Correct the result."
    raise ValueError(f"turn resolution failed: {last_error}")
