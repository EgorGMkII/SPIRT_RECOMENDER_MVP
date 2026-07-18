"""Structured resolution of one user turn against compact turn memory."""

import json
import re
from typing import Any

from pydantic import ValidationError

from sommelier.agent.contracts import TurnResolution
from sommelier.agent.memory import SessionMemory
from sommelier.agent.profile import UserProfile

RESOLVER_TURN_LIMIT = 6
RESOLVER_MESSAGE_LIMIT = 6
RECENT_USER_MESSAGE_CHARS = 500
RECENT_ASSISTANT_MESSAGE_CHARS = 800

RESOLVER_PROMPT = """Resolve the current message for a rum sommelier.
Return TurnResolution only.

REQUEST SCOPE:
- request_scope describes the CURRENT requested action, not the historical
  topic and not a tool choice;
- product: recommend or explain a rum product;
- cocktail: recommend concrete cocktails;
- recipe: ingredients or preparation for a cocktail;
- food_pairing: choose a product for food or explain a food pairing;
- cart: add, delete or show cart items;
- profile: remember or correct a durable preference;
- catalog_listing: explicitly list every known rum product or every known
  cocktail by name; this is catalog navigation, not a recommendation;
- conversation: smalltalk or a clarification question without a catalog task;
- a follow-up may change scope when the user explicitly changes the action.
  Example: food_pairing -> cocktail for "which cocktails can I make from it?";
- a short constraint-only follow-up such as "давай сладкие варианты",
  "тогда покрепче" or "more fresh ones" MUST inherit the previous scope,
  action and resolved object, changing only the new constraint.

INPUT CONTEXT:
- recent_messages contains only the last three completed user/assistant
  exchanges, with long messages truncated. Use it to understand pending
  actions, corrections and what the assistant offered to do next;
- turns contains at most the six newest compact turns;
- each turn.shown_results contains the catalog objects explicitly shown to
  the user in that answer, in display order. It is the authoritative source
  for resolving catalog pronouns and references such as "из них",
  "первый", "второй" and "последний";
- never treat an incidental product name from recent_messages as the current
  catalog object when it is absent from recent turn.shown_results.

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
  "этот ром", "как его приготовить", "давай сладкий");
- initial_request must exactly equal initial_request from one saved turn;
- effective_request is the complete updated request containing ONLY desired,
  positive properties and task context;
- negative_request is the complete updated temporary exclusions, or null.

PRONOUN AND REFERENT RESOLUTION IS RECENCY-FIRST:
- resolve "он", "она", "оно", "его", "её", "из него", "из неё",
  "этот ром", "этот напиток", "этот коктейль", "it", "this rum" and similar
  references to the MOST RECENT compatible object in shown_results;
- inspect saved turns from newest to oldest. Within a turn, shown_results keeps
  the exact order in which options were shown to the user. Skip profile-only
  turns, smalltalk and turns with empty shown_results; they do not erase the
  last shown objects;
- compatibility is determined by the requested action: "какие коктейли из
  него" requires the most recent product, while "как его приготовить" requires
  the most recent cocktail;
- recency has priority over topical similarity to an older conversation branch.
  Never select an older object merely because an older request used similar
  words such as "коктейли из";
- after resolving a reference, effective_request MUST contain the exact object
  name from shown_results. Never leave an unresolved pronoun in
  effective_request;
- for "из них"/"which of them" questions, keep the action scoped to the
  latest compatible shown_results set rather than inventing a broad new search;
- if the user explicitly names an object, that explicit name overrides a
  pronoun or implicit recent referent.

CORRECTIONS MUST PRESERVE THE PENDING ACTION:
- messages such as "нет, про X", "я говорю про X", "имел в виду X" and
  "not that one, I mean X" correct the object of the immediately preceding
  request;
- keep the action requested in the immediately preceding turn and replace only
  the target object. Do not fall back to an older action associated with X;
- a wrong assistant answer does not cancel the user's pending action. Correct
  the resolved effective_request and perform the originally requested action.

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
- Older shown product: "BACARDÍ Coconut rum"
  Most recent shown product: "BACARDÍ Spiced"
  Current: "А какие коктейли из него можно сделать?"
  => follow_up=true
  => effective_request: "Подобрать коктейли на основе BACARDÍ Spiced."
  BACARDÍ Coconut rum is WRONG: the most recent compatible shown product wins.
- Previous user action: "А какие коктейли из него можно сделать?"
  Assistant incorrectly discussed BACARDÍ Coconut rum.
  Current: "Нет, про BACARDÍ Spiced говорю."
  => follow_up=true
  => effective_request: "Подобрать коктейли на основе BACARDÍ Spiced."
  Preserve the cocktail-selection action. Do not return to food pairing or
  merely describe BACARDÍ Spiced.
- Previous effective request: "Подобрать коктейли на основе BACARDÍ Spiced."
  Previous scope: cocktail
  Current: "Давай сладкие варианты."
  => follow_up=true
  => request_scope=cocktail
  => effective_request:
     "Подобрать сладкие коктейли на основе BACARDÍ Spiced."
  Returning a sweet rum product is WRONG.

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


def _recent_shown_result_candidates(
    memory: SessionMemory,
    turn_limit: int = RESOLVER_TURN_LIMIT,
) -> list[dict[str, object]]:
    """Build a newest-first validation view from durable shown_results."""

    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    recent_turns = memory.turns[-turn_limit:]
    for turn_offset, turn in enumerate(reversed(recent_turns), start=1):
        for shown in turn.shown_results:
            key = (shown.kind, shown.id)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "kind": shown.kind,
                    "id": shown.id,
                    "name": shown.name,
                    "turn_offset": turn_offset,
                }
            )
    return candidates


def truncate_recent_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    truncated: list[dict[str, str]] = []
    for message in messages[-RESOLVER_MESSAGE_LIMIT:]:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        limit = (
            RECENT_USER_MESSAGE_CHARS
            if role == "user"
            else RECENT_ASSISTANT_MESSAGE_CHARS
        )
        truncated.append({"role": role, "content": content[:limit]})
    return truncated


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


def _contains_catalog_pronoun(user_request: str) -> bool:
    text = f" {_normalized_clause(user_request)} "
    return any(
        marker in text
        for marker in (
            " из него ",
            " из неё ",
            " его ",
            " её ",
            " этот ром ",
            " этот напиток ",
            " этот коктейль ",
            " from it ",
            " this rum ",
            " this cocktail ",
        )
    )


def _referent_kind_for_scope(scope: str) -> str | None:
    if scope == "cocktail":
        return "product"
    if scope == "recipe":
        return "cocktail"
    if scope in {"product", "food_pairing"}:
        return "product"
    return None


def _explicit_recent_object(
    user_request: str,
    candidates: list[dict[str, object]],
) -> dict[str, object] | None:
    user_tokens = set(_normalized_clause(user_request).split())
    generic = {"bacardi", "bacardí", "rum", "ром", "cocktail", "коктейль"}
    for candidate in candidates:
        name_tokens = set(_normalized_clause(str(candidate["name"])).split())
        distinctive = {
            token for token in name_tokens if len(token) >= 4 and token not in generic
        }
        if distinctive & user_tokens:
            return candidate
    return None


def _validate_scope_and_referent(
    result: TurnResolution,
    user_request: str,
    memory: SessionMemory,
) -> None:
    if not _contains_catalog_pronoun(user_request):
        return
    expected_kind = _referent_kind_for_scope(result.request_scope)
    candidates = [
        item
        for item in _recent_shown_result_candidates(memory)
        if expected_kind is None or item["kind"] == expected_kind
    ]
    if not candidates:
        return
    explicit_object = _explicit_recent_object(user_request, candidates)
    if explicit_object is None and any(
        token in _normalized_clause(user_request)
        for token in ("bacardi", "bacardí")
    ):
        # The user explicitly named a Bacardi object that may not have been
        # shown yet. The explicit name overrides pronoun recency.
        return
    expected_name = str((explicit_object or candidates[0])["name"])
    if _normalized_clause(expected_name) not in _normalized_clause(
        result.effective_request
    ):
        raise ValueError(
            "effective_request must resolve the pronoun to the most recent "
            f"compatible object: {expected_name!r}"
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
    _validate_scope_and_referent(result, user_request, memory)
    _validate_negative_separation(result)
    return result


def resolve_turn(
    *,
    user_request: str,
    memory: SessionMemory,
    profile: UserProfile,
    llm: Any,
    recent_messages: list[dict[str, str]] | None = None,
) -> TurnResolution:
    structured = llm.with_structured_output(TurnResolution, method="function_calling")
    payload = {
        "user_request": user_request,
        "recent_messages": truncate_recent_messages(list(recent_messages or [])),
        "turns": [
            turn.model_dump(mode="json")
            for turn in memory.turns[-RESOLVER_TURN_LIMIT:]
        ],
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
