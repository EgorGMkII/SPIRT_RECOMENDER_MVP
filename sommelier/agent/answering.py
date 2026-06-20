"""Final answer context construction and generation."""

from __future__ import annotations

import json

from sommelier.agent.node_utils import trace
from sommelier.agent.prompts import COCKTAIL_RESPONSE_PROMPT, SOMMELIER_RESPONSE_PROMPT
from sommelier.agent.state import AgentState


def _message_content(message) -> str:
    """Extract text content from a LangChain-style LLM response."""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(message, str):
        return message
    return str(message)


def build_answer_context(state: AgentState) -> dict:
    """Build a compact evidence payload for the final product answer LLM."""

    candidates = []
    for result in state.retrieval_results:
        profile = result.profile
        candidates.append(
            {
                "product_id": result.product_id,
                "name": profile.name,
                "brand": profile.brand,
                "category": profile.category,
                "display_description": profile.display_description,
                "tasting_summary": profile.tasting_summary,
                "flavor_tags": profile.flavor_tags,
                "usage_tags": profile.usage_tags,
                "cocktail_names": profile.cocktail_names[:8],
                "evidence_fields": profile.evidence_fields,
                "score": round(result.score, 4),
                "retrieval_sources": state.retrieval_sources.get(result.product_id, []),
            }
        )
    return {
        "user_message": state.user_message,
        "effective_user_message": state.effective_user_message,
        "is_followup": state.is_followup,
        "intent": state.parsed_intent.intent if state.parsed_intent else None,
        "direct_or_expanded_query": state.expanded_query or state.search_query,
        "food_pairing_caveat": state.retrieval_caveat,
        "user_profile": state.user_profile.model_dump(mode="json") if state.user_profile else None,
        "candidates": candidates,
    }


def build_cocktail_answer_context(state: AgentState) -> dict:
    """Build a compact evidence payload for cocktail answer generation."""

    candidates = []
    for result in state.cocktail_results:
        profile = result.profile
        candidates.append(
            {
                "cocktail_id": result.cocktail_id,
                "name": profile.name,
                "main_rum": profile.main_rum,
                "description": profile.description,
                "ingredients": profile.ingredients,
                "recipe_steps": profile.recipe_steps,
                "matched_tokens": result.matched_tokens,
                "score": round(result.score, 4),
            }
        )
    previous_turn = None
    if state.session_memory and state.session_memory.last_turn:
        last_turn = state.session_memory.last_turn
        previous_turn = {
            "user_message": last_turn.user_message,
            "effective_user_message": last_turn.effective_user_message,
            "intent": last_turn.intent,
            "final_answer": last_turn.final_answer,
            "candidates": [
                candidate.model_dump(mode="json")
                for candidate in last_turn.candidates
            ],
        }
    return {
        "user_message": state.user_message,
        "effective_user_message": state.effective_user_message,
        "is_followup": state.is_followup,
        "intent": state.parsed_intent.intent if state.parsed_intent else None,
        "previous_turn": previous_turn,
        "avoid_previous_candidates": state.avoid_previous_candidates,
        "avoid_candidate_ids": state.avoid_candidate_ids,
        "avoid_candidate_names": state.avoid_candidate_names,
        "normalized_cocktail_query": state.cocktail_query or state.search_query,
        "user_profile": state.user_profile.model_dump(mode="json") if state.user_profile else None,
        "cocktail_candidates": candidates,
    }


def build_sommelier_answer_prompt(state: AgentState) -> str:
    """Build the final product response prompt from validated context."""

    context_json = json.dumps(
        build_answer_context(state),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    return SOMMELIER_RESPONSE_PROMPT.format(context_json=context_json)


def build_cocktail_answer_prompt(state: AgentState) -> str:
    """Build the final cocktail response prompt from validated context."""

    context_json = json.dumps(
        build_cocktail_answer_context(state),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    return COCKTAIL_RESPONSE_PROMPT.format(context_json=context_json)


def generate_sommelier_answer(state: AgentState, llm=None) -> str:
    """Generate a sommelier-style answer from validated retrieval evidence."""

    active_llm = llm
    if active_llm is None:
        from llm_module import get_langchain_openai_chat_model

        active_llm = get_langchain_openai_chat_model()
    return _message_content(active_llm.invoke(build_sommelier_answer_prompt(state))).strip()


def generate_cocktail_answer(state: AgentState, llm=None) -> str:
    """Generate a cocktail answer from validated cocktail evidence."""

    active_llm = llm
    if active_llm is None:
        from llm_module import get_langchain_openai_chat_model

        active_llm = get_langchain_openai_chat_model()
    return _message_content(active_llm.invoke(build_cocktail_answer_prompt(state))).strip()


def _deterministic_answer(state: AgentState) -> str:
    """Generate a deterministic product answer when LLM writing is disabled."""

    lines = ["\u042f \u0431\u044b \u043d\u0430\u0447\u0430\u043b \u0441 \u044d\u0442\u0438\u0445 \u0432\u0430\u0440\u0438\u0430\u043d\u0442\u043e\u0432:"]
    for index, result in enumerate(state.retrieval_results[:2], start=1):
        profile = result.profile
        description = profile.display_description or profile.searchable_text
        lines.append(
            f"{index}. {profile.name} - {description} "
            f"(score={result.score:.3f})"
        )

    if state.expanded_query:
        lines.append(f"\u041f\u043e\u0438\u0441\u043a\u043e\u0432\u0430\u044f \u0444\u043e\u0440\u043c\u0443\u043b\u0438\u0440\u043e\u0432\u043a\u0430: {state.expanded_query}")
    elif state.retrieval_results:
        lines.append(f"\u041d\u043e\u0440\u043c\u0430\u043b\u0438\u0437\u043e\u0432\u0430\u043d\u043d\u044b\u0439 \u0437\u0430\u043f\u0440\u043e\u0441: {state.retrieval_results[0].normalized_query}")

    if state.retrieval_caveat:
        lines.append(f"\u0412\u0430\u0436\u043d\u043e: {state.retrieval_caveat}")

    return "\n".join(lines)


def _deterministic_cocktail_answer(state: AgentState) -> str:
    """Generate a deterministic cocktail answer when LLM writing is disabled."""

    lines = ["\u042f \u0431\u044b \u043d\u0430\u0447\u0430\u043b \u0441 \u044d\u0442\u0438\u0445 \u043a\u043e\u043a\u0442\u0435\u0439\u043b\u0435\u0439:"]
    for index, result in enumerate(state.cocktail_results[:2], start=1):
        profile = result.profile
        lines.append(f"{index}. {profile.name} - {profile.description} (score={result.score:.3f})")
        lines.append(f"   \u0420\u043e\u043c: {profile.main_rum}")
        if profile.ingredients:
            lines.append("   \u0418\u043d\u0433\u0440\u0435\u0434\u0438\u0435\u043d\u0442\u044b: " + "; ".join(profile.ingredients))
        if profile.recipe_steps:
            lines.append("   \u0428\u0430\u0433\u0438: " + " / ".join(profile.recipe_steps))
    if state.cocktail_query:
        lines.append(f"\u041d\u043e\u0440\u043c\u0430\u043b\u0438\u0437\u043e\u0432\u0430\u043d\u043d\u044b\u0439 \u043a\u043e\u043a\u0442\u0435\u0439\u043b\u044c\u043d\u044b\u0439 \u0437\u0430\u043f\u0440\u043e\u0441: {state.cocktail_query}")
    return "\n".join(lines)


def answer_generation_node(state: AgentState) -> AgentState:
    """Generate the final MVP answer from retrieved candidates."""

    if state.errors:
        state.final_answer = (
            "\u041d\u0435 \u0441\u043c\u043e\u0433 \u0432\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c \u043f\u043e\u0438\u0441\u043a \u043f\u043e \u0442\u0435\u043a\u0443\u0449\u0435\u043c\u0443 \u0438\u043d\u0434\u0435\u043a\u0441\u0443. "
            "\u041f\u0440\u043e\u0432\u0435\u0440\u044c, \u0447\u0442\u043e ProductSearchProfile \u0438 FAISS index \u0443\u0436\u0435 \u0441\u043e\u0431\u0440\u0430\u043d\u044b. "
            f"\u0422\u0435\u0445\u043d\u0438\u0447\u0435\u0441\u043a\u0430\u044f \u0434\u0435\u0442\u0430\u043b\u044c: {state.errors[-1]}"
        )
        trace(state, "answer_generation", {}, "error answer generated", status="error")
        return state

    if not state.retrieval_results and not state.cocktail_results:
        state.final_answer = "\u041d\u0435 \u043d\u0430\u0448\u0451\u043b \u043f\u043e\u0434\u0445\u043e\u0434\u044f\u0449\u0438\u0445 \u043a\u0430\u043d\u0434\u0438\u0434\u0430\u0442\u043e\u0432 \u0432 \u0442\u0435\u043a\u0443\u0449\u0435\u043c \u043a\u0430\u0442\u0430\u043b\u043e\u0433\u0435."
        trace(state, "answer_generation", {}, "empty answer generated")
        return state

    if state.cocktail_results:
        if state.use_llm_answer:
            try:
                state.final_answer = generate_cocktail_answer(state)
                trace(
                    state,
                    "answer_generation",
                    {
                        "candidate_count": len(state.cocktail_results),
                        "mode": "llm",
                        "answer_type": "cocktail",
                    },
                    "cocktail answer generated",
                )
                return state
            except Exception as exc:
                state.errors.append(f"llm cocktail answer generation failed: {exc}")
                trace(
                    state,
                    "answer_generation",
                    {
                        "candidate_count": len(state.cocktail_results),
                        "mode": "llm",
                        "answer_type": "cocktail",
                    },
                    f"llm cocktail answer failed, used deterministic fallback: {exc}",
                    status="error",
                )
        state.final_answer = _deterministic_cocktail_answer(state)
        trace(
            state,
            "answer_generation",
            {
                "candidate_count": len(state.cocktail_results),
                "mode": "deterministic",
                "answer_type": "cocktail",
            },
            "cocktail answer generated",
        )
        return state

    if state.use_llm_answer:
        try:
            state.final_answer = generate_sommelier_answer(state)
            trace(
                state,
                "answer_generation",
                {"candidate_count": len(state.retrieval_results), "mode": "llm"},
                "sommelier answer generated",
            )
            return state
        except Exception as exc:
            state.errors.append(f"llm answer generation failed: {exc}")
            trace(
                state,
                "answer_generation",
                {"candidate_count": len(state.retrieval_results), "mode": "llm"},
                f"llm answer failed, used deterministic fallback: {exc}",
                status="error",
            )

    state.final_answer = _deterministic_answer(state)
    trace(
        state,
        "answer_generation",
        {"candidate_count": len(state.retrieval_results), "mode": "deterministic"},
        "answer generated",
    )
    return state
