from sommelier.agent.nlu import build_intent_parse_prompt, parse_intent
from sommelier.agent.routing import route_intent


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeIntentLLM:
    def invoke(self, prompt: str) -> FakeMessage:
        assert "grilled food" in prompt
        return FakeMessage(
            '{"intent":"food_pairing","query":"grilled food dinner pairing",'
            '"confidence":0.91}'
        )


def test_route_food_pairing_intent_from_llm_parser() -> None:
    intent = parse_intent(
        "What rum should I pair with grilled food?",
        llm=FakeIntentLLM(),
        use_llm=True,
    )

    assert route_intent(intent) == "food_pairing"


def test_intent_prompt_prioritizes_recommendations_over_profile_update() -> None:
    prompt = build_intent_parse_prompt(
        "тогда посоветуй ром для коктейлей, но без сладкого профиля"
    )

    assert "Recommendation requests win over profile_update" in prompt
    assert "ром для коктейлей без сладкого профиля" in prompt
    assert '"intent":"search_products"' in prompt
