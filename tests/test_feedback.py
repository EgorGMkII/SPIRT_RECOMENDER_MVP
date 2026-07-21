from sommelier.agent.contracts import FeedbackResult
from sommelier.agent.feedback import FEEDBACK_PROMPT, classify_feedback


class RecordingFeedbackLLM:
    def __init__(self, result: FeedbackResult) -> None:
        self.result = result
        self.prompts: list[str] = []

    def with_structured_output(self, schema, method=None):
        assert schema is FeedbackResult
        assert method == "function_calling"
        return self

    def invoke(self, prompt: str) -> FeedbackResult:
        self.prompts.append(prompt)
        return self.result


def test_follow_up_classifier_receives_previous_answer_and_request() -> None:
    llm = RecordingFeedbackLLM(
        FeedbackResult(feedback="negative_feedback")
    )

    result = classify_feedback(
        user_request="Я не просил советовать",
        previous_assistant_answer="Попробуйте этот ром.",
        follow_up=True,
        llm=llm,
    )

    assert result.feedback == "negative_feedback"
    assert "Попробуйте этот ром." in llm.prompts[0]
    assert "Я не просил советовать" in llm.prompts[0]
    assert '"follow_up": true' in llm.prompts[0]


def test_new_request_does_not_send_previous_answer() -> None:
    llm = RecordingFeedbackLLM(FeedbackResult(feedback="neutral"))

    classify_feedback(
        user_request="Мне не нравится сладкий ром",
        previous_assistant_answer="SECRET PREVIOUS ANSWER",
        follow_up=False,
        llm=llm,
    )

    assert "SECRET PREVIOUS ANSWER" not in llm.prompts[0]
    assert '"previous_assistant_answer": null' in llm.prompts[0]


def test_prompt_defines_requested_labels_and_priority() -> None:
    assert "negative_feedback > purchase_intent > neutral" in FEEDBACK_PROMPT
    assert '"Хочу купить этот ром" -> purchase_intent' in FEEDBACK_PROMPT
    assert '"Ответ не тот, я не просил советовать" -> negative_feedback' in FEEDBACK_PROMPT
    assert '"Мне не нравится сладкий ром" -> neutral' in FEEDBACK_PROMPT
    assert '"Этот ром невкусный" -> neutral' in FEEDBACK_PROMPT


def test_invalid_structured_feedback_falls_back_to_neutral() -> None:
    llm = RecordingFeedbackLLM(None)

    result = classify_feedback(
        user_request="Давай сладкие варианты.",
        previous_assistant_answer="Вот варианты.",
        follow_up=True,
        llm=llm,
    )

    assert result.feedback == "neutral"


def test_invalid_structured_feedback_fallback_keeps_priority() -> None:
    llm = RecordingFeedbackLLM(None)

    result = classify_feedback(
        user_request="Ответ не тот, но хочу купить этот ром.",
        previous_assistant_answer="Я посоветовал другой ром.",
        follow_up=True,
        llm=llm,
    )

    assert result.feedback == "negative_feedback"
