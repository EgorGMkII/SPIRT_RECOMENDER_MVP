import shutil
from pathlib import Path
from uuid import uuid4

from sommelier.agent.profile import ProfileUpdate, UserProfile, apply_profile_update
from sommelier.agent.profile_store import load_user_profile, save_user_profile
from sommelier.agent.tools.profile_update import extract_profile_update, has_profile_signal


def test_profile_update_is_deterministic() -> None:
    profile = UserProfile(session_id="s1", liked_flavors=["vanilla"])
    update = ProfileUpdate(
        liked_flavors=["oak", "vanilla"],
        disliked_flavors=["coconut"],
        liked_cocktails=["Mojito"],
    )

    updated = apply_profile_update(profile, update)

    assert updated.liked_flavors == ["oak", "vanilla"]
    assert updated.disliked_flavors == ["coconut"]
    assert updated.liked_cocktails == ["mojito"]


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def invoke(self, prompt: str) -> FakeMessage:
        assert "люблю ваниль" in prompt
        return FakeMessage(
            '{"liked_flavors":["vanilla"],"disliked_flavors":["coconut"],'
            '"liked_cocktails":[],"disliked_cocktails":[],"ignored":[]}'
        )


def test_extract_profile_update_uses_llm_for_explicit_preferences() -> None:
    update = extract_profile_update(
        "люблю ваниль, не люблю кокос",
        llm=FakeLLM(),
        use_llm=True,
    )

    assert update.liked_flavors == ["vanilla"]
    assert update.disliked_flavors == ["coconut"]


class FakeDislikeSweetLLM:
    def invoke(self, prompt: str) -> FakeMessage:
        assert "не нравится сладкое" in prompt
        return FakeMessage(
            '{"liked_flavors":[],"disliked_flavors":["sweet"],'
            '"liked_cocktails":[],"disliked_cocktails":[],"ignored":[]}'
        )


def test_extract_profile_update_handles_russian_dislike_signal() -> None:
    update = extract_profile_update(
        "нет, мне не нравится сладкое",
        llm=FakeDislikeSweetLLM(),
        use_llm=True,
    )

    assert not has_profile_signal("нет, мне не нравится сладкое")
    assert update.disliked_flavors == ["sweet"]


def test_extract_profile_update_ignores_plain_requests_without_llm() -> None:
    update = extract_profile_update("дай рецепт мохито")

    assert update.liked_flavors == []
    assert update.ignored == ["дай рецепт мохито"]
    assert not has_profile_signal("дай рецепт мохито")


def test_profile_store_roundtrip() -> None:
    profile_dir = Path(".test_tmp") / f"profile-store-{uuid4().hex}"
    profile = UserProfile(session_id="Session 1", liked_flavors=["vanilla"])

    try:
        save_user_profile(profile, profile_dir=profile_dir)
        loaded = load_user_profile("Session 1", profile_dir=profile_dir)

        assert loaded == profile
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)
