"""Durable user preference profile models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class UserProfile(BaseModel):
    """Simple durable user preference profile."""

    session_id: str
    liked_flavors: list[str] = Field(default_factory=list)
    disliked_flavors: list[str] = Field(default_factory=list)
    liked_cocktails: list[str] = Field(default_factory=list)
    disliked_cocktails: list[str] = Field(default_factory=list)


class PreferencePatch(BaseModel):
    """Add/remove operations for one preference list."""

    model_config = ConfigDict(extra="forbid")
    add: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)


class ProfilePatch(BaseModel):
    """Validated patch proposed by the resolver."""

    model_config = ConfigDict(extra="forbid")
    liked_flavors: PreferencePatch = Field(default_factory=PreferencePatch)
    disliked_flavors: PreferencePatch = Field(default_factory=PreferencePatch)
    liked_cocktails: PreferencePatch = Field(default_factory=PreferencePatch)
    disliked_cocktails: PreferencePatch = Field(default_factory=PreferencePatch)


def _normalized_set(values: list[str]) -> set[str]:
    """Normalize preference values for deterministic patch operations."""

    return {" ".join(value.lower().split()) for value in values if value.strip()}


def _apply_preference_patch(existing: list[str], patch: PreferencePatch) -> list[str]:
    """Apply add/remove operations to one preference list."""

    values = _normalized_set(existing)
    values.difference_update(_normalized_set(patch.remove))
    values.update(_normalized_set(patch.add))
    return sorted(values)


def apply_profile_patch(profile: UserProfile, patch: ProfilePatch) -> UserProfile:
    """Apply a controlled patch and remove direct like/dislike conflicts."""

    profile.liked_flavors = _apply_preference_patch(
        profile.liked_flavors,
        patch.liked_flavors,
    )
    profile.disliked_flavors = _apply_preference_patch(
        profile.disliked_flavors,
        patch.disliked_flavors,
    )
    profile.liked_cocktails = _apply_preference_patch(
        profile.liked_cocktails,
        patch.liked_cocktails,
    )
    profile.disliked_cocktails = _apply_preference_patch(
        profile.disliked_cocktails,
        patch.disliked_cocktails,
    )

    liked_flavors = set(profile.liked_flavors)
    disliked_flavors = set(profile.disliked_flavors)
    for value in _normalized_set(patch.liked_flavors.add):
        disliked_flavors.discard(value)
    for value in _normalized_set(patch.disliked_flavors.add):
        liked_flavors.discard(value)

    liked_cocktails = set(profile.liked_cocktails)
    disliked_cocktails = set(profile.disliked_cocktails)
    for value in _normalized_set(patch.liked_cocktails.add):
        disliked_cocktails.discard(value)
    for value in _normalized_set(patch.disliked_cocktails.add):
        liked_cocktails.discard(value)

    profile.liked_flavors = sorted(liked_flavors)
    profile.disliked_flavors = sorted(disliked_flavors)
    profile.liked_cocktails = sorted(liked_cocktails)
    profile.disliked_cocktails = sorted(disliked_cocktails)
    return profile
    model_config = ConfigDict(extra="forbid")

    model_config = ConfigDict(extra="forbid")
