"""Durable user preference profile models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Simple durable user preference profile."""

    session_id: str
    liked_flavors: list[str] = Field(default_factory=list)
    disliked_flavors: list[str] = Field(default_factory=list)
    liked_cocktails: list[str] = Field(default_factory=list)
    disliked_cocktails: list[str] = Field(default_factory=list)


class ProfileUpdate(BaseModel):
    """Validated profile update extracted from a user message."""

    liked_flavors: list[str] = Field(default_factory=list)
    disliked_flavors: list[str] = Field(default_factory=list)
    liked_cocktails: list[str] = Field(default_factory=list)
    disliked_cocktails: list[str] = Field(default_factory=list)
    ignored: list[str] = Field(default_factory=list)


def _merge_unique(existing: list[str], new_values: list[str]) -> list[str]:
    """Merge lowercase unique preference values while preserving stable order."""

    values = [" ".join(value.lower().split()) for value in existing + new_values if value]
    return sorted({value for value in values if value})


def apply_profile_update(profile: UserProfile, update: ProfileUpdate) -> UserProfile:
    """Apply a deterministic profile update."""

    profile.liked_flavors = _merge_unique(profile.liked_flavors, update.liked_flavors)
    profile.disliked_flavors = _merge_unique(profile.disliked_flavors, update.disliked_flavors)
    profile.liked_cocktails = _merge_unique(profile.liked_cocktails, update.liked_cocktails)
    profile.disliked_cocktails = _merge_unique(
        profile.disliked_cocktails,
        update.disliked_cocktails,
    )
    return profile
