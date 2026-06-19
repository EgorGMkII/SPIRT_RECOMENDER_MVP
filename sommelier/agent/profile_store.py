"""JSON persistence for user preference profiles."""

from __future__ import annotations

import json
import re
from pathlib import Path

from sommelier.agent.profile import UserProfile

DEFAULT_PROFILE_DIR = Path("data/user_profiles")


def profile_slug(session_id: str) -> str:
    """Return a filesystem-safe profile slug."""

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", session_id).strip("-").lower()
    return slug or "default"


def profile_path(session_id: str, profile_dir: Path = DEFAULT_PROFILE_DIR) -> Path:
    """Return the JSON path for a session profile."""

    return profile_dir / f"{profile_slug(session_id)}.json"


def load_user_profile(
    session_id: str,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
) -> UserProfile:
    """Load a user profile or create an empty one."""

    path = profile_path(session_id, profile_dir=profile_dir)
    if not path.exists():
        return UserProfile(session_id=session_id)
    return UserProfile.model_validate_json(path.read_text(encoding="utf-8"))


def save_user_profile(
    profile: UserProfile,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
) -> Path:
    """Save a user profile as JSON."""

    path = profile_path(profile.session_id, profile_dir=profile_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
