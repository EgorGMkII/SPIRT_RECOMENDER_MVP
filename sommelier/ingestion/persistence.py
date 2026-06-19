"""Persistence helpers for ingestion artifacts."""

import json
import re
from pathlib import Path
from urllib.parse import urlparse
from pydantic import BaseModel


def slugify_url(url: str) -> str:
    """Create a stable filesystem slug for a URL."""

    parsed = urlparse(url)
    raw = f"{parsed.netloc}{parsed.path}".strip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-").lower()
    return slug or "page"


def save_text(path: Path, content: str) -> Path:
    """Write text content to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def save_json(path: Path, model: BaseModel | dict | list) -> Path:
    """Write JSON content to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, BaseModel):
        payload = model.model_dump(mode="json")
    else:
        payload = model
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
