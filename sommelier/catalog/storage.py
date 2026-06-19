"""Persistence helpers for catalog artifacts."""

import json
from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel, TypeAdapter

T = TypeVar("T", bound=BaseModel)


def load_models(path: Path, model: type[T]) -> list[T]:
    """Load a list of Pydantic models from JSON."""

    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return TypeAdapter(list[model]).validate_python(data)


def save_models(path: Path, models: list[BaseModel]) -> None:
    """Save a list of Pydantic models to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [model.model_dump(mode="json") for model in models]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
