"""Food pairing tool using inference-based query expansion."""

from pathlib import Path
from pydantic import BaseModel, Field
from sommelier.agent.schemas import ToolResult
from sommelier.retrieval.food_pairing_query import search_for_food_pairing


class FoodPairingInput(BaseModel):
    """Input for food pairing recommendations."""

    food_text: str
    top_k: int = Field(default=5, ge=1, le=20)
    index_dir: Path = Path("data/indexes")


def food_pairing(payload: FoodPairingInput) -> ToolResult:
    """Return rum candidates inferred from food-query expansion."""

    result = search_for_food_pairing(
        food_text=payload.food_text,
        top_k=payload.top_k,
        index_dir=payload.index_dir,
    )
    names = [item.profile.name for item in result.retrieval_results]
    summary = (
        f"Inferred pairing candidates for '{result.original_food_text}': "
        f"{', '.join(names) if names else 'no candidates found'}. "
        f"{result.caveat}"
    )
    return ToolResult(
        tool_name="food_pairing",
        summary=summary,
        metadata=result.model_dump(mode="json"),
    )
