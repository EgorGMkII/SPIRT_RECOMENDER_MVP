"""Food-for-rum tool interface."""

from pydantic import BaseModel
from sommelier.agent.schemas import ToolResult


class FoodForRumInput(BaseModel):
    """Input for food suggestions for a known rum."""

    product_id: str


def food_for_rum(payload: FoodForRumInput) -> ToolResult:
    """Return food suggestions for a rum product."""

    return ToolResult(tool_name="food_for_rum", summary="Food-for-rum tool stub.")
