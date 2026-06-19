"""Cocktail expansion tool interface."""

from pydantic import BaseModel
from sommelier.agent.schemas import ToolResult


class CocktailExpansionInput(BaseModel):
    """Input for cocktail suggestions."""

    product_id: str | None = None
    cocktail_style: str | None = None


def cocktail_expansion(payload: CocktailExpansionInput) -> ToolResult:
    """Return cocktail ideas for a rum or style."""

    return ToolResult(tool_name="cocktail_expansion", summary="Cocktail expansion tool stub.")
