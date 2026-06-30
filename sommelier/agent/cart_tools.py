"""Typed model tools for deterministic session-cart operations."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from sommelier.agent.memory import CartItem


class CartToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AddCartInput(CartToolInput):
    id: str = Field(min_length=1, max_length=300)
    amount: int = Field(default=1, ge=1, le=99)


class DeleteCartInput(CartToolInput):
    id: str = Field(min_length=1, max_length=300)


class ShowCartInput(CartToolInput):
    pass


class CartOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    items: list[CartItem] = Field(default_factory=list)


def _model_only_tool(**_: object) -> dict:
    """The graph executor owns cart state and returns the real output."""

    raise RuntimeError("cart tools must be executed by the graph executor")


add_cart = StructuredTool.from_function(
    func=_model_only_tool,
    name="add_cart",
    description=(
        "Add a Bacardi product to the session cart. Use an exact product id "
        "from a current search result or a product previously shown to the user. "
        "Adding an existing id increases its amount."
    ),
    args_schema=AddCartInput,
)
dellete_cart = StructuredTool.from_function(
    func=_model_only_tool,
    name="dellete_cart",
    description=(
        "Remove the complete product position with the exact id from the "
        "session cart."
    ),
    args_schema=DeleteCartInput,
)
show_cart = StructuredTool.from_function(
    func=_model_only_tool,
    name="show_cart",
    description="Return all current session cart items with id and amount.",
    args_schema=ShowCartInput,
)

CART_TOOLS = [add_cart, dellete_cart, show_cart]
CART_TOOL_MAP = {tool.name: tool for tool in CART_TOOLS}
