"""Product search tool interface."""

from sommelier.agent.schemas import ToolResult
from sommelier.retrieval.schemas import SearchRequest


def search_products(request: SearchRequest) -> ToolResult:
    """Search products using the future normalized-query vector retrieval path.

    TODO: Wire to ProductSearchProfile loading and FaissIndex.search.
    """

    return ToolResult(tool_name="search_products", summary=f"No catalog loaded yet for query: {request.query}")
