"""Tool tracing for agent observability."""

from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ToolTrace(BaseModel):
    """Single tool execution trace."""

    tool_name: str
    input: dict = Field(default_factory=dict)
    output_summary: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "success"


class ToolTracer:
    """Collect tool traces during an agent turn."""

    def __init__(self) -> None:
        self.traces: list[ToolTrace] = []

    def record(
        self,
        tool_name: str,
        tool_input: dict,
        output_summary: str,
        status: str = "success",
    ) -> ToolTrace:
        """Record and return a trace event."""

        trace = ToolTrace(
            tool_name=tool_name,
            input=tool_input,
            output_summary=output_summary,
            status=status,
        )
        self.traces.append(trace)
        return trace
