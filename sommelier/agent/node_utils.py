"""Shared helpers for agent graph nodes."""

from __future__ import annotations

from sommelier.agent.state import AgentState
from sommelier.agent.tracer import ToolTrace


def trace(
    state: AgentState,
    tool_name: str,
    tool_input: dict,
    output_summary: str,
    status: str = "success",
) -> None:
    """Append a trace event to agent state."""

    state.tool_traces.append(
        ToolTrace(
            tool_name=tool_name,
            input=tool_input,
            output_summary=output_summary,
            status=status,
        )
    )
