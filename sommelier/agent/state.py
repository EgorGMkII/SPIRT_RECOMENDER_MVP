"""Ephemeral state for one user request."""

from typing import Annotated, Any
from uuid import uuid4

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from sommelier.agent.contracts import (
    CocktailCandidate,
    FeedbackResult,
    FinalAnswerResult,
    ProductCandidate,
    TurnResolution,
)
from sommelier.agent.memory import SessionMemory
from sommelier.agent.profile import UserProfile
from sommelier.agent.tracer import ToolTrace


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    turn_id: str = Field(default_factory=lambda: uuid4().hex)
    user_request: str

    session_memory: SessionMemory | None = None
    user_profile: UserProfile | None = None
    turn_resolution: TurnResolution | None = None
    feedback_result: FeedbackResult | None = None

    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)
    tool_call_count: int = 0
    canonical_tool_calls: list[str] = Field(default_factory=list)
    cards: list[ProductCandidate | CocktailCandidate] = Field(default_factory=list)

    final_answer_result: FinalAnswerResult | None = None
    errors: list[str] = Field(default_factory=list)
    tool_traces: list[ToolTrace] = Field(default_factory=list)
