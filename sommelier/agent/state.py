"""LangGraph state models."""

from uuid import uuid4

from pydantic import BaseModel, Field
from sommelier.agent.memory import SessionMemory
from sommelier.agent.profile import UserProfile
from sommelier.agent.schemas import IntentType, ParsedIntent
from sommelier.agent.tracer import ToolTrace
from sommelier.retrieval.cocktail_search import CocktailSearchResult
from sommelier.retrieval.faiss_index import SearchResult


class AgentState(BaseModel):
    """State passed through the controlled recommendation workflow."""

    session_id: str
    turn_id: str = Field(default_factory=lambda: uuid4().hex)
    user_message: str
    effective_user_message: str | None = None
    is_followup: bool = False
    followup_intent: IntentType | None = None
    followup_reason: str | None = None
    avoid_previous_candidates: bool = False
    persist_artifacts: bool = True
    session_memory: SessionMemory | None = None
    parsed_intent: ParsedIntent | None = None
    user_profile: UserProfile | None = None
    use_llm_followup: bool = False
    use_llm_intent: bool = False
    use_llm_query: bool = False
    use_llm_food_query: bool = False
    use_llm_profile_update: bool = False
    tool_traces: list[ToolTrace] = Field(default_factory=list)
    search_query: str | None = None
    expanded_query: str | None = None
    retrieval_results: list[SearchResult] = Field(default_factory=list)
    retrieval_sources: dict[str, list[str]] = Field(default_factory=dict)
    retrieval_caveat: str | None = None
    cocktail_query: str | None = None
    cocktail_results: list[CocktailSearchResult] = Field(default_factory=list)
    avoid_candidate_ids: list[str] = Field(default_factory=list)
    avoid_candidate_names: list[str] = Field(default_factory=list)
    use_llm_answer: bool = False
    profile_updated: bool = False
    errors: list[str] = Field(default_factory=list)
    final_answer: str | None = None
