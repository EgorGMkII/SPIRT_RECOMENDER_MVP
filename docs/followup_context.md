# Follow-up context

Current runtime does not store a separate recent-object model.

Resolver receives:

- current user request;
- up to six latest `TurnMemory` entries;
- recent full dialogue snippets;
- `UserProfile`.

Catalog references are resolved directly from ordered
`TurnMemory.shown_results`. Full cards are not stored in memory; they are loaded
for the current turn by search tools or `lookup_by_ids`.

For the complete source of truth, see:

- `architecture.md`;
- `memory_model.md`;
- `agent_prompts.md`;
- `tool_contracts.md`.
