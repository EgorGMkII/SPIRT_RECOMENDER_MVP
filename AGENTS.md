# AGENTS.md

## Project Goal

Build a Python AI sommelier assistant, starting with rum recommendations. The assistant should recommend Bacardi rum products, explain why they fit a user's taste or food context, suggest pairings and cocktails, and maintain a lightweight user preference profile.

The system must be reliable and auditable: LLMs may extract information and write natural-language explanations, but deterministic Python code owns validation, search-profile construction, query normalization, vector retrieval orchestration, and profile updates.

## Coding Style

- Use Python 3.11+.
- Prefer small modules with clear ownership.
- Use Pydantic models for internal data contracts and API payloads.
- Keep functions deterministic where possible.
- Use explicit types for public functions, service boundaries, and tool inputs/outputs.
- Keep side effects at module edges: crawling, file I/O, API calls, and web handlers should delegate to typed services.
- Avoid hidden global state except immutable configuration constants.
- Write tests for deterministic behavior before adding broad integrations.

## Architecture Principles

- Build in vertical slices, not all at once.
- Keep ingestion, catalog storage, retrieval, agent orchestration, and web delivery separate.
- Treat crawled text, parsed product cards, normalized catalog data, and vector indexes as different artifacts.
- Store product data in structured JSON before adding database complexity.
- Keep prompts versioned and isolated from business logic.
- Prefer controlled workflows over open-ended agent loops.
- Make every tool call traceable for debugging.

## Agent And Tool Design

- Use LangGraph for a controlled workflow.
- Do not build an uncontrolled ReAct-style agent.
- Model graph state with Pydantic schemas.
- Model every tool input and output with Pydantic schemas.
- Tools should be narrow and deterministic where possible:
  - `search_products`
  - `food_pairing`
  - `cocktail_expansion`
  - `food_for_rum`
  - `profile_update`
- Tools should not mutate state directly unless that is their explicit purpose.
- Tool outputs should include enough structured evidence for final answer generation.
- The final response generator may use LLM wording, but should rely only on validated retrieval results and tool outputs.

## LLM Usage Constraints

- LLMs may parse raw product text into candidate structured data.
- LLMs may generate concise user-facing explanations from validated facts.
- LLMs may parse user intent into a controlled schema.
- LLMs must not invent products, product facts, prices, or availability.
- LLM-generated descriptors should not become retrieval rules.
- Lightweight tags may be extracted deterministically for metadata, debugging, analytics, or future filtering.
- Primary retrieval uses normalized natural-language queries and vector search over ProductSearchProfile text.
- Food pairing uses deterministic food-query expansion plus vector search. Do not claim direct Bacardi food-pairing evidence unless explicit source data exists.
- Deterministic code must perform:
  - schema validation;
  - ProductCard to ProductSearchProfile conversion;
  - query normalization;
  - FAISS lookup orchestration;
  - profile updates;
  - catalog/index persistence.

## Tests

The expected test command is:

```powershell
pytest
```

Until a test suite exists, verify documentation-only changes with:

```powershell
git diff -- AGENTS.md docs/project_map.md docs/architecture.md
```

Future tests should prioritize:

- Pydantic schema validation;
- ProductSearchProfile conversion;
- query normalization;
- vector retrieval behavior;
- food pairing rule matching;
- deterministic ranking;
- profile update behavior;
- API route contract tests.

## Safe Modification Guidelines

- Inspect existing files before editing.
- Keep edits scoped to the requested layer.
- Do not refactor unrelated files during feature work.
- Do not overwrite raw crawled data unless the task is explicitly rebuilding the catalog.
- Preserve generated artifacts if they are needed for reproducibility.
- Do not commit secrets, API keys, raw credentials, or local `.env` files.
- Prefer adding new tests for changed deterministic behavior.
- If a file has unrelated user edits, work with them rather than reverting them.

## Required Self-Review Format

After each task, report:

1. Files changed.
2. What changed and why.
3. Verification performed.
4. Risks or open questions.
5. Suggested next step.
