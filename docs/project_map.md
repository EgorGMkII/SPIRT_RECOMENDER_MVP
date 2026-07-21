# Карта текущей реализации

## Agent

```text
sommelier/agent/
  cart_tools.py      typed add/delete/show cart tool contracts
  feedback.py        independent structured feedback classifier
  contracts.py       TurnResolution, FinalAnswerResult и tool outputs
  graph.py           линейный LangGraph и transactional persistence
  memory.py          SessionMemory, TurnMemory, CatalogRef, ShownResult
  profile.py         UserProfile и ProfilePatch
  resolver.py        structured turn resolver
  search_runtime.py  product hybrid retrieval adapter
  search_tools.py    четыре catalog model tools
  state.py           временный AgentState одного запроса
  tool_agent.py      bind_tools loop и bounded executor
  tracer.py          trace contracts

sommelier/storage/
  database.py            SQLite connection, pragmas и schema
  session_repository.py  memory/profile/messages/traces transactions
```

Старые intent routing, `ActiveContext`, candidate pools, ReAct loop и tool
stubs отсутствуют.

## Catalog и retrieval

```text
sommelier/catalog/
  search_profiles.py
  cocktail_profiles.py
  build_search_profiles.py
  build_cocktail_profiles.py

sommelier/retrieval/
  faiss_index.py
  bm25_index.py
  cocktail_search.py
  query_normalizer.py
  cocktail_query_normalizer.py
  food_pairing_query.py
  build_faiss_index.py
```

Catalog и index artifacts находятся в `data/catalog/` и `data/indexes/`.

## Ingestion

`sommelier/ingestion/` содержит crawler, page extraction, structured LLM
parsers и persistence raw/parsed artifacts. Этот слой не входит в runtime
диалога.

## Web

- `web/app.py` создаёт FastAPI application;
- `web/api.py` запускает `run_agent_turn`;
- `web/schemas.py` возвращает `follow_up`, `request_scope`, `answer_mode`,
  `effective_request`, answer, profile, использованные candidates и traces;
- `web/static/` и `web/templates/` содержат UI.

## SQLite persistence

`sqlite_migration.md` описывает текущую SQLite-схему и отдельную загрузку
полного текста чата при перезагрузке. Persistence не меняет AgentState, graph
contracts, session ID или LLM context.

## Ручная проверка

`notebooks/dialog_testing.ipynb` и `scripts/agent_chat_cli.py` — актуальные
стенды диалогов, profile update, follow-up, search и `lookup_by_ids`.
