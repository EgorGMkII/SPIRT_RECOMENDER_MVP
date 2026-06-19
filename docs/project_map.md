# Project Map

## Directory Structure

Current target structure:

```text
spirt_test/
  sommelier/
    config.py
    ingestion/
    catalog/
    retrieval/
    agent/
    web/
  data/
    raw_pages/
    parsed_products/
    catalog/
      products/
      search_profiles/
    indexes/
  docs/
    project_map.md
    architecture.md
  tests/
  README.md
  AGENTS.md
  pyproject.toml
```

## Major Modules

### `sommelier/ingestion/`

Source-specific ingestion logic. The first source is Bacardi rum pages.

Important files:

- `crawler.py` for URL discovery and page downloading;
- `page_extract.py` for title, h1, metadata, clean text, and product links;
- `crawl_bacardi.py` for the crawl CLI;
- `llm_product_parser.py` for ProductCard extraction;
- `parse_products_llm.py` for ProductCard parsing CLI;
- `product_extraction_prompt.py` for the LLM extraction prompt.

### `sommelier/catalog/`

Catalog artifacts and transformations.

Important files:

- `search_profiles.py` for `ProductSearchProfile`, lightweight metadata tags, and ProductCard-to-profile conversion;
- `build_search_profiles.py` for the profile build CLI;
- `schemas.py` for earlier catalog contracts kept for compatibility.

### `sommelier/retrieval/`

Natural-language retrieval.

Important files:

- `query_normalizer.py` for deterministic query normalization;
- `food_pairing_query.py` for inference-based food query expansion;
- `faiss_index.py` for FAISS-compatible vector indexing and search;
- `build_faiss_index.py` for index build CLI.

Older tag-search/ranker modules are legacy scaffolding and should not be treated as the primary retrieval path.

### `sommelier/agent/`

Controlled LangGraph workflow and typed tools. Full final agent behavior is not implemented in this MVP retrieval slice.

### `sommelier/web/`

Minimal FastAPI layer. Web handlers should stay thin and delegate to catalog, retrieval, or agent services.

## Data Ingestion Pipeline

```text
Bacardi rum listing
  -> discover product URLs
  -> fetch product pages
  -> save raw HTML
  -> extract page records
  -> save parsed page JSON
  -> parse ProductCards with LLM
  -> validate with Pydantic
  -> save ProductCard JSON
```

Artifacts:

```text
data/raw_pages/
data/parsed_products/
data/catalog/products/
```

## Search Profile Pipeline

```text
ProductCard JSON
  -> deterministic ProductSearchProfile conversion
  -> clean searchable_text
  -> lightweight flavor/usage metadata tags
  -> save search profile JSON
```

Artifacts:

```text
data/catalog/search_profiles/
```

`searchable_text` excludes FAQ items, source metadata, recommended rums, extraction warnings, footer, menu, legal, and social text.

## Retrieval Pipeline

```text
user query
  -> normalize_query()
  -> embed normalized query
  -> FAISS-compatible vector search over ProductSearchProfile.searchable_text
  -> return top-k product candidates
```

Tags are secondary metadata for display/debugging/future filters. Retrieval should not primarily depend on tag matching.

## Food-Pairing Pipeline

There is no reliable direct Bacardi rum-to-food pairing database in the current source pages. MVP food pairing works by expanding food text into a rum search query and then using the same vector retrieval layer:

```text
food text
  -> normalize_food_pairing_query()
  -> expanded rum search query
  -> FAISS-compatible vector search
  -> candidate ProductSearchProfiles
  -> caveat: inferred, not source-backed
```

Do not add direct rum-food mappings unless explicit source data is available.

## Web API Layer

Initial endpoints:

- `POST /chat` for user messages;
- future catalog/index rebuild endpoints;
- future debug trace endpoints.

The web layer should not own recommendation logic.

## Schemas

Important schemas now live close to their layer:

- `sommelier.ingestion.llm_product_parser.ProductCard`;
- `sommelier.catalog.search_profiles.ProductSearchProfile`;
- `sommelier.retrieval.faiss_index.SearchResult`.

## Prompts

Product extraction prompt:

```text
sommelier/ingestion/product_extraction_prompt.py
```

Retrieval query normalization is deterministic and currently does not use prompts.

## Tests

Tests live in `tests/`.

Important test targets:

- page extraction and crawling fixtures;
- ProductCard parser with mocked LLM responses;
- ProductSearchProfile schema and conversion;
- exclusion of metadata/FAQ/recommended rums from `searchable_text`;
- deterministic query normalization;
- lightweight metadata tag extraction;
- FAISS-compatible vector retrieval with fake embeddings.
