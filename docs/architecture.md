# Architecture

## Overview

The project is a Python AI sommelier assistant for rum recommendations. It starts with Bacardi rum product pages and turns them into structured ProductCards, normalized semantic search profiles, and a vector index for natural-language retrieval.

The current MVP architecture is:

```text
Bacardi pages
  -> extracted page JSON
  -> LLM ProductCard JSON
  -> ProductSearchProfile JSON
  -> normalized user query
  -> embeddings
  -> FAISS-compatible vector search
  -> top-k candidate products
```

LLMs are used for structured product extraction and later may be used for explanations. Deterministic code owns validation, search-profile construction, query normalization, indexing, and retrieval orchestration.

## Ingestion Module

The ingestion module crawls and parses Bacardi rum pages from `https://www.bacardi.com/our-rums/`.

Responsibilities:

- discover product URLs;
- fetch product pages with polite HTTP settings;
- extract title, h1, clean text, metadata, source URL, and product links;
- preserve raw HTML artifacts;
- save extracted page records under `data/parsed_products/`;
- call the LLM parser only for ProductCard construction.

The crawler and page extractor are deterministic. The LLM parser must produce schema-compatible JSON and cannot bypass Pydantic validation.

## ProductCard Layer

ProductCards live under:

```text
data/catalog/products/
```

They are provenance-preserving records. They may contain:

- marketing copy;
- tasting notes;
- nose, palate, finish;
- process and serving text;
- cocktail names;
- related/recommended rums;
- FAQ items;
- source metadata;
- extraction warnings.

ProductCards are intentionally rich. They are not directly used as embedding text because they contain metadata, warnings, related products, and FAQ content that can pollute semantic retrieval.

## ProductSearchProfile Layer

ProductSearchProfiles live under:

```text
data/catalog/search_profiles/
```

`ProductSearchProfile.searchable_text` is the clean semantic representation used for embeddings. It includes only current-product fields:

- name;
- category;
- short description;
- marketing description;
- tasting notes;
- nose;
- palate;
- finish;
- process;
- how to serve;
- cocktail names.

It excludes:

- source metadata;
- FAQ items;
- recommended rums;
- extraction warnings;
- navigation, footer, legal, cookie, and social text.

Lightweight `flavor_tags` and `usage_tags` may be stored for display, debugging, analytics, or future filtering. They are secondary metadata and are not the primary retrieval mechanism.

## Query Normalization

User queries are normalized before embedding.

Example:

```text
User query:
I want a smooth rum with vanilla and oak notes that works well in cocktails.

Normalized query:
Rum with vanilla, oak flavors. Smooth and approachable profile. Suitable for cocktails and mixing drinks.
```

The MVP normalizer is deterministic and rule-based. It keeps the query concise and embedding-friendly without calling an LLM.

## FAISS-Compatible Index

The vector index stores embeddings for `ProductSearchProfile.searchable_text`.

Expected artifacts:

```text
data/indexes/product_faiss_metadata.json
data/indexes/product.faiss        # only when faiss is installed
```

If FAISS is unavailable, the project saves deterministic fallback vectors so tests and local demos can still run. This keeps the interface stable while avoiding mandatory native dependencies during early development.

## Retrieval Flow

The primary retrieval flow is:

```text
natural-language user query
  -> deterministic normalized query
  -> query embedding
  -> FAISS-compatible similarity search
  -> top-k ProductSearchProfile candidates
  -> optional LLM reranking/explanation later
```

Retrieval should rely primarily on semantic similarity between normalized query text and product profile text, not on tag matching.

## Food-Pairing Retrieval

Bacardi product pages currently do not provide a reliable direct food-pairing database. The MVP must not invent product-specific food pairing facts or claim that Bacardi recommends a specific rum for a specific food unless that appears in source data.

Food pairing is implemented as inference-based query expansion:

```text
food description
  -> deterministic food-to-rum search query expansion
  -> FAISS-compatible search over ProductSearchProfile.searchable_text
  -> top-k candidate products
  -> caveated recommendation
```

Example:

```text
Food query:
Ем шашлык из свинины, какой ром подойдёт?

Expanded query:
Rum suitable for grilled barbecue meat. Rich, smoky, spiced rum profile.
Rum with caramel, spice, oak or molasses for pork.
```

Final answers must phrase these as inferred recommendations, not as source-backed food pairing claims.

## Scoring

For the MVP, the vector similarity score is the candidate score. Later layers may add:

- LLM reranking over top-k candidates;
- explanation generation from candidate evidence;
- profile-aware boosts;
- explicit filters when the user provides hard constraints.

These additions should remain secondary to the normalized-query vector retrieval path unless a future design explicitly changes this.

## User Profile

The user profile stores durable preferences inferred from explicit user statements and confirmed interactions.

Possible fields:

- preferred flavor descriptions;
- disliked flavor descriptions;
- cocktail or sipping interests;
- avoided product characteristics;
- favorite products;
- interaction history summary.

Profile updates must be deterministic and traceable. Profile-aware retrieval is a later step.

## ToolTracer

`ToolTracer` records internal tool activity for debugging and evaluation.

Each trace event should include:

- session ID;
- turn ID;
- tool name;
- typed input;
- typed output summary;
- timestamps;
- success or error state;
- retrieval evidence where applicable.

## LangGraph Workflow

The agent should use a controlled LangGraph workflow. Full agent behavior is not part of this MVP retrieval slice.

Target future flow:

```text
receive message
  -> parse intent
  -> load profile
  -> normalize search query
  -> vector search
  -> optional rerank/explain
  -> generate final answer
  -> store trace/profile updates
```

The graph should allow deliberate branches, not unbounded autonomous tool recursion.
