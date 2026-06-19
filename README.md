
# AI Sommelier Assistant

Python MVP for an AI sommelier assistant focused on rum recommendations.

Current pipeline:

```text
Bacardi pages
  -> raw HTML
  -> parsed page JSON
  -> LLM ProductCard JSON
  -> ProductSearchProfile JSON
  -> normalized natural-language query
  -> FAISS-compatible vector search
  -> top-k candidate products
```

Retrieval is intentionally not tag-first. Lightweight flavor and usage tags may be stored as metadata, but the primary search path is semantic similarity between a normalized user query and `ProductSearchProfile.searchable_text`.

Food pairing is also retrieval-based. The current Bacardi source pages do not provide a reliable direct food-pairing database, so MVP food pairing expands a food description into a rum search query and returns vector-search candidates with a caveat that the pairing is inferred, not source-backed.

Useful commands:

```powershell
python -m sommelier.ingestion.crawl_bacardi --max-products 3
python -m sommelier.ingestion.parse_products_llm --limit 3 --force
python -m sommelier.catalog.build_search_profiles --force --use-llm-searchable-text
python -m sommelier.retrieval.build_faiss_index --force --use-openai-embeddings
python -B -m pytest -p no:cacheprovider
```

Web app:

```powershell
python -m uvicorn sommelier.web.app:app --host 127.0.0.1 --port 8000
```

Docker deployment on port 8012:

```bash
cp .env.example .env
# edit .env with real API credentials
docker compose up --build -d
```

Open:

```text
http://<vm-host>:8012
```

The container mounts `./data` to `/app/data`, so catalog/index files and runtime
session/profile/trace JSON files persist across restarts.
