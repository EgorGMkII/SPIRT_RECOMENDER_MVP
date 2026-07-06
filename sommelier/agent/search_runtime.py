"""Runtime product retrieval helpers used by the agent graph."""

from __future__ import annotations

from pathlib import Path

from sommelier.retrieval.bm25_index import Bm25Index
from sommelier.retrieval.faiss_index import FaissIndex, OpenAIEmbeddingProvider
from sommelier.retrieval.query_normalizer import normalize_query

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
DEFAULT_PROFILES_DIR = PROJECT_ROOT / "data" / "catalog" / "search_profiles"
DEFAULT_FAISS_TOP_K = 2
DEFAULT_BM25_TOP_K = 2


def load_agent_index(index_dir: Path = DEFAULT_INDEX_DIR) -> FaissIndex:
    """Load the product vector index used by the MVP agent."""

    return FaissIndex.load(index_dir, embedding_provider=OpenAIEmbeddingProvider())


def load_agent_bm25_index(profiles_dir: Path = DEFAULT_PROFILES_DIR) -> Bm25Index:
    """Load the lexical BM25 index used by the MVP agent."""

    return Bm25Index.load(profiles_dir)


def hybrid_search(
    query: str,
    normalize: bool = True,
    use_llm_query: bool = False,
    faiss_top_k: int = DEFAULT_FAISS_TOP_K,
    bm25_top_k: int = DEFAULT_BM25_TOP_K,
    excluded_product_ids: list[str] | None = None,
    limit: int | None = None,
) -> tuple[list, dict[str, list[str]], dict]:
    """Return FAISS and BM25 candidates with duplicate products merged."""

    excluded = set(excluded_product_ids or [])
    faiss_fetch_k = faiss_top_k + len(excluded)
    bm25_fetch_k = bm25_top_k + len(excluded)
    normalized_query = (
        normalize_query(query, use_llm=use_llm_query)
        if normalize
        else query
    )
    debug: dict = {"normalized_query": normalized_query}
    try:
        faiss_results = load_agent_index().search(
            normalized_query,
            top_k=faiss_fetch_k,
            normalize=False,
        )
        faiss_results = [
            result for result in faiss_results
            if result.product_id not in excluded
        ][:faiss_top_k]
        debug["faiss_error"] = None
    except Exception as exc:
        faiss_results = []
        debug["faiss_error"] = str(exc)

    try:
        bm25_results = load_agent_bm25_index().search(
            normalized_query,
            top_k=bm25_fetch_k,
            normalize=False,
        )
        bm25_results = [
            item for item in bm25_results
            if item.result.product_id not in excluded
        ][:bm25_top_k]
        debug["bm25_error"] = None
    except Exception as exc:
        bm25_results = []
        debug["bm25_error"] = str(exc)

    if not faiss_results and not bm25_results:
        errors = [
            message
            for message in (debug.get("faiss_error"), debug.get("bm25_error"))
            if message
        ]
        raise RuntimeError("; ".join(errors) or "hybrid retrieval returned no results")

    merged = []
    sources: dict[str, list[str]] = {}
    debug.update(
        {
            "faiss_top": [result.product_id for result in faiss_results],
            "bm25_top": [item.result.product_id for item in bm25_results],
            "bm25_matched_tokens": {
                item.result.product_id: item.matched_tokens for item in bm25_results
            },
        }
    )

    for result in faiss_results:
        sources.setdefault(result.product_id, []).append("faiss")
        merged.append(result)

    seen = {result.product_id for result in merged}
    for item in bm25_results:
        result = item.result
        sources.setdefault(result.product_id, []).append("bm25")
        if result.product_id not in seen:
            merged.append(result)
            seen.add(result.product_id)

    if limit is not None:
        merged = merged[:limit]
        kept_ids = {result.product_id for result in merged}
        sources = {
            product_id: product_sources
            for product_id, product_sources in sources.items()
            if product_id in kept_ids
        }
    debug["excluded_product_ids"] = sorted(excluded)
    return merged, sources, debug
