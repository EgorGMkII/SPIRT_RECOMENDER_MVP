"""Runtime product retrieval helpers used by the agent graph."""

from __future__ import annotations

from pathlib import Path

from sommelier.retrieval.bm25_index import Bm25Index
from sommelier.retrieval.faiss_index import FaissIndex, OpenAIEmbeddingProvider
from sommelier.retrieval.query_normalizer import normalize_query

DEFAULT_INDEX_DIR = Path("data/indexes")
DEFAULT_PROFILES_DIR = Path("data/catalog/search_profiles")
DEFAULT_FAISS_TOP_K = 2
DEFAULT_BM25_TOP_K = 2
DEFAULT_COCKTAIL_TOP_K = 5


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
) -> tuple[list, dict[str, list[str]], dict]:
    """Return FAISS and BM25 candidates with duplicate products merged."""

    normalized_query = (
        normalize_query(query, use_llm=use_llm_query)
        if normalize
        else query
    )
    debug: dict = {"normalized_query": normalized_query}
    try:
        faiss_results = load_agent_index().search(
            normalized_query,
            top_k=DEFAULT_FAISS_TOP_K,
            normalize=False,
        )
        debug["faiss_error"] = None
    except Exception as exc:
        faiss_results = []
        debug["faiss_error"] = str(exc)

    try:
        bm25_results = load_agent_bm25_index().search(
            normalized_query,
            top_k=DEFAULT_BM25_TOP_K,
            normalize=False,
        )
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

    return merged, sources, debug
