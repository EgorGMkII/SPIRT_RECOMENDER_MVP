"""CLI for building a FAISS-compatible product vector index."""

from __future__ import annotations

import argparse
from pathlib import Path

from sommelier.retrieval.faiss_index import OpenAIEmbeddingProvider, build_index_from_profiles


def build_parser() -> argparse.ArgumentParser:
    """Build command line parser."""

    parser = argparse.ArgumentParser(description="Build product vector index.")
    parser.add_argument("--profiles-dir", type=Path, default=Path("data/catalog/search_profiles"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/indexes"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--use-openai-embeddings",
        action="store_true",
        help="Build the index with real embeddings from llm_module.py.",
    )
    return parser


def main() -> None:
    """Run index build command."""

    args = build_parser().parse_args()
    metadata_file = args.index_dir / "product_faiss_metadata.json"
    if metadata_file.exists() and not args.force:
        raise SystemExit(f"Index already exists at {metadata_file}. Use --force to overwrite.")
    embedding_provider = OpenAIEmbeddingProvider() if args.use_openai_embeddings else None
    index = build_index_from_profiles(args.profiles_dir, embedding_provider=embedding_provider)
    index.save(args.index_dir)
    print(f"Indexed {len(index.profiles)} ProductSearchProfile file(s).")
    print(
        "Embeddings:",
        "OpenAIEmbeddingProvider" if args.use_openai_embeddings else "FakeEmbeddingProvider",
    )
    print(f"Metadata: {metadata_file}")
    faiss_file = args.index_dir / "product.faiss"
    if faiss_file.exists():
        print(f"FAISS index: {faiss_file}")
    else:
        print("FAISS package not available; saved deterministic fallback vectors only.")


if __name__ == "__main__":
    main()
