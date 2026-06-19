"""CLI for building ProductSearchProfile JSON files from ProductCards."""

from __future__ import annotations

import argparse
from pathlib import Path

from sommelier.catalog.search_profiles import build_search_profiles


def build_parser() -> argparse.ArgumentParser:
    """Build command line parser."""

    parser = argparse.ArgumentParser(description="Build semantic product search profiles.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/catalog/products"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/catalog/search_profiles"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--use-llm-searchable-text",
        action="store_true",
        help="Generate ProductSearchProfile.searchable_text with the configured LLM.",
    )
    return parser


def main() -> None:
    """Run profile build command."""

    args = build_parser().parse_args()
    outputs = build_search_profiles(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        force=args.force,
        limit=args.limit,
        use_llm_searchable_text=args.use_llm_searchable_text,
    )
    print(f"Built {len(outputs)} ProductSearchProfile file(s):")
    for path in outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()
