"""CLI for parsing extracted Bacardi pages into structured ProductCard JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from sommelier.ingestion.llm_product_parser import parse_directory


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description="Parse Bacardi product pages with an LLM.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/parsed_products"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/catalog/products"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    """Run the parser CLI."""

    args = build_parser().parse_args()
    summary = parse_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
