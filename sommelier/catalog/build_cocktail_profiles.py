"""CLI for building compact cocktail search profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

from sommelier.catalog.cocktail_profiles import build_cocktail_profiles


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description="Build CocktailSearchProfile JSON files.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/catalog/cocktails"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/catalog/cocktail_search_profiles"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    """Run the cocktail profile builder CLI."""

    args = build_parser().parse_args()
    files = build_cocktail_profiles(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        force=args.force,
        limit=args.limit,
    )
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
