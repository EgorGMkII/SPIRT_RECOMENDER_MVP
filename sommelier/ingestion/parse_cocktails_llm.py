"""CLI wrapper for parsing extracted Bacardi cocktail pages into CocktailCard JSON."""

from sommelier.ingestion.llm_cocktail_parser import main


if __name__ == "__main__":
    main()
