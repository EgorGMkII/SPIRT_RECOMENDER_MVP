"""Catalog access layer."""

from pathlib import Path
from sommelier.catalog.product_profiles import build_product_profile
from sommelier.catalog.schemas import CatalogEntry, ProductCard
from sommelier.catalog.storage import load_models


class ProductRepository:
    """Read product catalog data for retrieval services."""

    def __init__(self, catalog_path: Path) -> None:
        self.catalog_path = catalog_path

    def list_products(self) -> list[ProductCard]:
        """Return all product cards from storage."""

        return load_models(self.catalog_path, ProductCard)

    def list_entries(self) -> list[CatalogEntry]:
        """Return catalog entries with generated searchable profiles."""

        return [
            CatalogEntry(card=card, profile=build_product_profile(card))
            for card in self.list_products()
        ]
