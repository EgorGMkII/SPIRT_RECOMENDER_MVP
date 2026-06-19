"""Build final catalog entries from parsed products."""

from sommelier.catalog.product_profiles import build_product_profile
from sommelier.catalog.schemas import CatalogEntry, ProductCard


def build_catalog_entries(products: list[ProductCard]) -> list[CatalogEntry]:
    """Transform validated product cards into catalog entries."""

    return [
        CatalogEntry(card=product, profile=build_product_profile(product))
        for product in products
    ]
