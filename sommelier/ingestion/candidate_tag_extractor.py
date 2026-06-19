"""First-pass candidate tag extraction from parsed products."""

from sommelier.catalog.schemas import ProductCard


def extract_candidate_tags(products: list[ProductCard]) -> dict[str, set[str]]:
    """Collect possible descriptors before controlled vocabulary approval."""

    return {
        "product": {tag for product in products for tag in product.tags.product},
        "food": {tag for product in products for tag in product.tags.food},
    }
