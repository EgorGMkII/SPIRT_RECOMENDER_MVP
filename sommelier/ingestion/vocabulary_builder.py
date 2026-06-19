"""Controlled vocabulary generation from candidate tags."""

from sommelier.catalog.schemas import Tag, TagSpace
from sommelier.catalog.tag_vocabulary import allowed_values


def approve_candidate_tags(candidates: dict[str, set[str]]) -> list[Tag]:
    """Map candidate descriptors into the controlled vocabulary.

    Unknown values are intentionally ignored until a human or validation process
    approves them.
    """

    approved: list[Tag] = []
    for value in candidates.get("product", set()):
        if value in allowed_values(TagSpace.PRODUCT):
            approved.append(Tag(value=value, label=value.replace("_", " ").title(), space=TagSpace.PRODUCT))
    for value in candidates.get("food", set()):
        if value in allowed_values(TagSpace.FOOD):
            approved.append(Tag(value=value, label=value.replace("_", " ").title(), space=TagSpace.FOOD))
    return approved
