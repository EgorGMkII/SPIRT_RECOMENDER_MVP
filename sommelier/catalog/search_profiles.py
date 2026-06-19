"""Build normalized semantic search profiles from ProductCard JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, HttpUrl


class ProductSearchProfile(BaseModel):
    """Clean product representation optimized for embedding-based retrieval."""

    product_id: str
    source_url: HttpUrl
    brand: str
    name: str
    category: str | None = None
    short_description: str | None = None
    tasting_summary: str | None = None
    flavor_tags: list[str] = Field(default_factory=list)
    usage_tags: list[str] = Field(default_factory=list)
    cocktail_names: list[str] = Field(default_factory=list)
    searchable_text: str
    display_description: str | None = None
    evidence_fields: dict[str, str | list[str]] = Field(default_factory=dict)
    source_product_card_path: str
    warnings: list[str] = Field(default_factory=list)


FLAVOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "vanilla": ("vanilla",),
    "oak": ("oak", "oaky", "toasted oak"),
    "honey": ("honey",),
    "clove": ("clove",),
    "pepper": ("pepper",),
    "cinnamon": ("cinnamon",),
    "caramel": ("toffee", "caramel"),
    "floral": ("floral", "flower", "orange blossom", "rose", "lavender"),
    "fruity": ("fruit", "fruity"),
    "banana": ("banana",),
    "citrus": ("lime", "citrus", "orange"),
    "pineapple": ("pineapple",),
    "coconut": ("coconut",),
    "guava": ("guava",),
}

USAGE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cocktail": ("cocktail", "cocktails"),
    "mixer": ("mixing", "mixer", "mixes well", "mixed"),
    "sipping": ("sipping",),
    "neat": ("neat",),
    "highball": ("highball",),
}


def _clean_join(values: list[str | None]) -> str:
    """Join non-empty text values into a compact paragraph."""

    return " ".join(" ".join(value.split()) for value in values if value).strip()


def _unique(values: list[str]) -> list[str]:
    """Return unique values while preserving order."""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _message_content(message: Any) -> str:
    """Extract string content from an LLM response."""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(message, str):
        return message
    return str(message)


def _dedupe_sentences(text: str) -> str:
    """Remove repeated sentence-like chunks from generated text."""

    chunks = [
        chunk.strip(" \n\t.")
        for chunk in text.replace("\n", " ").split(".")
        if chunk.strip(" \n\t.")
    ]
    normalized_seen: set[str] = set()
    result: list[str] = []
    for chunk in chunks:
        normalized = " ".join(chunk.lower().split())
        if normalized not in normalized_seen:
            result.append(chunk)
            normalized_seen.add(normalized)
    return ". ".join(result).strip() + ("." if result else "")


def extract_lightweight_tags(text: str) -> tuple[list[str], list[str]]:
    """Extract lightweight metadata tags using deterministic keyword rules."""

    lowered = text.lower()
    flavor_tags = [
        tag
        for tag, aliases in FLAVOR_KEYWORDS.items()
        if any(alias in lowered for alias in aliases)
    ]
    usage_tags = [
        tag
        for tag, aliases in USAGE_KEYWORDS.items()
        if any(alias in lowered for alias in aliases)
    ]
    return flavor_tags, usage_tags


def build_deterministic_searchable_text(
    card: dict[str, Any],
    flavor_tags: list[str],
    usage_tags: list[str],
    cocktail_names: list[str],
    tasting_summary: str,
) -> str:
    """Build an organic concise retrieval text without an LLM."""

    name = card.get("name") or "Rum"
    category = card.get("category")
    description = card.get("short_description") or card.get("marketing_description")
    process = card.get("process")
    how_to_serve = card.get("how_to_serve")
    parts = [
        f"{name}.",
        f"Style: {category}." if category else None,
        f"Product profile: {description}." if description else None,
        f"Flavor profile: {tasting_summary}." if tasting_summary else None,
        f"Key flavor descriptors: {', '.join(flavor_tags)}." if flavor_tags else None,
        f"Best uses and serves: {', '.join(usage_tags)}." if usage_tags else None,
        f"Serve context: {how_to_serve}." if how_to_serve else None,
        f"Production/process notes: {process}." if process else None,
        f"Cocktail fit: {', '.join(cocktail_names[:8])}." if cocktail_names else None,
    ]
    return _dedupe_sentences(_clean_join(parts))


def build_searchable_text_prompt(
    card: dict[str, Any],
    flavor_tags: list[str],
    usage_tags: list[str],
    cocktail_names: list[str],
    tasting_summary: str,
) -> str:
    """Build prompt for LLM-generated embedding text."""

    payload = {
        "name": card.get("name"),
        "brand": card.get("brand"),
        "category": card.get("category"),
        "short_description": card.get("short_description"),
        "marketing_description": card.get("marketing_description"),
        "tasting_notes": card.get("tasting_notes"),
        "nose": card.get("nose"),
        "palate": card.get("palate"),
        "finish": card.get("finish"),
        "process": card.get("process"),
        "how_to_serve": card.get("how_to_serve"),
        "cocktail_names": cocktail_names,
        "flavor_tags": flavor_tags,
        "usage_tags": usage_tags,
        "tasting_summary": tasting_summary,
    }
    return (
        "Create one compact natural-language retrieval text for embedding search.\n"
        "The text must describe only the current rum product.\n"
        "Make it maximally informative for semantic search, but avoid repetition.\n"
        "Include product identity, style/category, flavor profile, aroma/palate/finish, "
        "texture or smoothness, serve style, cocktail suitability, and useful flavor/use tags "
        "in organic language.\n"
        "Do not include FAQ, source metadata, legal text, warnings, or recommended related rums.\n"
        "Do not invent facts. Prefer 70-130 words. Return plain text only, no JSON, no bullets.\n\n"
        f"PRODUCT DATA:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def generate_llm_searchable_text(
    card: dict[str, Any],
    flavor_tags: list[str],
    usage_tags: list[str],
    cocktail_names: list[str],
    tasting_summary: str,
    llm: Any | None = None,
) -> str:
    """Generate concise embedding text with the configured LLM."""

    active_llm = llm
    if active_llm is None:
        from llm_module import get_langchain_openai_chat_model

        active_llm = get_langchain_openai_chat_model()
    prompt = build_searchable_text_prompt(
        card=card,
        flavor_tags=flavor_tags,
        usage_tags=usage_tags,
        cocktail_names=cocktail_names,
        tasting_summary=tasting_summary,
    )
    text = _message_content(active_llm.invoke(prompt)).strip()
    text = text.strip("` \n\t")
    return _dedupe_sentences(text)


def product_card_to_search_profile(
    card: dict[str, Any],
    source_product_card_path: Path,
    use_llm_searchable_text: bool = False,
    llm: Any | None = None,
) -> ProductSearchProfile:
    """Convert one ProductCard dictionary into a semantic search profile."""

    cocktail_names = [str(name) for name in card.get("cocktail_names", []) if name]
    tasting_summary = _clean_join(
        [
            card.get("tasting_notes"),
            card.get("nose"),
            card.get("palate"),
            card.get("finish"),
        ]
    )
    tag_source_text = _clean_join(
        [
            card.get("name"),
            card.get("category"),
            card.get("short_description"),
            card.get("marketing_description"),
            card.get("tasting_notes"),
            card.get("nose"),
            card.get("palate"),
            card.get("finish"),
            card.get("process"),
            card.get("how_to_serve"),
            " ".join(cocktail_names),
        ]
    )
    flavor_tags, usage_tags = extract_lightweight_tags(tag_source_text)
    if use_llm_searchable_text:
        searchable_text = generate_llm_searchable_text(
            card=card,
            flavor_tags=flavor_tags,
            usage_tags=usage_tags,
            cocktail_names=cocktail_names,
            tasting_summary=tasting_summary,
            llm=llm,
        )
    else:
        searchable_text = build_deterministic_searchable_text(
            card=card,
            flavor_tags=flavor_tags,
            usage_tags=usage_tags,
            cocktail_names=cocktail_names,
            tasting_summary=tasting_summary,
        )
    display_description = card.get("short_description") or card.get("marketing_description")
    evidence_fields: dict[str, str | list[str]] = {
        "name": card.get("name") or "",
        "category": card.get("category") or "",
        "short_description": card.get("short_description") or "",
        "tasting_summary": tasting_summary,
        "how_to_serve": card.get("how_to_serve") or "",
        "cocktail_names": cocktail_names,
    }

    return ProductSearchProfile(
        product_id=str(card["product_id"]),
        source_url=card["source_url"],
        brand=str(card.get("brand") or "Bacardi"),
        name=str(card["name"]),
        category=card.get("category"),
        short_description=card.get("short_description"),
        tasting_summary=tasting_summary or None,
        flavor_tags=flavor_tags,
        usage_tags=usage_tags,
        cocktail_names=cocktail_names,
        searchable_text=searchable_text,
        display_description=display_description,
        evidence_fields=evidence_fields,
        source_product_card_path=str(source_product_card_path),
        warnings=list(card.get("extraction_warnings", [])),
    )


def build_search_profile_file(
    input_file: Path,
    output_dir: Path,
    force: bool = False,
    use_llm_searchable_text: bool = False,
    llm: Any | None = None,
) -> Path:
    """Build and save one ProductSearchProfile JSON file."""

    card = json.loads(input_file.read_text(encoding="utf-8"))
    profile = product_card_to_search_profile(
        card,
        input_file,
        use_llm_searchable_text=use_llm_searchable_text,
        llm=llm,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_file.stem}.json"
    if output_file.exists() and not force:
        return output_file
    output_file.write_text(
        json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file


def build_search_profiles(
    input_dir: Path,
    output_dir: Path,
    force: bool = False,
    limit: int | None = None,
    use_llm_searchable_text: bool = False,
    llm: Any | None = None,
) -> list[Path]:
    """Build search profile files for ProductCard JSON files."""

    product_files = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if limit is not None:
        product_files = product_files[:limit]
    return [
        build_search_profile_file(
            input_file,
            output_dir,
            force=force,
            use_llm_searchable_text=use_llm_searchable_text,
            llm=llm,
        )
        for input_file in product_files
    ]


def load_search_profiles(profiles_dir: Path) -> list[ProductSearchProfile]:
    """Load ProductSearchProfile JSON files from a directory."""

    profiles: list[ProductSearchProfile] = []
    for path in sorted(profiles_dir.glob("*.json")):
        profiles.append(
            ProductSearchProfile.model_validate_json(path.read_text(encoding="utf-8"))
        )
    return profiles
