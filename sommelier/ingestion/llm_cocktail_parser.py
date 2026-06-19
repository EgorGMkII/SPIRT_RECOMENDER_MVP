"""LLM-assisted conversion from Bacardi cocktail pages to CocktailCard JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from llm_module import get_langchain_openai_chat_model
from sommelier.ingestion.cocktail_extraction_prompt import build_cocktail_extraction_prompt
from sommelier.ingestion.persistence import save_json, slugify_url

MAX_TEXT_CHARS = 16000
PARSER_VERSION = "llm-cocktail-parser-v1"


class CocktailIngredient(BaseModel):
    """One cocktail ingredient line."""

    name: str
    amount: str | None = None


class CocktailRecipe(BaseModel):
    """Structured cocktail recipe."""

    servings: str | None = None
    prep_time: str | None = None
    difficulty: str | None = None
    ingredients: list[CocktailIngredient] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)


class CocktailCard(BaseModel):
    """Structured cocktail card produced from a Bacardi cocktail page."""

    cocktail_id: str
    source_url: HttpUrl
    brand: str = "Bacardi"
    name: str
    title: str | None = None
    main_rum: str | None = None
    short_description: str
    marketing_description: str | None = None
    recipe: CocktailRecipe
    glassware: str | None = None
    garnish: str | None = None
    method: str | None = None
    raw_text_excerpt: str
    source_metadata: dict[str, str] = Field(default_factory=dict)
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    extraction_warnings: list[str] = Field(default_factory=list)
    parser_version: str = PARSER_VERSION


class CocktailParseResult(BaseModel):
    """Result of parsing one cocktail page file."""

    input_file: str
    output_file: str | None = None
    cocktail_id: str | None = None
    skipped: bool = False
    dry_run: bool = False
    warnings: list[str] = Field(default_factory=list)


class CocktailParseDirectorySummary(BaseModel):
    """Summary of parsing extracted cocktail page records."""

    input_dir: str
    output_dir: str
    results: list[CocktailParseResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def is_cocktail_page_record(page_record: dict[str, Any]) -> bool:
    """Return whether an extracted page record looks like a Bacardi cocktail page."""

    source_url = str(page_record.get("source_url", ""))
    path = urlparse(source_url).path.strip("/").lower()
    return path.startswith("rum-cocktails/") and path != "rum-cocktails"


def _page_payload(page_record: dict[str, Any]) -> str:
    """Create compact JSON payload for the extraction prompt."""

    payload = {
        "source_url": page_record.get("source_url"),
        "title": page_record.get("title"),
        "h1": page_record.get("h1"),
        "metadata": page_record.get("metadata", {}),
        "clean_text": str(page_record.get("clean_text", ""))[:MAX_TEXT_CHARS],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _message_content(message: Any) -> str:
    """Extract text content from a LangChain/OpenAI-like message."""

    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return str(message)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from an LLM response."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _default_card_payload(page_record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Merge deterministic provenance into an LLM payload."""

    source_url = str(page_record.get("source_url") or payload.get("source_url") or "")
    clean_text = str(page_record.get("clean_text", ""))
    cocktail_id = str(payload.get("cocktail_id") or slugify_url(source_url)).removeprefix(
        "www-bacardi-com-rum-cocktails-"
    )
    return {
        **payload,
        "cocktail_id": cocktail_id,
        "source_url": source_url,
        "brand": payload.get("brand") or "Bacardi",
        "title": payload.get("title") or page_record.get("title"),
        "raw_text_excerpt": payload.get("raw_text_excerpt") or clean_text[:1200],
        "source_metadata": page_record.get("metadata", {}),
        "extraction_warnings": payload.get("extraction_warnings") or [],
    }


def parse_cocktail_record(
    page_record: dict[str, Any],
    llm: Any | None = None,
    max_retries: int = 1,
    use_structured_output: bool = True,
) -> CocktailCard:
    """Parse one extracted page record into a validated cocktail card."""

    active_llm = llm or get_langchain_openai_chat_model()
    prompt = build_cocktail_extraction_prompt(_page_payload(page_record))

    if use_structured_output and hasattr(active_llm, "with_structured_output"):
        try:
            structured_llm = active_llm.with_structured_output(
                CocktailCard,
                method="function_calling",
            )
            result = structured_llm.invoke(prompt)
            if isinstance(result, CocktailCard):
                payload = result.model_dump(mode="json")
            elif isinstance(result, BaseModel):
                payload = result.model_dump(mode="json")
            else:
                payload = dict(result)
            return CocktailCard.model_validate(_default_card_payload(page_record, payload))
        except Exception:
            pass

    errors: list[str] = []
    current_prompt = prompt
    for attempt in range(max_retries + 1):
        try:
            raw_response = _message_content(active_llm.invoke(current_prompt))
            payload = _extract_json_object(raw_response)
            return CocktailCard.model_validate(_default_card_payload(page_record, payload))
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            errors.append(str(exc))
            if attempt >= max_retries:
                raise ValueError(f"Could not parse cocktail card after retries: {errors}") from exc
            current_prompt = (
                f"{prompt}\n\nPrevious response failed validation with this error:\n"
                f"{exc}\nReturn corrected JSON only."
            )

    raise ValueError("Unreachable parser state.")


def parse_cocktail_file(
    input_file: Path,
    output_dir: Path,
    llm: Any | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> CocktailParseResult:
    """Parse one extracted cocktail page JSON file and optionally write a card."""

    page_record = json.loads(input_file.read_text(encoding="utf-8"))
    if not is_cocktail_page_record(page_record):
        return CocktailParseResult(input_file=str(input_file), skipped=True, warnings=["Not a cocktail page."])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{slugify_url(str(page_record['source_url']))}.json"
    if output_file.exists() and not force and not dry_run:
        return CocktailParseResult(
            input_file=str(input_file),
            output_file=str(output_file),
            skipped=True,
            warnings=["Output exists; use --force to overwrite."],
        )

    card = parse_cocktail_record(page_record, llm=llm)
    if not dry_run:
        save_json(output_file, card)
    return CocktailParseResult(
        input_file=str(input_file),
        output_file=str(output_file),
        cocktail_id=card.cocktail_id,
        dry_run=dry_run,
        warnings=card.extraction_warnings,
    )


def parse_directory(
    input_dir: Path,
    output_dir: Path,
    llm: Any | None = None,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> CocktailParseDirectorySummary:
    """Parse extracted cocktail page records into CocktailCard JSON files."""

    summary = CocktailParseDirectorySummary(input_dir=str(input_dir), output_dir=str(output_dir))
    files = sorted(
        path
        for path in input_dir.glob("*.json")
        if path.name != "bacardi_cocktail_crawl_summary.json"
    )
    attempted_cocktails = 0
    for path in files:
        try:
            page_record = json.loads(path.read_text(encoding="utf-8"))
            if not is_cocktail_page_record(page_record):
                summary.results.append(
                    CocktailParseResult(
                        input_file=str(path),
                        skipped=True,
                        warnings=["Not a cocktail page."],
                    )
                )
                continue
            if limit is not None and attempted_cocktails >= limit:
                break
            attempted_cocktails += 1
            result = parse_cocktail_file(
                path,
                output_dir=output_dir,
                llm=llm,
                force=force,
                dry_run=dry_run,
            )
            summary.results.append(result)
        except Exception as exc:
            summary.errors.append(f"{path}: {exc}")
    if not dry_run:
        save_json(output_dir.parent / "cocktail_catalog_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description="Parse Bacardi cocktail pages with an LLM.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/parsed_cocktails"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/catalog/cocktails"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    """Run the cocktail parser CLI."""

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
