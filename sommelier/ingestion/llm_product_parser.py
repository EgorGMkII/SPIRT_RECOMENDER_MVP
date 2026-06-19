"""LLM-assisted conversion from raw Bacardi page records to product cards."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from llm_module import get_langchain_openai_chat_model
from sommelier.ingestion.persistence import save_json, slugify_url
from sommelier.ingestion.product_extraction_prompt import build_product_extraction_prompt

MAX_TEXT_CHARS = 14000
PARSER_VERSION = "llm-product-parser-v1"


class FAQItem(BaseModel):
    """Question and answer extracted from a product page FAQ section."""

    question: str
    answer: str


class ProductCard(BaseModel):
    """Structured product card produced from a Bacardi product page."""

    product_id: str
    source_url: HttpUrl
    brand: str = "Bacardi"
    name: str
    title: str | None = None
    category: str | None = None
    short_description: str | None = None
    marketing_description: str | None = None
    tasting_notes: str | None = None
    nose: str | None = None
    palate: str | None = None
    finish: str | None = None
    process: str | None = None
    how_to_serve: str | None = None
    cocktail_names: list[str] = Field(default_factory=list)
    recommended_rums: list[str] = Field(default_factory=list)
    faq_items: list[FAQItem] = Field(default_factory=list)
    raw_text_excerpt: str
    source_metadata: dict[str, str] = Field(default_factory=dict)
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    extraction_warnings: list[str] = Field(default_factory=list)
    parser_version: str = PARSER_VERSION


class ParseResult(BaseModel):
    """Result of parsing one product page file."""

    input_file: str
    output_file: str | None = None
    product_id: str | None = None
    skipped: bool = False
    dry_run: bool = False
    warnings: list[str] = Field(default_factory=list)


class ParseDirectorySummary(BaseModel):
    """Summary of parsing a directory of extracted page records."""

    input_dir: str
    output_dir: str
    results: list[ParseResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def is_product_page_record(page_record: dict[str, Any]) -> bool:
    """Return whether an extracted page record looks like a Bacardi product page."""

    source_url = str(page_record.get("source_url", ""))
    path = urlparse(source_url).path.strip("/").lower()
    return path.startswith("our-rums/") and path != "our-rums"


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
    product_id = str(payload.get("product_id") or slugify_url(source_url)).removeprefix(
        "www-bacardi-com-our-rums-"
    )
    merged = {
        **payload,
        "product_id": product_id,
        "source_url": source_url,
        "brand": payload.get("brand") or "Bacardi",
        "title": payload.get("title") or page_record.get("title"),
        "raw_text_excerpt": payload.get("raw_text_excerpt") or clean_text[:1200],
        "source_metadata": page_record.get("metadata", {}),
        "extraction_warnings": payload.get("extraction_warnings") or [],
    }
    return merged


def _invoke_json_llm(llm: Any, prompt: str) -> str:
    """Invoke a LangChain-like LLM and return text content."""

    response = llm.invoke(prompt)
    return _message_content(response)


def parse_product_record(
    page_record: dict[str, Any],
    llm: Any | None = None,
    max_retries: int = 1,
    use_structured_output: bool = True,
) -> ProductCard:
    """Parse one extracted page record into a validated product card."""

    active_llm = llm or get_langchain_openai_chat_model()
    prompt = build_product_extraction_prompt(_page_payload(page_record))

    if use_structured_output and hasattr(active_llm, "with_structured_output"):
        try:
            structured_llm = active_llm.with_structured_output(
                ProductCard,
                method="function_calling",
            )
            result = structured_llm.invoke(prompt)
            if isinstance(result, ProductCard):
                payload = result.model_dump(mode="json")
            elif isinstance(result, BaseModel):
                payload = result.model_dump(mode="json")
            else:
                payload = dict(result)
            return ProductCard.model_validate(_default_card_payload(page_record, payload))
        except Exception:
            # Fall back to JSON mode below. Some proxy/model combinations do not
            # support LangChain structured output consistently.
            pass

    errors: list[str] = []
    current_prompt = prompt
    for attempt in range(max_retries + 1):
        try:
            raw_response = _invoke_json_llm(active_llm, current_prompt)
            payload = _extract_json_object(raw_response)
            return ProductCard.model_validate(_default_card_payload(page_record, payload))
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            errors.append(str(exc))
            if attempt >= max_retries:
                raise ValueError(f"Could not parse product card after retries: {errors}") from exc
            current_prompt = (
                f"{prompt}\n\nPrevious response failed validation with this error:\n"
                f"{exc}\nReturn corrected JSON only."
            )

    raise ValueError("Unreachable parser state.")


def parse_product_text(raw_text: str, source_url: str | None = None, llm: Any | None = None) -> ProductCard:
    """Parse raw text directly into a product card."""

    return parse_product_record(
        {
            "source_url": source_url or "https://www.bacardi.com/our-rums/unknown",
            "title": None,
            "h1": None,
            "metadata": {},
            "clean_text": raw_text,
        },
        llm=llm,
    )


def parse_product_file(
    input_file: Path,
    output_dir: Path,
    llm: Any | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> ParseResult:
    """Parse one extracted page JSON file and optionally write a ProductCard JSON."""

    page_record = json.loads(input_file.read_text(encoding="utf-8"))
    if not is_product_page_record(page_record):
        return ParseResult(input_file=str(input_file), skipped=True, warnings=["Not a product page."])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{slugify_url(str(page_record['source_url']))}.json"
    if output_file.exists() and not force and not dry_run:
        return ParseResult(
            input_file=str(input_file),
            output_file=str(output_file),
            skipped=True,
            warnings=["Output exists; use --force to overwrite."],
        )

    card = parse_product_record(page_record, llm=llm)
    if not dry_run:
        save_json(output_file, card)
    return ParseResult(
        input_file=str(input_file),
        output_file=str(output_file),
        product_id=card.product_id,
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
) -> ParseDirectorySummary:
    """Parse extracted page records from a directory into product card JSON files."""

    summary = ParseDirectorySummary(input_dir=str(input_dir), output_dir=str(output_dir))
    files = sorted(
        path
        for path in input_dir.glob("*.json")
        if path.name != "bacardi_crawl_summary.json"
    )
    attempted_products = 0
    for path in files:
        try:
            page_record = json.loads(path.read_text(encoding="utf-8"))
            if not is_product_page_record(page_record):
                summary.results.append(
                    ParseResult(
                        input_file=str(path),
                        skipped=True,
                        warnings=["Not a product page."],
                    )
                )
                continue
            if limit is not None and attempted_products >= limit:
                break
            attempted_products += 1
            result = parse_product_file(
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
        save_json(output_dir.parent / "catalog_summary.json", summary)
    return summary
