"""CLI for crawling Bacardi cocktail pages and saving ingestion artifacts."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, HttpUrl, TypeAdapter

from sommelier.config import get_settings
from sommelier.ingestion.crawler import CrawledPage, create_http_client, download_page
from sommelier.ingestion.page_extract import (
    ExtractedPageRecord,
    extract_cocktail_links,
    extract_page_record,
)
from sommelier.ingestion.persistence import save_json, save_text, slugify_url

BACARDI_COCKTAILS_URL = "https://www.bacardi.com/rum-cocktails/"
DEFAULT_MAX_LOAD_MORE_PAGES = 20
HTTP_URL_LIST_ADAPTER = TypeAdapter(list[HttpUrl])


class CocktailCrawlOutput(BaseModel):
    """Summary of files produced by a Bacardi cocktail crawl."""

    listing_url: HttpUrl
    raw_html_files: list[str] = Field(default_factory=list)
    parsed_json_files: list[str] = Field(default_factory=list)
    cocktail_urls: list[HttpUrl] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _extract_grid_params(html: str) -> dict[str, str]:
    """Extract Bacardi cocktails load-more AJAX params from listing HTML."""

    match = re.search(r"bacardi2020_cocktails_grid_params\s*=\s*(\{.*?\});", html)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1).replace("\\/", "/"))
    except json.JSONDecodeError:
        return {}
    return {
        "ajaxurl": str(payload.get("ajaxurl") or ""),
        "action": str(payload.get("action") or ""),
        "page": str(payload.get("page") or "1"),
        "panel_id": str(payload.get("panel_id") or ""),
        "panel_name": str(payload.get("panel_name") or ""),
    }


def _iter_text_values(payload: Any):
    """Yield string values recursively from an arbitrary JSON payload."""

    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from _iter_text_values(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_text_values(value)


def _response_text_values(response_text: str) -> list[str]:
    """Return response text plus any nested string values if response is JSON."""

    values = [response_text]
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return values
    values.extend(_iter_text_values(payload))
    return values


def discover_load_more_cocktail_urls(
    listing_html: str,
    listing_url: str,
    client,
    max_pages: int = DEFAULT_MAX_LOAD_MORE_PAGES,
) -> tuple[list[str], list[str]]:
    """Discover cocktail URLs from Bacardi's load-more AJAX endpoint."""

    params = _extract_grid_params(listing_html)
    if not params.get("ajaxurl") or not params.get("action"):
        return [], ["Cocktail load-more AJAX params were not found."]
    if not hasattr(client, "post"):
        return [], ["HTTP client does not support POST for cocktail load-more."]

    discovered: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    start_page = int(params.get("page") or "1") + 1
    for page in range(start_page, start_page + max_pages):
        data = {
            "action": params["action"],
            "page": str(page),
            "panel_id": params.get("panel_id", ""),
            "panel_name": params.get("panel_name", ""),
        }
        try:
            response = client.post(
                params["ajaxurl"],
                data=data,
                headers={
                    "Referer": listing_url,
                    "X-Requested-With": "XMLHttpRequest",
                },
            )
            response.raise_for_status()
        except Exception as exc:
            errors.append(f"Load-more cocktail page {page} failed: {exc}")
            break

        page_links: list[str] = []
        for text in _response_text_values(response.text):
            page_links.extend(extract_cocktail_links(text, listing_url))
        page_new_links = [url for url in page_links if url not in seen]
        for url in page_new_links:
            seen.add(url)
            discovered.append(url)

        if not page_new_links:
            break

    return discovered, errors


def save_cocktail_page_artifacts(
    page: CrawledPage,
    record: ExtractedPageRecord,
    raw_dir: Path,
    parsed_dir: Path,
) -> tuple[Path, Path]:
    """Save raw HTML and extracted JSON for one cocktail page."""

    slug = slugify_url(str(page.url))
    raw_path = save_text(raw_dir / f"{slug}.html", page.html)
    parsed_path = save_json(parsed_dir / f"{slug}.json", record)
    return raw_path, parsed_path


def crawl_bacardi_cocktails(
    listing_url: str = BACARDI_COCKTAILS_URL,
    raw_dir: Path | None = None,
    parsed_dir: Path | None = None,
    delay_seconds: float = 0.5,
    max_cocktails: int | None = None,
    max_load_more_pages: int = DEFAULT_MAX_LOAD_MORE_PAGES,
) -> CocktailCrawlOutput:
    """Crawl the Bacardi cocktail listing and discovered recipe pages."""

    settings = get_settings()
    raw_output_dir = raw_dir or settings.data_dir / "raw_cocktail_pages"
    parsed_output_dir = parsed_dir or settings.data_dir / "parsed_cocktails"
    output = CocktailCrawlOutput(listing_url=listing_url)

    with create_http_client() as client:
        listing_page = download_page(listing_url, client=client)
        listing_record = extract_page_record(listing_page.html, str(listing_page.url))
        raw_path, parsed_path = save_cocktail_page_artifacts(
            listing_page,
            listing_record,
            raw_output_dir,
            parsed_output_dir,
        )
        output.raw_html_files.append(str(raw_path))
        output.parsed_json_files.append(str(parsed_path))

        cocktail_urls = list(getattr(listing_record, "cocktail_links", []) or [])
        if not cocktail_urls:
            cocktail_urls = extract_cocktail_links(
                listing_page.html,
                str(listing_page.url),
            )
        if max_cocktails is None or len(cocktail_urls) < max_cocktails:
            load_more_urls, load_more_errors = discover_load_more_cocktail_urls(
                listing_page.html,
                str(listing_page.url),
                client=client,
                max_pages=max_load_more_pages,
            )
            output.errors.extend(load_more_errors)
            seen_urls = {str(url) for url in cocktail_urls}
            for url in load_more_urls:
                if url not in seen_urls:
                    cocktail_urls.append(url)
                    seen_urls.add(url)
        if max_cocktails is not None:
            cocktail_urls = cocktail_urls[:max_cocktails]
        output.cocktail_urls = HTTP_URL_LIST_ADAPTER.validate_python(cocktail_urls)

        for url in cocktail_urls:
            try:
                time.sleep(delay_seconds)
                page = download_page(str(url), client=client)
                record = extract_page_record(page.html, str(page.url))
                raw_path, parsed_path = save_cocktail_page_artifacts(
                    page,
                    record,
                    raw_output_dir,
                    parsed_output_dir,
                )
                output.raw_html_files.append(str(raw_path))
                output.parsed_json_files.append(str(parsed_path))
            except RuntimeError as exc:
                output.errors.append(str(exc))

    summary_path = save_json(parsed_output_dir / "bacardi_cocktail_crawl_summary.json", output)
    output.parsed_json_files.append(str(summary_path))
    return output


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Crawl Bacardi cocktail pages.")
    parser.add_argument("--listing-url", default=BACARDI_COCKTAILS_URL)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--parsed-dir", type=Path, default=None)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--max-cocktails", type=int, default=None)
    parser.add_argument("--max-load-more-pages", type=int, default=DEFAULT_MAX_LOAD_MORE_PAGES)
    return parser


def main() -> None:
    """Run the Bacardi cocktail ingestion CLI."""

    args = build_parser().parse_args()
    output = crawl_bacardi_cocktails(
        listing_url=args.listing_url,
        raw_dir=args.raw_dir,
        parsed_dir=args.parsed_dir,
        delay_seconds=args.delay_seconds,
        max_cocktails=args.max_cocktails,
        max_load_more_pages=args.max_load_more_pages,
    )
    print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
