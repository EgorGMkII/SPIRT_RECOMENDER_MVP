"""CLI for crawling Bacardi rum pages and saving ingestion artifacts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from pydantic import BaseModel, Field, HttpUrl

from sommelier.config import get_settings
from sommelier.ingestion.crawler import CrawledPage, create_http_client, download_page
from sommelier.ingestion.page_extract import ExtractedPageRecord, extract_page_record
from sommelier.ingestion.persistence import save_json, save_text, slugify_url

BACARDI_RUMS_URL = "https://www.bacardi.com/our-rums/"


class CrawlOutput(BaseModel):
    """Summary of files produced by a Bacardi crawl."""

    listing_url: HttpUrl
    raw_html_files: list[str] = Field(default_factory=list)
    parsed_json_files: list[str] = Field(default_factory=list)
    product_urls: list[HttpUrl] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def save_page_artifacts(
    page: CrawledPage,
    record: ExtractedPageRecord,
    raw_dir: Path,
    parsed_dir: Path,
) -> tuple[Path, Path]:
    """Save raw HTML and extracted JSON for one page."""

    slug = slugify_url(str(page.url))
    raw_path = save_text(raw_dir / f"{slug}.html", page.html)
    parsed_path = save_json(parsed_dir / f"{slug}.json", record)
    return raw_path, parsed_path


def crawl_bacardi(
    listing_url: str = BACARDI_RUMS_URL,
    raw_dir: Path | None = None,
    parsed_dir: Path | None = None,
    delay_seconds: float = 0.5,
    max_products: int | None = None,
) -> CrawlOutput:
    """Crawl the Bacardi rum listing and discovered product pages."""

    settings = get_settings()
    raw_output_dir = raw_dir or settings.data_dir / "raw_pages"
    parsed_output_dir = parsed_dir or settings.data_dir / "parsed_products"
    output = CrawlOutput(listing_url=listing_url)

    with create_http_client() as client:
        listing_page = download_page(listing_url, client=client)
        listing_record = extract_page_record(listing_page.html, str(listing_page.url))
        raw_path, parsed_path = save_page_artifacts(
            listing_page,
            listing_record,
            raw_output_dir,
            parsed_output_dir,
        )
        output.raw_html_files.append(str(raw_path))
        output.parsed_json_files.append(str(parsed_path))

        product_urls = listing_record.product_links
        if max_products is not None:
            product_urls = product_urls[:max_products]
        output.product_urls = product_urls

        for url in product_urls:
            try:
                time.sleep(delay_seconds)
                page = download_page(str(url), client=client)
                record = extract_page_record(page.html, str(page.url))
                raw_path, parsed_path = save_page_artifacts(
                    page,
                    record,
                    raw_output_dir,
                    parsed_output_dir,
                )
                output.raw_html_files.append(str(raw_path))
                output.parsed_json_files.append(str(parsed_path))
            except RuntimeError as exc:
                output.errors.append(str(exc))

    summary_path = save_json(parsed_output_dir / "bacardi_crawl_summary.json", output)
    output.parsed_json_files.append(str(summary_path))
    return output


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Crawl Bacardi rum pages.")
    parser.add_argument("--listing-url", default=BACARDI_RUMS_URL)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--parsed-dir", type=Path, default=None)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--max-products", type=int, default=None)
    return parser


def main() -> None:
    """Run the Bacardi ingestion CLI."""

    args = build_parser().parse_args()
    output = crawl_bacardi(
        listing_url=args.listing_url,
        raw_dir=args.raw_dir,
        parsed_dir=args.parsed_dir,
        delay_seconds=args.delay_seconds,
        max_products=args.max_products,
    )
    print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
