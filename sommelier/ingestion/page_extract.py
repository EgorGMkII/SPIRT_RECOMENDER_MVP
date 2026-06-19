"""HTML extraction utilities for raw product page text."""

from urllib.parse import urldefrag, urljoin, urlparse
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, HttpUrl


class ExtractedPageRecord(BaseModel):
    """Structured extraction result for a crawled page."""

    source_url: HttpUrl
    title: str | None = None
    h1: str | None = None
    clean_text: str
    product_links: list[HttpUrl] = Field(default_factory=list)
    cocktail_links: list[HttpUrl] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


def _soup(html: str) -> BeautifulSoup:
    """Create a BeautifulSoup parser for HTML."""

    return BeautifulSoup(html, "html.parser")


def extract_clean_text(html: str) -> str:
    """Extract clean visible text from product HTML.

    Script, style, SVG, and hidden-ish utility nodes are removed before text
    normalization.
    """

    soup = _soup(html)
    for node in soup(["script", "style", "noscript", "svg", "template"]):
        node.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def extract_title(html: str) -> str | None:
    """Extract the document title."""

    soup = _soup(html)
    if soup.title and soup.title.string:
        return " ".join(soup.title.string.split())
    return None


def extract_h1(html: str) -> str | None:
    """Extract the first h1 text."""

    soup = _soup(html)
    h1 = soup.find("h1")
    if h1:
        value = h1.get_text(separator=" ", strip=True)
        return " ".join(value.split()) or None
    return None


def extract_metadata(html: str) -> dict[str, str]:
    """Extract basic HTML metadata."""

    soup = _soup(html)
    metadata: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        key = tag.get("property") or tag.get("name")
        value = tag.get("content")
        if key and value:
            metadata[str(key)] = " ".join(str(value).split())
    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    if canonical and canonical.get("href"):
        metadata["canonical"] = str(canonical["href"])
    return metadata


def normalize_url(base_url: str, href: str) -> str | None:
    """Normalize a discovered link and remove fragments."""

    if not href:
        return None
    href = href.strip()
    if href.startswith(("mailto:", "tel:", "javascript:")):
        return None
    absolute = urljoin(base_url, href)
    absolute, _fragment = urldefrag(absolute)
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return absolute.rstrip("/")


def is_bacardi_product_url(url: str) -> bool:
    """Return whether a URL looks like a Bacardi rum product page."""

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower().strip("/")
    if not host.endswith("bacardi.com"):
        return False
    if "our-rums" not in path:
        return False
    return path != "our-rums" and path != "our-rums/"


def is_bacardi_cocktail_url(url: str) -> bool:
    """Return whether a URL looks like a Bacardi cocktail recipe page."""

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower().strip("/")
    if not host.endswith("bacardi.com"):
        return False
    if "rum-cocktails" not in path:
        return False
    return path != "rum-cocktails" and path != "rum-cocktails/"


def extract_product_links(html: str, source_url: str) -> list[str]:
    """Extract normalized Bacardi rum product links from HTML."""

    soup = _soup(html)
    links: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        normalized = normalize_url(source_url, str(anchor["href"]))
        if normalized and is_bacardi_product_url(normalized):
            links.add(normalized)
    return sorted(links)


def extract_cocktail_links(html: str, source_url: str) -> list[str]:
    """Extract normalized Bacardi cocktail recipe links from HTML."""

    soup = _soup(html)
    links: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        normalized = normalize_url(source_url, str(anchor["href"]))
        if normalized and is_bacardi_cocktail_url(normalized):
            links.add(normalized)
    return sorted(links)


def extract_page_record(html: str, source_url: str) -> ExtractedPageRecord:
    """Extract title, h1, clean text, product links, and metadata from HTML."""

    return ExtractedPageRecord(
        source_url=source_url,
        title=extract_title(html),
        h1=extract_h1(html),
        clean_text=extract_clean_text(html),
        product_links=extract_product_links(html, source_url),
        cocktail_links=extract_cocktail_links(html, source_url),
        metadata=extract_metadata(html),
    )
