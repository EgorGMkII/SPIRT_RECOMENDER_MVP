"""Website crawling interfaces for product discovery and page downloading."""

from datetime import datetime, timezone
from typing import Protocol
from pydantic import BaseModel, Field, HttpUrl
import httpx

from sommelier.ingestion.page_extract import extract_product_links

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36 AI-Sommelier-Assistant/0.1"
)
DEFAULT_TIMEOUT_SECONDS = 15.0


class CrawledPage(BaseModel):
    """Downloaded page artifact."""

    url: HttpUrl
    html: str
    status_code: int = 200
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HttpClient(Protocol):
    """Small protocol so tests can provide fake network clients."""

    def get(self, url: str) -> httpx.Response:
        """Fetch a URL."""


def decode_response_text(response: httpx.Response) -> str:
    """Decode HTML response text with a UTF-8 first policy.

    Bacardi pages are UTF-8, and some environments can still infer a legacy
    encoding from headers or proxy behavior.
    """

    try:
        return response.content.decode("utf-8")
    except UnicodeDecodeError:
        return response.text


def create_http_client(
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
) -> httpx.Client:
    """Create a polite HTTP client for crawling public pages."""

    return httpx.Client(
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=True,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )


def download_page(url: str, client: HttpClient | None = None) -> CrawledPage:
    """Download a product page with timeout, user-agent, and error handling."""

    owns_client = client is None
    active_client = client or create_http_client()
    try:
        response = active_client.get(url)
        response.raise_for_status()
        return CrawledPage(
            url=str(response.url),
            html=decode_response_text(response),
            status_code=response.status_code,
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    finally:
        if owns_client and hasattr(active_client, "close"):
            active_client.close()


def discover_product_urls(listing_url: str, client: HttpClient | None = None) -> list[str]:
    """Download a listing page and discover normalized product URLs."""

    page = download_page(listing_url, client=client)
    return extract_product_links(page.html, str(page.url))
