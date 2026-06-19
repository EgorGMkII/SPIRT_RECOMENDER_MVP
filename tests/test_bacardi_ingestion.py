import json
import shutil
from pathlib import Path
from uuid import uuid4

from sommelier.ingestion.crawler import CrawledPage, decode_response_text, discover_product_urls
from sommelier.ingestion.crawl_bacardi import save_page_artifacts
from sommelier.ingestion.page_extract import extract_clean_text, extract_page_record, extract_product_links


FIXTURE_HTML = """
<!doctype html>
<html>
  <head>
    <title>Bacardi Rums</title>
    <meta name="description" content="Explore Bacardi rum products">
    <meta property="og:title" content="Our Rums">
    <link rel="canonical" href="https://www.bacardi.com/our-rums/">
    <style>.hidden { display: none; }</style>
    <script>window.noise = true;</script>
  </head>
  <body>
    <h1>Our Rums</h1>
    <p>Find a rum for mojitos and daiquiris.</p>
    <a href="/our-rums/bacardi-carta-blanca/">Carta Blanca</a>
    <a href="https://www.bacardi.com/our-rums/bacardi-spiced/#details">Spiced</a>
    <a href="/cocktails/mojito/">Mojito</a>
    <a href="mailto:test@example.com">Mail</a>
  </body>
</html>
"""


class FakeClient:
    def get(self, url: str):
        import httpx

        return httpx.Response(
            200,
            text=FIXTURE_HTML,
            request=httpx.Request("GET", url),
        )


def test_product_link_extraction_normalizes_bacardi_rum_urls() -> None:
    links = extract_product_links(FIXTURE_HTML, "https://www.bacardi.com/our-rums/")

    assert links == [
        "https://www.bacardi.com/our-rums/bacardi-carta-blanca",
        "https://www.bacardi.com/our-rums/bacardi-spiced",
    ]


def test_discover_product_urls_uses_injected_client() -> None:
    links = discover_product_urls("https://www.bacardi.com/our-rums/", client=FakeClient())

    assert "https://www.bacardi.com/our-rums/bacardi-spiced" in links


def test_response_decoding_prefers_utf8() -> None:
    import httpx

    response = httpx.Response(
        200,
        content="BACARDÍ Carta Blanca".encode("utf-8"),
        headers={"content-type": "text/html; charset=iso-8859-1"},
        request=httpx.Request("GET", "https://www.bacardi.com/our-rums/"),
    )

    assert decode_response_text(response) == "BACARDÍ Carta Blanca"


def test_clean_text_extraction_removes_script_and_style() -> None:
    text = extract_clean_text(FIXTURE_HTML)

    assert "Find a rum for mojitos" in text
    assert "window.noise" not in text
    assert ".hidden" not in text


def test_saved_json_matches_extracted_page_schema() -> None:
    tmp_path = Path(".test_tmp") / f"ingestion-{uuid4().hex}"
    page = CrawledPage(url="https://www.bacardi.com/our-rums/", html=FIXTURE_HTML)
    record = extract_page_record(FIXTURE_HTML, "https://www.bacardi.com/our-rums/")

    try:
        raw_path, parsed_path = save_page_artifacts(
            page,
            record,
            raw_dir=tmp_path / "raw_pages",
            parsed_dir=tmp_path / "parsed_products",
        )

        payload = json.loads(parsed_path.read_text(encoding="utf-8"))

        assert raw_path.exists()
        assert payload["source_url"] == "https://www.bacardi.com/our-rums/"
        assert payload["title"] == "Bacardi Rums"
        assert payload["h1"] == "Our Rums"
        assert payload["metadata"]["description"] == "Explore Bacardi rum products"
        assert payload["product_links"] == [
            "https://www.bacardi.com/our-rums/bacardi-carta-blanca",
            "https://www.bacardi.com/our-rums/bacardi-spiced",
        ]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
