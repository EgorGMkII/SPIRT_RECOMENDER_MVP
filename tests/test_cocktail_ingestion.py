import json
import shutil
from pathlib import Path
from uuid import uuid4

import httpx

from sommelier.ingestion.crawl_bacardi_cocktails import crawl_bacardi_cocktails
from sommelier.ingestion.llm_cocktail_parser import (
    CocktailCard,
    is_cocktail_page_record,
    parse_cocktail_record,
)
from sommelier.ingestion.page_extract import (
    extract_cocktail_links,
    extract_page_record,
    is_bacardi_cocktail_url,
)


class FakeClient:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.posts: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def get(self, url: str) -> httpx.Response:
        html = self.pages[url]
        return httpx.Response(
            200,
            content=html.encode("utf-8"),
            request=httpx.Request("GET", url),
            headers={"content-type": "text/html; charset=utf-8"},
        )

    def post(self, url: str, data: dict, headers: dict | None = None) -> httpx.Response:
        self.posts.append({"url": url, "data": data, "headers": headers or {}})
        page = data["page"]
        if page == "2":
            html = '<a href="https://www.bacardi.com/rum-cocktails/pina-colada/">Pina Colada</a>'
        else:
            html = ""
        return httpx.Response(
            200,
            content=html.encode("utf-8"),
            request=httpx.Request("POST", url),
            headers={"content-type": "text/html; charset=utf-8"},
        )

    def close(self) -> None:
        pass


class FakeLLM:
    def invoke(self, prompt: str) -> str:
        assert "Mojito" in prompt
        return json.dumps(
            {
                "cocktail_id": "mojito",
                "source_url": "https://www.bacardi.com/rum-cocktails/mojito/",
                "brand": "Bacardi",
                "name": "Mojito",
                "title": "Mojito Rum Cocktail",
                "main_rum": "BACARDI Carta Blanca",
                "short_description": "A bright mint and lime rum highball.",
                "marketing_description": "A crisp classic made for warm evenings.",
                "recipe": {
                    "servings": "1",
                    "prep_time": "5 minutes",
                    "difficulty": "easy",
                    "ingredients": [
                        {"name": "BACARDI Carta Blanca", "amount": "50 ml"},
                        {"name": "lime juice", "amount": "25 ml"},
                    ],
                    "steps": ["Build over ice.", "Top with soda."],
                },
                "glassware": "highball",
                "garnish": "mint sprig",
                "method": "build",
                "raw_text_excerpt": "Mojito BACARDI Carta Blanca lime mint soda",
                "source_metadata": {},
                "extraction_confidence": 0.9,
                "extraction_warnings": [],
            }
        )


def test_cocktail_url_detection_and_extraction() -> None:
    html = """
    <a href="/rum-cocktails/mojito/">Mojito</a>
    <a href="https://www.bacardi.com/rum-cocktails/daiquiri/">Daiquiri</a>
    <a href="/rum-cocktails/">Listing</a>
    <a href="/our-rums/carta-blanca-rum/">Rum</a>
    """

    links = extract_cocktail_links(html, "https://www.bacardi.com/rum-cocktails/")

    assert is_bacardi_cocktail_url("https://www.bacardi.com/rum-cocktails/mojito/")
    assert not is_bacardi_cocktail_url("https://www.bacardi.com/rum-cocktails/")
    assert links == [
        "https://www.bacardi.com/rum-cocktails/daiquiri",
        "https://www.bacardi.com/rum-cocktails/mojito",
    ]


def test_extract_page_record_includes_cocktail_links() -> None:
    record = extract_page_record(
        '<html><head><title>Cocktails</title></head><body><a href="/rum-cocktails/mojito/">Mojito</a></body></html>',
        "https://www.bacardi.com/rum-cocktails/",
    )

    assert str(record.cocktail_links[0]) == "https://www.bacardi.com/rum-cocktails/mojito"


def test_crawl_bacardi_cocktails_with_fake_client(monkeypatch) -> None:
    listing = "https://www.bacardi.com/rum-cocktails/"
    mojito = "https://www.bacardi.com/rum-cocktails/mojito"
    ajax_params = r"""
    <script>
    var bacardi2020_cocktails_grid_params = {"ajaxurl":"https:\/\/www.bacardi.com\/wp-admin\/admin-ajax.php","action":"bacardi2020_cocktails_grid_load_more","page":"1","panel_id":"bacardi2020-cocktails-grid","panel_name":"panel--type-bacardi2020-cocktails-grid"};
    </script>
    """
    pages = {
        listing: f'<html><a href="{mojito}">Mojito</a>{ajax_params}</html>',
        mojito: "<html><h1>Mojito</h1><p>BACARDI Carta Blanca lime mint soda.</p></html>",
        "https://www.bacardi.com/rum-cocktails/pina-colada": (
            "<html><h1>Pina Colada</h1><p>BACARDI rum pineapple coconut.</p></html>"
        ),
    }
    monkeypatch.setattr(
        "sommelier.ingestion.crawl_bacardi_cocktails.create_http_client",
        lambda: FakeClient(pages),
    )
    work_dir = Path(".test_tmp") / f"cocktail-crawl-{uuid4().hex}"
    try:
        output = crawl_bacardi_cocktails(
            listing_url=listing,
            raw_dir=work_dir / "raw",
            parsed_dir=work_dir / "parsed",
            delay_seconds=0,
        )

        assert len(output.cocktail_urls) == 2
        assert len(output.raw_html_files) == 3
        assert (work_dir / "parsed" / "bacardi_cocktail_crawl_summary.json").exists()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_parse_cocktail_record_with_fake_llm() -> None:
    page_record = {
        "source_url": "https://www.bacardi.com/rum-cocktails/mojito/",
        "title": "Mojito Rum Cocktail",
        "h1": "Mojito",
        "metadata": {},
        "clean_text": "Mojito BACARDI Carta Blanca lime mint soda.",
    }

    card = parse_cocktail_record(
        page_record,
        llm=FakeLLM(),
        use_structured_output=False,
    )

    assert isinstance(card, CocktailCard)
    assert card.cocktail_id == "mojito"
    assert card.main_rum == "BACARDI Carta Blanca"
    assert card.recipe.ingredients[0].name == "BACARDI Carta Blanca"
    assert card.recipe.steps == ["Build over ice.", "Top with soda."]


def test_is_cocktail_page_record_skips_listing() -> None:
    assert is_cocktail_page_record(
        {"source_url": "https://www.bacardi.com/rum-cocktails/mojito/"}
    )
    assert not is_cocktail_page_record(
        {"source_url": "https://www.bacardi.com/rum-cocktails/"}
    )
