import json
import shutil
from pathlib import Path
from uuid import uuid4

from sommelier.ingestion.llm_product_parser import (
    ProductCard,
    parse_directory,
    parse_product_file,
    parse_product_record,
)


PAGE_RECORD = {
    "source_url": "https://www.bacardi.com/our-rums/carta-blanca-rum/",
    "title": "BACARDI Superior Rum | White Rum | BACARDI Global",
    "h1": "BACARDI Carta Blanca",
    "clean_text": (
        "OUR RUMS COCKTAILS FAQ BACARDI Carta Blanca A sublime rum for cocktails. "
        "Tasting Notes Nose Almonds and fruit Palate Smooth and creamy "
        "Finish Dry, clean, fresh Filtered to perfection This distinctive spirit "
        "is aged in American white oak barrels and shaped through charcoal. "
        "The Perfect Mixer Ideal for mixing. ABOUT US CONTACT US COOKIE POLICY"
    ),
    "metadata": {
        "description": "Savor the original premium white rum.",
        "keywords": "white rum",
    },
    "product_links": [],
}


VALID_LLM_PAYLOAD = {
    "product_id": "carta-blanca-rum",
    "brand": "Bacardi",
    "name": "BACARDI Carta Blanca",
    "category": "white rum",
    "short_description": "A sublime rum for cocktails.",
    "marketing_description": "A sublime rum for cocktails.",
    "tasting_notes": "Floral and fruity.",
    "nose": "Almonds and fruit",
    "palate": "Smooth and creamy",
    "finish": "Dry, clean, fresh",
    "process": "Aged in American white oak barrels and shaped through charcoal.",
    "how_to_serve": "Ideal for mixing.",
    "cocktail_names": ["Mojito", "Daiquiri"],
    "recommended_rums": [],
    "faq_items": [{"question": "What is white rum?", "answer": "A light-bodied rum."}],
    "raw_text_excerpt": "BACARDI Carta Blanca A sublime rum for cocktails.",
    "extraction_confidence": 0.91,
    "extraction_warnings": ["Navigation and footer content detected and ignored."],
}


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def invoke(self, prompt: str) -> FakeMessage:
        response = self.responses[self.calls]
        self.calls += 1
        return FakeMessage(response)


def _tmp_dir() -> Path:
    return Path(".test_tmp") / f"llm-parser-{uuid4().hex}"


def test_product_card_validation() -> None:
    card = ProductCard.model_validate(
        {
            **VALID_LLM_PAYLOAD,
            "source_url": PAGE_RECORD["source_url"],
            "title": PAGE_RECORD["title"],
            "source_metadata": PAGE_RECORD["metadata"],
        }
    )

    assert card.name == "BACARDI Carta Blanca"
    assert card.extraction_confidence == 0.91


def test_parser_output_validation_and_source_url_preservation() -> None:
    llm = FakeLLM([json.dumps(VALID_LLM_PAYLOAD)])

    card = parse_product_record(PAGE_RECORD, llm=llm, use_structured_output=False)

    assert str(card.source_url) == PAGE_RECORD["source_url"]
    assert card.title == PAGE_RECORD["title"]
    assert card.source_metadata == PAGE_RECORD["metadata"]


def test_retry_on_malformed_json() -> None:
    llm = FakeLLM(["not json", json.dumps(VALID_LLM_PAYLOAD)])

    card = parse_product_record(PAGE_RECORD, llm=llm, max_retries=1, use_structured_output=False)

    assert card.product_id == "carta-blanca-rum"
    assert llm.calls == 2


def test_extraction_warning_handling() -> None:
    llm = FakeLLM([json.dumps(VALID_LLM_PAYLOAD)])

    card = parse_product_record(PAGE_RECORD, llm=llm, use_structured_output=False)

    assert card.extraction_warnings == ["Navigation and footer content detected and ignored."]


def test_file_writing() -> None:
    tmp = _tmp_dir()
    input_dir = tmp / "parsed_products"
    output_dir = tmp / "catalog" / "products"
    input_dir.mkdir(parents=True)
    input_file = input_dir / "carta.json"
    input_file.write_text(json.dumps(PAGE_RECORD), encoding="utf-8")
    llm = FakeLLM([json.dumps(VALID_LLM_PAYLOAD)])

    try:
        result = parse_product_file(input_file, output_dir=output_dir, llm=llm, force=True)
        assert result.output_file is not None
        payload = json.loads(Path(result.output_file).read_text(encoding="utf-8"))
        assert payload["name"] == "BACARDI Carta Blanca"
        assert payload["source_url"] == PAGE_RECORD["source_url"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_parse_directory_writes_catalog_summary() -> None:
    tmp = _tmp_dir()
    input_dir = tmp / "parsed_products"
    output_dir = tmp / "catalog" / "products"
    input_dir.mkdir(parents=True)
    (input_dir / "listing.json").write_text(
        json.dumps({**PAGE_RECORD, "source_url": "https://www.bacardi.com/our-rums/"}),
        encoding="utf-8",
    )
    (input_dir / "carta.json").write_text(json.dumps(PAGE_RECORD), encoding="utf-8")
    llm = FakeLLM([json.dumps(VALID_LLM_PAYLOAD)])

    try:
        summary = parse_directory(input_dir, output_dir, llm=llm, limit=1, force=True)
        assert len([result for result in summary.results if not result.skipped]) == 1
        assert (tmp / "catalog" / "catalog_summary.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_carta_blanca_fixture_extraction() -> None:
    llm = FakeLLM([json.dumps(VALID_LLM_PAYLOAD)])

    card = parse_product_record(PAGE_RECORD, llm=llm, use_structured_output=False)

    assert card.name == "BACARDI Carta Blanca"
    assert card.category == "white rum"
    assert card.nose == "Almonds and fruit"
