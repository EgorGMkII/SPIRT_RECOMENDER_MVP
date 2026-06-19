"""Prompt builder for Bacardi cocktail recipe extraction."""

from __future__ import annotations


def build_cocktail_extraction_prompt(page_payload_json: str) -> str:
    """Build prompt for LLM extraction of a clean CocktailCard."""

    return f"""
You extract clean structured cocktail recipe cards from Bacardi cocktail pages.

Return exactly one JSON object matching this schema:
{{
  "cocktail_id": "stable lowercase id, usually URL slug",
  "source_url": "source URL",
  "brand": "Bacardi",
  "name": "cocktail name",
  "title": "page title or null",
  "main_rum": "primary Bacardi rum named by the page, or null if not explicit",
  "short_description": "1 concise informative sentence written by you from the page data",
  "marketing_description": "marketing or mood copy from the page, cleaned and concise, or null",
  "recipe": {{
    "servings": "serving count or null",
    "prep_time": "prep time or null",
    "difficulty": "difficulty or null",
    "ingredients": [
      {{"name": "ingredient name", "amount": "amount text or null"}}
    ],
    "steps": ["ordered preparation step"]
  }},
  "glassware": "glassware or null",
  "garnish": "garnish or null",
  "method": "shake/build/stir/blend/etc. or null",
  "raw_text_excerpt": "short relevant excerpt from source text",
  "source_metadata": {{}},
  "extraction_confidence": 0.0,
  "extraction_warnings": []
}}

Rules:
- Use the source page as the source of truth for recipe ingredients and steps.
- The short_description may be composed by you, but must not invent recipe facts.
- If the page does not explicitly name a main rum, set main_rum to null and add a warning.
- Exclude navigation, footer, legal, social, unrelated products, and generic menu text.
- Keep descriptions compact and useful for recommendation/retrieval.
- Preserve ingredient amounts exactly when present.
- Return JSON only, no markdown.

PAGE RECORD:
{page_payload_json}
""".strip()
