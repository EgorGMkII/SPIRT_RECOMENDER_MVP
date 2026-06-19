"""Prompt template for LLM-based Bacardi product extraction."""

PRODUCT_EXTRACTION_PROMPT = """
You extract structured Bacardi rum product data from noisy webpage text.

Return only schema-compatible JSON. Do not wrap the JSON in Markdown.
Do not hallucinate. Extract only information supported by the page text.
Prefer null, empty strings, or empty arrays over guessing.

Never invent ABV, age statement, country, price, barrel type, awards, or
ingredients unless explicitly present in the provided page text.

Ignore navigation, footer, legal text, cookie text, newsletter blocks, social
media blocks, and unrelated related-product lists. Preserve useful marketing
copy about the current product.

Separate marketing_description from tasting_notes:
- marketing_description is broad product positioning and descriptive copy.
- tasting_notes, nose, palate, and finish are sensory details from tasting
  sections.

Required JSON object fields:
- product_id: stable lowercase slug for this product
- source_url
- brand
- name
- title
- category
- short_description
- marketing_description
- tasting_notes
- nose
- palate
- finish
- process
- how_to_serve
- cocktail_names
- recommended_rums
- faq_items
- raw_text_excerpt
- source_metadata
- extraction_confidence
- extraction_warnings

faq_items must be a list of objects with "question" and "answer".
extraction_confidence must be a number from 0 to 1.
extraction_warnings must list ambiguity, noisy pages, or incomplete sections.

Few-shot example 1:
INPUT TEXT:
BACARDI Carta Blanca
A sublime rum for cocktails.
Tasting Notes
Nose Almonds and fruit
Palate Smooth and creamy
Finish Dry, clean, fresh
Filtered to perfection
This distinctive spirit is aged in American white oak barrels and shaped through
a secret blend of charcoal for a distinctive smoothness.
The Perfect Mixer
BACARDI Carta Blanca is a light and aromatic white rum with delicate floral and
fruity notes, ideal for mixing.
OUTPUT JSON:
{
  "name": "BACARDI Carta Blanca",
  "category": "white rum",
  "marketing_description": "A sublime rum for cocktails.",
  "nose": "Almonds and fruit",
  "palate": "Smooth and creamy",
  "finish": "Dry, clean, fresh",
  "process": "This distinctive spirit is aged in American white oak barrels and shaped through a secret blend of charcoal for a distinctive smoothness.",
  "how_to_serve": "Ideal for mixing.",
  "extraction_warnings": []
}

Few-shot example 2:
INPUT TEXT:
OUR RUMS COCKTAILS FAQ BACARDI Spiced
A rum with vanilla and cinnamon notes.
Nose Vanilla and cinnamon
Palate Sweet spices
Finish Black pepper and fudge
ABOUT US CONTACT US COOKIE POLICY
OUTPUT JSON:
{
  "name": "BACARDI Spiced",
  "marketing_description": "A rum with vanilla and cinnamon notes.",
  "nose": "Vanilla and cinnamon",
  "palate": "Sweet spices",
  "finish": "Black pepper and fudge",
  "extraction_warnings": ["Navigation and footer content detected and ignored."]
}
""".strip()


def build_product_extraction_prompt(page_payload: str) -> str:
    """Build the final extraction prompt for a page payload."""

    return f"{PRODUCT_EXTRACTION_PROMPT}\n\nPAGE PAYLOAD:\n{page_payload}\n\nReturn JSON only."
