# Tool contracts

Runtime tools are narrow model tools over local state/catalog data. They do not
persist durable memory directly; persistence happens only after a successful
final answer.

## Catalog tools

Available catalog tools:

```text
search_products
search_products_for_food
search_cocktails
lookup_by_ids
list_catalog
```

Search tools return up to seven full answer-safe cards for the current turn.
Search output is deduplicated against `shown_results` from the last two saved
turns, so “ещё варианты” avoids immediate repeats.

### search_products

Search Bacardi rum products using a positive query.

```python
search_products(query: str, limit: int = 7)
```

Use for rum/product recommendations and choosing a rum for a named cocktail.

### search_products_for_food

Search rum products for edible food pairing.

```python
search_products_for_food(
    food_description: str,
    search_query: str,
    limit: int = 7,
)
```

Use only for dishes, ingredients, meals or cuisine. A named cocktail is not
food; choosing rum for Old Fashioned/Mojito/Daiquiri uses `search_products`.

### search_cocktails

Search cocktail recipes with a positive query.

```python
search_cocktails(query: str, limit: int = 7)
```

Use for new cocktail candidates or narrowed cocktail recommendations, including
“какие коктейли из X” and constraint follow-ups.

### lookup_by_ids

Load full local cards for already available references.

```python
lookup_by_ids(
    kind: Literal["product", "cocktail"],
    ids: list[str],  # 1..10
)
```

Allowed refs are:

- any `{kind,id}` in saved `TurnMemory.shown_results`;
- any `{kind,id}` already present in `AgentState.cards` for the current turn.

Lookup is local and deterministic; it does not run retrieval or mutate memory.
If some requested ids are not allowed, executor performs partial success:

- allowed ids are loaded in requested order;
- rejected ids are omitted;
- tool output includes `rejected_ids` and a caveat;
- only an entirely invalid lookup returns `lookup_ref_not_allowed`.

Use for recipes, explanations, comparisons, and property filtering over already
shown objects.

### list_catalog

Return the complete compact list of product or cocktail names.

```python
list_catalog(kind: Literal["product", "cocktail"])
```

This is catalog navigation, not recommendation. It returns ids/names only,
does not create full cards, and final answer must return `shown_refs=[]`.

## Cart tools

```text
add_cart:
  input:  {id: str, amount: int = 1}
  output: {action: "added", items: [{id, amount}]}

dellete_cart:
  input:  {id: str}
  output: {action: "deleted", items: [{id, amount}]}

show_cart:
  input:  {}
  output: {action: "shown", items: [{id, amount}]}
```

Cart accepts product IDs only. `add_cart` is allowed for a product shown in
saved memory or found in the current turn. Repeated add increases amount;
`dellete_cart` removes the item.

## Execution limits and validation

- Native model tool calling uses `bind_tools`.
- One tool call per AIMessage.
- Maximum two tool calls per user turn.
- Duplicate, unknown and over-budget calls are blocked.
- Tool errors are returned as safe `ToolMessage` envelopes.
- Tool-agent plain text is sanitized to an internal `"No tool needed."` marker
  so it cannot leak as a pseudo-answer into final generation.
