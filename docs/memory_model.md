# Turn-based memory

## Durable state

Session JSON содержит только:

```text
schema_version = 4
session_id
turns
cart
```

`turns` — последние 12 успешных ходов.
`cart` — список `CartItem {id, amount}` с product IDs.

```python
TurnMemory(
    follow_up: bool,
    user_request: str,
    initial_request: str,
    effective_request: str,
    negative_request: str | None,
    assistant_summary: str,
    shown_results: list[ShownResult],
)
```

`shown_results` содержит максимум десять объектов в порядке упоминания.
Промпт просит обычно показывать не больше четырёх вариантов, но память не
режет уже явно показанные пользователю объекты:

```json
{
  "kind": "cocktail",
  "id": "mojito",
  "name": "Mojito",
  "summary": "Коктейль с лаймом и мятой."
}
```

Других persisted списков catalog IDs нет. Полные карточки в память не
записываются.

Повторный `add_cart` увеличивает `amount` существующей позиции. Корзина не
хранит полные карточки, цены или availability.

## Смысл полей

- `user_request` — дословное сообщение текущего хода.
- `initial_request` — первый запрос связанной цепочки.
- `effective_request` — полный актуальный позитивный запрос.
- `negative_request` — временные запреты цепочки.
- `assistant_summary` — компактное summary ответа текущего хода.
- `shown_results` — только реально упомянутые catalog objects.

## Полные карточки

Search tools возвращают полные карточки в `AgentState.cards` текущего хода.
Для ранее показанного объекта agent вызывает `lookup_by_ids` по reference из
`shown_results`. `cards` не сохраняется на диск.

## Profile и transaction

`UserProfile` хранится отдельно. Profile patch и новый TurnMemory записываются
только после успешной валидации финального ответа. Schema ниже 4 начинается с
пустой conversational memory; профиль сохраняется.
