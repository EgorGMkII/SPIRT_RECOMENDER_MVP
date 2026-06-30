# AI Sommelier Assistant

AI-сомелье для рекомендаций рома и коктейлей на базе локального каталога
Bacardi. Оркестрация построена на контролируемом LangGraph workflow:
детерминированный Python-код отвечает за retrieval, валидацию, память,
профиль и persistence, а LLM разрешает контекст хода, выбирает разрешённый
tool и формулирует ответ по проверенным карточкам.

## Возможности

- рекомендации рома по вкусу и назначению;
- подбор рома к блюду;
- поиск коктейлей по стилю и ингредиентам;
- рецепты и пояснения по ранее показанным объектам;
- session-корзина товаров с количеством;
- компактная turn-based память и пользовательский профиль;
- trace каждого tool call.

## Runtime flow

```text
START
  -> load memory/profile
  -> resolve and validate turn
  -> apply optional profile patch to working copy
  -> tool-calling agent
       <-> search_products
       <-> search_products_for_food
       <-> search_cocktails
       <-> lookup_by_id
       <-> add_cart / dellete_cart / show_cart
  -> generate and validate answer
  -> build TurnMemory
  -> atomically persist memory/profile/transcript/traces to SQLite
  -> END
```

`follow_up` не является отдельной веткой graph. Resolver использует его только
для связи нового сообщения с одним из сохранённых ходов. Один пользовательский
запрос допускает не более двух tool calls, по одному call в `AIMessage`.

## Память

`SessionMemory` хранит последние 12 `TurnMemory`. Каждый успешный ход содержит:

- исходный `user_request`;
- `follow_up`, `initial_request` и позитивный `effective_request`;
- временный `negative_request`;
- короткий `assistant_summary`;
- до трёх упорядоченных `shown_results` с `{kind, id, name, summary}`.

Также `SessionMemory.cart` хранит позиции `{id, amount}`. В корзину добавляются
только product IDs.

Полные catalog cards в `SessionMemory` не сохраняются. Новый поиск возвращает
их на текущий ход, а `lookup_by_id` загружает полную карточку ранее показанного
объекта. Memory, profile, полный transcript и traces атомарно сохраняются в
`data/sommelier.sqlite3` только после успешного финального ответа. Transcript
используется web-интерфейсом и не передаётся агенту.

## Retrieval

- продукты: гибридный FAISS/BM25 retrieval;
- сочетания с едой: детерминированное расширение food query и поиск продуктов;
- коктейли: отдельный BM25-индекс;
- негативные ограничения не добавляются в search query, а учитываются при
  выборе финального ответа.

Ingestion, catalog cards, search profiles и indexes являются отдельными
артефактами. Команды их построения находятся в `sommelier/ingestion`,
`sommelier/catalog` и `sommelier/retrieval`.

## Запуск

```powershell
python -m uvicorn sommelier.web.app:app --reload
```

Web UI: `http://127.0.0.1:8000/`.

При Docker-запуске runtime SQLite хранится в named volume
`sommelier-runtime`, а не в Windows bind mount `./data`. Путь задаётся через
`SOMMELIER_DB_PATH=/app/runtime/sommelier.sqlite3`.

Для ручной проверки диалогов используется
`notebooks/dialog_testing.ipynb`.

## Тесты

```powershell
pytest
```

Тесты используют fake LLM и локальные retrieval doubles; реальные model calls
и интернет не требуются.
