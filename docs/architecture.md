# Архитектура turn-based агента

## Статус

Это основной source of truth для работающего runtime.

## Цель

Хранить память как последовательность компактных пользовательских ходов.
Убрать `ActiveContext`, relation routing и несколько параллельных коллекций
catalog references.

Агент должен уметь:

- понять, продолжает ли сообщение предыдущую задачу;
- восстановить полный актуальный запрос;
- найти новые продукты или коктейли;
- понять ссылки «первый», «второй», «последний» по упорядоченным результатам
  предыдущего ответа;
- загрузить полные карточки выбранных ранее объектов через `lookup_by_ids`;
- добавить, удалить или показать product IDs в session-корзине;
- сохранить компактную запись нового хода.

## Полный graph

```text
START
  -> load_memory_and_profile
  -> resolve_turn
  -> validate_turn_resolution
  -> classify_feedback
  -> apply_optional_profile_patch
  -> tool_calling_agent

tool_calling_agent
  -> tool call есть -> execute_tool -> tool_calling_agent
  -> tool call нет и есть evidence -> generate_answer
  -> tool call нет и evidence нет -> generate_soft_answer

generate_answer
  -> validate hard FinalAnswerResult
  -> build_turn_memory
  -> persist
  -> persist_feedback
  -> END

generate_soft_answer
  -> validate soft FinalAnswerResult
  -> build_turn_memory
  -> persist
  -> persist_feedback
  -> END
```

В graph нет:

- `ActiveContext`;
- route по `new_request`, `follow_up`, `profile_update`, `other`;
- `last_retrieval_refs`;
- предварительной загрузки полных карточек из памяти;
- отдельного `dialog_summary`.

## Durable memory

```python
class CatalogRef(BaseModel):
    kind: Literal["product", "cocktail"]
    id: str


class ShownResult(CatalogRef):
    name: str
    summary: str


class CartItem(BaseModel):
    id: str
    amount: int


class TurnMemory(BaseModel):
    follow_up: bool
    user_request: str
    initial_request: str
    effective_request: str
    negative_request: str | None
    assistant_summary: str
    shown_results: list[ShownResult]


class SessionMemory(BaseModel):
    schema_version: Literal[4] = 4
    session_id: str
    turns: list[TurnMemory]
    cart: list[CartItem]
```

Правила:

- `turns` хранит последние 12 успешных ходов;
- `shown_results` сохраняет порядок объектов в конкретном ответе;
- `cart` хранит product IDs и количество без полных карточек и цен;
- в одном ходе не более десяти `shown_results`; prompt обычно ограничивает
  пользовательский ответ четырьмя catalog objects, но память не должна резать
  уже показанное;
- полные product/cocktail cards в session JSON не сохраняются;
- неуспешный ход не добавляется в `turns`;
- `UserProfile` хранится отдельно, как сейчас.

## TurnResolution

Первый structured LLM получает:

- текущий сырой `user_request`;
- последние 12 `TurnMemory`;
- `UserProfile`.

Возвращает:

```python
class TurnResolution(BaseModel):
    follow_up: bool
    initial_request: str
    effective_request: str
    negative_request: str | None
    cart_action: Literal["add", "delete", "show"] | None
    profile_patch: ProfilePatch | None
    reasoning_note: str
```

Правила нового запроса:

```text
follow_up = false
initial_request = текущий user_request
effective_request = самостоятельная позитивная формулировка текущего запроса
negative_request = временные запреты текущего запроса или null
```

Правила продолжения:

```text
follow_up = true
initial_request = initial_request связанного предыдущего хода
effective_request = полный обновлённый позитивный запрос
negative_request = полный обновлённый набор временных запретов или null
```

`profile_patch` независим от `follow_up`. Resolver:

- не выбирает tool;
- не создаёт search query;
- не выбирает catalog ID;
- не отвечает пользователю.

`cart_action` фиксирует запрошенную операцию, но не выбирает tool arguments.
Python валидирует явные cart-команды и не позволяет tool loop завершиться без
успешного соответствующего cart tool.

Pure profile update и smalltalk используют ту же линейную ветку. Tool-calling
agent просто не вызывает search tool, если каталог не нужен.

## Контекст tool-calling agent

Агент получает один JSON-блок:

```text
current user request
TurnResolution
последние TurnMemory
текущая cart
UserProfile
tool results текущего хода
remaining tool budget
```

Порядок прошлых вариантов задаётся только порядком `shown_results` внутри
соответствующего `TurnMemory`:

```json
{
  "assistant_summary": "Предложены Mojito и Old Cuban.",
  "shown_results": [
    {"kind": "cocktail", "id": "mojito", "name": "Mojito", "summary": "..."},
    {"kind": "cocktail", "id": "old-cuban", "name": "Old Cuban", "summary": "..."}
  ]
}
```

Для запроса «как приготовить первый из последних вариантов?» модель определяет:

```text
последний подходящий TurnMemory
-> shown_results[0]
-> {kind: cocktail, id: mojito}
  -> lookup_by_ids(kind="cocktail", ids=["mojito"])
```

Полные карточки прошлых ходов заранее в prompt не загружаются.

## Tools

Доступны четыре read-only catalog tools и три cart tools:

```text
search_products
search_products_for_food
search_cocktails
lookup_by_ids
list_catalog
add_cart
dellete_cart
show_cart
```

### Search tools

Контракты трёх search tools остаются текущими. Они возвращают полноценные
answer-safe cards.

Search query строится только из позитивного `effective_request`.
`negative_request` не передаётся в BM25/FAISS и учитывается при выборе ответа.

Executor автоматически удаляет из search output references из
`shown_results` двух последних сохранённых ходов. Модель не передаёт excluded
IDs.

### list_catalog

Для явных запросов полного списка resolver устанавливает
`request_scope="catalog_listing"`, после чего agent вызывает
`list_catalog(kind)`. Tool возвращает весь локальный каталог в компактном виде
`{kind,id,name}` без retrieval и полных карточек. Final answer перечисляет все
полученные имена, но не считает их рекомендациями: `shown_refs=[]`, а в
память не добавляются `shown_results`.

### lookup_by_ids

```python
class LookupByIdsInput(BaseModel):
    kind: Literal["product", "cocktail"]
    ids: list[str]  # 1..10


class LookupByIdsOutput(BaseModel):
    cards: list[ProductCandidate | CocktailCandidate]
    rejected_ids: list[str] = []
```

Правила:

- tool выполняет точный локальный lookup, а не retrieval;
- каждый `{kind, id}` должен присутствовать в `shown_results` памяти либо в
  cards текущего хода;
- если часть ids недоступна, executor отбрасывает её и возвращает
  `rejected_ids`;
- если все ids недоступны, возвращается безопасная tool error;
- tool не изменяет память и профиль;
- возвращаются полные карточки доступных refs в порядке запроса.

Максимум два model tool calls на пользовательский ход суммарно. Search tool уже
возвращает полные карточки, поэтому lookup после search обычно не нужен.

### Cart tools

```python
add_cart(id: str, amount: int = 1)
dellete_cart(id: str)
show_cart()
```

`add_cart` принимает только product ID из показанных результатов памяти или
текущего search output. Если ID ещё неизвестен, агент использует
`search_products`, затем `add_cart`. Повторное добавление увеличивает amount.
`dellete_cart` удаляет позицию целиком, `show_cart` возвращает текущие
`{id, amount}`.

Cart tools меняют только рабочую `SessionMemory`. Изменение становится durable
в общем узле `persist` после успешной генерации ответа; failed turn оставляет
корзину на диске без изменений.

`lookup_by_ids` не считается выполнением cart action. Если model пытается
ответить без обязательного cart tool, executor делает один retry с validation
feedback. После повторного отказа ход завершается safe error без сохранения.

## AgentState одного хода

```python
class AgentState(BaseModel):
    session_id: str
    turn_id: str
    user_request: str

    session_memory: SessionMemory
    user_profile: UserProfile
    turn_resolution: TurnResolution | None

    messages: list[BaseMessage]
    tool_call_count: int
    canonical_tool_calls: list[str]
    cards: list[ProductCandidate | CocktailCandidate]
    answer_mode: Literal["hard", "soft"]

    final_answer_result: FinalAnswerResult | None
    errors: list[str]
    tool_traces: list[ToolTrace]
```

`cards` содержит только полные карточки, возвращённые tools в текущем ходе.
Поле начинается пустым при каждом новом сообщении и не сохраняется на диск.

В state нет:

- `persist_artifacts`;
- `original_memory`, `original_profile`;
- `context_cards`, `available_cards`;
- `current_turn_refs`;
- `resolver_result`;
- `ActiveContext`.

LLM и stores передаются как graph dependencies/config, а не хранятся в state.

## Final answer и сохранение хода

Final answer generator получает:

- текущий `TurnResolution`;
- последние `TurnMemory`;
- summaries ранее показанных объектов;
- полные cards, возвращённые tools текущего хода;
- tool errors и food-pairing caveat;
- `UserProfile`.

Hard final answer возвращает:

```python
class FinalAnswerResult(BaseModel):
    answer: str
    shown_refs: list[CatalogRef]
    assistant_summary: str
```

Python валидирует `shown_refs`:

- reference присутствует в cards текущего хода;
- утверждения, требующие полной карточки, разрешены только после search/lookup;
- schema допускает до 10 refs, prompt обычно просит не более четырёх объектов;
- kind и id существуют в каталоге.

Soft final answer используется, когда tool-agent не вызвал tool и evidence
cards/tool output нет. Он отвечает только по памяти/summary/recent dialogue,
всегда возвращает `shown_refs=[]`, не добавляет новых catalog facts и просит
пользователя уточнить, нужно ли подобрать/проверить по карточкам.

После успешной валидации Python строит:

```python
TurnMemory(
    follow_up=resolution.follow_up,
    user_request=state.user_request,
    initial_request=resolution.initial_request,
    effective_request=resolution.effective_request,
    negative_request=resolution.negative_request,
    assistant_summary=final.assistant_summary,
    shown_results=shown_results_from_shown_refs,
)
```

Новый `TurnMemory` добавляется в память только после успешного ответа. Memory и
profile сохраняются транзакционно.

## Traces

Минимальные события:

```text
load_memory_and_profile
resolve_turn
apply_profile_patch, если был patch
tool_call, для каждого search/lookup
generate_answer
generate_soft_answer, если включился memory-only fallback
persist или rollback
```

Prompts, API keys и полные секреты в traces не сохраняются.
