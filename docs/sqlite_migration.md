# SQLite runtime persistence

## Цель

Заменить набор runtime JSON/JSONL-файлов одной SQLite-базой, не меняя логику
агента:

- `AgentState` остаётся временным состоянием одного запроса;
- `SessionMemory`, `UserProfile`, `ToolTrace` и их Pydantic-контракты не
  меняются;
- graph, resolver, tools, prompts и retrieval не меняются;
- `session_id` продолжает создаваться браузером и храниться в `localStorage`;
- memory/profile/cart загружаются перед ходом;
- durable state записывается только после успешного финального ответа;
- полный текст чата хранится отдельно и не попадает в prompt агента.

SQLite-файл:

```text
data/sommelier.sqlite3
```

При deployment каталог `data/` должен находиться на persistent volume.
Путь можно переопределить переменной `SOMMELIER_DB_PATH`.

В Docker Desktop нельзя размещать WAL-базу в Windows bind mount
`./data:/app/data`: такой mount может не поддерживать необходимые SQLite
locking/SHM операции. Compose использует отдельный Linux named volume:

```text
SOMMELIER_DB_PATH=/app/runtime/sommelier.sqlite3
sommelier-runtime:/app/runtime
```

Bind mount `/app/data` остаётся для catalog/index artifacts.

## Что заменяет SQLite

Текущие артефакты:

```text
data/sessions/{session_id}.json
data/user_profiles/{session_id}.json
data/traces/{session_id}.jsonl
```

заменяются таблицами:

```text
sessions
messages
traces
feedback_events
```

Catalog, ingestion и retrieval artifacts в SQLite не переносятся:

```text
data/catalog/
data/indexes/
data/parsed_products/
data/parsed_cocktails/
```

## Схема базы

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    memory_json TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    UNIQUE (session_id, turn_id, role)
);

CREATE INDEX IF NOT EXISTS messages_session_order
ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS traces_session_order
ON traces(session_id, id);
```

`memory_json` содержит результат:

```python
state.session_memory.model_dump_json()
```

Включая:

```text
turns
cart
```

`profile_json` содержит:

```python
state.user_profile.model_dump_json()
```

Полный ответ `FinalAnswerResult.answer` сохраняется только в `messages` и не
добавляется в `SessionMemory`.

## Граница между памятью агента и историей интерфейса

### Агент читает

```text
sessions.memory_json
sessions.profile_json
```

### Web-интерфейс читает

```text
messages
```

### Агент не читает

```text
messages
traces
```

Таким образом, размер полного диалога не увеличивает LLM context. Ограничение
`SessionMemory.turns` остаётся прежним.

## Модули persistence

Добавить:

```text
sommelier/storage/
  __init__.py
  database.py
  session_repository.py
```

### database.py

Отвечает только за:

- путь к SQLite-файлу;
- создание connection;
- включение `foreign_keys`, WAL и `busy_timeout`;
- создание таблиц;
- управление короткими транзакциями.

Не содержит Pydantic-моделей и логики агента.

### session_repository.py

Предоставляет typed-функции:

```python
load_session_memory(session_id: str) -> SessionMemory
load_user_profile(session_id: str) -> UserProfile
load_messages(session_id: str) -> list[dict[str, str]]
load_trace_events(session_id: str) -> list[dict]

persist_successful_turn(
    *,
    session_id: str,
    turn_id: str,
    memory: SessionMemory,
    profile: UserProfile,
    user_message: str,
    assistant_message: str,
    traces: list[ToolTrace],
) -> None
```

Имена load-функций можно сохранить прежними, чтобы graph и debug API
потребовали минимальных изменений.

## Загрузка состояния перед ходом

Текущий node:

```text
load_memory_and_profile
```

сохраняет контракт и начинает читать SQLite:

```python
memory = repository.load_session_memory(state.session_id)
profile = repository.load_user_profile(state.session_id)
```

Если строки `sessions` ещё нет:

```python
memory = SessionMemory(session_id=session_id)
profile = UserProfile(session_id=session_id)
```

Создавать строку в БД на этом этапе необязательно. Она будет создана после
первого успешного ответа.

## Сохранение успешного хода

Нельзя сохранять `AgentState` после каждого node. SQLite обновляется один раз в
конце успешного graph:

```text
generate_answer
-> validate
-> build_turn_memory
-> persist_successful_turn
-> END
```

Одна короткая SQL-транзакция должна:

1. создать либо обновить строку `sessions`;
2. записать новый `memory_json`;
3. записать новый `profile_json`;
4. добавить исходный `state.user_request` в `messages`;
5. добавить полный `state.final_answer_result.answer` в `messages`;
6. добавить traces текущего `turn_id`;
7. выполнить `COMMIT`.

Пример порядка:

```python
with connection:
    connection.execute(
        """
        INSERT INTO sessions (
            session_id, memory_json, profile_json,
            version, created_at, updated_at
        )
        VALUES (?, ?, ?, 1, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            memory_json = excluded.memory_json,
            profile_json = excluded.profile_json,
            version = sessions.version + 1,
            updated_at = excluded.updated_at
        """,
        (...),
    )

    connection.executemany(
        """
        INSERT INTO messages (
            session_id, turn_id, role, content, created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (...),
    )

    connection.executemany(
        """
        INSERT INTO traces (
            session_id, turn_id, trace_json, created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (...),
    )
```

Если любая операция падает, SQLite выполняет rollback всей транзакции. На
диске остаются согласованные memory, profile, cart, messages и traces прошлого
успешного хода.

`UNIQUE(session_id, turn_id, role)` защищает messages от повторной записи при
повторе одного и того же `turn_id`.

## Изменения существующих модулей

### memory_store.py

Удалён. Ограничение количества turns перенесено в `agent/memory.py`, чтение и
запись выполняет `SessionRepository`.

### profile_store.py

Удалён. Профиль читается и записывается через `SessionRepository`.

### trace_store.py

Удалён. Debug read работает с таблицей `traces`, append выполняется только
внутри общей транзакции успешного хода.

### graph.py

Меняется только persistence edge:

```python
persist_successful_turn(
    session_id=state.session_id,
    turn_id=state.turn_id,
    memory=state.session_memory,
    profile=state.user_profile,
    user_message=state.user_request,
    assistant_message=state.final_answer_result.answer,
    traces=state.tool_traces,
)
```

Resolver, tool loop и final answer generation не затрагиваются.

## Загрузка полного текста чата при перезагрузке

### Идентификатор

Текущий код остаётся:

```javascript
let sessionId = localStorage.getItem("sommelier_session_id");
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem("sommelier_session_id", sessionId);
}
```

При F5, закрытии и повторном открытии браузера используется прежний
`session_id`, пока пользователь не очистит данные сайта или не сменит
браузер/устройство.

### API

Добавить response contracts:

```python
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]
```

Добавить endpoint:

```text
GET /api/sessions/{session_id}/messages
```

Он выполняет:

```sql
SELECT role, content
FROM messages
WHERE session_id = ?
ORDER BY id;
```

Если session отсутствует или messages пусты:

```json
{
  "session_id": "...",
  "messages": []
}
```

Это нормальный ответ, а не 404.

### Frontend

При старте страницы:

```text
прочитать session_id из localStorage
-> GET /api/sessions/{session_id}/messages
-> очистить локальный контейнер сообщений
-> отрисовать messages по порядку
-> если messages пусты, показать стандартное приветствие
```

Пример:

```javascript
async function loadChatHistory() {
  const response = await fetch(
    `/api/sessions/${encodeURIComponent(sessionId)}/messages`
  );
  if (!response.ok) {
    throw new Error(`History request failed: ${response.status}`);
  }

  const payload = await response.json();
  messages.replaceChildren();

  if (payload.messages.length === 0) {
    appendMessage("assistant", DEFAULT_GREETING);
    return;
  }

  for (const message of payload.messages) {
    appendMessage(message.role, message.content);
  }
}
```

После отправки нового сообщения frontend продолжает немедленно добавлять user
message и полученный assistant answer в DOM. Повторно запрашивать всю историю
после каждого хода не нужно.

Приветствие интерфейса не сохраняется в `messages`, потому что это UI-текст, а
не ответ агента.

## Реализованные этапы

### Этап 1. Repository и schema

- добавить `database.py`;
- добавить `session_repository.py`;
- создать SQLite schema при startup или первом connection;
- написать unit-тесты repository во временной БД.

Runtime использует SQLite без JSON fallback.

### Этап 2. Read path

- переключить `load_session_memory`;
- переключить `load_user_profile`;
- переключить debug reads traces;
- проверить пустую и существующую session.

### Этап 3. Transactional write

- заменить JSON persistence одним `persist_successful_turn`;
- проверить rollback при ошибке каждой операции;
- проверить отсутствие messages у failed turn;
- проверить сохранение cart и profile вместе с memory.

### Этап 4. Chat history API

- добавить response schemas;
- добавить `GET /api/sessions/{session_id}/messages`;
- добавить API-тест пустой и заполненной истории.

### Этап 5. Frontend reload

- вынести приветствие в константу;
- добавить `loadChatHistory`;
- вызывать его при загрузке страницы;
- не добавлять приветствие поверх восстановленного диалога;
- проверить F5 и повторный запуск браузера с тем же `localStorage`.

## Тесты

Обязательные проверки:

- новая session возвращает пустые `SessionMemory`, `UserProfile` и messages;
- существующие memory/profile/cart корректно восстанавливаются;
- успешный ход атомарно сохраняет session, два messages и traces;
- failed turn ничего не записывает;
- повтор одного `turn_id` не дублирует messages;
- полный assistant answer есть только в `messages`, но отсутствует в
  `SessionMemory`;
- `GET messages` сохраняет порядок;
- reload frontend не создаёт новый `session_id`;
- agent prompt не получает строки из `messages`;
- полный `pytest` не использует сеть и реальный LLM.

## Ограничения SQLite

Эта схема рассчитана на один сервер приложения и persistent local disk.
SQLite подходит для текущего этапа и убирает разрозненные runtime JSON.

При нескольких экземплярах приложения с отдельными файловыми системами нужно
перейти на PostgreSQL. Pydantic-контракты и graph при этом останутся прежними:
заменится только реализация repository.
