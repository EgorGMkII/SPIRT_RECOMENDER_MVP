# Текущий runtime

```text
START
  -> load_memory_and_profile
  -> resolve_turn
  -> validate_turn_resolution
  -> classify_feedback
  -> apply_optional_profile_patch
  -> tool_calling_agent
  <-> execute_tool
  -> generate_answer
  -> build_turn_memory
  -> persist
  -> persist_feedback
  -> END
```

Graph линейный: routing по relation и `ActiveContext` отсутствуют.

На каждый user request создаётся новый `AgentState`. Его `cards` содержит только
полные карточки, возвращённые tools в этом ходе, и после хода уничтожается.

Durable memory — последние 12 `TurnMemory`. Единственные persisted catalog
references находятся в упорядоченных `shown_results`.

Runtime persistence находится в `data/sommelier.sqlite3`:

- `sessions` хранит компактные `SessionMemory` и `UserProfile`;
- `messages` хранит полный transcript только для web UI;
- `traces` хранит технические события.

Успешный ход записывает все три части одной SQLite-транзакцией. Failed turn
ничего не сохраняет.
