# Независимая feedback-аналитика

Контур классифицирует каждый пользовательский запрос для статистики и не
участвует в работе основного агента.

Значения: `neutral`, `purchase_intent`, `negative_feedback`.
`negative_feedback` означает только критику ответа или поведения бота.
Негативное мнение о продукте, вкусе или коктейле остаётся `neutral`.
При смешанном сигнале действует приоритет
`negative_feedback > purchase_intent > neutral`.

После валидации `TurnResolution` classifier получает текущий `user_request`,
`follow_up` и последний полный ответ assistant только при `follow_up=true`.
Предыдущий ответ читается из `messages` внутри узла и не сохраняется в
`AgentState`. `FeedbackResult` не передаётся resolver, tool-agent, profile
updater или answer generator.

Classifier делает один structured LLM-вызов. При ошибке событие не создаётся,
`neutral` не подставляется, основной ход продолжается без нового `state.errors`.

Сохранение выполняется отдельной SQLite-транзакцией:

```text
persist    -> persist_feedback -> END
safe_error -> persist_feedback -> END
```

`turn_success=true` только после успешного основного `persist`. Повторный
`turn_id` не создаёт дубль. Ошибка аналитической записи лишь логируется.

Статистика доступна через:

```http
GET /api/analytics/feedback
GET /api/analytics/feedback?session_id=<id>
```

Ответ содержит общий счётчик, три label-счётчика и число successful/failed
turns. Endpoint пока является debug-интерфейсом без отдельной авторизации.
