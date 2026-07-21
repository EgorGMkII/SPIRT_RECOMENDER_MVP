# LLM node contracts

Prompts are isolated in runtime modules, but their behavioral contracts are:

## Resolver

Structured output: `TurnResolution`.

```text
follow_up
request_scope
initial_request
effective_request
negative_request
cart_action
optional profile_patch
reasoning_note
```

`request_scope` is the current action:

```text
product | cocktail | recipe | food_pairing | cart | profile |
catalog_listing | conversation
```

Resolver does not choose tools, catalog ids or user-facing answers.

Rules:

- `effective_request` is compact positive normalization, not an answer plan.
- Temporary exclusions stay only in `negative_request`.
- Resolver must not add option counts, serving advice or explanation depth
  unless the user asked for them.
- Follow-up pronouns and ordinals resolve against saved `shown_results` from
  recent `TurnMemory`.
- For follow-up, Python owns durable `initial_request` normalization; the LLM
  may copy it, but validation can replace it with the saved root request.
- Explicit profile statements produce `profile_patch` and do not imply search.

## Tool-agent

The tool-agent uses native `bind_tools` and chooses one of four modes:

1. Search new candidates.
2. Look up already shown objects.
3. Cart action.
4. No tool for profile/smalltalk/clarification.

It receives the current request, `TurnResolution`, compact turns, profile/cart
state and previous tool messages. It does not receive the full transcript.

Important constraints:

- use `search_cocktails` for new/narrowed cocktail candidates;
- use `lookup_by_ids` for recipes, comparisons, explanations and filtering
  already shown objects;
- use only saved `shown_results` or current cards for lookup ids;
- after successful search, do not immediately lookup the same fresh cards just
  to inspect them;
- never write tool names or JSON arguments as plain text; use native tool calls;
- never write user-facing recommendations/recipes from this node.

If the model returns plain text without tool calls, runtime sanitizes it to the
internal marker `"No tool needed."`.

## Hard final answer

Structured output: `FinalAnswerResult`.

```text
answer
shown_refs
assistant_summary
```

Hard mode is used when current cards/tool output exist.

Rules:

- catalog facts come only from current full cards and tool messages;
- `shown_refs` may contain only ids from current cards;
- `shown_refs` lists every catalog object explicitly named to the user, in
  first-mention order;
- schema allows up to 10 refs so memory does not forget objects already shown;
- prompt asks to normally show no more than 4 objects;
- catalog listing returns `shown_refs=[]`;
- recipe details require current cocktail card;
- cart changes require successful cart tool output.

If hard structured generation fails, runtime returns a safe soft fallback
instead of exposing a generic failure to the user.

## Soft final answer

Soft mode is used when tool-agent returned no tool call and no current evidence
cards/tool output exist.

Allowed sources:

- `TurnMemory` summaries and `shown_results`;
- recent dialogue for continuity;
- user profile/cart state.

Strict rules:

- always `shown_refs=[]`;
- do not introduce new catalog objects;
- do not provide recipes, ingredient quantities, prices, availability, ABV or
  exact catalog facts;
- ask one concise question that nudges the user toward clarification or a
  concrete lookup/search when evidence is needed.

Soft generation also has a safe fallback when structured output is invalid.

## Feedback classifier

Independent analytics-only structured output:

```text
neutral | purchase_intent | negative_feedback
```

`negative_feedback` means criticism of the bot answer/behavior only. Negative
opinions about products/tastes are neutral. If structured output is invalid,
deterministic fallback rules classify the message without affecting the agent
answer.
