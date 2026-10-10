# rabt-shared

The API contract between the Rabt **AI engine** (owner: Awaiz) and the **integration backend** (owner: Huda).

| Path | What |
|---|---|
| `rabt_shared/` | Pydantic v2 models — the source of truth |
| `jsonschema/` | Generated JSON Schemas + `tool_catalog.json` for non-Python consumers (e.g. TypeScript) |
| `fixtures/` | Example request/response payloads for the tuition-booking flow (`manifest.json` maps each file to its schema) |

Human-readable spec: [`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md).

```bash
# Python consumers
uv pip install -e shared        # or: pip install -e shared
python -c "from rabt_shared import AgentRespondRequest; print(AgentRespondRequest.model_json_schema()['title'])"

# Regenerate JSON Schemas after changing a model (CI/tests fail if they are stale)
uv run python -m rabt_shared.jsonschema_export

# TypeScript types (optional)
npx json-schema-to-typescript shared/jsonschema/agent_response.schema.json > agent_response.d.ts
```

Changing a model is a contract change: follow the process in [`CLAUDE.md`](../CLAUDE.md#contract-change-process).
