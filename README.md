# OpenJanus

A browser-based agent that works with a local SQLite database and can propose its own tools and skills.
Runs on Claude, Gemini, or a local Ollama model (Qwen, Gemma, ...), selectable per chat.

## Run
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add the keys for the providers you want
.venv/bin/python -m agent_app # http://127.0.0.1:8000
.venv/bin/python -m pytest
```
Ollama: `ollama serve` and pull a tool-capable model (`ollama pull qwen3:8b`).

## How it fits together
| Piece | Where |
|---|---|
| Provider-neutral LLM layer | `agent_app/llm/` |
| Agent loop, tool loading | `agent_app/agent.py` |
| Tool pattern (`Params` + `run(params, ctx)`) | `agent_app/sdk.py` |
| Built-in tools: database / extension builder | `agent_app/tools/` |
| Built-in skills | `skills/*/SKILL.md` |
| Pydantic model specs -> DDL, validators, exportable source | `agent_app/db/models.py` |
| Pending/approved lifecycle | `agent_app/extensions/manager.py` |
| Self-inspection (catalog, overlap and database audits; read-only) | `agent_app/introspection.py`, `agent_app/tools/introspect.py`, `skills/self-review` |
| Lint for generated code | `agent_app/extensions/lint.py` |
| Chat history (JSON per chat in `data/sessions/`) | `agent_app/sessions.py` |

**Data model.** Tables are created only via `define_model` from a `ModelSpec`. Every insert/update is validated by a
Pydantic model built from that spec, and `export_models` returns Pydantic source, JSON Schema and DDL for migration.
Raw SQL from the agent is read-only (`mode=ro` connection + SQLite authorizer + timeout).

**Your own tables (CSV import and learning).** *Import CSV…* in the header creates a typed table (headers become
snake_case; ints/floats/dates are detected). Any table without a model is read-only for the agent until it is adopted:
`profile_table` (what is in each column) → `find_relationships` (likely foreign keys by value overlap + naming, in both
directions, with confidence) → `adopt_table` (registers a model; the data is untouched) → `annotate_model` (stores
descriptions and `references`). The `table-onboarding` skill drives this; tools can read the result at runtime via
`ctx.db.get_spec(...)`. Renaming columns/tables to fit snake_case only happens when the agent passes explicit renames
(the skill tells it to ask you first), and a rejected call changes nothing.

**Approval flow.** `propose_tool` / `propose_skill` write to `data/extensions/pending/`, which is never imported or
listed to the model. Only the UI's Approve button (token-protected endpoint) moves an item to `approved/`, recording
its SHA-256; a file edited after approval is not loaded. The agent has no tool that can approve.

**Self-review.** The `self-review` skill drives three read-only tools: `list_capabilities` (every tool and skill: origin, status,
parameters, usage, which skills reference it), `audit_capabilities` (overlapping tools/skills, skills that reference missing tools,
unused or poorly described generated tools, an overlong tool list) and `audit_database` (unmodelled tables, model/table drift,
orphaned references, missing keys, near-duplicate tables, empty/all-NULL/constant columns, unindexed references). The
**Capabilities** tab shows the same catalog and runs both audits. Findings are advisory: the agent cannot retire tools or alter
the schema; it reports and you decide. Overlap detection is lexical (name/description similarity), so treat it as a pointer.
"Never used" only counts chats saved since history was added.

**Themes.** The header's theme menu switches between *Dark* (default), *Pastel light* and *Hacker* (terminal look). Themes are CSS
variable sets at the top of `agent_app/static/index.html` (`:root[data-theme="..."]`), saved in the browser's localStorage; add a
new one by defining another variable block and an `<option>`.

## Security notes (read before approving tools)
- The lint is **not a sandbox**. Approved tools run in-process, so read the source before approving.
  The Approvals panel shows detected capabilities (`network`, `database-write`, ...).
- Generated tools only get `ctx.db` and `ctx.http`. `ctx.http` blocks loopback/private addresses (a DNS-rebinding gap remains).
- The server binds to 127.0.0.1 and all state-changing requests need a per-process token embedded in the served page.

## Known limits
- No schema migrations on existing models (define a new model and copy rows); no foreign keys/composite keys/min-max constraints yet.
- Tables are read-only to the agent until adopted. Tables whose columns are all TEXT (e.g. sqlite `.import`) stay string-typed
  after adoption (the agent is told to CAST in queries); there is no type-conversion tool yet.
- `references` on adopted tables are logical metadata (SQLite cannot add foreign keys to an existing table); on tables made by
  `define_model` they are real, enforced foreign keys.
- `find_relationships` scans every table pair and is capped at 2000 comparisons, so it is meant for small/medium local databases.
- Chats are saved after each reply and restored on restart (the header *History* menu; the newest opens on load). Gemini's
  thought signatures aren't persisted, so a resumed Gemini chat is rebuilt from text and tool calls. There is no delete button yet
  (`DELETE /api/sessions/{id}` exists). Small local models may need several tries to write a valid tool; lint errors are fed back to them.
