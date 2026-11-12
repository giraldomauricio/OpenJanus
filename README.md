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

**Query results.** When a tool returns a table (`{columns, rows}`, e.g. `query`) with 2+ rows, the chat shows a preview with
**Download CSV** and, only when it makes sense, **Chart**. The CSV is built in the browser from the rows the tool returned (up to 1000;
a notice appears if the query hit its limit) and neutralises spreadsheet formulas (`=`, `+`, `-`, `@`). A chart is offered when there
is one point per row and a numeric column: dates -> line, labels (<= 50, unique) -> bar, two numeric columns -> scatter. Lists of
names, ids, single values and free text get no chart. Charts are dependency-free SVG (`renderChart` in `index.html`).

**Schema changes.** `alter_model` adds, drops, renames and changes fields (type, required, unique, default, references) and
`convert_column` turns TEXT columns into integer / number / boolean / date / datetime. Both are **dry-run by default**: the change is
executed in a transaction and rolled back, so the report (`plan`, `problems`, `failures`, before/after `sample`) is exact; `apply: true`
repeats it for real after taking a backup (`data/backups/`, newest 10 kept). Conversions are strict: values that do not parse are listed
(use `date_format` for non-ISO dates, `on_error: "null"` to opt into data loss). Tables made by `define_model` are rebuilt when needed
(constraints and foreign keys kept); your own adopted tables only get SQLite's native add/rename/drop column and a column swap for
`convert_column` (the column moves to the end), never a rebuild. The agent is told to ask you before applying.

**Stopping.** The Send button becomes **Stop** (or press Esc) while a reply is running. The model connection is dropped (even while a
reasoning model is silent), tools not yet started are skipped, the partial reply is kept, and the chat continues normally.

**Exports.** Query results offer a CSV of the rows returned; each table in the *Database* tab has *Download CSV* for the whole
table (streamed, UTF-8 with BOM, formula-looking text neutralised).

**Themes.** The header's theme menu switches between *Dark* (default), *Pastel light* and *Hacker* (terminal look). Themes are CSS
variable sets at the top of `agent_app/static/index.html` (`:root[data-theme="..."]`), saved in the browser's localStorage; the logo is `agent_app/static/logo.png` (recoloured per theme with a CSS filter); add a
new one by defining another variable block and an `<option>`.

## Security notes (read before approving tools)
- The lint is **not a sandbox**. Approved tools run in-process, so read the source before approving.
  The Approvals panel shows detected capabilities (`network`, `database-write`, ...).
- Generated tools only get `ctx.db` and `ctx.http`. `ctx.http` blocks loopback/private addresses; the host is resolved once and the connection is pinned to the checked IP (TLS still verifies the hostname), so DNS rebinding cannot swap it.
- The server binds to 127.0.0.1 and all state-changing requests need a per-process token embedded in the served page.

## Logs, token usage and prompt caching
- The server logs to the console **and** to `data/logs/openjanus.log` (rotating, 2 MB x 3), including uvicorn's own lines: each question
  (provider, model, size), every model call (time, tokens, cache hits), every tool call (name, outcome, time), approvals, package
  imports/exports, outbound HTTP (host and status only) and security events (bad tokens, blocked addresses, which secret *name* went
  to which host). Your messages, answers, tool results and secret values are never logged; the text of an *error* is, and it can name a
  table or column. `OPENJANUS_LOG_LEVEL=DEBUG` also logs tool arguments (truncated); `OPENJANUS_LOG_FILE=0` turns the file off.
- Under each answer the chat shows the tokens used for that question (`11.4k in (11.4k from cache, 99%) · 49 out · 2 model calls`).
- **Prompt caching** (`OPENJANUS_PROMPT_CACHE`, on by default): Claude requests carry a cache marker, so the tool list and system prompt
  (about 11k tokens) and the earlier conversation are re-read at a fraction of the price on every step of a tool loop. Gemini caches
  repeated prefixes on its own and Ollama reuses its prompt state; only Claude and Gemini report how much was served from cache.
  Caches last about 5 minutes, and approving or retiring a tool or skill refreshes the tool list once.

## Known limits
- No composite keys or min/max constraints yet. Model changes go through `alter_model` / `convert_column` (see *Schema changes*).
- Tables are read-only to the agent until adopted. Tables whose columns are all TEXT (e.g. sqlite `.import`) stay string-typed
  after adoption until you ask for `convert_column`.
- `references` on adopted tables are logical metadata (SQLite cannot add foreign keys to an existing table); on tables made by
  `define_model` they are real, enforced foreign keys.
- `find_relationships` scans every table pair and is capped at 2000 comparisons, so it is meant for small/medium local databases.
- Chats are saved after each reply and restored on restart (the header *History* menu; the newest opens on load). Each saved chat
  remembers its last provider/model (restored when you reopen it) and every message is captioned with the model that answered.
  The 🗑 button deletes the current chat. Gemini's thought signatures are kept within a run but not saved, so a resumed Gemini chat
  is rebuilt from text and tool calls.
- Replies stream token by token (`stream()` in each provider; providers that only implement `generate()` fall back to one chunk).
  Only the final text is saved. Claude and Gemini streaming is covered by tests with faked clients, not by live API calls.
- Small local models may need several tries to write a valid tool; lint errors are fed back to them.

## Customising the UI (`ui.json`)

Copy `ui.example.json` to `ui.json` (or point `OPENJANUS_UI_CONFIG` at another file). The page picks up edits within ~3 seconds.
All keys are optional: `theme` (dark/light/hacker), `colors` (bg, panel, line, text, muted, accent, ok, bad, warn, code_bg, user_bg,
on_accent), `fonts` (body, mono, size), `radius`, `topbar` (visible, title, show = list of items, in display order),
`chat.placeholder`, `sidebar` (position, width, tabs, default_tab). Invalid values are ignored and listed in a banner under the top
bar. The Approvals tab can't be hidden.
