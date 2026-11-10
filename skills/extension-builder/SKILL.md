---
name: extension-builder
description: Use when the user asks for a capability you do not have, or wants a reusable workflow, and you should propose a new tool or skill for them to approve.
tools: propose_tool, propose_skill, list_extensions
---
# Proposing new tools and skills

Everything you create is saved as **pending**. It is not importable or visible to you until the user reviews the source
in the Approvals panel and approves it. After proposing, tell the user clearly what you created and why, then stop;
do not try to call it. Never claim a pending tool or skill is available.

## When to create what
- **Tool** = code that does something (fetch data from a URL, transform data, compute, write rows).
- **Skill** = instructions for how/when to do a recurring task, possibly combining existing tools.
- Prefer an existing tool or skill if one fits (`list_extensions`). Prefer a skill if no new code is needed.
- To CHANGE an approved tool or skill, propose it again under the SAME name. The user sees it as a replacement and compares it
  with the current version; the old one keeps working until they approve, then it is archived. The user can also retire an approved
  tool or skill themselves in the Approvals panel; you cannot.
- Keep each tool small and single-purpose.
- NEVER simulate, mock or hard-code data. A tool must fetch or compute real results. If that is impossible under the rules below, tell the user instead of proposing it.

## Tool pattern (mandatory)
Pass `name`, `description`, `rationale` and `code`. The `code` must define exactly:

```python
class Params(BaseModel):
    symbol: str = Field(description="Ticker symbol, e.g. AAPL")

def run(params: Params, ctx: ToolContext) -> dict:
    ...
    return {"ok": True}   # JSON-serialisable
```
`BaseModel`, `Field` and `ToolContext` are already imported. Rules enforced by a lint (violations are rejected):
- Allowed imports: pydantic, typing, json, re, math, datetime, decimal, statistics, collections, itertools, functools,
  csv, io, textwrap, urllib.parse, html, hashlib, base64, uuid, enum, string.
- No `open`, `eval`, `exec`, `getattr`, no underscore-prefixed attributes, no other imports. No files, no subprocesses.
- HTTP only via `ctx.http.get(url, params=..., headers=...)` / `ctx.http.post(url, json=...)`. The response has
  `.status`, `.text` and `.json()`. Local/private addresses are blocked.
- Database only via `ctx.db`: `query(sql, params)`, `describe(table)`, `list_tables()`, `get_spec(model)` (columns, descriptions,
  `references`), `profile_table(table)`, `find_relationships(table)`, `insert(model, rows, on_conflict)`,
  `update(model, values, where)`, `delete(model, where)`, `define_model(ModelSpec)`.
  Never write SQL for writes.
- Raise `ValueError("helpful message")` for expected failures. Do not catch and hide errors.
- Never hard-code secrets and NEVER make a token/password/API key a tool parameter: parameters travel through the chat. See "Secrets".

### Secrets (API keys, tokens, passwords)
The user keeps them in the server's environment (the `.env` file). You cannot read them and must never ask the user to paste one.
Declare which ones the tool needs and where they may be sent, at module level, and put a placeholder in a request HEADER:

```python
SECRETS = {"BITBUCKET_ACCESS_TOKEN": ["api.bitbucket.org"]}   # env var name -> exact hosts it may be sent to (https only)

def run(params: Params, ctx: ToolContext) -> dict:
    r = ctx.http.get("https://api.bitbucket.org/2.0/repositories/" + params.workspace,
                     headers={"Authorization": "Bearer {secret:BITBUCKET_ACCESS_TOKEN}"})
```
- `{secret:NAME}` inserts the value; `{basic:USERNAME:NAME}` inserts base64("USERNAME:value") for Basic auth
  (`"Authorization": "Basic {basic:" + params.username + ":BITBUCKET_APP_PASSWORD}"`; a username is not secret, so it may be a parameter).
- Placeholders work in headers only. The real value is filled in by the app when the request is sent: your code never sees it,
  and it is only sent to the declared hosts, including after redirects.
- Use the real variable name the user mentioned (e.g. BITBUCKET_ACCESS_TOKEN) and the API's real host. Say in the rationale which
  variables the user must set in `.env` (then restart OpenJanus). A missing variable gives a clear error naming it.
- The reviewer sees the declared secrets and hosts when approving.

### Tools that work with the user's own tables
Read their meaning at runtime with `ctx.db.get_spec("table")` instead of hard-coding it, and follow `references` for joins.
See the `table-onboarding` skill; the table must already be adopted.

### Tools that store data
1. Make sure the table exists via the `define_model` built-in tool BEFORE proposing the tool (see `sqlite-database` skill),
   so the tool only needs `ctx.db.insert("model_name", rows)`.
2. Parse the source into dicts whose keys match the model's field names. `insert` validates with Pydantic.
3. Use a unique/primary key plus `on_conflict="replace"` or `"ignore"` so the tool can be re-run safely.
4. Return counts (`inserted`, `skipped`) so the user can see what happened.

### Example
```python
class Params(BaseModel):
    base: str = Field("EUR", description="3-letter base currency")

def run(params: Params, ctx: ToolContext) -> dict:
    r = ctx.http.get("https://api.frankfurter.app/latest", params={"from": params.base})
    if r.status != 200:
        raise ValueError(f"API returned {r.status}")
    data = r.json()
    rows = [{"base": data["base"], "quote": q, "rate": v, "date": data["date"]} for q, v in data["rates"].items()]
    return ctx.db.insert("exchange_rate", rows, on_conflict="replace")
```

## Skill pattern
Pass `name` (kebab-case), `description` (ONE line saying *when* to use it), `instructions` (markdown) and `tools`
(names of tools it uses). Structure the instructions as: purpose, step-by-step workflow, rules/edge cases, example.
Do not include frontmatter; it is generated.
