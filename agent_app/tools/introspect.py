"""Built-in, read-only tools that let the agent inspect itself and its database."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..introspection import audit_capabilities as _audit_caps, audit_database as _audit_db
from ..sdk import AgentContext, tool


class ListCapParams(BaseModel):
    kind: Literal["all", "tools", "skills"] = "all"
    name: str | None = Field(None, description="Give one tool or skill name to also get its full parameters/instructions")


@tool("list_capabilities",
      "List EVERYTHING you can do: built-in and generated tools and skills (active and pending), with origin, capabilities, "
      "how often each tool was used, and which skills reference it. Pass `name` for one item's full parameter list.")
def list_capabilities(params: ListCapParams, ctx: AgentContext) -> Any:
    cat = ctx.agent.catalog()
    if params.name:
        for t in cat["tools"]:
            if t["name"] == params.name:
                return {"kind": "tool", **t}
        for s in cat["skills"]:
            if s["name"] == params.name:
                body = ctx.skills.all().get(s["name"])
                return {"kind": "skill", **s, "instructions": body.body if body else None}
        raise ValueError(f"No tool or skill named '{params.name}'")
    out: dict[str, Any] = {"load_problems": cat["load_problems"]}
    if params.kind in ("all", "tools"):
        out["tools"] = [{k: v for k, v in t.items() if k != "params"} | {"param_names": [p["name"] for p in t["params"]]}
                        for t in cat["tools"]]
    if params.kind in ("all", "skills"):
        out["skills"] = cat["skills"]
    return out


class NoParams(BaseModel):
    pass


@tool("audit_capabilities",
      "Review your own tools and skills for overlap/redundancy, skills that reference missing tools, generated tools that are "
      "never used or poorly described, and an overly long tool list. Read-only: returns findings with suggestions, changes nothing.")
def audit_capabilities(params: NoParams, ctx: AgentContext) -> Any:
    return _audit_caps(ctx.agent.catalog())


@tool("audit_database",
      "Review the database for unmodelled tables, drift between models and tables, orphaned references, missing primary keys, "
      "near-duplicate tables, empty/all-NULL/constant columns, undocumented models and unindexed references. Read-only.")
def audit_database(params: NoParams, ctx: AgentContext) -> Any:
    return _audit_db(ctx.db)
