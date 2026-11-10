"""Built-in tools for creating new tools and skills. Everything created here is PENDING until the user approves it."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..extensions import ProposalError
from ..sdk import AgentContext, tool


class ProposeToolParams(BaseModel):
    name: str = Field(description="snake_case tool name, e.g. fetch_exchange_rates")
    description: str = Field(description="What the tool does and when to call it (10-400 chars)")
    code: str = Field(description="Python source defining `class Params(BaseModel)` and `def run(params, ctx)`. "
                                  "Imports of BaseModel/Field/ToolContext are already provided.")
    rationale: str = Field(description="Why the user needs this tool; shown to the reviewer")


@tool("propose_tool",
      "Propose a new tool. It is saved as PENDING and cannot be used until the user approves it in the Approvals panel. "
      "Load the 'extension-builder' skill first. Contract for `code`: define `class Params(BaseModel)` (fields with "
      "Field(description=...)) and `def run(params: Params, ctx: ToolContext) -> dict`. BaseModel, Field, ToolContext are "
      "pre-imported. Network ONLY via ctx.http.get(url, params=, headers=) / ctx.http.post(url, json=) (response has "
      ".status/.text/.json()); `requests`/`httpx`/`os` imports are rejected. Database ONLY via ctx.db.query(sql, args), "
      "ctx.db.insert(model_name, rows, on_conflict), ctx.db.update(...), ctx.db.delete(...). Create the table first with "
      "the define_model tool. NEVER simulate, mock or hard-code data in a tool: if the real source cannot be reached "
      "within these rules, tell the user instead of proposing it.")
def propose_tool(params: ProposeToolParams, ctx: AgentContext) -> Any:
    try:
        r = ctx.extensions.propose_tool(params.name, params.description, params.code, params.rationale)
    except ProposalError as e:
        raise ValueError(str(e)) from e
    note = ("Saved as pending. Tell the user to review and approve it; you cannot call it until then.")
    if r.get("replaces_approved"):
        note += (" It REPLACES the currently approved tool of the same name: that version keeps working until the user approves this "
                 "one, which then takes over (the old one is archived).")
    return {**r, "note": note}


class ProposeSkillParams(BaseModel):
    name: str = Field(description="kebab-case skill name, e.g. currency-conversion")
    description: str = Field(description="One line: WHEN should this skill be loaded (10-300 chars)")
    instructions: str = Field(description="Markdown instructions: workflow, rules, examples")
    tools: list[str] = Field(default_factory=list, description="Names of tools this skill relies on")


@tool("propose_skill",
      "Propose a new skill (reusable instructions). Saved as PENDING until the user approves it. "
      "Read the 'extension-builder' skill first.")
def propose_skill(params: ProposeSkillParams, ctx: AgentContext) -> Any:
    try:
        r = ctx.extensions.propose_skill(params.name, params.description, params.instructions, params.tools)
    except ProposalError as e:
        raise ValueError(str(e)) from e
    return {**r, "note": "Saved as pending. It becomes available only after the user approves it."
            + (" It REPLACES the approved skill of the same name; the old one keeps working until then." if r.get("replaces_approved") else "")}


class LoadSkillParams(BaseModel):
    name: str


@tool("load_skill", "Load the full instructions of a skill listed in the system prompt.")
def load_skill(params: LoadSkillParams, ctx: AgentContext) -> Any:
    skills = ctx.skills.all()
    if params.name not in skills:
        raise ValueError(f"Unknown skill '{params.name}'. Available: {sorted(skills)}")
    s = skills[params.name]
    return {"name": s.name, "tools": s.tools, "instructions": s.body}


class NoParams(BaseModel):
    pass


@tool("list_extensions", "List only the GENERATED tools/skills by approval status (pending vs approved). Use list_capabilities for everything.")
def list_extensions(params: NoParams, ctx: AgentContext) -> Any:
    fields = ("kind", "name", "description", "capabilities")
    return {s: [{k: i.get(k) for k in fields} for i in ctx.extensions.list_items(s)]  # type: ignore[arg-type]
            for s in ("pending", "approved")}
