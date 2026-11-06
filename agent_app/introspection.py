"""Self-inspection: what OpenJanus can do, where its tools/skills overlap, and whether the database is healthy.

Everything here is read-only and advisory. Findings are {severity, area, kind, subjects, detail, suggestion}; the agent
reports them and the user decides, so nothing is changed or deleted automatically.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Any, Callable

from .db import Database
from .db.models import quote
from .skills import Skill, parse_skill

_STOP = set("a an the of to for and or in on with from by is are be as at this that it its use used when you your "
            "any all into via using can will if not no than then them they their".split())
_SEV = {"high": 0, "medium": 1, "low": 2, "info": 3}
LARGE_TOOLSET = 25  # beyond this, small local models start picking the wrong tool


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " ").replace("-", " "))
    # 5-letter prefixes act as a crude stem: technology/technologies/technical, detect/detector/detects all collapse
    return {w[:5] for w in words if w not in _STOP and len(w) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def finding(severity: str, area: str, kind: str, subjects: list[str], detail: str, suggestion: str = "") -> dict[str, Any]:
    return {"severity": severity, "area": area, "kind": kind, "subjects": subjects, "detail": detail,
            "suggestion": suggestion}


def _sorted(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(findings, key=lambda f: (_SEV[f["severity"]], f["area"], f["kind"], f["subjects"]))


def _summary(findings: list[dict[str, Any]]) -> dict[str, int]:
    return {s: sum(1 for f in findings if f["severity"] == s) for s in _SEV}


# ---- catalog ---------------------------------------------------------------------------------------------------

def build_catalog(tools: dict[str, Any], skills: dict[str, Skill], pending: list[dict[str, Any]],
                  pending_skill_source: Callable[[str], str], capabilities: dict[str, list[str]],
                  usage: dict[str, int], load_problems: list[str]) -> dict[str, Any]:
    """tools/skills = what is active right now; pending = extensions.list_items('pending')."""
    skills_using: dict[str, list[str]] = {}
    for s in skills.values():
        for t in s.tools:
            skills_using.setdefault(t, []).append(s.name)

    def params_of(t: Any) -> list[dict[str, Any]]:
        sch = t.schema()
        req = set(sch.get("required", []))
        return [{"name": n, "type": p.get("type") or "|".join(x.get("type", "?") for x in p.get("anyOf", [])) or "any",
                 "required": n in req, "description": p.get("description", "")}
                for n, p in sch.get("properties", {}).items()]

    tool_rows = [{"name": t.name, "origin": t.origin, "status": "active", "description": t.description,
                  "params": params_of(t), "capabilities": capabilities.get(t.name, []),
                  "used_by_skills": skills_using.get(t.name, []), "uses": usage.get(t.name, 0)}
                 for t in tools.values()]
    skill_rows = [{"name": s.name, "origin": s.origin, "status": "active", "description": s.description,
                   "tools": s.tools} for s in skills.values()]
    for it in pending:
        if it["kind"] == "tool":
            tool_rows.append({"name": it["name"], "origin": "generated", "status": "pending",
                              "description": it.get("description", ""), "params": [],
                              "capabilities": it.get("capabilities", []), "used_by_skills": [], "uses": 0})
        else:
            try:
                tl = parse_skill(pending_skill_source(it["name"]), "generated").tools
            except Exception:
                tl = []
            skill_rows.append({"name": it["name"], "origin": "generated", "status": "pending",
                               "description": it.get("description", ""), "tools": tl})
    return {"tools": tool_rows, "skills": skill_rows, "load_problems": load_problems}


# ---- capability audit ------------------------------------------------------------------------------------------

def audit_capabilities(cat: dict[str, Any]) -> dict[str, Any]:
    f: list[dict[str, Any]] = []
    tools, skills = cat["tools"], cat["skills"]
    active_tools = {t["name"] for t in tools if t["status"] == "active"}
    pending_tools = {t["name"] for t in tools if t["status"] == "pending"}

    for p in cat["load_problems"]:
        f.append(finding("high", "tools", "load-problem", [], p, "Fix or reject the item; it is not available to the agent."))

    # tool overlap
    for a, b in combinations(tools, 2):
        builtin_pair = a["origin"] == b["origin"] == "builtin"
        name_sim = _jaccard(_tokens(a["name"]), _tokens(b["name"]))
        desc_sim = _jaccard(_tokens(a["description"]), _tokens(b["description"]))
        pa = {p["name"] for p in a["params"]}
        pb = {p["name"] for p in b["params"]}
        same_inputs = len(pa) >= 2 and pa == pb
        score = max(name_sim, desc_sim)
        if (builtin_pair and score < 0.7) or (not builtin_pair and score < 0.4 and not (same_inputs and desc_sim >= 0.25)):
            continue
        why = [f"name similarity {name_sim:.0%}", f"description similarity {desc_sim:.0%}"] + (["identical parameters"] if same_inputs else [])
        f.append(finding("medium" if score >= 0.6 or same_inputs else "low", "tools", "overlapping-tools", [a["name"], b["name"]],
                         "Possible overlap (" + ", ".join(why) + ").",
                         "Compare what they do. If redundant, keep one (reject the pending one, or ask the user to retire the approved one); "
                         "if distinct, sharpen both descriptions so the model can tell them apart."))

    # skill overlap
    for a, b in combinations(skills, 2):
        d = _jaccard(_tokens(a["description"]), _tokens(b["description"]))
        ta, tb = set(a["tools"]), set(b["tools"])
        t = _jaccard(ta, tb)
        if d >= 0.4 or (t >= 0.6 and len(ta & tb) >= 2):
            f.append(finding("medium" if d >= 0.55 else "low", "skills", "overlapping-skills", [a["name"], b["name"]],
                             f"Descriptions {d:.0%} similar; they share tools {sorted(ta & tb)}.",
                             "Merge them, or make each description say clearly WHEN it applies."))

    # skills pointing at tools that do not exist
    for s in skills:
        for t in s["tools"]:
            if t in active_tools:
                continue
            if t in pending_tools:
                f.append(finding("low", "skills", "skill-needs-pending-tool", [s["name"], t],
                                 f"Skill '{s['name']}' relies on '{t}', which is still pending approval.", "Review and approve the tool."))
            else:
                f.append(finding("medium", "skills", "skill-references-missing-tool", [s["name"], t],
                                 f"Skill '{s['name']}' lists tool '{t}', which does not exist.", "Fix the skill's tools list or propose the tool."))

    for t in tools:
        if t["origin"] != "generated" or t["status"] != "active":
            continue
        if not t["used_by_skills"]:
            f.append(finding("info", "tools", "tool-not-in-any-skill", [t["name"]],
                             "No skill mentions this tool, so the model only finds it through the raw tool list.",
                             "Mention it in a skill's tools list, or note it in a new skill."))
        if t["uses"] == 0:
            f.append(finding("low", "tools", "tool-never-used", [t["name"]],
                             "Not called in any saved chat.", "Candidate for removal if it is not needed."))
        if len(t["description"]) < 30:
            f.append(finding("low", "tools", "weak-description", [t["name"]],
                             f"Description is only {len(t['description'])} chars; the model chooses tools from it.", "Say what it does, its input and output."))
        if "network" in t["capabilities"] and "database-write" in t["capabilities"]:
            f.append(finding("info", "tools", "network-and-db-write", [t["name"]],
                             "Fetches from the network and writes to the database.", "Make sure you reviewed the source for what it stores."))

    n_active = len(active_tools)
    if n_active > LARGE_TOOLSET:
        f.append(finding("info", "tools", "large-toolset", [], f"{n_active} tools are offered to the model on every turn.",
                         "Small local models degrade with long tool lists; consolidate or retire rarely used tools."))
    if pending_tools or any(s["status"] == "pending" for s in skills):
        f.append(finding("info", "tools", "pending-review", sorted(pending_tools | {s['name'] for s in skills if s['status'] == 'pending'}),
                         "Items are waiting for approval.", "Review them in the Approvals panel."))
    f = _sorted(f)
    return {"tools": len(tools), "skills": len(skills), "summary": _summary(f), "findings": f}


# ---- database audit --------------------------------------------------------------------------------------------

def audit_database(db: Database) -> dict[str, Any]:
    f: list[dict[str, Any]] = []
    tables = db.list_tables()
    cols: dict[str, list[str]] = {}
    rows = {t["table"]: t["rows"] for t in tables}

    for t in tables:
        name, n = t["table"], t["rows"]
        pcols = db.pragma_columns(name)
        cols[name] = [c["name"] for c in pcols]
        spec = db.get_spec(name)
        indexes = db.pragma_indexes(name)
        indexed = {i["columns"][0] for i in indexes if i["columns"]} | {c["name"] for c in pcols if c["pk"]}

        if not t["managed"]:
            f.append(finding("medium", "database", "table-without-model", [name],
                             "No model: the agent can read it but not write it, and its columns are undocumented.",
                             "Run the table-onboarding skill (profile_table, adopt_table, annotate_model)."))
        if n == 0:
            f.append(finding("low", "database", "empty-table", [name], "The table has no rows.", "Drop it if it is unused (ask the user first)."))
        if not any(c["pk"] for c in pcols):
            f.append(finding("medium", "database", "no-primary-key", [name], "No primary key, so rows cannot be addressed reliably.",
                             "Create a new model with a key and copy the rows (no in-place migration yet)."))

        if spec:
            actual = set(cols[name])
            declared = {x.name for x in spec.fields}
            if declared - actual or actual - declared:
                f.append(finding("high", "database", "model-drift", [name],
                                 f"Model and table disagree. Only in model: {sorted(declared - actual)}; only in table: {sorted(actual - declared)}.",
                                 "Re-adopt or recreate the model so the validators match the data."))
            if not spec.description:
                f.append(finding("info", "database", "undocumented-model", [name], "The model has no description.", "Use annotate_model."))
            bare = [x.name for x in spec.fields if not x.description]
            if bare and len(bare) >= max(2, len(spec.fields) // 2):
                f.append(finding("info", "database", "undocumented-fields", [name], f"{len(bare)}/{len(spec.fields)} fields lack descriptions.",
                                 "Use annotate_model so later queries and tools can rely on the meaning of each column."))
            refs = {x.name: x.references for x in spec.fields if x.references}
        else:
            refs = {}
        for c in cols[name]:
            if c.endswith("_id") and c not in refs and not any(x["pk"] and x["name"] == c for x in pcols):
                f.append(finding("low", "database", "possible-missing-reference", [f"{name}.{c}"],
                                 "Looks like a foreign key but no reference is recorded.", "Run find_relationships, then annotate_model."))

        for col, target in refs.items():
            tt, tc = target.split(".")
            if tt in rows and col in cols[name]:
                orphans = db.query(f"SELECT COUNT(*) AS n FROM {quote(name)} WHERE {quote(col)} IS NOT NULL AND {quote(col)} NOT IN "
                                   f"(SELECT {quote(tc)} FROM {quote(tt)} WHERE {quote(tc)} IS NOT NULL)")["rows"][0]["n"]
                if orphans:
                    f.append(finding("high", "database", "orphaned-references", [f"{name}.{col}"],
                                     f"{orphans} row(s) point at a missing {target}.", "Fix or delete those rows (confirm with the user)."))
            if n >= 1000 and col not in indexed:
                f.append(finding("low", "database", "unindexed-reference", [f"{name}.{col}"],
                                 f"Joins on this column scan {n} rows.", "Add an index (needs a manual change; the agent cannot run DDL)."))

        if 0 < n <= 200_000 and len(cols[name]) <= 60:  # per-column quality in one pass
            sel = ", ".join(f"COUNT({quote(c)}) AS n{i}, COUNT(DISTINCT {quote(c)}) AS d{i}" for i, c in enumerate(cols[name]))
            r = db.query(f"SELECT {sel} FROM {quote(name)}")["rows"][0]
            for i, c in enumerate(cols[name]):
                if r[f"n{i}"] == 0:
                    f.append(finding("low", "database", "all-null-column", [f"{name}.{c}"], "Every value is NULL.", "Candidate to drop or fill."))
                elif r[f"d{i}"] == 1 and n > 1:
                    f.append(finding("info", "database", "constant-column", [f"{name}.{c}"], "Every non-null value is identical.", "Likely carries no information."))

    for a, b in combinations(cols, 2):
        sim = _jaccard(set(cols[a]), set(cols[b]))
        same_name = _tokens(a) == _tokens(b)
        if len(cols[a]) >= 3 and (sim >= 0.8 or same_name):
            f.append(finding("medium", "database", "similar-tables", [a, b],
                             f"Column sets are {sim:.0%} alike" + ("; names differ only by plural/format" if same_name else "") + ".",
                             "Check whether they hold the same entity and should be merged."))
    f = _sorted(f)
    return {"tables": len(tables), "summary": _summary(f), "findings": f}
