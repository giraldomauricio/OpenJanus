"""Static checks on LLM-written tool code and skills.

IMPORTANT: this is a *lint*, not a sandbox. It rejects obviously dangerous constructs
and tells the reviewer what the tool can do, but the real control is that a human reads
the source and approves it before it is ever imported.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

MAX_SOURCE_CHARS = 20_000

ALLOWED_MODULES = {
    "pydantic", "typing", "json", "re", "math", "datetime", "decimal", "statistics", "collections",
    "itertools", "functools", "csv", "io", "textwrap", "urllib.parse", "html", "hashlib", "base64",
    "uuid", "enum", "string", "agent_app.sdk",
}
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "memoryview",
}
DB_READ = {"query", "describe", "list_tables", "table_names", "get_spec", "all_specs", "export_models",
           "profile_table", "find_relationships"}
DB_WRITE = {"insert", "update", "delete", "define_model"}

NET_MODULES = {"requests", "httpx", "urllib", "aiohttp", "socket", "http"}
TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
SKILL_NAME = re.compile(r"^[a-z][a-z0-9-]{2,40}$")


@dataclass
class LintResult:
    errors: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _module_allowed(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in ALLOWED_MODULES)


def lint_tool_source(source: str) -> LintResult:
    res = LintResult()
    if len(source) > MAX_SOURCE_CHARS:
        res.errors.append(f"Source too long ({len(source)} > {MAX_SOURCE_CHARS} chars)")
        return res
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        res.errors.append(f"Syntax error line {e.lineno}: {e.msg}")
        return res

    caps: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if not _module_allowed(a.name):
                    res.errors.append(f"Import not allowed: {a.name}" + (" (use ctx.http instead)" if a.name.split(".")[0] in NET_MODULES else ""))
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or not _module_allowed(node.module):
                res.errors.append(f"Import not allowed: {'.' * node.level}{node.module or ''}" + (
                    " (use ctx.http instead)" if (node.module or "").split(".")[0] in NET_MODULES else ""))
            elif any(a.name == "*" for a in node.names):
                res.errors.append("Wildcard imports are not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                res.errors.append(f"Use of '{node.id}' is not allowed (line {node.lineno})")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                res.errors.append(f"Access to private/dunder attribute '{node.attr}' is not allowed (line {node.lineno})")
            v = node.value
            if isinstance(v, ast.Name) and v.id == "ctx" and node.attr == "http":
                caps.add("network")
            if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id == "ctx" and v.attr == "db":
                if node.attr in DB_WRITE:
                    caps.add("database-write")
                elif node.attr in DB_READ:
                    caps.add("database-read")
                else:
                    res.errors.append(f"ctx.db.{node.attr} is not part of the allowed database API")
            if isinstance(v, ast.Name) and v.id == "ctx" and node.attr not in ("db", "http"):
                res.errors.append(f"ctx.{node.attr} does not exist. Only ctx.db (query/insert/update/delete/describe/list_tables) "
                                  "and ctx.http (get/post) are available")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            res.errors.append("global/nonlocal are not allowed")

    top = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    params = top.get("Params")
    if not isinstance(params, ast.ClassDef) or not any(
        isinstance(b, ast.Name) and b.id == "BaseModel" for b in params.bases
    ):
        res.errors.append("Must define top-level `class Params(BaseModel)`")
    run = top.get("run")
    if not isinstance(run, ast.FunctionDef) or len(run.args.args) != 2:
        res.errors.append("Must define top-level `def run(params, ctx)` taking exactly two arguments")
    res.errors = list(dict.fromkeys(res.errors))
    res.capabilities = sorted(caps)
    return res


def lint_skill(name: str, description: str, instructions: str) -> LintResult:
    res = LintResult()
    if not SKILL_NAME.match(name):
        res.errors.append("Skill name must be kebab-case: letters, digits, hyphens; 3-41 chars; start with a letter")
    if not 10 <= len(description) <= 300 or "\n" in description:
        res.errors.append("Description must be a single line of 10-300 characters saying WHEN to use the skill")
    if not 20 <= len(instructions) <= MAX_SOURCE_CHARS:
        res.errors.append("Instructions must be 20-20000 characters")
    return res
