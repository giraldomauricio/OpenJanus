"""The one pattern every tool follows.

    class Params(BaseModel): ...          # Pydantic input model -> JSON schema for the LLM
    def run(params: Params, ctx) -> Any   # returns something JSON-serialisable

Built-in tools register with ``@tool(name, description)``. Generated tools are plain
modules exposing ``NAME``, ``DESCRIPTION``, ``Params`` and ``run`` (see skills/extension-builder).
"""
from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

if typing.TYPE_CHECKING:
    from .db import Database
    from .extensions import ExtensionManager
    from .http_safe import SafeHttp
    from .skills import SkillStore


class ToolContext:
    """What generated tools receive. Deliberately small: the database and a guarded HTTP client."""

    def __init__(self, db: "Database", http: "SafeHttp"):
        self.db = db
        self.http = http


class _Narrow:
    """A read-only view exposing exactly the named methods of `target` and nothing else (no `.path`, no `.scoped`, no internals).

    The lint already rejects aliases and private attributes, but this is the control that does not depend on spotting a pattern in
    source code: even a tool that got past the lint can only call what is listed here."""
    __slots__ = ("_names", "_get")

    def __init__(self, target: Any, names: frozenset[str]):
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_get", lambda n: getattr(target, n))

    def __getattr__(self, name: str) -> Any:
        if name not in self._names:
            raise AttributeError(f"ctx has no '{name}'. Allowed: {', '.join(sorted(self._names))}")
        return self._get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ctx is read-only")


DB_METHODS = frozenset({"query", "describe", "list_tables", "table_names", "get_spec", "all_specs", "export_models",
                        "profile_table", "find_relationships", "insert", "update", "delete", "define_model"})
HTTP_METHODS = frozenset({"get", "post"})


def restricted_context(db: "Database", http: "SafeHttp") -> ToolContext:
    """What a GENERATED tool receives: the database and HTTP client narrowed to their documented methods."""
    return ToolContext(_Narrow(db, DB_METHODS), _Narrow(http, HTTP_METHODS))  # type: ignore[arg-type]


class AgentContext(ToolContext):
    """Extra capabilities available to trusted built-in tools only."""

    def __init__(self, db: "Database", http: "SafeHttp", extensions: "ExtensionManager", skills: "SkillStore"):
        super().__init__(db, http)
        self.extensions = extensions
        self.skills = skills
        self.agent: Any = None  # set by Agent; only trusted built-in tools see it


@dataclass
class Tool:
    name: str
    description: str
    params: type[BaseModel]
    fn: Callable[[Any, Any], Any]
    origin: str  # "builtin" | "generated"
    secrets: dict[str, list[str]] = field(default_factory=dict)  # generated tools: env var name -> hosts it may be sent to

    def schema(self) -> dict[str, Any]:
        return self.params.model_json_schema()

    def call(self, raw_args: dict[str, Any], ctx: ToolContext) -> Any:
        return self.fn(self.params.model_validate(raw_args), ctx)


BUILTIN_TOOLS: dict[str, Tool] = {}


def tool(name: str, description: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        first = next(iter(inspect.signature(fn).parameters))
        params_cls = typing.get_type_hints(fn)[first]
        BUILTIN_TOOLS[name] = Tool(name, description, params_cls, fn, "builtin")
        return fn

    return deco
