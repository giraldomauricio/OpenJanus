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
