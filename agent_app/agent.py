from __future__ import annotations

import importlib.util
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from pydantic import BaseModel, ValidationError

from . import config
from .db import Database
from .db.store import format_validation_error
from .extensions import ExtensionManager
from .extensions.lint import lint_tool_source
from .http_safe import SafeHttp
from .introspection import build_catalog
from .tables import as_table
from .llm import LLMProvider, ToolDef, Turn
from .sdk import BUILTIN_TOOLS, AgentContext, Tool, ToolContext
from .skills import SkillStore
from . import tools as _builtin  # noqa: F401  (registers built-in tools)

STOPPED = "[Stopped by the user.]"

SYSTEM_PROMPT = """You are a data assistant working with a local SQLite database through tools.

Your areas of expertise are described by skills you load on demand with `load_skill`:
{skills}

Rules:
- Load the relevant skill BEFORE doing non-trivial work in its area.
- Never guess at schema or data; inspect with the tools.
- Tables the user created or imported themselves have no model yet. Learn them (table-onboarding skill) before relying on them.
- You can propose new tools and skills, but they are saved as PENDING and are unusable until the user approves them
  in the Approvals panel. After proposing, explain what you created and ask the user to review it. Do not call a
  tool that is not in your tool list, and never say a pending item is available.
- Never create a tool that simulates, mocks or hard-codes data. If a real data source cannot be reached within the
  tool rules, say so instead of proposing a fake tool.
- If a tool call fails, read the error, fix the input, and retry at most twice before asking the user.
- Report only what tool results confirm. If a call failed and was not fixed, say so plainly and do not describe it as done.
  Do not state facts about a table (keys, relationships) that the tools did not show.
- Be concise. Report concrete results (counts, names), not intentions."""


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    title: str = ""
    provider: str = ""  # last provider/model used in this chat (each user event also records its own)
    model: str = ""
    created: float = field(default_factory=time.time)
    updated: float = 0.0
    cancel: threading.Event = field(default_factory=threading.Event)  # set by the Stop button
    events: list[dict[str, Any]] = field(default_factory=list)  # what the UI displays, replayed when a chat is reopened


def _load_generated(name: str, path: Path) -> Tool:
    spec = importlib.util.spec_from_file_location(f"generated_tools.{name}", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not (isinstance(mod.Params, type) and issubclass(mod.Params, BaseModel)):
        raise TypeError("Params must be a Pydantic model")
    return Tool(mod.NAME, mod.DESCRIPTION, mod.Params, mod.run, "generated")


class Agent:
    def __init__(self, db: Database, extensions: ExtensionManager, skills: SkillStore):
        self.db, self.extensions, self.skills = db, extensions, skills
        http = SafeHttp()
        self.tool_ctx = ToolContext(db, http)
        self.agent_ctx = AgentContext(db, http, extensions, skills)
        self._generated_cache: dict[tuple[str, int], Tool] = {}
        self.usage: Callable[[], dict[str, int]] = lambda: {}  # tool name -> times called in saved chats (set by the server)
        self.agent_ctx.agent = self  # lets the self-inspection tools see the live tool list

    def catalog(self) -> dict[str, Any]:
        """Everything the agent can do right now (built-in, approved, pending), for the UI and the introspection tools."""
        tools, problems = self.build_tools()
        caps = {i["name"]: i.get("capabilities", []) for i in self.extensions.list_items("approved") if i["kind"] == "tool"}
        return build_catalog(tools, self.skills.all(), self.extensions.list_items("pending"),
                             lambda n: self.extensions.get_source("skill", n, "pending"), caps, self.usage(), problems)

    def build_tools(self) -> tuple[dict[str, Tool], list[str]]:
        """Built-in tools + approved generated tools (integrity-checked, re-linted, then imported)."""
        tools = dict(BUILTIN_TOOLS)
        problems = list(self.extensions.integrity_problems())
        for name, path in self.extensions.approved_tool_files():
            key = (name, path.stat().st_mtime_ns)
            try:
                if key not in self._generated_cache:
                    lint = lint_tool_source(path.read_text())
                    if not lint.ok:
                        raise ValueError("; ".join(lint.errors))
                    self._generated_cache[key] = _load_generated(name, path)
                t = self._generated_cache[key]
                if t.name in tools:
                    raise ValueError("name collides with an existing tool")
                tools[t.name] = t
            except Exception as e:  # a broken tool must not take the agent down
                problems.append(f"tool '{name}' failed to load: {e}")
        return tools, problems

    def system_prompt(self) -> str:
        lines = [f"- {s.name}: {s.description}" for s in self.skills.all().values()]
        return SYSTEM_PROMPT.format(skills="\n".join(lines))

    def _execute(self, tools: dict[str, Tool], name: str, args: dict[str, Any]) -> tuple[bool, str]:
        return self._run(tools, name, args)[:2]

    def _run(self, tools: dict[str, Tool], name: str, args: dict[str, Any]) -> tuple[bool, str, Any]:
        """(ok, text for the model, raw result). The raw result lets the UI offer CSV/charts without re-parsing truncated text."""
        t = tools.get(name)
        if t is None:
            return False, f"Unknown tool '{name}'. Available: {sorted(tools)}", None
        ctx = self.agent_ctx if t.origin == "builtin" else self.tool_ctx
        result = None
        try:
            result = t.call(args, ctx)
            text = json.dumps(result, default=str, ensure_ascii=False)
            ok = True
        except ValidationError as e:
            ok, text = False, "Invalid arguments: " + format_validation_error(e)
        except Exception as e:
            ok, text = False, f"{type(e).__name__}: {e}"
        if len(text) > config.MAX_TOOL_RESULT_CHARS:
            text = text[: config.MAX_TOOL_RESULT_CHARS] + f'... [truncated, {len(text)} chars total]'
        return ok, text, result if ok else None

    @staticmethod
    def _stopped(session: Session, partial: str = "") -> Iterator[dict[str, Any]]:
        """Close the turn cleanly after a Stop so the stored conversation stays valid for every provider."""
        partial = partial.rstrip()
        session.messages.append({"role": "assistant", "tool_calls": [], "raw": None,
                                 "text": (partial + "\n\n" if partial else "") + STOPPED})
        if partial:
            yield {"type": "text", "text": partial}
        yield {"type": "cancelled"}

    def run(self, session: Session, user_text: str, provider: LLMProvider) -> Iterator[dict[str, Any]]:
        session.cancel.clear()
        session.messages.append({"role": "user", "content": user_text})
        tools, problems = self.build_tools()
        for p in problems:
            yield {"type": "warning", "text": p}
        defs = [ToolDef(t.name, t.description, t.schema()) for t in tools.values()]
        system = self.system_prompt()
        for _ in range(config.MAX_STEPS):
            if session.cancel.is_set():
                yield from self._stopped(session)
                return
            turn, partial, stream = None, "", provider.stream(system, session.messages, defs)
            try:
                for part in stream:
                    if isinstance(part, Turn):
                        turn = part
                    elif part:
                        partial += part
                        yield {"type": "text_delta", "text": part}  # live preview only; the "text" event below is authoritative
                    if session.cancel.is_set():
                        break
                if turn is None and not session.cancel.is_set():
                    raise RuntimeError("The model returned nothing")
            except Exception as e:
                if not session.cancel.is_set():  # a Stop closes the connection on purpose, which surfaces as an error here
                    yield {"type": "error", "text": f"{provider.name}: {e}"}
                    return
            finally:
                stream.close()
            if session.cancel.is_set():
                yield from self._stopped(session, turn.text if turn else partial)
                return
            session.messages.append({"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls,
                                     "raw": turn.raw})
            if turn.text:
                yield {"type": "text", "text": turn.text}
            if not turn.tool_calls:
                yield {"type": "done"}
                return
            results = []
            for c in turn.tool_calls:
                if session.cancel.is_set():  # every tool call needs a result, even the ones that never ran
                    results.append({"id": c.id, "name": c.name, "content": "Not run: the user pressed Stop.", "is_error": True})
                    continue
                yield {"type": "tool_call", "id": c.id, "name": c.name, "args": c.args}
                ok, content, raw = self._run(tools, c.name, c.args)
                results.append({"id": c.id, "name": c.name, "content": content, "is_error": not ok})
                ev = {"type": "tool_result", "id": c.id, "name": c.name, "ok": ok, "content": content}
                table = as_table(raw)  # UI-only: lets the chat offer a preview, CSV download and chart
                if table:
                    ev["table"] = table
                yield ev
            session.messages.append({"role": "tool", "results": results})
        yield {"type": "error", "text": f"Stopped after {config.MAX_STEPS} steps without a final answer."}
