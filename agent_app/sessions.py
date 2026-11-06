"""Chat persistence: one JSON file per session under data/sessions/."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from .agent import Session
from .llm import ToolCall

_ID = re.compile(r"^[0-9a-f]{32}$")


def _encode_message(m: dict[str, Any]) -> dict[str, Any]:
    if m["role"] == "assistant":
        # `raw` is provider-specific (e.g. Gemini thought signatures) and not serializable; providers rebuild without it.
        return {"role": "assistant", "text": m["text"],
                "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in m["tool_calls"]]}
    return m


def _decode_message(m: dict[str, Any]) -> dict[str, Any]:
    if m["role"] == "assistant":
        return {"role": "assistant", "text": m["text"], "raw": None,
                "tool_calls": [ToolCall(c["id"], c["name"], c["args"]) for c in m["tool_calls"]]}
    return m


class SessionStore:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path | None:
        return self.dir / f"{sid}.json" if _ID.match(sid) else None  # ids are hex only: no path traversal

    def save(self, s: Session) -> None:
        if not s.events:
            return  # never persist empty chats
        path = self._path(s.id)
        if path is None:
            return
        s.updated = time.time()
        data = {"id": s.id, "title": s.title, "created": s.created, "updated": s.updated,
                "events": s.events, "messages": [_encode_message(m) for m in s.messages]}
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, default=str, ensure_ascii=False)
        os.replace(tmp, path)  # atomic: a crash mid-write cannot corrupt the saved chat

    def load(self, sid: str) -> Session | None:
        path = self._path(sid)
        if path is None or not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            return Session(id=d["id"], title=d.get("title", ""), created=d.get("created", 0), updated=d.get("updated", 0),
                           events=d.get("events", []), messages=[_decode_message(m) for m in d["messages"]])
        except (ValueError, KeyError, TypeError):
            return None

    def list(self) -> list[dict[str, Any]]:
        out = []
        for p in self.dir.glob("*.json"):
            try:
                d = json.loads(p.read_text())
                out.append({"id": d["id"], "title": d.get("title") or "Untitled", "updated": d.get("updated", 0)})
            except (ValueError, KeyError):
                continue
        return sorted(out, key=lambda x: x["updated"], reverse=True)

    def tool_usage(self) -> dict[str, int]:
        """How often each tool was called across all saved chats."""
        counts: dict[str, int] = {}
        for p in self.dir.glob("*.json"):
            try:
                for ev in json.loads(p.read_text()).get("events", []):
                    if ev.get("type") == "tool_call":
                        counts[ev["name"]] = counts.get(ev["name"], 0) + 1
            except (ValueError, KeyError, TypeError):
                continue
        return counts

    def delete(self, sid: str) -> bool:
        path = self._path(sid)
        if path is None or not path.exists():
            return False
        path.unlink()
        return True
