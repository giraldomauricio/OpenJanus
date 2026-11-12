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
        self._cache: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}

    def _path(self, sid: str) -> Path | None:
        return self.dir / f"{sid}.json" if _ID.match(sid) else None  # ids are hex only: no path traversal

    def save(self, s: Session) -> None:
        if not s.events:
            return  # never persist empty chats
        path = self._path(s.id)
        if path is None:
            return
        s.updated = time.time()
        data = {"id": s.id, "title": s.title, "provider": s.provider, "model": s.model, "created": s.created, "updated": s.updated,
                "events": s.events, "messages": [_encode_message(m) for m in s.messages]}
        self._write(path, data)

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, default=str, ensure_ascii=False)
        os.replace(tmp, path)  # atomic: a crash mid-write cannot corrupt the saved chat

    def rename(self, sid: str, title: str) -> bool:
        """Change only the title of a saved chat (keeps `updated`, so the history order does not change)."""
        path = self._path(sid)
        if path is None or not path.exists():
            return False
        try:
            d = json.loads(path.read_text())
        except ValueError:
            return False
        d["title"] = title
        self._write(path, d)
        return True

    def load(self, sid: str) -> Session | None:
        path = self._path(sid)
        if path is None or not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            return Session(id=d["id"], title=d.get("title", ""), provider=d.get("provider", ""), model=d.get("model", ""), created=d.get("created", 0), updated=d.get("updated", 0),
                           events=d.get("events", []), messages=[_decode_message(m) for m in d["messages"]])
        except (ValueError, KeyError, TypeError):
            return None

    def _summary(self, p: Path) -> dict[str, Any] | None:
        """Title/time/model and tool-call counts of one saved chat. Cached by (mtime, size), so the history menu and the usage
        statistics only re-read chats that changed instead of every chat on every refresh."""
        try:
            st = p.stat()
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        hit = self._cache.get(p.name)
        if hit and hit[0] == key:
            return hit[1]
        try:
            d = json.loads(p.read_text())
            counts: dict[str, int] = {}
            for ev in d.get("events", []):
                if isinstance(ev, dict) and ev.get("type") == "tool_call":
                    counts[ev["name"]] = counts.get(ev["name"], 0) + 1
            summ = {"id": d["id"], "title": d.get("title") or "Untitled", "updated": d.get("updated", 0),
                    "provider": d.get("provider", ""), "model": d.get("model", ""), "tools": counts}
        except (ValueError, KeyError, TypeError, OSError):
            return None
        self._cache[p.name] = (key, summ)
        return summ

    def _summaries(self) -> list[dict[str, Any]]:
        paths = list(self.dir.glob("*.json"))
        live = {p.name for p in paths}
        for gone in set(self._cache) - live:  # deleted chats
            del self._cache[gone]
        return [x for x in (self._summary(p) for p in paths) if x]

    def list(self) -> list[dict[str, Any]]:
        out = [{k: v for k, v in x.items() if k != "tools"} for x in self._summaries()]
        return sorted(out, key=lambda x: x["updated"], reverse=True)

    def tool_usage(self) -> dict[str, int]:
        """How often each tool was called across all saved chats."""
        counts: dict[str, int] = {}
        for x in self._summaries():
            for name, n in x["tools"].items():
                counts[name] = counts.get(name, 0) + n
        return counts

    def exists(self, sid: str) -> bool:
        path = self._path(sid)
        return path is not None and path.exists()

    def delete(self, sid: str) -> bool:
        path = self._path(sid)
        if path is None or not path.exists():
            return False
        path.unlink()
        return True
