from __future__ import annotations

import re
import uuid
from typing import Any

import httpx

from .. import config
from .base import LLMProvider, ToolCall, ToolDef, Turn

_THINK = re.compile(r"<think>.*?</think>", re.S)


class OllamaProvider(LLMProvider):
    """Talks to a local Ollama server (Gemma, Qwen, ...). The model must support tool calling."""

    name = "ollama"

    @staticmethod
    def _convert(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m["text"]}
                if m["tool_calls"]:
                    msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.args}} for c in m["tool_calls"]]
                out.append(msg)
            else:
                out += [{"role": "tool", "tool_name": r["name"],
                         "content": ("ERROR: " if r["is_error"] else "") + r["content"]} for r in m["results"]]
        return out

    def generate(self, system: str, messages: list[dict[str, Any]], tools: list[ToolDef]) -> Turn:
        body = {
            "model": self.model,
            "stream": False,
            "messages": self._convert(system, messages),
            "tools": [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                         "parameters": t.schema}} for t in tools],
            "options": {"num_ctx": config.OLLAMA_NUM_CTX},  # default context is too small for the tool schemas
        }
        try:
            r = httpx.post(f"{config.OLLAMA_HOST}/api/chat", json=body, timeout=600)
        except httpx.ConnectError as e:
            raise RuntimeError(f"Cannot reach Ollama at {config.OLLAMA_HOST}. Is `ollama serve` running?") from e
        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:500]}")
        msg = r.json()["message"]
        turn = Turn(text=_THINK.sub("", msg.get("content") or "").strip())
        for c in msg.get("tool_calls") or []:
            fn = c["function"]
            args = fn.get("arguments") or {}
            turn.tool_calls.append(ToolCall(uuid.uuid4().hex[:12], fn["name"], args if isinstance(args, dict) else {}))
        return turn
