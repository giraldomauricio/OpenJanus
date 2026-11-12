from __future__ import annotations

import json
import re
import uuid
from typing import Any, Iterator

import httpx

from .. import config
from .. import images
from .base import LLMProvider, ToolCall, ToolDef, Turn, Usage, collect

_THINK = re.compile(r"<think>.*?</think>", re.S)


class ThinkFilter:
    """Removes <think>...</think> from a text stream, even when a tag is split across chunks."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buf, self.inside = "", False

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out = []
        while True:
            tag = self.CLOSE if self.inside else self.OPEN
            i = self.buf.find(tag)
            if i >= 0:
                if not self.inside:
                    out.append(self.buf[:i])
                self.buf, self.inside = self.buf[i + len(tag):], not self.inside
                continue
            keep = 0  # hold back a possible partial tag at the end of the buffer
            for n in range(min(len(tag) - 1, len(self.buf)), 0, -1):
                if tag.startswith(self.buf[-n:]):
                    keep = n
                    break
            if not self.inside:
                out.append(self.buf[: len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep:]
            return "".join(out)

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return rest


class OllamaProvider(LLMProvider):
    """Talks to a local Ollama server (Gemma, Qwen, ...). The model must support tool calling."""

    name = "ollama"

    @staticmethod
    def _convert(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                imgs, missing = images.resolve(m)
                msg = {"role": "user", "content": m["content"] + (("\n" + images.missing_note(missing)) if missing else "")}
                if imgs:
                    msg["images"] = [images.b64(data) for _, data in imgs]
                out.append(msg)
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
        return collect(self.stream(system, messages, tools))

    def _require_vision(self, messages: list[dict[str, Any]]) -> None:
        """Ollama silently ignores images for text-only models, which makes the model invent a description: refuse instead."""
        if not any(m.get("images") for m in messages if m["role"] == "user"):
            return
        try:
            caps = httpx.post(f"{config.OLLAMA_HOST}/api/show", json={"model": self.model}, timeout=10).json().get("capabilities")
        except Exception:
            return  # cannot tell: let Ollama try
        if isinstance(caps, list) and "vision" not in caps:
            raise RuntimeError(f"The Ollama model '{self.model}' cannot read images. Choose a vision model (e.g. a llava or qwen-vl "
                               "variant) or switch the provider to Claude or Gemini for this chat.")

    def _stream(self, system: str, messages: list[dict[str, Any]], tools: list[ToolDef]) -> Iterator[str | Turn]:
        self._require_vision(messages)
        body = {
            "model": self.model,
            "stream": True,
            "messages": self._convert(system, messages),
            "tools": [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                         "parameters": t.schema}} for t in tools],
            "options": {"num_ctx": config.OLLAMA_NUM_CTX},  # default context is too small for the tool schemas
        }
        turn, raw_text, think = Turn(), "", ThinkFilter()
        try:
            with httpx.Client(timeout=httpx.Timeout(600, connect=10)) as client, self.closing_on_cancel(client.close), \
                    client.stream("POST", f"{config.OLLAMA_HOST}/api/chat", json=body) as r:
                if r.status_code != 200:
                    r.read()
                    raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:500]}")
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if d.get("error"):
                        raise RuntimeError(f"Ollama error: {d['error']}")
                    if d.get("done"):  # the last line carries the counters
                        turn.usage = Usage(d.get("prompt_eval_count") or 0, d.get("eval_count") or 0)
                    msg = d.get("message") or {}
                    if msg.get("content"):
                        raw_text += msg["content"]
                        if (shown := think.feed(msg["content"])):
                            yield shown
                    for c in msg.get("tool_calls") or []:
                        fn = c["function"]
                        args = fn.get("arguments") or {}
                        turn.tool_calls.append(ToolCall(uuid.uuid4().hex[:12], fn["name"], args if isinstance(args, dict) else {}))
        except httpx.ConnectError as e:
            raise RuntimeError(f"Cannot reach Ollama at {config.OLLAMA_HOST}. Is `ollama serve` running?") from e
        if (rest := think.flush()):
            yield rest
        turn.text = _THINK.sub("", raw_text).strip()
        yield turn
