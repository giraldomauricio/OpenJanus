from __future__ import annotations

from typing import Any, Iterator

from .. import config, images
from .base import LLMProvider, ToolCall, ToolDef, Turn, Usage, collect


class AnthropicProvider(LLMProvider):
    name = "claude"

    def __init__(self, model: str):
        super().__init__(model)
        import os

        import anthropic

        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env and restart the server.")
        self._anthropic = anthropic
        # Explicit key + base URL: the SDK would otherwise also read ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN from the shell.
        self.client = anthropic.Anthropic(api_key=key, base_url=config.ANTHROPIC_BASE_URL)

    @staticmethod
    def _convert(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                imgs, missing = images.resolve(m)
                if imgs or missing:
                    blocks = [{"type": "image", "source": {"type": "base64", "media_type": mime, "data": images.b64(data)}} for mime, data in imgs]
                    out.append({"role": "user", "content": blocks + [{"type": "text", "text": m["content"] + ("\n" + images.missing_note(missing) if missing else "")}]})
                else:
                    out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks: list[dict[str, Any]] = []
                if m["text"]:
                    blocks.append({"type": "text", "text": m["text"]})
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"], "is_error": r["is_error"]}
                    for r in m["results"]
                ]})
        return out

    @staticmethod
    def _usage(u: Any) -> Usage:
        """The API reports uncached input separately from cache reads/writes; fold them into one whole-prompt figure."""
        read, write = getattr(u, "cache_read_input_tokens", None) or 0, getattr(u, "cache_creation_input_tokens", None) or 0
        return Usage((u.input_tokens or 0) + read + write, u.output_tokens or 0, read, write)

    def generate(self, system: str, messages: list[dict[str, Any]], tools: list[ToolDef]) -> Turn:
        return collect(self.stream(system, messages, tools))

    def _stream(self, system: str, messages: list[dict[str, Any]], tools: list[ToolDef]) -> Iterator[str | Turn]:
        with self.closing_on_cancel(lambda: self.client.close()), self.client.messages.stream(
            model=self.model,
            max_tokens=8192,
            system=system,
            messages=self._convert(messages),
            tools=[{"name": t.name, "description": t.description, "input_schema": t.schema} for t in tools],
            # Prompt caching: one marker on the request caches the stable prefix (tools + system) and the growing conversation, so
            # each step of a tool loop re-reads them at a fraction of the price instead of paying full input rates again.
            **({"cache_control": {"type": "ephemeral"}} if config.PROMPT_CACHE else {}),
        ) as s:
            yield from s.text_stream
            final = s.get_final_message()
        turn = Turn(usage=self._usage(final.usage) if getattr(final, "usage", None) else None)
        for b in final.content:
            if b.type == "text":
                turn.text += b.text
            elif b.type == "tool_use":
                turn.tool_calls.append(ToolCall(b.id, b.name, dict(b.input)))
        yield turn
