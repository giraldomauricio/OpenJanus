from __future__ import annotations

from typing import Any

from .base import LLMProvider, ToolCall, ToolDef, Turn


class AnthropicProvider(LLMProvider):
    name = "claude"

    def __init__(self, model: str):
        super().__init__(model)
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    @staticmethod
    def _convert(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
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

    def generate(self, system: str, messages: list[dict[str, Any]], tools: list[ToolDef]) -> Turn:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system,
            messages=self._convert(messages),
            tools=[{"name": t.name, "description": t.description, "input_schema": t.schema} for t in tools],
        )
        turn = Turn()
        for b in resp.content:
            if b.type == "text":
                turn.text += b.text
            elif b.type == "tool_use":
                turn.tool_calls.append(ToolCall(b.id, b.name, dict(b.input)))
        return turn
