from __future__ import annotations

import os
import uuid
from typing import Any

from .base import LLMProvider, ToolCall, ToolDef, Turn


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str):
        super().__init__(model)
        from google import genai
        from google.genai import types

        self.types = types
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=key)

    def _convert(self, messages: list[dict[str, Any]]) -> list[Any]:
        t = self.types
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append(t.Content(role="user", parts=[t.Part(text=m["content"])]))
            elif m["role"] == "assistant":
                if isinstance(m.get("raw"), t.Content):  # keeps thought signatures intact
                    out.append(m["raw"])
                    continue
                parts = [t.Part(text=m["text"])] if m["text"] else []
                parts += [t.Part(function_call=t.FunctionCall(name=c.name, args=c.args)) for c in m["tool_calls"]]
                out.append(t.Content(role="model", parts=parts))
            else:
                out.append(t.Content(role="user", parts=[
                    t.Part.from_function_response(
                        name=r["name"], response={"error": r["content"]} if r["is_error"] else {"result": r["content"]})
                    for r in m["results"]
                ]))
        return out

    def generate(self, system: str, messages: list[dict[str, Any]], tools: list[ToolDef]) -> Turn:
        t = self.types
        config = t.GenerateContentConfig(
            system_instruction=system,
            tools=[t.Tool(function_declarations=[
                t.FunctionDeclaration(name=x.name, description=x.description, parameters_json_schema=x.schema)
                for x in tools
            ])],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = self.client.models.generate_content(model=self.model, contents=self._convert(messages), config=config)
        turn = Turn()
        if not resp.candidates or not resp.candidates[0].content:
            return turn
        content = resp.candidates[0].content
        turn.raw = content
        for p in content.parts or []:
            if p.function_call:
                fc = p.function_call
                turn.tool_calls.append(ToolCall(fc.id or uuid.uuid4().hex[:12], fc.name, dict(fc.args or {})))
            elif p.text and not getattr(p, "thought", False):
                turn.text += p.text
        return turn
