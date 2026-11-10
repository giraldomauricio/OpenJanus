from __future__ import annotations

import os
import uuid
from typing import Any, Iterator

from .. import images
from .base import LLMProvider, ToolCall, ToolDef, Turn, collect


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
                imgs, missing = images.resolve(m)
                note = ("\n" + images.missing_note(missing)) if missing else ""
                out.append(t.Content(role="user", parts=[t.Part.from_bytes(data=data, mime_type=mime) for mime, data in imgs]
                                     + [t.Part(text=m["content"] + note)]))
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
        return collect(self.stream(system, messages, tools))

    def _stream(self, system: str, messages: list[dict[str, Any]], tools: list[ToolDef]) -> Iterator[str | Turn]:
        t = self.types
        config = t.GenerateContentConfig(
            system_instruction=system,
            tools=[t.Tool(function_declarations=[
                t.FunctionDeclaration(name=x.name, description=x.description, parameters_json_schema=x.schema)
                for x in tools
            ])],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        turn, parts = Turn(), []
        for chunk in self.client.models.generate_content_stream(model=self.model, contents=self._convert(messages), config=config):
            if not chunk.candidates or not chunk.candidates[0].content:
                continue
            for p in chunk.candidates[0].content.parts or []:
                parts.append(p)  # kept in order so thought signatures survive in turn.raw
                if p.function_call:
                    fc = p.function_call
                    turn.tool_calls.append(ToolCall(fc.id or uuid.uuid4().hex[:12], fc.name, dict(fc.args or {})))
                elif p.text and not getattr(p, "thought", False):
                    turn.text += p.text
                    yield p.text
        if parts:
            turn.raw = t.Content(role="model", parts=parts)
        yield turn
