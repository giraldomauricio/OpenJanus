"""Provider-neutral conversation format.

    {"role": "user", "content": str}
    {"role": "assistant", "text": str, "tool_calls": [ToolCall], "raw": provider-specific or None}
    {"role": "tool", "results": [{"id", "name", "content", "is_error"}]}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


@dataclass
class ToolDef:
    name: str
    description: str
    schema: dict[str, Any]


class LLMProvider(ABC):
    name: str

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, system: str, messages: list[dict[str, Any]], tools: list[ToolDef]) -> Turn: ...
