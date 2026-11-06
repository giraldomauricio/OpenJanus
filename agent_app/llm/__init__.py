from __future__ import annotations

import os

import httpx

from .. import config
from .base import LLMProvider, ToolCall, ToolDef, Turn

PROVIDERS = ("claude", "gemini", "ollama")


def make_provider(name: str, model: str | None = None) -> LLMProvider:
    model = model or config.DEFAULT_MODELS.get(name)
    if name == "claude":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model)
    if name == "gemini":
        from .gemini_provider import GeminiProvider
        return GeminiProvider(model)
    if name == "ollama":
        from .ollama_provider import OllamaProvider
        return OllamaProvider(model)
    raise ValueError(f"Unknown provider '{name}'. Choose one of {PROVIDERS}")


def provider_status() -> dict[str, dict]:
    return {
        "claude": {"model": config.DEFAULT_MODELS["claude"], "configured": bool(os.getenv("ANTHROPIC_API_KEY"))},
        "gemini": {"model": config.DEFAULT_MODELS["gemini"],
                   "configured": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))},
        "ollama": {"model": config.DEFAULT_MODELS["ollama"], "configured": True},
    }


def ollama_models() -> list[str]:
    try:
        r = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


__all__ = ["LLMProvider", "ToolCall", "ToolDef", "Turn", "make_provider", "provider_status", "ollama_models"]
