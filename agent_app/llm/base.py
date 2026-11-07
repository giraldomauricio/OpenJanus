"""Provider-neutral conversation format.

    {"role": "user", "content": str}
    {"role": "assistant", "text": str, "tool_calls": [ToolCall], "raw": provider-specific or None}
    {"role": "tool", "results": [{"id", "name", "content", "is_error"}]}
"""
from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


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
    cancel: threading.Event | None = None  # set by the server; when it fires the provider drops its connection

    def __init__(self, model: str):
        self.model = model

    @contextmanager
    def closing_on_cancel(self, closer):
        """While active, calling ``closer`` as soon as ``self.cancel`` is set. This unblocks a read that is waiting on a slow
        model (or one still loading) instead of waiting for the next chunk; the resulting error is expected by the caller."""
        if self.cancel is None:
            yield
            return
        done = threading.Event()

        def watch() -> None:
            while not done.is_set():
                if self.cancel.wait(0.2):
                    if not done.is_set():
                        try:
                            closer()
                        except Exception:
                            pass
                    return

        threading.Thread(target=watch, daemon=True).start()
        try:
            yield
        finally:
            done.set()

    @abstractmethod
    def generate(self, system: str, messages: list[dict[str, Any]], tools: list[ToolDef]) -> Turn: ...

    def stream(self, system: str, messages: list[dict[str, Any]], tools: list[ToolDef]) -> Iterator["str | Turn"]:
        """Yields text deltas (str) as they are produced, then the complete Turn as the LAST item.

        Providers implement ``_stream`` (or just ``generate``); this wrapper makes any of them honour Stop."""
        return self.interruptible(self._stream(system, messages, tools))

    def _stream(self, system: str, messages: list[dict[str, Any]], tools: list[ToolDef]) -> Iterator["str | Turn"]:
        """Default: no streaming (one delta, then the Turn), so a provider only needs ``generate``."""
        turn = self.generate(system, messages, tools)
        if turn.text:
            yield turn.text
        yield turn

    def interruptible(self, it: Iterator["str | Turn"]) -> Iterator["str | Turn"]:
        """Pull from ``it`` in a helper thread and poll, so a Stop is honoured even while ``it`` is blocked on the network
        (a reasoning model can stay silent for minutes). After a Stop this generator simply ends; the helper thread
        closes ``it`` (dropping the connection) as soon as its pending read returns."""
        if self.cancel is None:
            yield from it
            return
        box: queue.Queue = queue.Queue()
        stop = threading.Event()

        def pump() -> None:
            try:
                for item in it:
                    if stop.is_set():
                        break
                    box.put(("item", item))
                box.put(("end", None))
            except BaseException as e:  # forwarded to the consumer below
                box.put(("err", e))
            finally:
                try:
                    it.close()  # type: ignore[attr-defined]
                except Exception:
                    pass

        threading.Thread(target=pump, daemon=True).start()
        try:
            while True:
                try:
                    kind, val = box.get(timeout=0.15)
                except queue.Empty:
                    if self.cancel.is_set():
                        return
                    continue
                if kind == "item":
                    yield val
                elif kind == "err":
                    raise val
                else:
                    return
        finally:
            stop.set()


def collect(parts: Iterator["str | Turn"]) -> Turn:
    """Drain a stream() and return its final Turn."""
    turn = None
    for p in parts:
        if isinstance(p, Turn):
            turn = p
    if turn is None:
        raise RuntimeError("The model stream ended without a result")
    return turn
