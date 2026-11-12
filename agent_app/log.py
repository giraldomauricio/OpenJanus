"""Application logging: one `openjanus` logger tree, shown on the console and written to data/logs/openjanus.log (rotating).

What is logged: requests, model calls (timing and token counts), tool calls (name, outcome, timing), approvals, imports/exports,
outbound HTTP (host and status only), and security events (bad tokens, blocked addresses, which secret NAME was sent to which host).
What is NEVER logged: your messages, model answers, successful tool results, API keys or secret values. Tool arguments appear only
at OPENJANUS_LOG_LEVEL=DEBUG, truncated. One caveat: the text of an ERROR (a failed tool, a failed model call) is logged, and such
messages can mention names from your database (a table or column that does not exist, say) but not your question or the answer.

Usage in a module:  log = logging.getLogger("openjanus.tools")
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from . import config

ROOT = "openjanus"


class _Component(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.component = record.name.removeprefix(ROOT + ".") if record.name != ROOT else "app"
        return True


def setup(level: str | None = None, log_dir: Path | None = None, file: bool | None = None) -> logging.Logger:
    """Attach handlers (idempotent). Called when the server starts, not on import, so tests and tools stay quiet."""
    root = logging.getLogger(ROOT)
    if getattr(root, "_configured", False):
        return root
    lvl = getattr(logging, (level or config.LOG_LEVEL).upper(), logging.INFO)
    root.setLevel(lvl)
    root.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(component)-9s %(message)s", "%Y-%m-%d %H:%M:%S")
    flt = _Component()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    console.addFilter(flt)
    root.addHandler(console)
    if config.LOG_FILE_ENABLED if file is None else file:
        directory = Path(log_dir or config.LOG_DIR)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(directory / "openjanus.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            fh.addFilter(flt)
            root.addHandler(fh)
            for name in ("uvicorn.error", "uvicorn.access"):  # the web server's own lines land in the same file
                logging.getLogger(name).addHandler(fh)
        except OSError as e:  # a read-only disk must not stop the app
            root.warning("Log file unavailable (%s); logging to the console only", e)
    root._configured = True  # type: ignore[attr-defined]
    return root


def short(sid: str) -> str:
    return sid[:8]


def clip(value: object, n: int = 200) -> str:
    s = str(value).replace("\n", " ")
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n})"
