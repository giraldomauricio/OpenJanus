"""Detects tabular tool results ({"columns": [...], "rows": [{...}]}, e.g. from `query`) for the UI."""
from __future__ import annotations

from typing import Any

MAX_ROWS = 1000
MAX_CELL_CHARS = 2000


def _cell(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float)):
        return v
    s = v if isinstance(v, str) else str(v)
    return s if len(s) <= MAX_CELL_CHARS else s[:MAX_CELL_CHARS] + "..."


def as_table(result: Any) -> dict[str, Any] | None:
    """Compact, bounded copy for the browser, or None when the result is not a table with at least one row."""
    if not isinstance(result, dict):
        return None
    cols, rows = result.get("columns"), result.get("rows")
    if not (isinstance(cols, list) and cols and all(isinstance(c, str) for c in cols)
            and isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows)):
        return None
    return {"columns": cols, "rows": [[_cell(r.get(c)) for c in cols] for r in rows[:MAX_ROWS]],
            "truncated": bool(result.get("truncated")) or len(rows) > MAX_ROWS}
