"""Pure helpers for learning a column's type and a sane name from raw data."""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Iterable

_INT = re.compile(r"^[+-]?(0|[1-9]\d{0,17})$")  # leading zeros (zip codes, ids like 007) stay strings
_FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def classify(v: Any) -> str:
    """integer | number | date | datetime | string"""
    if isinstance(v, bool):
        return "string"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if not isinstance(v, str):
        return "string"
    s = v.strip()
    if _INT.match(s):
        return "integer"
    if _FLOAT.match(s) and not (len(s) > 1 and s[0] == "0" and s[1].isdigit()):
        return "number"
    if _DATE.match(s):
        try:
            date.fromisoformat(s)
            return "date"
        except ValueError:
            return "string"
    if len(s) >= 16 and ("T" in s or " " in s):
        try:
            datetime.fromisoformat(s)
            return "datetime"
        except ValueError:
            return "string"
    return "string"


def infer_type(values: Iterable[Any]) -> str:
    kinds = {classify(v) for v in values if v is not None and v != ""}
    if not kinds:
        return "string"
    if kinds == {"integer"}:
        return "integer"
    if kinds <= {"integer", "number"}:
        return "number"
    if kinds == {"date"}:
        return "date"
    if kinds <= {"date", "datetime"}:
        return "datetime"
    return "string"


def suggest_name(raw: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw.strip())
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "col"
    if not s[0].isalpha():
        s = "c_" + s
    return s[:40].rstrip("_")


def dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            n = f"{n[:36]}_{seen[n]}"
        seen.setdefault(n, 1)
        out.append(n)
    return out


def affinity(declared: str) -> tuple[str, str | None]:
    """(sqlite affinity family, natural model type or None) for a declared column type."""
    d = declared.upper()
    if "BOOL" in d:
        return "integer", "boolean"
    if "JSON" in d:
        return "text", "json"
    if "INT" in d:
        return "integer", "integer"
    if any(x in d for x in ("CHAR", "CLOB", "TEXT")):
        return "text", None
    if "BLOB" in d:
        return "blob", None
    if d == "":
        return "none", None
    if any(x in d for x in ("REAL", "FLOA", "DOUB")):
        return "real", "number"
    if "DATETIME" in d or "TIMESTAMP" in d:
        return "numeric", "datetime"
    if "DATE" in d:
        return "numeric", "date"
    return "numeric", "number"


# model types a column of each affinity family may be declared as (overrides are checked against this)
COMPATIBLE = {
    "integer": {"integer", "boolean"},
    "text": {"string", "date", "datetime", "json"},
    "real": {"number"},
    "numeric": {"number", "integer", "date", "datetime"},
    "none": {"string", "integer", "number", "date", "datetime", "json"},
}
