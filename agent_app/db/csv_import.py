"""User-initiated CSV import (UI button). Creates a typed table; the agent then adopts it as a model."""
from __future__ import annotations

import csv
import io
from typing import Any

from .models import quote
from .profile import dedupe, infer_type, suggest_name

MAX_ROWS = 200_000
DECLARED = {"integer": "INTEGER", "number": "REAL", "date": "DATE", "datetime": "DATETIME", "string": "TEXT"}


def parse_csv(raw: bytes) -> tuple[list[str], list[str], list[list[Any]], list[str]]:
    """-> (original headers, column names, typed rows, declared types)"""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError("CSV must be UTF-8 encoded") from e
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV is empty") from None
    names = dedupe([suggest_name(h) if h.strip() else f"column_{i + 1}" for i, h in enumerate(header)])
    rows = []
    for n, r in enumerate(reader, start=2):
        if not any(x.strip() for x in r):
            continue
        if len(r) != len(header):
            raise ValueError(f"Line {n} has {len(r)} fields, expected {len(header)}")
        rows.append([x if x.strip() else None for x in r])
        if len(rows) > MAX_ROWS:
            raise ValueError(f"More than {MAX_ROWS} rows; import a smaller file")
    types = []
    for i in range(len(names)):
        t = infer_type(r[i] for r in rows)
        types.append(t)
        conv = {"integer": int, "number": float}.get(t)
        if conv:
            for r in rows:
                if r[i] is not None:
                    r[i] = conv(r[i].strip())
    return header, names, rows, [DECLARED[t] for t in types]


def import_csv(db: Any, raw: bytes, table: str) -> dict[str, Any]:
    _, names, rows, declared = parse_csv(raw)
    table = suggest_name(table)
    if table.startswith("sqlite_") or table == "_models" or table in db.table_names():
        raise ValueError(f"A table named '{table}' already exists or is reserved; choose another name")
    ddl = f"CREATE TABLE {quote(table)} (" + ", ".join(f"{quote(n)} {d}" for n, d in zip(names, declared)) + ")"
    with db._rw() as c:
        c.execute(ddl)
        c.executemany(f"INSERT INTO {quote(table)} VALUES ({', '.join('?' * len(names))})", rows)
    return {"table": table, "rows": len(rows), "columns": [{"name": n, "declared_type": d} for n, d in zip(names, declared)]}
