"""Streaming file import (CSV or JSON) for files too big to hold in memory (hundreds of MB).

CSV: comma, semicolon, tab or pipe separated, UTF-8, a header row; blank cells become NULL. JSON, accepted shapes: a top-level array of objects `[{...}, {...}]`, JSON Lines (one object per line), or an object wrapping such an array
(`{"results": [{...}]}`). Records are read one at a time; memory stays small whatever the file size.

Two passes over the file: (1) learn the columns and their types without keeping any rows, (2) load them in batches inside ONE
transaction, so a failure or a Cancel leaves the database exactly as it was (a new table is not left behind half-filled).
Nested objects become `parent_child` columns; arrays (and objects nested deeper than 3 levels) are stored as JSON text.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterator

from .models import quote
from .profile import classify, dedupe, suggest_name

MAX_COLUMNS = 500
MAX_DEPTH = 3
MAX_RECORD_CHARS = 64_000_000      # a single record bigger than this is treated as invalid JSON, which also bounds memory
CHUNK = 1 << 20
LINE_PROBE = 2 << 20             # how far into the file to look for the end of the first line when telling JSON Lines from an object
BATCH = 5000
CSV_EXTENSIONS = (".csv", ".tsv")
EXTENSIONS = (".json", ".jsonl", ".ndjson", *CSV_EXTENSIONS)
DECLARED = {"integer": "INTEGER", "number": "REAL", "date": "DATE", "datetime": "DATETIME", "string": "TEXT"}
_DEC = json.JSONDecoder()

Progress = Callable[[str, int, int, int], None]   # (phase 'scan'|'load', rows so far, bytes read, file size)


class Cancelled(Exception):
    """Raised when the caller asked to stop; the transaction is rolled back."""


class ImportFileError(ValueError):
    pass


def resolve_path(raw: str) -> Path:
    p = Path(raw.strip()).expanduser()
    if not p.is_absolute():
        raise ImportFileError("Give the full path of the file, e.g. /Users/you/data/findings.json")
    p = p.resolve()
    if p.suffix.lower() not in EXTENSIONS:
        raise ImportFileError(f"Only {', '.join(EXTENSIONS)} files can be imported this way")
    if not p.is_file():
        raise ImportFileError(f"No such file: {p}")
    if p.stat().st_size == 0:
        raise ImportFileError("The file is empty")
    return p


class _Stream:
    """A buffered text stream with `raw_decode` that refills as needed (the standard library cannot decode incrementally)."""

    def __init__(self, path: Path):
        self.f = open(path, "rb")
        self.text = io.TextIOWrapper(self.f, encoding="utf-8-sig", errors="strict")
        self.buf, self.pos, self.eof, self.dropped = "", 0, False, 0
        self.total = path.stat().st_size

    def close(self) -> None:
        self.text.close()

    @property
    def bytes_read(self) -> int:
        try:
            return min(self.f.tell(), self.total)
        except ValueError:
            return self.total

    def fill(self, compact: bool = True) -> None:
        if self.eof:
            return
        try:
            chunk = self.text.read(CHUNK)
        except UnicodeDecodeError as e:
            raise ImportFileError("The file must be UTF-8 encoded") from e
        if not chunk:
            self.eof = True
            return
        if compact and self.pos > 0:  # drop what was already consumed (keeps the buffer, and so memory, small)
            self.dropped += self.pos
            self.buf, self.pos = self.buf[self.pos:], 0
        self.buf += chunk

    def peek(self, offset: int = 0) -> str:
        """Next non-whitespace char at or after pos+offset ('' at end of file)."""
        i = self.pos + offset
        while True:
            while i < len(self.buf) and self.buf[i] in " \t\r\n":
                i += 1
            if i < len(self.buf):
                return self.buf[i]
            if self.eof:
                return ""
            self.fill(compact=False)

    def skip_ws(self) -> str:
        c = self.peek()
        while self.pos < len(self.buf) and self.buf[self.pos] in " \t\r\n":
            self.pos += 1
        return c

    def expect(self, ch: str) -> None:
        if self.skip_ws() != ch:
            raise ImportFileError(f"Invalid JSON near character {self.dropped + self.pos}: expected '{ch}'")
        self.pos += 1

    def value(self) -> Any:
        self.skip_ws()
        while True:
            try:
                v, end = _DEC.raw_decode(self.buf, self.pos)
                if end < len(self.buf) or self.eof:
                    self.pos = end
                    return v
                # decoded exactly up to the end of the buffer: a number may be cut in half, so read more and decode again
            except json.JSONDecodeError as e:
                incomplete = e.msg.startswith("Unterminated") or e.pos >= len(self.buf) - 8
                if self.eof or not incomplete:
                    raise ImportFileError(f"Invalid JSON near character {self.dropped + e.pos}: {e.msg}") from e
            if len(self.buf) - self.pos > MAX_RECORD_CHARS:
                raise ImportFileError("A single JSON value is larger than 64 MB, or the JSON is invalid here")
            self.fill()


def _array_items(s: _Stream) -> Iterator[dict]:
    s.expect("[")
    if s.skip_ws() == "]":
        return
    while True:
        v = s.value()
        if not isinstance(v, dict):
            raise ImportFileError("Every item of the array must be a JSON object")
        yield v
        c = s.skip_ws()
        s.pos += 1
        if c == "]":
            return
        if c != ",":
            raise ImportFileError(f"Invalid JSON near character {s.dropped + s.pos}: expected ',' or ']'")


class Records:
    """Iterate the records of a file of any supported shape. `bytes_read` / `total` give progress; call close() when done."""

    def __init__(self, path: Path, records_key: str | None = None):
        self.s, self.key = _Stream(path), records_key
        self.total = self.s.total

    @property
    def bytes_read(self) -> int:
        return self.s.bytes_read

    def close(self) -> None:
        self.s.close()

    def _looks_like_jsonl(self) -> bool:
        s = self.s
        while s.buf.find("\n", s.pos) < 0 and len(s.buf) - s.pos < LINE_PROBE and not s.eof:
            s.fill(compact=False)
        end = s.buf.find("\n", s.pos)
        if end < 0:
            return False
        try:
            first = json.loads(s.buf[s.pos:end])
        except ValueError:
            return False
        return isinstance(first, dict) and s.peek(end - s.pos) == "{"

    def __iter__(self) -> Iterator[dict]:
        s = self.s
        c = s.skip_ws()
        if c == "":
            return  # blank file: the caller reports "no records"
        if c == "[":
            yield from _array_items(s)
        elif c == "{" and self._looks_like_jsonl():
            while s.skip_ws():
                v = s.value()
                if not isinstance(v, dict):
                    raise ImportFileError("Every line of a JSON Lines file must be a JSON object")
                yield v
        elif c == "{":  # an object wrapping the records: walk its members until the array of objects
            s.expect("{")
            found = False
            while s.skip_ws() != "}":
                key = s.value()
                if not isinstance(key, str):
                    raise ImportFileError("Invalid JSON: an object key must be a string")
                s.expect(":")
                if s.skip_ws() == "[" and s.peek(1) == "{" and self.key in (None, key):
                    found = True
                    yield from _array_items(s)
                    break
                s.value()  # some other member: read and discard it
                if s.skip_ws() == ",":
                    s.pos += 1
            if not found:
                raise ImportFileError("No array of objects found inside the top-level object"
                                      + (f" under the key '{self.key}'" if self.key else ""))
        else:
            raise ImportFileError("A JSON file must start with '[' (array of records) or '{' (JSON Lines, or an object holding the array)")


class CsvRecords:
    """Rows of a CSV file as dicts, with the same interface as `Records` (iterate, bytes_read, total, close)."""

    MAX_FIELD = 50_000_000

    def __init__(self, path: Path, records_key: str | None = None):
        self.f = open(path, "rb")
        self.text = io.TextIOWrapper(self.f, encoding="utf-8-sig", errors="strict", newline="")
        self.total = path.stat().st_size

    @property
    def bytes_read(self) -> int:
        try:
            return min(self.f.tell(), self.total)
        except ValueError:
            return self.total

    def close(self) -> None:
        self.text.close()

    @staticmethod
    def _delimiter(first_line: str) -> str:
        """The separator that splits the header into the most columns (quoted separators do not count); comma on a tie or none."""
        best, best_n = ",", 1
        for d in ",;\t|":
            try:
                n = len(next(csv.reader([first_line], delimiter=d)))
            except (csv.Error, StopIteration):
                continue
            if n > best_n:
                best, best_n = d, n
        return best

    def __iter__(self) -> Iterator[dict]:
        csv.field_size_limit(self.MAX_FIELD)
        reader = None
        try:
            first = self.text.readline()
            if not first.strip():
                return
            delimiter = self._delimiter(first)
            self.text.seek(0)
            reader = csv.reader(self.text, delimiter=delimiter)
            header = next(reader)
            seen: dict[str, int] = {}
            keys = []
            for i, h in enumerate(header):  # unique raw keys even when the header repeats or has blanks (names are cleaned later)
                k = h.strip() or f"column_{i + 1}"
                seen[k] = seen.get(k, 0) + 1
                keys.append(k if seen[k] == 1 else f"{k}__{seen[k]}")
            for row in reader:
                if not any(x.strip() for x in row):
                    continue
                if len(row) != len(keys):
                    raise ImportFileError(f"Line {reader.line_num} has {len(row)} fields, expected {len(keys)}")
                yield {k: (v if v.strip() else None) for k, v in zip(keys, row)}  # blank cell = NULL, as in the small CSV import
        except UnicodeDecodeError as e:
            raise ImportFileError("The file must be UTF-8 encoded") from e
        except csv.Error as e:
            raise ImportFileError(f"Invalid CSV near line {reader.line_num if reader else 1}: {e}") from e


def open_records(path: Path, records_key: str | None = None) -> "Records | CsvRecords":
    return (CsvRecords if path.suffix.lower() in CSV_EXTENSIONS else Records)(path, records_key)


# ------------------------------------------------------------------------------------------------ flattening and types
def flatten(rec: dict, prefix: str = "", depth: int = 0, out: dict | None = None) -> dict[str, Any]:
    out = {} if out is None else out
    for k, v in rec.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict) and v and depth < MAX_DEPTH:
            flatten(v, name + "_", depth + 1, out)
        elif isinstance(v, (dict, list)):
            out[name] = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        else:
            out[name] = v
    return out


def _kind(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int) and abs(v) >= 2 ** 63:
        return "string"  # does not fit SQLite's 64-bit integer: keep the digits as text
    return classify(v) if v != "" else None


def _type(kinds: set[str]) -> str:
    if not kinds:
        return "string"
    if kinds == {"boolean"}:
        return "integer"  # stored as 0/1
    if "boolean" in kinds:
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


def _cell(v: Any, target: str) -> Any:
    """Convert a JSON value for a column whose type is integer | number | string (dates are text)."""
    if v is None:
        return None
    if target == "integer":
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        return int(v.strip()) if isinstance(v, str) and v.strip() else (None if v == "" else int(v))
    if target == "number":
        if isinstance(v, bool):
            return float(v)
        return float(v.strip()) if isinstance(v, str) and v.strip() else (None if v == "" else float(v))
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _target_of_declared(declared: str) -> str:
    d = (declared or "").upper()
    if "INT" in d:
        return "integer"
    if any(x in d for x in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return "number"
    return "string"


# ------------------------------------------------------------------------------------------------------ the two passes
def scan(path: Path, records_key: str | None = None, limit: int | None = None, progress: Progress | None = None,
         cancelled: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    """Pass 1: columns (in first-seen order), their types and the record count. Keeps no rows."""
    kinds: dict[str, set[str]] = {}
    n = 0
    rd = open_records(path, records_key)
    try:
        for rec in rd:
            for k, v in flatten(rec).items():
                ks = kinds.setdefault(k, set())
                if len(kinds) > MAX_COLUMNS:
                    raise ImportFileError(f"More than {MAX_COLUMNS} different columns: the keys look like data, not field names")
                if (kd := _kind(v)) is not None:
                    ks.add(kd)
            n += 1
            if n % 2000 == 0:
                if cancelled():
                    raise Cancelled()
                if progress:
                    progress("scan", n, rd.bytes_read, rd.total)
            if limit is not None and n >= limit:
                break
    finally:
        rd.close()
    if n == 0:
        raise ImportFileError("The file has no data rows" if path.suffix.lower() in CSV_EXTENSIONS else "The file contains no records")
    raw_names = list(kinds)
    return {"rows": n, "raw_names": raw_names, "names": dedupe([suggest_name(k) for k in raw_names]),
            "types": [_type(kinds[k]) for k in raw_names]}


def load(db: Any, path: Path, table: str, mode: str, records_key: str | None = None, progress: Progress | None = None,
         cancelled: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    """Import `path` into `table`. mode: new (create; error if it exists) | append | replace (delete all rows first, keep the schema)."""
    if mode not in ("new", "append", "replace"):
        raise ImportFileError("mode must be new, append or replace")
    table = suggest_name(table)
    if table.startswith("sqlite_") or table == "_models":
        raise ImportFileError(f"'{table}' is a reserved name")
    exists = table in db.table_names()
    if mode == "new" and exists:
        raise ImportFileError(f"A table named '{table}' already exists. Choose another name, or use append / replace.")
    if mode != "new" and not exists:
        raise ImportFileError(f"There is no table named '{table}' to {mode}. Use mode 'new' to create it.")

    info = scan(path, records_key, progress=progress, cancelled=cancelled)
    raw_names, cols, types = info["raw_names"], info["names"], info["types"]
    if mode == "new":
        targets = [t if t in ("integer", "number") else "string" for t in types]
        declared = [DECLARED[t] for t in types]
        ddl = f"CREATE TABLE {quote(table)} (" + ", ".join(f"{quote(n)} {d}" for n, d in zip(cols, declared)) + ")"
    else:
        existing = {c["name"]: c["type"] for c in db.pragma_columns(table)}
        unknown = [n for n in cols if n not in existing]
        if unknown:
            raise ImportFileError(f"The file has columns that table '{table}' does not have: {', '.join(unknown[:10])}"
                                  + (" ..." if len(unknown) > 10 else "") + ". Import into a new table instead.")
        declared = [existing[n] for n in cols]
        targets, ddl = [_target_of_declared(d) for d in declared], ""
    index = {k: i for i, k in enumerate(raw_names)}
    sql = f"INSERT INTO {quote(table)} ({', '.join(map(quote, cols))}) VALUES ({', '.join('?' * len(cols))})"
    removed, n = 0, 0
    rd = open_records(path, records_key)
    try:
        with db._rw() as c:
            c.execute("BEGIN IMMEDIATE")  # DDL included: a failure or Cancel rolls everything back
            if ddl:
                c.execute(ddl)
            elif mode == "replace":
                removed = c.execute(f"SELECT COUNT(*) FROM {quote(table)}").fetchone()[0]
                c.execute(f"DELETE FROM {quote(table)}")
            batch: list[list[Any]] = []
            for rec in rd:
                row: list[Any] = [None] * len(cols)
                for k, v in flatten(rec).items():
                    i = index.get(k)
                    if i is None:
                        continue
                    try:
                        row[i] = _cell(v, targets[i])
                    except (ValueError, TypeError) as e:
                        raise ImportFileError(f"Record {n + 1}, column '{cols[i]}': cannot store {v!r} as {targets[i]}") from e
                batch.append(row)
                n += 1
                if len(batch) >= BATCH:
                    c.executemany(sql, batch)
                    batch.clear()
                    if cancelled():
                        raise Cancelled()
                    if progress:
                        progress("load", n, rd.bytes_read, rd.total)
            if batch:
                c.executemany(sql, batch)
    except sqlite3.Error as e:
        raise ImportFileError(f"The database refused the data (nothing was changed): {e}") from e
    finally:
        rd.close()
    return {"table": table, "mode": mode, "rows": n, "replaced_rows": removed if mode == "replace" else None,
            "columns": [{"name": c_, "declared_type": d} for c_, d in zip(cols, declared)]}
