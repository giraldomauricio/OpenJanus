"""SQLite access. All agent-facing writes go through validated ``ModelSpec`` models."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import ValidationError

from .introspect import IntrospectionMixin
from .migrate import MigrationMixin
from .models import ModelSpec, quote

_ALLOWED_READ_ACTIONS = {
    sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE,
}
META_TABLE = "_models"


def _authorizer(action: int, *_: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_READ_ACTIONS else sqlite3.SQLITE_DENY


def format_validation_error(e: ValidationError, prefix: str = "") -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(value)"
        parts.append(f"{prefix}field '{loc}': {err['msg']}")
    return "; ".join(parts)


class Database(IntrospectionMixin, MigrationMixin):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._rw() as c:
            c.execute(
                f"CREATE TABLE IF NOT EXISTS {META_TABLE} "
                "(name TEXT PRIMARY KEY, spec TEXT NOT NULL, created_at TEXT NOT NULL)"
            )

    # --- connections -------------------------------------------------------

    @contextmanager
    def _rw(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def _ro(self) -> Iterator[sqlite3.Connection]:
        """Read-only connection with an authorizer that only permits SELECT."""
        conn = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.set_authorizer(_authorizer)
        try:
            yield conn
        finally:
            conn.close()

    # --- reads -------------------------------------------------------------

    def query(self, sql: str, params: list | dict | None = None, limit: int = 200) -> dict[str, Any]:
        """Run one read-only statement. Raises on anything that is not a SELECT."""
        deadline = time.monotonic() + 5

        def _timeout() -> int:
            return 1 if time.monotonic() > deadline else 0

        with self._ro() as conn:
            conn.set_progress_handler(_timeout, 100_000)
            try:
                cur = conn.execute(sql.strip().rstrip(";"), params or [])
                rows = cur.fetchmany(limit + 1)
            except sqlite3.DatabaseError as e:
                raise ValueError(f"SQL rejected: {e}") from e
            truncated = len(rows) > limit
            cols = [d[0] for d in cur.description or []]
            return {"columns": cols, "rows": [dict(r) for r in rows[:limit]], "truncated": truncated}

    def export_csv(self, table: str, chunk: int = 500) -> Iterator[str]:
        """The whole table as CSV text, streamed in chunks (UTF-8 BOM for Excel; formula-looking text is neutralised)."""
        import csv
        import io
        import re

        num = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")

        def cell(v: Any) -> Any:
            if v is None:
                return ""
            if isinstance(v, bytes):
                return f"<blob {len(v)} bytes>"
            if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r") and not num.match(v.strip()):
                return "'" + v
            return v

        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\r\n")
        with self._ro() as conn:
            cur = conn.execute(f"SELECT * FROM {quote(table)}")
            writer.writerow([cell(d[0]) for d in cur.description])
            yield "\ufeff" + buf.getvalue()
            while rows := cur.fetchmany(chunk):
                buf.seek(0)
                buf.truncate()
                writer.writerows([[cell(v) for v in r] for r in rows])
                yield buf.getvalue()

    def table_names(self) -> list[str]:
        r = self.query(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            f"AND name != '{META_TABLE}' ORDER BY name"
        )
        return [row["name"] for row in r["rows"]]

    def list_tables(self) -> list[dict[str, Any]]:
        out = []
        for name in self.table_names():
            count = self.query(f"SELECT COUNT(*) AS n FROM {quote(name)}")["rows"][0]["n"]
            out.append({"table": name, "rows": count, "managed": self.get_spec(name) is not None})
        return out

    def describe(self, table: str) -> dict[str, Any]:
        if table not in self.table_names():
            raise ValueError(f"No such table: {table}. Existing: {self.table_names()}")
        cols = self.pragma_columns(table)
        spec = self.get_spec(table)
        referenced_by = [{"model": s.name, "field": f.name} for s in self.all_specs() for f in s.fields
                         if f.references and f.references.split(".")[0] == table]
        return {
            "table": table,
            "managed_by_model": spec is not None,
            "columns": [{"name": c["name"], "sqlite_type": c["type"], "not_null": bool(c["notnull"]),
                         "primary_key": bool(c["pk"])} for c in cols],
            "model": spec.model_dump() if spec else None,
            "referenced_by": referenced_by,
        }

    # --- model registry ----------------------------------------------------

    def get_spec(self, name: str) -> ModelSpec | None:
        r = self.query("SELECT spec FROM _models WHERE name = ?", [name])["rows"]
        return ModelSpec.model_validate_json(r[0]["spec"]) if r else None

    def all_specs(self) -> list[ModelSpec]:
        return [ModelSpec.model_validate_json(r["spec"])
                for r in self.query("SELECT spec FROM _models ORDER BY name")["rows"]]

    def _require_spec(self, name: str) -> ModelSpec:
        spec = self.get_spec(name)
        if spec is None:
            hint = "It exists but was not created via define_model; writes need a model." \
                if name in self.table_names() else f"Defined models: {[s.name for s in self.all_specs()]}"
            raise ValueError(f"No model named '{name}'. {hint}")
        return spec

    def define_model(self, spec: ModelSpec) -> dict[str, Any]:
        existing = self.get_spec(spec.name)
        if existing:
            if existing == spec:
                return {"status": "unchanged", **spec.export()}
            raise ValueError(
                f"Model '{spec.name}' already exists with a different definition. "
                f"Existing: {existing.model_dump_json()}. To change an existing model use alter_model "
                "(add/drop/rename/change fields) or convert_column (change a column's type)."
            )
        if spec.name in self.table_names():
            raise ValueError(f"A table named '{spec.name}' already exists that was not created from a model. "
                             "Use adopt_table to register it instead.")
        self._validate_references(spec)
        with self._rw() as c:
            c.execute(spec.to_sqlite_ddl())
            c.execute(f"INSERT INTO {META_TABLE} (name, spec, created_at) VALUES (?, ?, ?)",
                      (spec.name, spec.model_dump_json(), datetime.now(timezone.utc).isoformat()))
        return {"status": "created", **spec.export()}

    def export_models(self, name: str | None = None) -> list[dict[str, Any]]:
        specs = [self._require_spec(name)] if name else self.all_specs()
        return [s.export() for s in specs]

    # --- validated writes --------------------------------------------------

    @staticmethod
    def _to_db(spec: ModelSpec, values: dict[str, Any]) -> dict[str, Any]:
        out = {}
        types = {f.name: f.type for f in spec.fields}
        for k, v in values.items():
            if v is None and types[k] == "integer" and spec.pk and spec.pk.name == k:
                continue  # let sqlite assign the id
            out[k] = json.dumps(v) if types[k] == "json" and v is not None else v
        return out

    def insert(self, model: str, rows: list[dict[str, Any]],
               on_conflict: Literal["error", "ignore", "replace"] = "error") -> dict[str, Any]:
        spec = self._require_spec(model)
        Model = spec.pydantic_model()
        validated, errors = [], []
        for i, row in enumerate(rows):
            try:
                validated.append(self._to_db(spec, Model.model_validate(row).model_dump(mode="json")))
            except ValidationError as e:
                errors.append(format_validation_error(e, prefix=f"row {i}: "))
        if errors:
            raise ValueError("Nothing inserted. " + " | ".join(errors[:20]))
        verb = {"error": "INSERT", "ignore": "INSERT OR IGNORE", "replace": "INSERT OR REPLACE"}[on_conflict]
        inserted, last_id = 0, None
        try:
            with self._rw() as c:
                for v in validated:
                    cols = list(v)
                    sql = (f"{verb} INTO {quote(model)} ({', '.join(map(quote, cols))}) "
                           f"VALUES ({', '.join('?' * len(cols))})") if cols else \
                        f"{verb} INTO {quote(model)} DEFAULT VALUES"
                    cur = c.execute(sql, [v[k] for k in cols])
                    if cur.rowcount > 0:
                        inserted += 1
                        last_id = cur.lastrowid
        except sqlite3.IntegrityError as e:
            raise ValueError(f"Nothing inserted: {e}") from e
        return {"inserted": inserted, "skipped": len(validated) - inserted, "last_row_id": last_id}

    def _where(self, spec: ModelSpec, where: dict[str, Any]) -> tuple[str, list[Any]]:
        if not where:
            raise ValueError("'where' must contain at least one column filter")
        known = {f.name for f in spec.fields}
        clauses, args = [], []
        for col, val in where.items():
            if col not in known:
                raise ValueError(f"Unknown column '{col}' in where. Columns: {sorted(known)}")
            if val is None:
                clauses.append(f"{quote(col)} IS NULL")
            else:
                clauses.append(f"{quote(col)} = ?")
                args.append(val)
        return " AND ".join(clauses), args

    def update(self, model: str, values: dict[str, Any], where: dict[str, Any]) -> dict[str, Any]:
        spec = self._require_spec(model)
        if not values:
            raise ValueError("'values' is empty")
        try:
            clean = spec.pydantic_model(partial=True).model_validate(values).model_dump(mode="json", exclude_unset=True)
        except ValidationError as e:
            raise ValueError(format_validation_error(e)) from e
        clean = {k: json.dumps(v) if next(f for f in spec.fields if f.name == k).type == "json" and v is not None
                 else v for k, v in clean.items()}
        w_sql, w_args = self._where(spec, where)
        sets = ", ".join(f"{quote(k)} = ?" for k in clean)
        try:
            with self._rw() as c:
                cur = c.execute(f"UPDATE {quote(model)} SET {sets} WHERE {w_sql}", [*clean.values(), *w_args])
                return {"updated": cur.rowcount}
        except sqlite3.IntegrityError as e:
            raise ValueError(f"Nothing updated: {e}") from e

    def delete(self, model: str, where: dict[str, Any]) -> dict[str, Any]:
        spec = self._require_spec(model)
        w_sql, w_args = self._where(spec, where)
        with self._rw() as c:
            cur = c.execute(f"DELETE FROM {quote(model)} WHERE {w_sql}", w_args)
            return {"deleted": cur.rowcount}
