"""Schema evolution for models: ``alter_model`` (add / drop / rename / change fields) and ``convert_column`` (text -> typed).

Safety design
  * Every change runs inside ONE transaction on a dedicated connection. A dry run executes the whole change and rolls it
    back, so the report (rows copied, conversion failures, constraint violations) is exactly what apply would do.
  * ``apply`` re-runs the same change for real, after taking a full backup of the database (``data/backups/``, newest 10 kept).
  * Tables created by ``define_model`` ("managed") are rebuilt when needed (create new table, copy, drop, rename), which
    keeps constraints and real foreign keys. Adopted tables (the user's own) are never rebuilt, because that could lose
    indexes, triggers or constraints we cannot see; only SQLite's native ADD/RENAME/DROP COLUMN are used on them.
  * Conversions are strict: a value that does not parse is reported, never silently turned into 0 or NULL
    (``on_error="null"`` is an explicit opt-in).
"""
from __future__ import annotations

import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field, TypeAdapter

from .models import FieldSpec, FieldType, ModelSpec, SQLITE_TYPES, quote

MAX_REBUILD_ROWS = 500_000
KEEP_BACKUPS = 10


# ---- strict value conversion ---------------------------------------------------------------------------------------

class ConversionError(ValueError):
    pass


_NUM = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_TRUE, _FALSE = {"true", "t", "yes", "y", "1"}, {"false", "f", "no", "n", "0"}


def _num_text(s: str) -> str:
    s = s.strip()
    return s.replace(",", "") if _THOUSANDS.match(s) else s


def convert_value(v: Any, to: str, date_format: str | None = None) -> Any:
    """Convert one stored value to type ``to`` or raise ConversionError. Empty strings and NULL become NULL."""
    if v is None:
        return None
    if isinstance(v, bytes):
        raise ConversionError("binary data cannot be converted")
    if isinstance(v, str):
        v = v.strip()
        if v == "":
            return None
    if to == "string":
        return v if isinstance(v, str) else str(v)
    if to in ("integer", "number"):
        if isinstance(v, str):
            t = _num_text(v)
            if not _NUM.match(t):
                raise ConversionError(f"{v!r} is not a number")
            f = float(t)
            if to == "number":
                return f
            if f != int(f):
                raise ConversionError(f"{v!r} is not a whole number")
            i = int(t) if re.fullmatch(r"[+-]?\d+", t) else int(f)
            if not -(2**63) <= i < 2**63:
                raise ConversionError(f"{str(v)[:30]!r} is too large for an integer column (SQLite integers are 64-bit); keep it as text")
            return i
        f = float(v)
        if to == "number":
            return f
        if f != int(f):
            raise ConversionError(f"{v!r} is not a whole number")
        if not -(2**63) <= int(f) < 2**63:
            raise ConversionError(f"{v!r} is too large for an integer column")
        return int(f)
    if to == "boolean":
        if isinstance(v, (int, float)) and v in (0, 1):
            return int(v)
        if isinstance(v, str) and v.lower() in _TRUE | _FALSE:
            return 1 if v.lower() in _TRUE else 0
        raise ConversionError(f"{v!r} is not a boolean (use true/false, yes/no or 1/0)")
    if to in ("date", "datetime"):
        if not isinstance(v, str):
            raise ConversionError(f"{v!r} is not a date")
        try:
            if date_format:
                d = datetime.strptime(v, date_format)
            elif _ISO_DATE.match(v):
                d = datetime.fromisoformat(v.replace("Z", "+00:00")) if len(v) > 10 else datetime.fromisoformat(v)
            else:
                raise ValueError
        except ValueError:
            hint = f"does not match date_format {date_format!r}" if date_format else \
                "is not ISO (YYYY-MM-DD); pass date_format, e.g. '%m/%d/%Y' or '%d/%m/%Y'"
            raise ConversionError(f"{v!r} {hint}") from None
        return d.date().isoformat() if to == "date" else d.isoformat()
    raise ConversionError(f"conversion to {to} is not supported")


_CONVERTIBLE_TO = {"string", "integer", "number", "boolean", "date", "datetime"}


# ---- operations ----------------------------------------------------------------------------------------------------

class AddField(BaseModel):
    op: Literal["add_field"]
    field: FieldSpec = Field(description="The new column. Required columns on a table with rows need a default to fill them.")


class DropField(BaseModel):
    op: Literal["drop_field"]
    name: str


class RenameField(BaseModel):
    op: Literal["rename_field"]
    name: str
    new_name: str = Field(description="snake_case")


class ChangeField(BaseModel):
    op: Literal["change_field"]
    name: str
    type: FieldType | None = Field(None, description="Convert the column to this type (strict; failures are reported)")
    required: bool | None = None
    unique: bool | None = None
    default: str | int | float | bool | None = Field(None, description="Set the default (also fills existing NULLs)")
    clear_default: bool = False
    description: str | None = None
    references: str | None = Field(None, description="'table.column' to add a foreign key, '' to remove it")
    date_format: str | None = Field(None, description="strptime format for text -> date/datetime, e.g. '%m/%d/%Y'")
    on_error: Literal["fail", "null"] = Field("fail", description="'null' turns values that cannot be converted into NULL (data loss)")


# A plain union (anyOf) rather than a tagged one: the OpenAPI-only "discriminator" keyword is rejected by some providers' schema validators.
Op = AddField | DropField | RenameField | ChangeField
_OP = TypeAdapter(Op)


class _Problem(Exception):
    """A reason the change cannot be applied (reported to the caller, not raised out of alter_model)."""

    def __init__(self, msg: str, failures: list[dict[str, Any]] | None = None):
        super().__init__(msg)
        self.failures = failures or []


def _friendly(e: Exception) -> str:
    """SQLite's message without our internal temp-table name, plus a hint for the usual causes."""
    msg = str(e).replace("__new.", ".")
    hint = {"UNIQUE": " (some existing rows have duplicate values)", "NOT NULL": " (some existing rows have no value)",
            "CHECK": " (some existing values violate the column's check)"}
    return f"SQLite rejected the change: {msg}" + next((h for k, h in hint.items() if k in msg), "")


def _sql_literal(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def _db_default(f: FieldSpec) -> Any:
    return int(f.default) if isinstance(f.default, bool) else f.default


class MigrationMixin:
    # relies on Database: path, _require_spec, all_specs, query, table_names, _validate_references

    # --- plumbing ----------------------------------------------------------

    @contextmanager
    def _migration_conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None)  # type: ignore[attr-defined]  # manual BEGIN/COMMIT
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=OFF")  # required while rebuilding; integrity is checked explicitly before commit
        try:
            yield conn
        finally:
            conn.close()

    def _backup(self, label: str) -> str:
        folder = self.path.parent / "backups"  # type: ignore[attr-defined]
        folder.mkdir(exist_ok=True)
        dest = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{re.sub(r'[^a-z0-9_]+', '_', label.lower())}.db"
        src, dst = sqlite3.connect(self.path), sqlite3.connect(dest)  # type: ignore[attr-defined]
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        for old in sorted(folder.glob("*.db"))[:-KEEP_BACKUPS]:
            old.unlink(missing_ok=True)
        return str(dest)

    @staticmethod
    def _save_spec_conn(conn: sqlite3.Connection, spec: ModelSpec) -> None:
        conn.execute("UPDATE _models SET spec = ? WHERE name = ?", (spec.model_dump_json(), spec.name))

    @staticmethod
    def _revalidate(spec: ModelSpec) -> ModelSpec:
        return ModelSpec.model_validate(spec.model_dump())

    @staticmethod
    def _count(conn: sqlite3.Connection, table: str, where: str = "1") -> int:
        return conn.execute(f"SELECT COUNT(*) FROM {quote(table)} WHERE {where}").fetchone()[0]

    def _referencing(self, others: list[ModelSpec], spec: ModelSpec, field: str) -> list[tuple[ModelSpec, FieldSpec]]:
        target = f"{spec.name}.{field}"
        return [(s, f) for s in [*others, spec] for f in s.fields if f.references == target]

    # --- alter_model -------------------------------------------------------

    def alter_model(self, model: str, ops: list[Any], apply: bool = False) -> dict[str, Any]:
        spec = self._require_spec(model)  # type: ignore[attr-defined]
        parsed = [_OP.validate_python(o) for o in ops]
        if not parsed:
            raise ValueError("ops is empty")
        result = self._run_ops(model, spec, parsed, commit=False)
        if apply and not result["problems"]:
            backup = self._backup(f"before-alter-{model}")
            result = self._run_ops(model, spec, parsed, commit=True)
            result["backup"] = backup
        result["dry_run"] = not result["applied"]
        if not result["applied"] and not result["problems"]:
            result["next"] = "Nothing was changed. Show this plan to the user; if they agree, call again with apply=true."
        return result

    def _run_ops(self, model: str, spec: ModelSpec, ops: list[Any], commit: bool) -> dict[str, Any]:
        others = [s for s in self.all_specs() if s.name != model]  # type: ignore[attr-defined]
        report: dict[str, Any] = {"applied": False, "plan": [], "problems": [], "notes": [], "failures": [], "rows": 0}
        with self._migration_conn() as conn:
            fk_before = len(conn.execute("PRAGMA foreign_key_check").fetchall())
            conn.execute("BEGIN")
            try:
                if sqlite3.sqlite_version_info < (3, 35, 0):
                    raise _Problem(f"SQLite {sqlite3.sqlite_version} is too old for this (3.35+ needed)")
                report["rows"] = self._count(conn, model)
                for op in ops:
                    spec = self._apply_op(conn, spec, op, others, report)
                fk_after = conn.execute("PRAGMA foreign_key_check").fetchall()
                if len(fk_after) > fk_before:
                    raise _Problem(f"The change would leave {len(fk_after) - fk_before} row(s) pointing at missing parent rows "
                                   "(foreign key violations); fix those rows first")
                report["notes"] = list(dict.fromkeys(report["notes"]))  # a multi-step change would repeat the "rebuilt" note
                report["model"] = spec.model_dump()
                report["ddl"] = None if spec.adopted else spec.to_sqlite_ddl()
            except _Problem as e:
                report["problems"].append(str(e))
                report["failures"] += e.failures
            except sqlite3.DatabaseError as e:
                report["problems"].append(_friendly(e))
            if commit and not report["problems"]:
                conn.execute("COMMIT")
                report["applied"] = True
            else:
                conn.execute("ROLLBACK")
        return report

    def _apply_op(self, conn: sqlite3.Connection, spec: ModelSpec, op: Any, others: list[ModelSpec],
                  report: dict[str, Any]) -> ModelSpec:
        by_name = {f.name: f for f in spec.fields}

        def need(name: str) -> FieldSpec:
            if name not in by_name:
                raise _Problem(f"No field '{name}' in {spec.name}. Fields: {list(by_name)}")
            return by_name[name]

        rows = self._count(conn, spec.name)
        new = spec.model_copy(deep=True)

        # ---- add_field
        if isinstance(op, AddField):
            f = op.field
            if f.name in by_name:
                raise _Problem(f"Field '{f.name}' already exists")
            if f.primary_key:
                raise _Problem("A primary key cannot be added to an existing table; define a new model and copy the rows")
            if rows and f.required and f.default is None:
                raise _Problem(f"{spec.name} has {rows} rows and '{f.name}' is required but has no default to fill them. "
                               "Give it a default, or make it optional (required=false)")
            new.fields.append(f.model_copy(deep=True))
            new = self._revalidate(new)
            if f.references:
                self._validate_references(new)  # type: ignore[attr-defined]
            if spec.adopted:
                if f.unique:
                    raise _Problem("A UNIQUE column cannot be added to an existing (adopted) table; define a new model and copy the rows")
                sql = f"ALTER TABLE {quote(spec.name)} ADD COLUMN {quote(f.name)} {SQLITE_TYPES[f.type]}"
                if f.default is not None:
                    sql += f" NOT NULL DEFAULT {_sql_literal(_db_default(f))}"
                if f.references:
                    sql += f" REFERENCES {quote(f.references.split('.')[0])}({quote(f.references.split('.')[1])})"
                conn.execute(sql)
                if f.required and f.default is None:
                    report["notes"].append(f"'{f.name}' is required by the model but the database column is nullable (SQLite cannot add "
                                           "a NOT NULL column without a default); the app enforces it on every write")
            else:
                self._rebuild(conn, spec, new, report, fill={f.name: _db_default(f)})
            report["plan"].append(f"add field '{f.name}' ({f.type}, {'required' if f.required else 'optional'}"
                                  f"{', default ' + repr(f.default) if f.default is not None else ''})"
                                  + (f"; existing {rows} rows get the default" if rows and f.default is not None else
                                     f"; existing {rows} rows get NULL" if rows else ""))

        # ---- drop_field
        elif isinstance(op, DropField):
            f = need(op.name)
            if f.primary_key:
                raise _Problem(f"'{f.name}' is the primary key and cannot be dropped")
            if len(spec.fields) == 1:
                raise _Problem("A model needs at least one field")
            refs = [f"{s.name}.{x.name}" for s, x in self._referencing(others, spec, f.name)]
            if refs:
                raise _Problem(f"'{spec.name}.{f.name}' is referenced by {refs}; remove those references first")
            lost = self._count(conn, spec.name, f"{quote(f.name)} IS NOT NULL")
            new.fields = [x for x in new.fields if x.name != f.name]
            new = self._revalidate(new)
            if spec.adopted:
                conn.execute(f"ALTER TABLE {quote(spec.name)} DROP COLUMN {quote(f.name)}")
            else:
                self._rebuild(conn, spec, new, report)
            report["plan"].append(f"drop field '{f.name}' ({lost} stored values will be deleted; a backup is taken first)")

        # ---- rename_field
        elif isinstance(op, RenameField):
            f = need(op.name)
            FieldSpec(name=op.new_name)  # validates snake_case
            if op.new_name in by_name:
                raise _Problem(f"Field '{op.new_name}' already exists")
            conn.execute(f"ALTER TABLE {quote(spec.name)} RENAME COLUMN {quote(f.name)} TO {quote(op.new_name)}")
            old_ref, new_ref = f"{spec.name}.{f.name}", f"{spec.name}.{op.new_name}"
            for x in new.fields:
                if x.name == f.name:
                    x.name = op.new_name
                if x.references == old_ref:
                    x.references = new_ref
            new = self._revalidate(new)
            for s in others:  # other models that point at this column follow the rename (SQLite already rewrote their FK clauses)
                changed = False
                for x in s.fields:
                    if x.references == old_ref:
                        x.references, changed = new_ref, True
                if changed:
                    self._save_spec_conn(conn, s)
                    report["plan"].append(f"update model '{s.name}': references now point to {new_ref}")
            report["plan"].append(f"rename field '{f.name}' to '{op.new_name}' (data untouched)")

        # ---- change_field
        else:
            f = need(op.name)
            nf = next(x for x in new.fields if x.name == f.name)
            changes: list[str] = []
            if op.description is not None and op.description != (f.description or ""):
                nf.description = op.description or None
                changes.append("description")
            if op.clear_default:
                nf.default = None
                changes.append("clear default")
            if op.default is not None:
                nf.default = op.default
                changes.append(f"default = {op.default!r}")
            if op.required is not None and op.required != f.required:
                if f.primary_key:
                    raise _Problem("The primary key's required flag cannot be changed")
                nf.required = op.required
                changes.append(f"required = {op.required}")
            if op.unique is not None and op.unique != f.unique:
                if f.primary_key:
                    raise _Problem("The primary key is already unique")
                if spec.adopted:
                    raise _Problem("UNIQUE cannot be added to an existing (adopted) table; define a new model and copy the rows")
                nf.unique = op.unique
                changes.append(f"unique = {op.unique}")
            if op.references is not None and (op.references or None) != f.references:
                nf.references = op.references or None
                changes.append(f"references = {op.references or None}")
            conv: dict[str, Any] = {}
            if op.type is not None and op.type != f.type:
                if spec.adopted:
                    raise _Problem("Changing the type of a column in an adopted table is done with the convert_column tool")
                if f.primary_key:
                    raise _Problem("The primary key's type cannot be changed")
                if op.type not in _CONVERTIBLE_TO or (f.type == "json" and op.type != "string"):
                    raise _Problem(f"Converting {f.type} to {op.type} is not supported")
                if self._referencing(others, spec, f.name):
                    raise _Problem(f"'{f.name}' is referenced by other models; its type cannot change")
                nf.type = op.type
                conv[f.name] = lambda v, to=op.type, fmt=op.date_format: convert_value(v, to, fmt)
                changes.append(f"type {f.type} -> {op.type}")
            if not changes:
                raise _Problem(f"change_field '{f.name}': nothing to change")
            new = self._revalidate(new)
            if nf.references and nf.references != f.references:
                self._validate_references(new)  # type: ignore[attr-defined]
            if spec.adopted:
                # only app-level metadata can change; the database column itself is untouched
                if op.references:
                    report["notes"].append("references on an adopted table are logical metadata (SQLite cannot add foreign keys)")
                if op.required is not None or op.default is not None:
                    report["notes"].append("required/default on an adopted table are enforced by the app, not by the database column")
            else:
                if (op.required or op.default is not None) and f.name != (spec.pk.name if spec.pk else None):
                    empty = self._count(conn, spec.name, f"{quote(f.name)} IS NULL")
                    if empty and nf.default is None:
                        raise _Problem(f"{empty} rows have no value for '{f.name}'. Set a default (it fills them) or fill them first")
                self._rebuild(conn, spec, new, report, conv=conv, on_error=op.on_error)
            report["plan"].append(f"change field '{f.name}': " + ", ".join(changes))

        self._save_spec_conn(conn, new)
        return new

    def _rebuild(self, conn: sqlite3.Connection, old: ModelSpec, new: ModelSpec, report: dict[str, Any],
                 conv: dict[str, Any] | None = None, fill: dict[str, Any] | None = None, on_error: str = "fail") -> None:
        """Create the new table, copy every row (converting where asked), drop the old table, rename. Managed tables only."""
        conv, fill = conv or {}, fill or {}
        name, tmp = old.name, f"{old.name}__new"
        n = self._count(conn, name)
        if n > MAX_REBUILD_ROWS:
            raise _Problem(f"{name} has {n} rows; rebuilding is limited to {MAX_REBUILD_ROWS}")
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND tbl_name=?", (name,)).fetchone():
            raise _Problem(f"{name} has triggers, which a rebuild would drop; not changed")
        indexes = [r["sql"] for r in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (name,))]
        conn.execute(new.to_sqlite_ddl().replace(f"CREATE TABLE {quote(name)}", f"CREATE TABLE {quote(tmp)}", 1))
        old_cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({quote(name)})")}
        failures: list[dict[str, Any]] = []
        nulled = 0
        out: list[list[Any]] = []
        for r in conn.execute(f"SELECT rowid AS _rid, * FROM {quote(name)}"):
            row: list[Any] = []
            for f in new.fields:
                v = r[f.name] if f.name in old_cols else fill.get(f.name)
                if f.name in conv and f.name in old_cols:
                    try:
                        v = conv[f.name](v)
                    except ConversionError as e:
                        if on_error == "null":
                            v, nulled = None, nulled + 1
                        else:
                            failures.append({"row": r["_rid"], "value": v if not isinstance(v, str) else v[:60], "error": str(e)})
                            v = None
                if v is None and f.default is not None:
                    v = _db_default(f)
                row.append(v)
            out.append(row)
        if failures:
            shown = failures[:10]
            raise _Problem(f"{len(failures)} value(s) cannot be converted (first {len(shown)} shown). Fix the data, pass a correct "
                           "date_format, or use on_error='null' to turn them into NULL", shown)
        cols = ", ".join(quote(f.name) for f in new.fields)
        conn.executemany(f"INSERT INTO {quote(tmp)} ({cols}) VALUES ({', '.join('?' * len(new.fields))})", out)
        conn.execute(f"DROP TABLE {quote(name)}")
        conn.execute(f"ALTER TABLE {quote(tmp)} RENAME TO {quote(name)}")
        for sql in indexes:
            try:
                conn.execute(sql)
            except sqlite3.DatabaseError:
                report["notes"].append(f"an index could not be recreated after the rebuild (it used a changed column): {sql[:80]}")
        if nulled:
            report["notes"].append(f"{nulled} value(s) could not be converted and became NULL (on_error='null')")
        report["notes"].append(f"{name} was rebuilt ({n} rows copied); constraints and foreign keys are kept")

    # --- convert_column ----------------------------------------------------

    def convert_column(self, table: str, column: str, to_type: str, date_format: str | None = None,
                       on_error: str = "fail", apply: bool = False) -> dict[str, Any]:
        spec = self._require_spec(table)  # type: ignore[attr-defined]
        f = next((x for x in spec.fields if x.name == column), None)
        if f is None:
            raise ValueError(f"No column '{column}' in {table}. Columns: {[x.name for x in spec.fields]}")
        if to_type not in _CONVERTIBLE_TO:
            raise ValueError(f"Cannot convert to '{to_type}'. Supported: {sorted(_CONVERTIBLE_TO)}")
        if f.type == to_type:
            return {"applied": False, "dry_run": True, "problems": [], "notes": [f"'{column}' is already {to_type}; nothing to do"]}
        if not spec.adopted:  # tables made by define_model: the rebuild path keeps constraints and foreign keys
            r = self.alter_model(table, [ChangeField(op="change_field", name=column, type=to_type,  # type: ignore[arg-type]
                                                     date_format=date_format, on_error=on_error)], apply)
            r["method"] = "table rebuild"
            return r
        result = self._convert_native(table, spec, f, to_type, date_format, on_error, commit=False)
        if apply and not result["problems"]:
            backup = self._backup(f"before-convert-{table}-{column}")
            result = self._convert_native(table, spec, f, to_type, date_format, on_error, commit=True)
            result["backup"] = backup
        result["dry_run"] = not result["applied"]
        if not result["applied"] and not result["problems"]:
            result["next"] = "Nothing was changed. Show the sample to the user; if they agree, call again with apply=true."
        return result

    def _convert_native(self, table: str, spec: ModelSpec, f: FieldSpec, to_type: str, date_format: str | None,
                        on_error: str, commit: bool) -> dict[str, Any]:
        """Adopted tables: add a typed column, fill it from the converted values, drop the old column, rename.
        The column moves to the end of the table; the user's other columns, indexes and constraints are untouched."""
        report: dict[str, Any] = {"applied": False, "problems": [], "notes": [], "failures": [], "method": "native column swap",
                                  "converted": 0, "nulled": 0, "sample": []}
        tmp = f"{f.name}__conv"
        with self._migration_conn() as conn:
            conn.execute("BEGIN")
            try:
                if sqlite3.sqlite_version_info < (3, 35, 0):
                    raise _Problem(f"SQLite {sqlite3.sqlite_version} is too old for this (3.35+ needed)")
                if f.primary_key or f.unique:
                    raise _Problem(f"'{f.name}' is a primary key or unique column; converting it needs a table rebuild, which is not "
                                   "done on your own tables. Define a new model and copy the data instead")
                if any(x.references == f"{table}.{f.name}" for s in self.all_specs() for x in s.fields):  # type: ignore[attr-defined]
                    raise _Problem(f"'{f.name}' is referenced by another model; its type cannot change")
                updates, failures = [], []
                for r in conn.execute(f"SELECT rowid AS _rid, {quote(f.name)} AS v FROM {quote(table)}"):
                    try:
                        nv = convert_value(r["v"], to_type, date_format)
                    except ConversionError as e:
                        if on_error == "null":
                            nv = None
                            report["nulled"] += 1
                        else:
                            failures.append({"row": r["_rid"], "value": r["v"] if not isinstance(r["v"], str) else r["v"][:60], "error": str(e)})
                            continue
                    updates.append((nv, r["_rid"]))
                    if nv is not None:
                        report["converted"] += 1
                        if len(report["sample"]) < 5 and {"from": r["v"], "to": nv} not in report["sample"]:
                            report["sample"].append({"from": r["v"], "to": nv})
                if failures:
                    shown = failures[:10]
                    raise _Problem(f"{len(failures)} value(s) in '{f.name}' cannot be converted to {to_type} (first {len(shown)} shown). "
                                   "Fix the data, pass the right date_format, or use on_error='null' to turn them into NULL", shown)
                conn.execute(f"ALTER TABLE {quote(table)} ADD COLUMN {quote(tmp)} {SQLITE_TYPES[to_type]}")
                conn.executemany(f"UPDATE {quote(table)} SET {quote(tmp)} = ? WHERE rowid = ?", updates)
                try:
                    conn.execute(f"ALTER TABLE {quote(table)} DROP COLUMN {quote(f.name)}")
                except sqlite3.DatabaseError as e:
                    raise _Problem(f"SQLite cannot replace '{f.name}' ({e}); it is probably indexed or used in a constraint, "
                                   "which this tool does not rewrite on your own tables") from e
                conn.execute(f"ALTER TABLE {quote(table)} RENAME COLUMN {quote(tmp)} TO {quote(f.name)}")
                new = spec.model_copy(deep=True)
                nf = next(x for x in new.fields if x.name == f.name)
                nf.type = to_type
                if nf.required:
                    nf.required = False
                    report["notes"].append(f"'{f.name}' was NOT NULL; the rebuilt column is nullable (SQLite cannot add NOT NULL without a default)")
                new.fields = [x for x in new.fields if x.name != f.name] + [nf]
                new = self._revalidate(new)
                self._save_spec_conn(conn, new)
                report["notes"].append(f"'{f.name}' moved to the end of the table's columns")
                if report["nulled"]:
                    report["notes"].append(f"{report['nulled']} value(s) could not be converted and became NULL (on_error='null')")
                report["model_field"] = nf.model_dump()
            except _Problem as e:
                report["problems"].append(str(e))
                report["failures"] += e.failures
            except sqlite3.DatabaseError as e:
                report["problems"].append(_friendly(e))
            if commit and not report["problems"]:
                conn.execute("COMMIT")
                report["applied"] = True
            else:
                conn.execute("ROLLBACK")
        return report
