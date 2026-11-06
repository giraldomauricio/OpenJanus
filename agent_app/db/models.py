"""Database-agnostic model definitions.

A ``ModelSpec`` is the single source of truth for a table. From it we derive:
  * the SQLite DDL,
  * a Pydantic model used to validate every write,
  * portable Python source + JSON Schema, so the data can be moved to another database
    (SQLAlchemy, Postgres, ...) without reverse-engineering the SQLite file.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

FieldType = Literal["string", "integer", "number", "boolean", "datetime", "date", "json"]

IDENT = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

PY_TYPES: dict[str, Any] = {
    "string": str, "integer": int, "number": float, "boolean": bool,
    "datetime": datetime, "date": date, "json": Any,
}
PY_TYPE_NAMES = {
    "string": "str", "integer": "int", "number": "float", "boolean": "bool",
    "datetime": "datetime", "date": "date", "json": "Any",
}
SQLITE_TYPES = {
    "string": "TEXT", "integer": "INTEGER", "number": "REAL", "boolean": "INTEGER",
    "datetime": "TEXT", "date": "TEXT", "json": "TEXT",
}


def quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


class FieldSpec(BaseModel):
    name: str = Field(description="snake_case column name")
    type: FieldType = "string"
    required: bool = Field(True, description="False makes the column nullable")
    primary_key: bool = False
    unique: bool = False
    default: str | int | float | bool | None = Field(None, description="Applied by the app when the value is omitted")
    description: str | None = None
    references: str | None = Field(None, description="'table.column' this field points to (a foreign key)")

    @field_validator("name")
    @classmethod
    def _ident(cls, v: str) -> str:
        if not IDENT.match(v):
            raise ValueError("must be snake_case: start with a letter; letters, digits, underscores; max 40 chars")
        return v

    @field_validator("references")
    @classmethod
    def _ref(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*", v):
            raise ValueError("must look like 'table.column'")
        return v


class ModelSpec(BaseModel):
    name: str = Field(description="snake_case table name")
    description: str | None = None
    fields: list[FieldSpec] = Field(min_length=1)
    adopted: bool = Field(False, description="True when the table already existed and was registered, not created, by a model")

    @field_validator("name")
    @classmethod
    def _ident(cls, v: str) -> str:
        if not IDENT.match(v) or v.startswith("sqlite_"):
            raise ValueError("must be snake_case, start with a letter, and not start with 'sqlite_'")
        return v

    @model_validator(mode="after")
    def _check(self) -> "ModelSpec":
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("duplicate field names")
        pks = [f for f in self.fields if f.primary_key]
        if len(pks) > 1:
            raise ValueError("composite primary keys are not supported; use one primary_key field")
        if not pks and not self.adopted:
            existing = next((f for f in self.fields if f.name == "id"), None)
            if existing:
                existing.primary_key = True
            else:
                self.fields = [FieldSpec(name="id", type="integer", required=False, primary_key=True,
                                         description="Auto-generated id"), *self.fields]
        for f in self.fields:
            if f.primary_key and f.type == "integer":
                f.required = False  # sqlite assigns rowid when omitted
        return self

    # --- derived artefacts -------------------------------------------------

    @property
    def class_name(self) -> str:
        return "".join(p.capitalize() for p in self.name.split("_"))

    @property
    def pk(self) -> FieldSpec | None:
        return next((f for f in self.fields if f.primary_key), None)

    def to_sqlite_ddl(self) -> str:
        cols = []
        for f in self.fields:
            parts = [quote(f.name), SQLITE_TYPES[f.type]]
            if f.primary_key:
                parts.append("PRIMARY KEY")
                if f.type != "integer":
                    parts.append("NOT NULL")
            else:
                if f.required or f.default is not None:
                    parts.append("NOT NULL")
                if f.unique:
                    parts.append("UNIQUE")
            if f.references:
                rt, rc = f.references.split(".")
                parts.append(f"REFERENCES {quote(rt)}({quote(rc)})")
            if f.type == "boolean":
                parts.append(f"CHECK ({quote(f.name)} IN (0, 1))")
            cols.append("  " + " ".join(parts))
        return f"CREATE TABLE {quote(self.name)} (\n" + ",\n".join(cols) + "\n)"

    def pydantic_model(self, partial: bool = False) -> type[BaseModel]:
        """Validation model. ``partial=True`` makes every field optional (for updates)."""
        defs: dict[str, Any] = {}
        for f in self.fields:
            t = PY_TYPES[f.type]
            if partial:
                defs[f.name] = (t | None, Field(None, description=f.description))
            elif f.default is not None:
                defs[f.name] = (t, Field(f.default, description=f.description))
            elif f.required:
                defs[f.name] = (t, Field(..., description=f.description))
            else:
                defs[f.name] = (t | None, Field(None, description=f.description))
        return create_model(self.class_name, __config__=ConfigDict(extra="forbid"), **defs)

    def python_source(self) -> str:
        lines = [
            "from datetime import date, datetime",
            "from typing import Any",
            "",
            "from pydantic import BaseModel, Field",
            "",
            "",
            f"class {self.class_name}(BaseModel):",
        ]
        if self.description:
            lines.append(f'    """{self.description}"""')
            lines.append("")
        for f in self.fields:
            t = PY_TYPE_NAMES[f.type]
            if f.default is not None:
                ann, default = t, repr(f.default)
            elif f.required:
                ann, default = t, "..."
            else:
                ann, default = f"{t} | None", "None"
            extra = f", description={f.description!r}" if f.description else ""
            tags: dict[str, Any] = {}
            if f.primary_key:
                tags["primary_key"] = True
            if f.unique:
                tags["unique"] = True
            if f.references:
                tags["references"] = f.references
            if tags:
                extra += f", json_schema_extra={tags!r}"
            lines.append(f"    {f.name}: {ann} = Field({default}{extra})")
        return "\n".join(lines) + "\n"

    def export(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adopted": self.adopted,
            "spec": self.model_dump(),
            "python_source": self.python_source(),
            "json_schema": self.pydantic_model().model_json_schema(),
            "sqlite_ddl": self.to_sqlite_ddl(),
        }
