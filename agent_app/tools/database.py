"""Built-in tools for the local SQLite database."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..db import ModelSpec
from ..db.migrate import Op
from ..db.models import FieldType
from ..sdk import AgentContext, tool


class NoParams(BaseModel):
    pass


@tool("list_tables", "List all tables in the database with row counts and whether each is managed by a defined model.")
def list_tables(params: NoParams, ctx: AgentContext) -> Any:
    return ctx.db.list_tables()


class DescribeParams(BaseModel):
    table: str


@tool("describe_table", "Show columns of a table and, if it was created via define_model, its model definition.")
def describe_table(params: DescribeParams, ctx: AgentContext) -> Any:
    return ctx.db.describe(params.table)


class QueryParams(BaseModel):
    sql: str = Field(description="A single read-only SELECT statement. Use ? placeholders for values.")
    args: list[str | int | float | bool | None] = Field(default_factory=list, description="Values for ? placeholders")
    limit: int = Field(100, ge=1, le=1000)


@tool("query", "Run a read-only SQL SELECT against the database. Writes are rejected; use the record tools for changes.")
def query(params: QueryParams, ctx: AgentContext) -> Any:
    return ctx.db.query(params.sql, params.args, params.limit)


@tool("define_model",
      "Create a table from a structured model definition (name, fields with types). The definition is stored so the "
      "schema can be exported as Pydantic code / JSON Schema / DDL. This is the ONLY way to create tables. "
      "If no primary key is given, an integer `id` primary key is added.")
def define_model(params: ModelSpec, ctx: AgentContext) -> Any:
    r = ctx.db.define_model(params)
    return {"status": r["status"], "ddl": r["sqlite_ddl"], "model": r["spec"]}


class InsertParams(BaseModel):
    model: str = Field(description="Name of a model created with define_model")
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=500)
    on_conflict: Literal["error", "ignore", "replace"] = "error"


@tool("insert_records", "Insert rows into a model's table. Every row is validated against the Pydantic model first; "
                        "if any row is invalid nothing is inserted.")
def insert_records(params: InsertParams, ctx: AgentContext) -> Any:
    return ctx.db.insert(params.model, params.rows, params.on_conflict)


class UpdateParams(BaseModel):
    model: str
    values: dict[str, Any] = Field(description="Columns to set")
    where: dict[str, Any] = Field(description="Equality filters (AND-ed); must not be empty")


@tool("update_records", "Update rows matching equality filters. Values are validated against the model.")
def update_records(params: UpdateParams, ctx: AgentContext) -> Any:
    return ctx.db.update(params.model, params.values, params.where)


class DeleteParams(BaseModel):
    model: str
    where: dict[str, Any] = Field(description="Equality filters (AND-ed); must not be empty")


@tool("delete_records", "Delete rows matching equality filters. Confirm with the user before deleting more than a few rows.")
def delete_records(params: DeleteParams, ctx: AgentContext) -> Any:
    return ctx.db.delete(params.model, params.where)


class ExportParams(BaseModel):
    model: str | None = Field(None, description="Model name, or omit to export all models")


@tool("export_models", "Export model definitions as Pydantic source, JSON Schema and SQLite DDL for migrating the data elsewhere.")
def export_models(params: ExportParams, ctx: AgentContext) -> Any:
    return ctx.db.export_models(params.model)


class ProfileParams(BaseModel):
    table: str


@tool("profile_table",
      "Learn what a table contains: per column the inferred type, null/distinct counts, min/max, examples, value "
      "distribution for low-cardinality columns, and whether it is a unique key. Works on any table.")
def profile_table(params: ProfileParams, ctx: AgentContext) -> Any:
    return ctx.db.profile_table(params.table)


class RelParams(BaseModel):
    table: str
    min_overlap: float = Field(0.8, ge=0.5, le=1.0, description="Minimum share of distinct values that must match")


@tool("find_relationships",
      "Find likely foreign-key relationships between a table and all others, in both directions, using value overlap and "
      "column naming. Returns evidence with a confidence level; confirm 'medium'/'low' ones with the user.")
def find_relationships(params: RelParams, ctx: AgentContext) -> Any:
    return ctx.db.find_relationships(params.table, params.min_overlap)


class AdoptParams(BaseModel):
    table: str = Field(description="Existing table that has no model yet (e.g. imported from CSV)")
    description: str | None = Field(None, description="What the table represents")
    field_descriptions: dict[str, str] = Field(default_factory=dict, description="column -> meaning")
    type_overrides: dict[str, FieldType] = Field(default_factory=dict, description="column -> model type; must fit the declared SQLite type")
    rename_columns: dict[str, str] = Field(default_factory=dict, description="old -> new snake_case name. ALTERS the user's table: confirm first")
    rename_table_to: str | None = Field(None, description="New snake_case table name. ALTERS the user's table: confirm first")
    references: dict[str, str] = Field(default_factory=dict, description="column -> 'table.column' for relationships you already confirmed")


@tool("adopt_table",
      "Register an existing, user-created table as a model WITHOUT changing its data: infers types, keys and uniqueness, "
      "checks that existing rows validate, and enables validated writes. Fails with suggested renames if names are not snake_case.")
def adopt_table(params: AdoptParams, ctx: AgentContext) -> Any:
    r = ctx.db.adopt_table(params.table, params.description, params.field_descriptions, params.type_overrides,
                           params.rename_columns, params.rename_table_to)
    if params.references:
        try:
            r["references_recorded"] = _annotate(ctx, r["model"]["name"], None, {}, params.references)["notes"] or "ok"
        except ValueError as e:  # adoption itself succeeded; report instead of failing the whole call
            r["notes"].append(f"Adopted, but references were NOT recorded: {e}")
    return r


class AnnotateParams(BaseModel):
    model: str
    description: str | None = Field(None, description="What the table represents")
    field_descriptions: dict[str, str] = Field(default_factory=dict, description="column -> what it means")
    references: dict[str, str] = Field(default_factory=dict,
                                       description="column -> 'table.column' it points to, e.g. {\"customer_id\": \"customers.customer_id\"}")


def _annotate(ctx: AgentContext, model: str, description: str | None, field_descriptions: dict[str, str],
              references: dict[str, str]) -> Any:
    notes: dict[str, dict[str, str]] = {}
    for col, text in field_descriptions.items():
        notes.setdefault(col, {})["description"] = text
    for col, ref in references.items():
        notes.setdefault(col, {})["references"] = ref
    return ctx.db.annotate_model(model, description, notes)


@tool("annotate_model",
      "Record what you learned about a model: table description, column meanings (field_descriptions) and relationships "
      "(references: column -> 'table.column'). Stored with the model so future chats, tools and skills can rely on it. "
      "Does not change table data. Use an empty string to clear a value.")
def annotate_model(params: AnnotateParams, ctx: AgentContext) -> Any:
    return _annotate(ctx, params.model, params.description, params.field_descriptions, params.references)


class AlterParams(BaseModel):
    model: str = Field(description="Name of the model/table to change")
    ops: list[Op] = Field(min_length=1, description=(
        "Changes, applied in order. Each has an `op`: "
        '{"op":"add_field","field":{"name":"note","type":"string","required":false}} | '
        '{"op":"drop_field","name":"note"} | '
        '{"op":"rename_field","name":"old","new_name":"new"} | '
        '{"op":"change_field","name":"qty","required":true,"default":0} (also: type, unique, description, references, '
        "date_format, on_error)"))
    apply: bool = Field(False, description="false = DRY RUN: executes the change inside a transaction and rolls it back, so the "
                                           "report is exact. Set true ONLY after the user has seen the dry run and agreed.")


@tool("alter_model",
      "Change an existing model's schema: add, drop, rename or change fields (type, required, unique, default, references). "
      "ALWAYS call it first with apply=false (dry run), show the user the plan, problems and any failures, and call with apply=true only "
      "after they agree (the app refuses apply unless an identical dry run was shown and the user has replied since). A backup is taken automatically on apply. Tables created by define_model are rebuilt when needed; the user's own "
      "(adopted) tables only support add/rename/drop field and metadata changes. Use convert_column to change a column's type on those.")
def alter_model(params: AlterParams, ctx: AgentContext) -> Any:
    return ctx.db.alter_model(params.model, params.ops, params.apply)


class ConvertParams(BaseModel):
    table: str
    column: str
    to_type: Literal["integer", "number", "boolean", "date", "datetime", "string"]
    date_format: str | None = Field(None, description="strptime format for text -> date/datetime when the text is not ISO, "
                                                      "e.g. '%m/%d/%Y' (US) or '%d/%m/%Y'. NEVER guess day-vs-month order: ask the user.")
    on_error: Literal["fail", "null"] = Field("fail", description="'fail' (default) reports values that do not parse; 'null' turns "
                                                                  "them into NULL, which loses data: only with the user's consent")
    apply: bool = Field(False, description="false = DRY RUN with a before/after sample; true only after the user agreed")


@tool("convert_column",
      "Convert a column's stored values to a real type (integer, number, boolean, date, datetime) for tables whose columns are all TEXT "
      "(e.g. imported with .import) or any other column. Strict: values that do not parse are listed, never silently zeroed. "
      "ALWAYS dry-run first (apply=false), show the user the sample and failures, then apply=true after they agree (the app refuses apply "
      "unless an identical dry run was shown and the user has replied since); a backup is taken automatically. Handles thousands separators (1,234) and dates via date_format. Needs a model (adopt_table first).")
def convert_column(params: ConvertParams, ctx: AgentContext) -> Any:
    return ctx.db.convert_column(params.table, params.column, params.to_type, params.date_format, params.on_error, params.apply)
