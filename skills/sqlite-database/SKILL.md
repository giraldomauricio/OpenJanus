---
name: sqlite-database
description: Use for any request to store, look up, change, summarise or export data in the local SQLite database, including creating tables for data coming from files, APIs or the user.
tools: list_tables, describe_table, query, define_model, insert_records, update_records, delete_records, export_models, profile_table, find_relationships, adopt_table, annotate_model
---
# Working with the local database

## Workflow
1. Start with `list_tables` / `describe_table` so you never guess at schema.
2. **Reading**: use `query` with a single SELECT and `?` placeholders. It is read-only; writes are rejected.
3. **New data source → new table**: design a model first, then call `define_model`.
4. **Writing**: use `insert_records`, `update_records`, `delete_records`. These validate every row against a
   Pydantic model generated from the model definition; if validation fails, read the error, fix the data, retry.
5. **Migration**: `export_models` returns Pydantic source, JSON Schema and DDL so the data can move to another database.

## Designing a model
- Names are `snake_case` (tables and columns). Use singular, descriptive table names (`exchange_rate`).
- Field types are database-agnostic: `string`, `integer`, `number`, `boolean`, `datetime`, `date`, `json`.
  Use `datetime`/`date` with ISO-8601 strings, not free text. Use `json` only for genuinely nested data.
- Give every field a short `description`. Mark optional fields `required: false`.
- If the source has a natural key (e.g. a symbol, an external id) declare it `primary_key: true`;
  otherwise omit it and an auto-increment `id` is added.
- If a field points at another table, set `references: "table.column"` (a real foreign key; the target must be a primary key
  or unique column). Create the referenced model first.
- Add `unique: true` where duplicates would be a bug, then use `on_conflict: "ignore"` or `"replace"` when re-importing.
- Schema changes are not supported yet. If the user wants a different shape, define a new model and copy rows.

## Rules
- Tables are created ONLY through `define_model`. Never try raw DDL. Tables the user brought in themselves are registered with
  `adopt_table` (load the `table-onboarding` skill).
- Before deleting more than a few rows, or updating without a precise `where`, confirm with the user.
- Insert in batches of at most 500 rows. Report how many rows were inserted/skipped.
