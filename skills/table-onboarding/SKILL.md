---
name: table-onboarding
description: Use when the user imported or created a table themselves (CSV or JSON import, existing database) and wants you to understand it, build a model for it, relate it to other tables, or build tools and skills around it.
tools: list_tables, profile_table, find_relationships, adopt_table, annotate_model, describe_table, export_models, convert_column
---
# Learning a table the user created

A table that shows `managed: false` in `list_tables` has no model yet. Until it is adopted you can read it but not write
to it, and nothing about what its columns mean is recorded.

## Workflow
1. `list_tables` to find it; `describe_table` for columns.
2. `profile_table`: look at inferred types, null counts, distinct counts, unique keys, and value distributions.
   Low-cardinality columns with `value_counts` are usually categories or codes: note what each value likely means.
3. `find_relationships`: shows which columns look like foreign keys to other tables (and which tables point at this one).
   - `high`: both value overlap and naming agree. Safe to record.
   - `medium`/`low`: tell the user the evidence (overlap %, unmatched values) and ask before recording.
4. `adopt_table`: registers the model without touching the data. Pass `description`, `field_descriptions` and any
   `type_overrides` you are confident about.
   - If it fails because names are not snake_case, it returns suggested renames. Renaming alters the user's table,
     so show the user the mapping and get a yes first.
   - Read `notes` in the result (TEXT columns that look numeric, rows that do not validate) and tell the user.
   - If columns are TEXT but really hold numbers/dates/booleans, offer `convert_column` (dry run first, show the sample, ask the
     user, then apply). For dates that are not ISO, ask for the order (day-first or month-first) before choosing `date_format`.
5. `annotate_model`: record relationships with `references` (e.g. `{"customer_id": "customers.customer_id"}`) and meanings with
   `field_descriptions`, refined with what the user told you. Check the result: if it errors, fix and retry, and if it
   still fails, tell the user it was NOT recorded. This is how knowledge persists for later chats, tools and skills.
6. Summarise for the user: what the table is, the key, the relationships found, open questions (ambiguous codes, dirty rows).

## Using what you learned
- Join along recorded `references` (`describe_table` shows them, and `referenced_by` shows the reverse direction).
- Write through `insert_records` / `update_records` now that the model exists; rows are validated.
- When proposing a **tool** for this data, do not hard-code column meanings. The tool can read them at runtime with
  `ctx.db.get_spec("table")` and `ctx.db.profile_table("table")`.
- When proposing a **skill** for this data (e.g. "how to analyse orders"), put the learned facts in the instructions:
  join paths, meaning of codes, quirks, example queries. Base them on tool output, never on guesses.

## Rules
- Never claim a relationship or column meaning without evidence from the tools or the user. Say "looks like" and give numbers.
- Do not modify, delete or rename user data unless the user agreed in this conversation.
- Ask short, specific questions about ambiguity rather than guessing (e.g. "status has values 1/2/3; what do they mean?").
