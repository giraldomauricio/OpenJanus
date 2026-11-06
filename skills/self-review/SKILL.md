---
name: self-review
description: Use when the user asks what you can do, wants your tools/skills or database reviewed, checked for overlap or redundancy, optimized or cleaned up, or asks whether a new tool is really needed.
tools: list_capabilities, audit_capabilities, audit_database, list_extensions, describe_table, profile_table, find_relationships, annotate_model
---
# Reviewing yourself and the database

All audit tools are **read-only and advisory**. You report; the user decides. Never claim you changed, retired or merged anything:
you cannot remove tools or skills, and generated ones can only be retired by the user (reject/delete in the Approvals panel).

## Capabilities review
1. `audit_capabilities` for the findings, and `list_capabilities` when you need to explain what exists.
2. For each `overlapping-tools` / `overlapping-skills` finding, call `list_capabilities` with `name` for BOTH items and compare
   parameters and instructions before judging. Description similarity is a hint, not proof: say whether they are
   truly redundant, complementary, or just badly described.
3. Recommend one of: keep both and sharpen descriptions / merge into one (propose a replacement via `extension-builder`; the user
   approves) / retire one. Say which and why, in a sentence each.
4. Treat `tool-never-used` as weak evidence: it only reflects saved chats.
5. Before proposing a new tool, run this check yourself: does an existing tool or skill already cover it?

## Database review
1. `audit_database`. Start with `high` findings (model drift, orphaned references), then `medium`.
2. Verify before advising: confirm a finding with `describe_table`, `query` or `profile_table` rather than repeating it blindly.
   For `similar-tables`, compare rows/columns with `query` before suggesting a merge.
3. Fixes you can do yourself: `annotate_model` (descriptions, references), `adopt_table` (after asking the user).
   Fixes you cannot do: dropping tables, adding indexes, altering columns, fixing orphaned rows without the user's confirmation.
   Describe those and let the user act.

## Reporting
Group by severity, lead with what matters, give counts and names. Skip `info` items unless asked. If there are no findings, say so.
