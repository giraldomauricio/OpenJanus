"""Learning from tables the user created elsewhere (CSV imports, other tools).

Adopting a table registers a ModelSpec for it *without touching its data*, so from then on it gets the same
validated writes, exports and relationship metadata as tables made with define_model.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .models import IDENT, FieldSpec, ModelSpec, quote
from .profile import COMPATIBLE, affinity, infer_type, suggest_name

SAMPLE = 5000
MAX_OVERLAP_QUERIES = 2000


class IntrospectionMixin:
    # relies on Database: query, _ro, _rw, table_names, get_spec, all_specs, _require_spec

    # --- sqlite metadata ---------------------------------------------------

    def pragma_columns(self, table: str) -> list[dict[str, Any]]:
        with self._ro() as conn:  # type: ignore[attr-defined]
            conn.set_authorizer(None)  # PRAGMA on a mode=ro connection is safe
            return [dict(r) for r in conn.execute(f"PRAGMA table_info({quote(table)})")]

    def pragma_indexes(self, table: str) -> list[dict[str, Any]]:
        """[{name, unique, columns}] for every index on the table."""
        out = []
        with self._ro() as conn:  # type: ignore[attr-defined]
            conn.set_authorizer(None)
            for idx in conn.execute(f"PRAGMA index_list({quote(table)})").fetchall():
                cols = [r["name"] for r in conn.execute(f"PRAGMA index_info({quote(idx['name'])})").fetchall()]
                out.append({"name": idx["name"], "unique": bool(idx["unique"]), "columns": cols})
        return out

    def _unique_columns(self, table: str) -> set[str]:
        """Columns that are a single-column primary key or have a single-column UNIQUE index."""
        cols = self.pragma_columns(table)
        out = {c["name"] for c in cols if c["pk"]} if sum(1 for c in cols if c["pk"]) == 1 else set()
        with self._ro() as conn:  # type: ignore[attr-defined]
            conn.set_authorizer(None)
            for idx in conn.execute(f"PRAGMA index_list({quote(table)})").fetchall():
                if idx["unique"]:
                    info = conn.execute(f"PRAGMA index_info({quote(idx['name'])})").fetchall()
                    if len(info) == 1:
                        out.add(info[0]["name"])
        return out

    def _validate_references(self, spec: ModelSpec) -> None:
        for f in spec.fields:
            if not f.references:
                continue
            t, c = f.references.split(".")
            if t == spec.name:
                target = {x.name: x for x in spec.fields}.get(c)
                ok = bool(target and (target.primary_key or target.unique))
            elif t in self.table_names():  # type: ignore[attr-defined]
                ok = c in self._unique_columns(t)
            else:
                raise ValueError(f"'{f.name}' references unknown table '{t}'")
            if not ok:
                raise ValueError(f"'{f.name}' references {f.references}, which must be a primary key or unique column")

    # --- profiling ---------------------------------------------------------

    def profile_table(self, table: str) -> dict[str, Any]:
        if table not in self.table_names():  # type: ignore[attr-defined]
            raise ValueError(f"No such table: {table}. Existing: {self.table_names()}")  # type: ignore[attr-defined]
        cols = self.pragma_columns(table)
        t = quote(table)
        total = self.query(f"SELECT COUNT(*) AS n FROM {t}")["rows"][0]["n"]  # type: ignore[attr-defined]
        unique = self._unique_columns(table)
        out = []
        for c in cols:
            q = quote(c["name"])
            s = self.query(f"SELECT COUNT({q}) AS nn, COUNT(DISTINCT {q}) AS d, MIN({q}) AS lo, MAX({q}) AS hi "  # type: ignore[attr-defined]
                           f"FROM {t}")["rows"][0]
            sample = [r["v"] for r in self.query(  # type: ignore[attr-defined]
                f"SELECT {q} AS v FROM {t} WHERE {q} IS NOT NULL LIMIT {SAMPLE}", limit=SAMPLE)["rows"]]
            entry: dict[str, Any] = {
                "column": c["name"], "declared_type": c["type"] or "(none)", "inferred_type": infer_type(sample),
                "nulls": total - s["nn"], "distinct": s["d"], "min": s["lo"], "max": s["hi"],
                "is_unique_key": c["name"] in unique or (total > 0 and s["nn"] == total and s["d"] == total),
                "examples": [r["v"] for r in self.query(  # type: ignore[attr-defined]
                    f"SELECT DISTINCT {q} AS v FROM {t} WHERE {q} IS NOT NULL LIMIT 5")["rows"]],
            }
            if 0 < s["d"] <= 12:  # low cardinality: probably a category/code, show the distribution
                entry["value_counts"] = self.query(  # type: ignore[attr-defined]
                    f"SELECT {q} AS value, COUNT(*) AS n FROM {t} GROUP BY {q} ORDER BY n DESC")["rows"]
            out.append(entry)
        return {"table": table, "rows": total, "columns": out,
                "note": f"Types inferred from up to {SAMPLE} non-null values per column."}

    # --- relationships -----------------------------------------------------

    def _column_stats(self, table: str) -> dict[str, dict[str, Any]]:
        cols = self.pragma_columns(table)
        if not cols:
            return {}
        agg = ", ".join(f"COUNT({quote(c['name'])}), COUNT(DISTINCT {quote(c['name'])})" for c in cols)
        row = list(self.query(f"SELECT COUNT(*), {agg} FROM {quote(table)}")["rows"][0].values())  # type: ignore[attr-defined]
        rows = row[0]
        unique = self._unique_columns(table)
        return {c["name"]: {"rows": rows, "non_null": row[1 + 2 * i], "distinct": row[2 + 2 * i],
                            "is_key": rows > 0 and (c["name"] in unique or (row[1 + 2 * i] == rows == row[2 + 2 * i]))}
                for i, c in enumerate(cols)}

    def _overlap(self, t1: str, c1: str, t2: str, c2: str) -> tuple[int, int]:
        q1, q2 = quote(c1), quote(c2)
        r = self.query(  # type: ignore[attr-defined]
            f"WITH s AS (SELECT DISTINCT CAST({q1} AS TEXT) AS v FROM {quote(t1)} WHERE {q1} IS NOT NULL LIMIT {SAMPLE}) "
            f"SELECT COUNT(*) AS total, COALESCE(SUM(v IN (SELECT CAST({q2} AS TEXT) FROM {quote(t2)} "
            f"WHERE {q2} IS NOT NULL)), 0) AS hit FROM s")["rows"][0]
        return r["total"], r["hit"]

    @staticmethod
    def _name_hint(col: str, target_table: str, target_col: str) -> bool:
        n = lambda s: s.replace("_", "").lower()  # noqa: E731
        return n(col) in {n(target_col), n(target_table) + n(target_col), n(target_table.rstrip("s")) + n(target_col)}

    def find_relationships(self, table: str, min_overlap: float = 0.8) -> dict[str, Any]:
        tables = self.table_names()  # type: ignore[attr-defined]
        if table not in tables:
            raise ValueError(f"No such table: {table}. Existing: {tables}")
        stats = {t: self._column_stats(t) for t in tables}
        found: list[dict[str, Any]] = []
        budget = MAX_OVERLAP_QUERIES

        def scan(src_t: str, dst_t: str, direction: str) -> None:
            nonlocal budget
            for sc, ss in stats[src_t].items():
                if ss["non_null"] == 0 or ss["distinct"] < 2:
                    continue
                for dc, ds in stats[dst_t].items():
                    if not ds["is_key"]:
                        continue
                    if budget <= 0:
                        return
                    budget -= 1
                    total, hit = self._overlap(src_t, sc, dst_t, dc)
                    ratio = hit / total if total else 0
                    hint = self._name_hint(sc, dst_t, dc) or (src_t == dst_t)
                    if ratio < min_overlap:
                        continue
                    if hint and ratio >= 0.95:
                        conf = "high"
                    elif hint:
                        conf = "medium"
                    elif ratio >= 0.95 and ss["distinct"] >= 10:
                        conf = "low"
                    else:
                        continue
                    found.append({"direction": direction, "from": f"{src_t}.{sc}", "to": f"{dst_t}.{dc}",
                                  "overlap": round(ratio, 3), "distinct_values_checked": total,
                                  "unmatched_values": total - hit, "name_match": hint, "confidence": conf})

        for other in tables:
            if other == table:
                continue
            scan(table, other, "outgoing")   # this.col looks like a foreign key to other.key
            scan(other, table, "incoming")   # other.col looks like a foreign key to this.key
        order = {"high": 0, "medium": 1, "low": 2}
        found.sort(key=lambda r: (order[r["confidence"]], -r["overlap"]))
        return {"table": table, "relationships": found,
                "note": "Evidence only (value overlap + naming). 'low' means coincidence is possible: confirm with the user. "
                        + ("Search budget exhausted; results are partial." if budget <= 0 else "")}

    # --- adopting and annotating ------------------------------------------

    def adopt_table(self, table: str, description: str | None = None,
                    field_descriptions: dict[str, str] | None = None,
                    type_overrides: dict[str, str] | None = None,
                    rename_columns: dict[str, str] | None = None,
                    rename_table_to: str | None = None) -> dict[str, Any]:
        field_descriptions, type_overrides, rename_columns = field_descriptions or {}, type_overrides or {}, rename_columns or {}
        tables = self.table_names()  # type: ignore[attr-defined]
        if table not in tables:
            raise ValueError(f"No such table: {table}. Existing: {tables}")
        if self.get_spec(table):  # type: ignore[attr-defined]
            raise ValueError(f"'{table}' already has a model")
        cols = self.pragma_columns(table)
        old_names = [c["name"] for c in cols]
        for k in rename_columns:
            if k not in old_names:
                raise ValueError(f"rename_columns: no column '{k}'. Columns: {old_names}")
        final = {n: rename_columns.get(n, n) for n in old_names}
        new_table = rename_table_to or table

        bad_cols = {n: suggest_name(n) for n, f in final.items() if not IDENT.match(f)}
        bad_table = not IDENT.match(new_table) or new_table.startswith("sqlite_")
        if bad_cols or bad_table:
            raise ValueError(
                "Models need snake_case names (lowercase letters, digits, underscores). Suggested "
                f"rename_columns={json.dumps(bad_cols)}"
                + (f", rename_table_to={json.dumps(suggest_name(new_table))}" if bad_table else "")
                + ". Renaming changes the user's table, so confirm with the user first, then call adopt_table again.")
        if len(set(final.values())) != len(final):
            raise ValueError("rename_columns would produce duplicate column names")
        if new_table != table and new_table in tables:
            raise ValueError(f"A table named '{new_table}' already exists")
        for k in list(type_overrides) + list(field_descriptions):
            if k not in final.values() and k not in final:
                raise ValueError(f"No column '{k}'. Columns after renaming: {list(final.values())}")

        renamed = {o: n for o, n in final.items() if o != n}
        # Everything below reads the table under its ORIGINAL names; renames are applied last, only if all
        # validation passed, so a rejected request never leaves the user's table half-modified.
        cols = self.pragma_columns(table)
        pk_cols = [c for c in cols if c["pk"]]
        unique = self._unique_columns(table)
        notes: list[str] = []
        if len(pk_cols) > 1:
            notes.append("Composite primary key is not supported by models; no primary_key flagged.")
        fields = []
        for c in cols:
            old, declared = c["name"], c["type"]
            name = final[old]
            family, hint = affinity(declared)
            if family == "blob":
                raise ValueError(f"Column '{old}' is BLOB, which models do not support")
            sample = [r["v"] for r in self.query(  # type: ignore[attr-defined]
                f"SELECT {quote(old)} AS v FROM {quote(table)} WHERE {quote(old)} IS NOT NULL LIMIT {SAMPLE}",
                limit=SAMPLE)["rows"]]
            inferred = infer_type(sample)
            if family == "text" and hint is None:
                ftype = inferred if inferred in ("date", "datetime") else "string"
                if inferred in ("integer", "number"):
                    notes.append(f"'{name}' is declared TEXT and every value looks {inferred}; kept as string because "
                                 f"the stored values are text. Use CAST({name} AS {'INTEGER' if inferred == 'integer' else 'REAL'}) in queries.")
            elif family == "none":
                ftype = inferred
            else:
                ftype = hint or "string"
            ov = type_overrides.get(name) or type_overrides.get(old)
            if ov:
                if ov not in COMPATIBLE[family]:
                    raise ValueError(f"type_overrides['{name}']='{ov}' is incompatible with declared type "
                                     f"'{declared}'. Allowed: {sorted(COMPATIBLE[family])}")
                ftype = ov
            is_pk = len(pk_cols) == 1 and bool(c["pk"])
            fields.append(FieldSpec(
                name=name, type=ftype, required=bool(c["notnull"]) or is_pk, primary_key=is_pk,
                unique=old in unique and not is_pk, description=field_descriptions.get(name) or field_descriptions.get(old)))
        spec = ModelSpec(name=new_table, description=description, fields=fields, adopted=True)

        # does the existing data actually satisfy the model we just inferred?
        Model = spec.pydantic_model()
        rows = self.query(f"SELECT * FROM {quote(table)}", limit=500)["rows"]  # type: ignore[attr-defined]
        bad = 0
        first = ""
        for r in rows:
            try:
                Model.model_validate({final[k]: v for k, v in r.items()})
            except ValidationError as e:
                bad += 1
                first = first or "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:2])
        if bad:
            notes.append(f"{bad} of {len(rows)} sampled existing rows do not validate against the inferred model "
                         f"(e.g. {first}). Consider type_overrides or ask the user about dirty data.")
        if renamed or new_table != table:
            with self._rw() as c:  # type: ignore[attr-defined]  # one transaction: all renames or none
                for o, n in renamed.items():
                    c.execute(f"ALTER TABLE {quote(table)} RENAME COLUMN {quote(o)} TO {quote(n)}")
                if new_table != table:
                    c.execute(f"ALTER TABLE {quote(table)} RENAME TO {quote(new_table)}")
        self._save_spec(spec, insert=True)
        return {"status": "adopted", "renamed_columns": renamed, "renamed_table": new_table if new_table != table else None,
                "model": spec.model_dump(), "notes": notes,
                "next": "Call profile_table / find_relationships, then annotate_model to record what the columns mean."}

    def _save_spec(self, spec: ModelSpec, insert: bool = False) -> None:
        from datetime import datetime, timezone
        with self._rw() as c:  # type: ignore[attr-defined]
            if insert:
                c.execute("INSERT INTO _models (name, spec, created_at) VALUES (?, ?, ?)",
                          (spec.name, spec.model_dump_json(), datetime.now(timezone.utc).isoformat()))
            else:
                c.execute("UPDATE _models SET spec = ? WHERE name = ?", (spec.model_dump_json(), spec.name))

    def annotate_model(self, model: str, description: str | None = None,
                       fields: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        spec = self._require_spec(model)  # type: ignore[attr-defined]
        by_name = {f.name: f for f in spec.fields}
        notes: list[str] = []
        for fname, note in (fields or {}).items():
            if fname not in by_name:
                raise ValueError(f"No field '{fname}' in {model}. Fields: {list(by_name)}")
            f = by_name[fname]
            if note.get("description") is not None:
                f.description = note["description"] or None
            if "references" in note and note["references"] is not None:
                ref = note["references"]
                if ref == "":
                    f.references = None
                else:
                    t, _, c = ref.partition(".")
                    FieldSpec(name=fname, references=ref)  # format check
                    if t not in self.table_names():  # type: ignore[attr-defined]
                        raise ValueError(f"references: unknown table '{t}'")
                    if c not in {x["name"] for x in self.pragma_columns(t)}:
                        raise ValueError(f"references: table '{t}' has no column '{c}'")
                    f.references = ref
                    if not spec.adopted:
                        notes.append(f"{model} was created by define_model, so '{ref}' is recorded as metadata only; "
                                     "the existing table does not enforce it.")
                    elif c not in self._unique_columns(t):
                        notes.append(f"{ref} is not a unique key in the database, so this is a logical relationship only.")
        if description is not None:
            spec.description = description or None
        self._save_spec(ModelSpec.model_validate(spec.model_dump()))
        return {"status": "updated", "model": spec.model_dump(), "notes": notes}
