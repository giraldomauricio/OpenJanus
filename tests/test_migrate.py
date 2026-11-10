import json
import sqlite3

import pytest

from agent_app.db import Database, ModelSpec
from agent_app.db.migrate import ConversionError, convert_value


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "t.db")


def snapshot(db):
    """Everything a dry run must not change: the real schema, the stored model specs and all rows."""
    with sqlite3.connect(db.path) as c:
        schema = c.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name").fetchall()
        specs = c.execute("SELECT name, spec FROM _models ORDER BY name").fetchall()
        data = {t: c.execute(f'SELECT * FROM "{t}"').fetchall() for (t,) in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    return schema, specs, data


def make_things(db, rows=3):
    db.define_model(ModelSpec(name="thing", fields=[
        {"name": "label", "type": "string"}, {"name": "qty", "type": "string", "required": False}]))
    db.insert("thing", [{"label": f"t{i}", "qty": str(i * 1000)} for i in range(rows)])


def op_add(**f):
    return {"op": "add_field", "field": f}


# ---- value conversion -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("v,to,fmt,expected", [
    ("42", "integer", None, 42), (" 1,234 ", "integer", None, 1234), ("12.0", "integer", None, 12), (7.0, "integer", None, 7),
    ("3.5", "number", None, 3.5), ("1,234.56", "number", None, 1234.56), ("-2e3", "number", None, -2000.0),
    ("yes", "boolean", None, 1), ("No", "boolean", None, 0), (1, "boolean", None, 1),
    ("2024-03-05", "date", None, "2024-03-05"), ("2024-03-05T10:20:30", "date", None, "2024-03-05"),
    ("03/04/2024", "date", "%m/%d/%Y", "2024-03-04"), ("03/04/2024", "date", "%d/%m/%Y", "2024-04-03"),
    ("2024-03-05", "datetime", None, "2024-03-05T00:00:00"), ("05/03/2024 14:30", "datetime", "%d/%m/%Y %H:%M", "2024-03-05T14:30:00"),
    ("Thu, 05 Nov 2026 22:12:46 +0530", "datetime", "%a, %d %b %Y %H:%M:%S %z", "2026-11-05T22:12:46+05:30"),   # RSS-style dates
    ("Thu, 05 Nov 2026 22:12:46 +0530", "date", "%a, %d %b %Y %H:%M:%S %z", "2026-11-05"),
    (5, "string", None, "5"), ("", "integer", None, None), (None, "date", None, None), ("  ", "number", None, None),
])
def test_convert_value_ok(v, to, fmt, expected):
    assert convert_value(v, to, fmt) == expected


@pytest.mark.parametrize("v,to,fmt", [
    ("abc", "integer", None), ("1.5", "integer", None), ("12abc", "number", None), ("1,5", "number", None), ("nan", "number", None),
    ("maybe", "boolean", None), (2, "boolean", None), ("03/04/2024", "date", None), ("2024-13-45", "date", None),
    ("03/04/2024", "date", "%Y-%m-%d"), (b"x", "string", None), (5, "date", None),
    ("340282366920938463463374607431768211455", "integer", None), (1e30, "integer", None),   # would overflow SQLite's 64-bit integers
])
def test_convert_value_rejects_instead_of_guessing(v, to, fmt):
    with pytest.raises(ConversionError):
        convert_value(v, to, fmt)


# ---- alter_model on tables made by define_model (rebuild path) ------------------------------------------------

def test_dry_run_changes_nothing_and_apply_matches_it(db):
    make_things(db)
    before = snapshot(db)
    r = db.alter_model("thing", [op_add(name="note", type="string", required=False)])
    assert r["dry_run"] and not r["applied"] and not r["problems"] and r["plan"] and r["rows"] == 3
    assert "next" in r and snapshot(db) == before
    assert not (db.path.parent / "backups").exists()                    # no backup for a dry run
    r2 = db.alter_model("thing", [op_add(name="note", type="string", required=False)], apply=True)
    assert r2["applied"] and r2["backup"] and r2["plan"] == r["plan"]
    cols = [c["name"] for c in db.pragma_columns("thing")]
    assert cols == ["id", "label", "qty", "note"] and db.get_spec("thing").fields[-1].name == "note"
    assert db.query("SELECT label, qty, note FROM thing ORDER BY id")["rows"][1] == {"label": "t1", "qty": "1000", "note": None}
    db.insert("thing", [{"label": "new", "note": "hello"}])               # validators use the new model
    with pytest.raises(ValueError):
        db.insert("thing", [{"label": "x", "bogus": 1}])
    with sqlite3.connect(r2["backup"]) as b:                              # the backup is the state BEFORE the change
        assert "note" not in [x[1] for x in b.execute("PRAGMA table_info(thing)")]


def test_add_required_field_needs_a_default_to_fill_existing_rows(db):
    make_things(db)
    r = db.alter_model("thing", [op_add(name="status", type="string", required=True)])
    assert any("required but has no default" in p for p in r["problems"])
    r = db.alter_model("thing", [op_add(name="status", type="string", required=True, default="new")], apply=True)
    assert r["applied"]
    assert {x["status"] for x in db.query("SELECT status FROM thing")["rows"]} == {"new"}
    assert "NOT NULL" in db.export_models("thing")[0]["sqlite_ddl"]


def test_failed_step_rolls_back_the_whole_change(db):
    make_things(db)
    before = snapshot(db)
    r = db.alter_model("thing", [op_add(name="ok_col", type="string", required=False), {"op": "drop_field", "name": "nope"}], apply=True)
    assert r["problems"] and not r["applied"] and snapshot(db) == before
    assert not (db.path.parent / "backups").exists()                    # nothing applied -> nothing backed up


def test_drop_field_rules(db):
    db.define_model(ModelSpec(name="thing", fields=[{"name": "code", "type": "string", "unique": True},
                                                    {"name": "qty", "type": "string", "required": False}]))
    db.insert("thing", [{"code": f"c{i}", "qty": str(i)} for i in range(3)])
    db.define_model(ModelSpec(name="child", fields=[{"name": "thing_code", "type": "string", "references": "thing.code"}]))
    assert any("primary key" in p for p in db.alter_model("thing", [{"op": "drop_field", "name": "id"}])["problems"])
    assert any("referenced by" in p and "child.thing_code" in p for p in db.alter_model("thing", [{"op": "drop_field", "name": "code"}])["problems"])
    r = db.alter_model("thing", [{"op": "drop_field", "name": "qty"}], apply=True)
    assert r["applied"] and [c["name"] for c in db.pragma_columns("thing")] == ["id", "code"]
    assert db.query("SELECT code FROM thing ORDER BY id")["rows"][2]["code"] == "c2"
    assert any("stored values will be deleted" in p for p in db.alter_model("child", [{"op": "drop_field", "name": "thing_code"}])["plan"])


def test_rename_updates_referencing_models_and_keeps_fk_enforcement(db):
    make_things(db)
    db.define_model(ModelSpec(name="child", fields=[{"name": "thing_id", "type": "integer", "references": "thing.id"}]))
    r = db.alter_model("thing", [{"op": "rename_field", "name": "id", "new_name": "thing_pk"}], apply=True)
    assert r["applied"] and any("child" in p for p in r["plan"])
    assert db.get_spec("child").fields[1].references == "thing.thing_pk"
    db.insert("child", [{"thing_id": 1}])
    with pytest.raises(ValueError, match="FOREIGN KEY"):
        db.insert("child", [{"thing_id": 999}])                         # the real foreign key still works after the rename
    assert any("already exists" in p for p in db.alter_model("thing", [{"op": "rename_field", "name": "label", "new_name": "qty"}])["problems"])
    with pytest.raises(ValueError):
        db.alter_model("thing", [{"op": "rename_field", "name": "label", "new_name": "Bad Name"}])


def test_change_type_is_strict_reports_failures_and_can_null_them(db):
    make_things(db)
    db.insert("thing", [{"label": "bad", "qty": "n/a"}, {"label": "fmt", "qty": "1,234"}])
    ch = {"op": "change_field", "name": "qty", "type": "integer"}
    before = snapshot(db)
    r = db.alter_model("thing", [ch], apply=True)
    assert r["problems"] and not r["applied"] and r["failures"][0]["value"] == "n/a" and snapshot(db) == before
    r = db.alter_model("thing", [{**ch, "on_error": "null"}], apply=True)
    assert r["applied"] and any("became NULL" in n for n in r["notes"])
    rows = {x["label"]: x["qty"] for x in db.query("SELECT label, qty FROM thing")["rows"]}
    assert rows == {"t0": 0, "t1": 1000, "t2": 2000, "bad": None, "fmt": 1234}
    assert db.get_spec("thing").fields[2].type == "integer" and "qty\" INTEGER" in db.export_models("thing")[0]["sqlite_ddl"]
    assert db.insert("thing", [{"label": "typed", "qty": 5}])["inserted"] == 1
    with pytest.raises(ValueError):
        db.insert("thing", [{"label": "typed", "qty": "five"}])         # now validated as an integer


def test_change_required_unique_default(db):
    make_things(db)
    db.insert("thing", [{"label": "dup"}, {"label": "dup"}])           # qty is NULL for these
    assert any("no value" in p for p in db.alter_model("thing", [{"op": "change_field", "name": "qty", "required": True}])["problems"])
    r = db.alter_model("thing", [{"op": "change_field", "name": "qty", "required": True, "default": "0"}], apply=True)
    assert r["applied"] and [x["qty"] for x in db.query("SELECT qty FROM thing WHERE label = 'dup'")["rows"]] == ["0", "0"]   # NULLs backfilled
    r = db.alter_model("thing", [{"op": "change_field", "name": "label", "unique": True}])
    assert r["problems"] and "UNIQUE" in r["problems"][0] and "duplicate" in r["problems"][0] and "__new" not in r["problems"][0]
    assert any("nothing to change" in p for p in db.alter_model("thing", [{"op": "change_field", "name": "label"}])["problems"])
    assert any("primary key" in p for p in db.alter_model("thing", [{"op": "change_field", "name": "id", "type": "string"}])["problems"])


def test_change_references_adds_a_real_foreign_key(db):
    make_things(db)
    db.define_model(ModelSpec(name="child", fields=[{"name": "thing_ref", "type": "integer", "required": False}]))
    db.insert("child", [{"thing_ref": 1}, {"thing_ref": 77}])
    r = db.alter_model("child", [{"op": "change_field", "name": "thing_ref", "references": "thing.id"}])
    assert r["problems"] and "missing parent" in r["problems"][0]          # orphan row 77 blocks it
    db.delete("child", {"thing_ref": 77})
    assert db.alter_model("child", [{"op": "change_field", "name": "thing_ref", "references": "thing.id"}], apply=True)["applied"]
    with pytest.raises(ValueError, match="FOREIGN KEY"):
        db.insert("child", [{"thing_ref": 500}])


def test_rebuild_keeps_user_indexes_and_refuses_tables_with_triggers(db):
    make_things(db)
    with sqlite3.connect(db.path) as c:
        c.execute('CREATE INDEX idx_label ON thing(label)')
    assert db.alter_model("thing", [op_add(name="n2", type="integer", required=False)], apply=True)["applied"]
    with sqlite3.connect(db.path) as c:
        assert c.execute("SELECT 1 FROM sqlite_master WHERE name='idx_label'").fetchone()
        c.execute("CREATE TRIGGER trg AFTER INSERT ON thing BEGIN SELECT 1; END")
    r = db.alter_model("thing", [op_add(name="n3", type="integer", required=False)])
    assert r["problems"] and "triggers" in r["problems"][0]


def test_backups_are_pruned(db):
    from agent_app.db import migrate
    make_things(db)
    (db.path.parent / "backups").mkdir()
    for i in range(14):
        (db.path.parent / "backups" / f"2000010{i % 10}-0000{i:02d}-old.db").write_bytes(b"x")
    db.alter_model("thing", [op_add(name="z", type="string", required=False)], apply=True)
    assert len(list((db.path.parent / "backups").glob("*.db"))) == migrate.KEEP_BACKUPS


# ---- adopted tables (the user's own): native operations only --------------------------------------------------

def adopt_raw(db, rows=3):
    with sqlite3.connect(db.path) as c:
        c.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, product TEXT NOT NULL, amount TEXT, sold_on TEXT, code TEXT UNIQUE)")
        c.execute("CREATE INDEX idx_product ON sales(product)")
        c.executemany("INSERT INTO sales (product, amount, sold_on, code) VALUES (?,?,?,?)",
                      [(f"p{i}", f"{(i + 1) * 1000:,}", f"0{i + 1}/15/2024", f"c{i}") for i in range(rows)])
    db.adopt_table("sales")


def test_adopted_tables_support_native_add_rename_drop_but_never_a_rebuild(db):
    adopt_raw(db)
    assert db.get_spec("sales").adopted
    r = db.alter_model("sales", [op_add(name="region", type="string", required=False), op_add(name="score", type="integer", required=True, default=5)], apply=True)
    assert r["applied"] and all("rebuilt" not in n for n in r["notes"])
    assert db.query("SELECT region, score FROM sales")["rows"][0] == {"region": None, "score": 5}
    assert db.alter_model("sales", [{"op": "rename_field", "name": "product", "new_name": "item"}], apply=True)["applied"]
    assert db.alter_model("sales", [{"op": "drop_field", "name": "region"}], apply=True)["applied"]
    with sqlite3.connect(db.path) as c:   # the user's own index followed the rename; nothing else was touched
        assert "item" in c.execute("SELECT sql FROM sqlite_master WHERE name='idx_product'").fetchone()[0]
    assert any("UNIQUE" in p for p in db.alter_model("sales", [op_add(name="u", type="string", required=False, unique=True)])["problems"])
    assert any("required but has no default" in p for p in db.alter_model("sales", [op_add(name="r2", type="string", required=True)])["problems"])
    assert any("convert_column" in p for p in db.alter_model("sales", [{"op": "change_field", "name": "amount", "type": "integer"}])["problems"])
    r = db.alter_model("sales", [{"op": "change_field", "name": "amount", "description": "Sale amount in USD"}], apply=True)
    assert r["applied"] and db.get_spec("sales").fields[2].description == "Sale amount in USD"


def test_convert_column_on_an_adopted_text_table(db):
    adopt_raw(db)
    before = snapshot(db)
    r = db.convert_column("sales", "amount", "integer")
    assert not r["applied"] and r["dry_run"] and r["converted"] == 3 and r["sample"][0] == {"from": "1,000", "to": 1000}
    assert snapshot(db) == before
    r = db.convert_column("sales", "sold_on", "date")
    assert r["problems"] and r["failures"][0]["value"] == "01/15/2024" and "date_format" in r["problems"][0]
    assert db.convert_column("sales", "sold_on", "date", date_format="%d/%m/%Y")["problems"]          # month 15 is not a month
    r = db.convert_column("sales", "sold_on", "date", date_format="%m/%d/%Y", apply=True)
    assert r["applied"] and r["backup"] and r["sample"][0] == {"from": "01/15/2024", "to": "2024-01-15"}
    assert db.convert_column("sales", "amount", "integer", apply=True)["applied"]
    assert db.query("SELECT amount, sold_on FROM sales ORDER BY id")["rows"] == [
        {"amount": 1000, "sold_on": "2024-01-15"}, {"amount": 2000, "sold_on": "2024-02-15"}, {"amount": 3000, "sold_on": "2024-03-15"}]
    spec = db.get_spec("sales")
    assert [f.name for f in spec.fields][-2:] == ["sold_on", "amount"] and {f.name: f.type for f in spec.fields}["amount"] == "integer"
    assert [c["name"] for c in db.pragma_columns("sales")] == [f.name for f in spec.fields]          # spec matches the real table
    assert db.query("SELECT product FROM sales ORDER BY id")["rows"][2]["product"] == "p2"           # other columns untouched
    with sqlite3.connect(db.path) as c:
        assert c.execute("SELECT 1 FROM sqlite_master WHERE name='idx_product'").fetchone()           # the user's index survived
        assert c.execute("SELECT typeof(amount) FROM sales LIMIT 1").fetchone()[0] == "integer"        # really stored as numbers now
    assert db.insert("sales", [{"product": "x", "amount": 5, "code": "zz"}])["inserted"] == 1


def test_convert_column_refusals_and_null_option(db):
    adopt_raw(db)
    with sqlite3.connect(db.path) as c:
        c.execute("INSERT INTO sales (product, amount, code) VALUES ('bad', 'n/a', 'c9')")
    assert "n/a" in str(db.convert_column("sales", "amount", "number")["failures"])
    r = db.convert_column("sales", "amount", "number", on_error="null", apply=True)
    assert r["applied"] and r["nulled"] == 1 and any("became NULL" in n for n in r["notes"])
    assert db.query("SELECT amount FROM sales WHERE product='bad'")["rows"][0]["amount"] is None
    assert any("primary key or unique" in p for p in db.convert_column("sales", "code", "integer")["problems"])
    assert any("primary key or unique" in p for p in db.convert_column("sales", "id", "string")["problems"])
    assert db.convert_column("sales", "amount", "number")["notes"][0].endswith("nothing to do")
    with pytest.raises(ValueError, match="No column"):
        db.convert_column("sales", "nope", "integer")
    with pytest.raises(ValueError, match="No model"):
        with sqlite3.connect(db.path) as c:
            c.execute("CREATE TABLE raw (a TEXT)")
        db.convert_column("raw", "a", "integer")                                                       # unadopted: needs a model first


def test_convert_column_on_a_managed_table_uses_the_rebuild_and_keeps_constraints(db):
    make_things(db)
    r = db.convert_column("thing", "qty", "integer", apply=True)
    assert r["applied"] and r["method"] == "table rebuild"
    assert [c["name"] for c in db.pragma_columns("thing")] == ["id", "label", "qty"]                   # column order kept
    assert db.query("SELECT qty FROM thing ORDER BY id")["rows"][1]["qty"] == 1000
    assert "NOT NULL" in db.export_models("thing")[0]["sqlite_ddl"]                                    # label stays NOT NULL


def test_tools_accept_the_json_an_llm_would_send(db):
    from agent_app.agent import Agent
    from agent_app import config
    from agent_app.extensions import ExtensionManager
    from agent_app.skills import SkillStore
    make_things(db)
    ext = ExtensionManager(db.path.parent / "ext")
    agent = Agent(db, ext, SkillStore(config.BUILTIN_SKILLS_DIR, ext))
    tools, _ = agent.build_tools()
    ok, text = agent._execute(tools, "alter_model", {"model": "thing", "ops": [
        {"op": "add_field", "field": {"name": "note", "type": "string", "required": False}}]})
    assert ok and '"dry_run": true' in text and "note" not in [c["name"] for c in db.pragma_columns("thing")]
    ok, text = agent._execute(tools, "alter_model", {"model": "thing", "ops": [{"op": "explode"}]})
    assert not ok and "Invalid arguments" in text
    ok, text = agent._execute(tools, "convert_column", {"table": "thing", "column": "qty", "to_type": "integer"})
    assert ok and '"dry_run": true' in text


def test_audit_suggests_converting_text_columns_that_are_really_typed(db):
    from agent_app.introspection import audit_database
    adopt_raw(db)
    assert db.alter_model("sales", [op_add(name="zipcode", type="string", required=False),
                                    op_add(name="plain", type="string", required=False)], apply=True)["applied"]
    with sqlite3.connect(db.path) as c:
        c.execute("UPDATE sales SET zipcode = '01234', plain = 'x'")
    found = {f["subjects"][0]: f["detail"] for f in audit_database(db)["findings"] if f["kind"] == "text-column-looks-typed"}
    assert "sales.sold_on" not in found                      # 01/15/2024 is not ISO: not suggested blindly
    assert "parse as integer" in found["sales.amount"]       # "1,000", "2,000", ...
    assert "sales.zipcode" not in found and "sales.plain" not in found and "sales.code" not in found   # identifiers / real text are left alone
    db.convert_column("sales", "amount", "integer", apply=True)
    assert "sales.amount" not in {f["subjects"][0] for f in audit_database(db)["findings"] if f["kind"] == "text-column-looks-typed"}


def test_long_digit_strings_are_treated_as_identifiers_not_numbers(db):
    from agent_app.introspection import audit_database
    adopt_raw(db)
    assert db.alter_model("sales", [op_add(name="cert", type="string", required=False)], apply=True)["applied"]
    with sqlite3.connect(db.path) as c:
        c.execute("UPDATE sales SET cert = '3402823669209384634633746074317' || id")
    assert "sales.cert" not in {f["subjects"][0] for f in audit_database(db)["findings"] if f["kind"] == "text-column-looks-typed"}
    r = db.convert_column("sales", "cert", "integer")
    assert r["problems"] and "too large" in r["failures"][0]["error"]


def test_multi_step_changes_report_each_note_once(db):
    make_things(db)
    r = db.alter_model("thing", [op_add(name="a", type="string", required=False), op_add(name="b", type="string", required=False)])
    assert len(r["plan"]) == 2 and sum("was rebuilt" in n for n in r["notes"]) == 1


def test_apply_is_refused_until_the_user_has_replied_after_an_identical_dry_run(db, tmp_path):
    from agent_app import config
    from agent_app.agent import Agent, Session
    from agent_app.extensions import ExtensionManager
    from agent_app.llm.base import LLMProvider, ToolCall, Turn
    from agent_app.skills import SkillStore
    make_things(db)
    ext = ExtensionManager(tmp_path / "ext")
    agent = Agent(db, ext, SkillStore(config.BUILTIN_SKILLS_DIR, ext))

    class Script(LLMProvider):
        name = "fake"
        def __init__(self, turns): super().__init__("m"); self.turns = list(turns)
        def generate(self, *a): return self.turns.pop(0)

    args = {"table": "thing", "column": "qty", "to_type": "integer"}
    call = lambda i, **kw: Turn(tool_calls=[ToolCall(str(i), "convert_column", {**args, **kw})])
    s = Session()

    def run(text, *turns):
        res = [e for e in agent.run(s, text, Script([*turns, Turn(text="ok")])) if e["type"] == "tool_result"]
        return [(r["ok"], json.loads(r["content"]) if r["ok"] else r["content"]) for r in res]

    # 1. straight to apply, then dry run + apply in the SAME turn: both refused / not applied
    out = run("convert qty", call(1, apply=True), call(2), call(3, apply=True))
    assert not out[0][0] and "no dry run" in out[0][1]
    assert out[1][0] and out[1][1]["dry_run"] and not out[1][1]["applied"]
    assert not out[2][0] and "has not replied" in out[2][1]
    assert db.get_spec("thing").fields[2].type == "string"
    # 2. a different change than the one reviewed is refused even after the user replied
    out = run("yes, go ahead", call(4, apply=True, on_error="null"))
    assert not out[0][0] and "no dry run" in out[0][1]
    # 3. the same change after the user replied: applied (and the approval is consumed)
    out = run("yes, go ahead", call(5, apply=True))
    assert out[0][0] and out[0][1]["applied"] and db.get_spec("thing").fields[2].type == "integer"
    assert s.dry_runs == {}
    # 4. a dry run with problems is not an approvable plan
    db.insert("thing", [{"label": "bad", "qty": 5}])
    out = run("now label", call(6, column="label", to_type="integer"))
    assert out[0][1]["problems"] and s.dry_runs == {}
