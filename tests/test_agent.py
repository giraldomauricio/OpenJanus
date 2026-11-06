import json

import pytest

from agent_app import config
from agent_app.agent import Agent, Session
from agent_app.db import Database, ModelSpec
from agent_app.extensions import ExtensionManager, ProposalError
from agent_app.extensions.lint import lint_tool_source
from agent_app.llm.base import LLMProvider, ToolCall, Turn
from agent_app.sdk import BUILTIN_TOOLS
from agent_app.skills import SkillStore

GOOD_TOOL = '''
class Params(BaseModel):
    n: int = Field(description="how many")

def run(params: Params, ctx: ToolContext) -> dict:
    return ctx.db.insert("thing", [{"label": f"t{i}"} for i in range(params.n)])
'''

SPEC = {"name": "thing", "fields": [{"name": "label", "type": "string"}, {"name": "qty", "type": "integer", "required": False}]}


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "t.db")
    skills = None
    ext = ExtensionManager(tmp_path / "ext", lambda: set(BUILTIN_TOOLS), lambda: skills.builtin_names())
    skills = SkillStore(config.BUILTIN_SKILLS_DIR, ext)
    return db, ext, skills, Agent(db, ext, skills)


def test_model_roundtrip_and_validation(env):
    db, *_ = env
    r = db.define_model(ModelSpec(**SPEC))
    assert 'PRIMARY KEY' in r["sqlite_ddl"] and "class Thing(BaseModel)" in r["python_source"]
    assert db.insert("thing", [{"label": "a"}, {"label": "b", "qty": 2}])["inserted"] == 2
    with pytest.raises(ValueError, match="row 1"):
        db.insert("thing", [{"label": "ok"}, {"label": "x", "qty": "many"}])
    assert db.query("SELECT COUNT(*) n FROM thing")["rows"][0]["n"] == 2  # failed batch inserted nothing
    with pytest.raises(ValueError, match="extra|Extra"):
        db.insert("thing", [{"label": "x", "bogus": 1}])
    assert db.update("thing", {"qty": 9}, {"label": "a"})["updated"] == 1
    with pytest.raises(ValueError):
        db.delete("thing", {})


@pytest.mark.parametrize("sql", ["DROP TABLE thing", "INSERT INTO thing(label) VALUES('x')",
                                 "ATTACH DATABASE '/tmp/x.db' AS x", "PRAGMA writable_schema=1",
                                 "SELECT 1; DROP TABLE thing"])
def test_query_is_read_only(env, sql):
    db, *_ = env
    db.define_model(ModelSpec(**SPEC))
    with pytest.raises(ValueError):
        db.query(sql)
    assert db.table_names() == ["thing"]


@pytest.mark.parametrize("bad", [
    "import os\n" + GOOD_TOOL,
    GOOD_TOOL.replace("return ctx", "open('/etc/passwd'); return ctx"),
    GOOD_TOOL.replace("ctx.db.insert", "ctx.db._rw"),
    GOOD_TOOL.replace("ctx.db.insert", "ctx.extensions.approve"),
    GOOD_TOOL.replace("ctx.db.insert", "getattr(ctx, 'x').insert"),
    "def run(params, ctx): pass",
])
def test_lint_rejects(bad):
    assert not lint_tool_source("from pydantic import BaseModel\n" + bad).ok


def test_lint_reports_capabilities():
    assert lint_tool_source("from pydantic import BaseModel\n" + GOOD_TOOL).capabilities == ["database-write"]


class Scripted(LLMProvider):
    name = "fake"

    def __init__(self, turns):
        super().__init__("fake")
        self.turns = list(turns)
        self.seen_tools = []

    def generate(self, system, messages, tools):
        self.seen_tools.append({t.name for t in tools})
        return self.turns.pop(0)


def events(agent, text, provider, session=None):
    return list(agent.run(session or Session(), text, provider))


def test_proposed_tool_unavailable_until_approved(env):
    db, ext, skills, agent = env
    db.define_model(ModelSpec(**SPEC))
    s = Session()
    propose = Turn(tool_calls=[ToolCall("1", "propose_tool", {
        "name": "make_things", "description": "Insert n placeholder things", "code": GOOD_TOOL, "rationale": "test"})])
    call = lambda: Turn(tool_calls=[ToolCall("2", "make_things", {"n": 3})])
    final = Turn(text="done")

    # 1. propose -> pending; calling it in the same session fails (unknown tool)
    p = Scripted([propose, call(), final])
    ev = events(agent, "make a tool", p, s)
    results = [e for e in ev if e["type"] == "tool_result"]
    assert results[0]["ok"] and "pending_approval" in results[0]["content"]
    assert not results[1]["ok"] and "Unknown tool" in results[1]["content"]
    assert "make_things" not in p.seen_tools[-1]
    assert [i["name"] for i in ext.list_items("pending")] == ["make_things"]
    assert db.query("SELECT COUNT(*) n FROM thing")["rows"][0]["n"] == 0

    # 2. user approves -> next turn the tool exists and runs with validated DB writes
    ext.approve("tool", "make_things")
    p = Scripted([call(), final])
    ev = events(agent, "use it", p)
    assert "make_things" in p.seen_tools[0]
    assert [e for e in ev if e["type"] == "tool_result"][0]["ok"]
    assert db.query("SELECT COUNT(*) n FROM thing")["rows"][0]["n"] == 3


def test_reject_and_tamper(env):
    db, ext, skills, agent = env
    db.define_model(ModelSpec(**SPEC))
    ext.propose_tool("t_one", "does one thing here", GOOD_TOOL, "r")
    ext.reject("tool", "t_one")
    assert ext.list_items("pending") == [] and ext.list_items("approved") == []

    ext.propose_tool("t_two", "does two things here", GOOD_TOOL, "r")
    ext.approve("tool", "t_two")
    path = ext._content("tool", "t_two", "approved")
    path.write_text(path.read_text() + "\n# edited after approval\n")
    tools, problems = agent.build_tools()
    assert "t_two" not in tools and any("modified after approval" in p for p in problems)


def test_name_collisions_and_skill_flow(env):
    db, ext, skills, agent = env
    with pytest.raises(ProposalError, match="built-in"):
        ext.propose_tool("query", "shadow the query tool", GOOD_TOOL, "r")
    with pytest.raises(ProposalError, match="exists"):
        ext.propose_skill("sqlite-database", "shadow a builtin skill ok", "x" * 30, [])
    ext.propose_skill("my-skill", "Use when the user wants things", "Step 1: do the thing properly.", ["query"])
    assert "my-skill" not in skills.all()
    ext.approve("skill", "my-skill")
    assert skills.all()["my-skill"].tools == ["query"]


def test_builtin_skills_load(env):
    assert {"sqlite-database", "extension-builder"} <= set(env[2].all())


# --- user-created tables: import, adopt, learn, relate ---------------------------------------------

from agent_app.db.csv_import import import_csv

CUSTOMERS = b"Customer ID,Full Name,Signup Date,Tier\n1,Ann,2024-01-05,gold\n2,Bob,2024-02-10,silver\n3,Cy,2024-03-01,gold\n"
ORDERS = (b"Order ID,customerId,Amount,Placed At\n" + b"".join(
    f"{i},{(i % 3) + 1},{i * 2.5},2024-04-{i % 28 + 1:02d}T10:00:00\n".encode() for i in range(1, 31)))


def test_csv_import_types_and_adopt(env):
    db, *_ = env
    r = import_csv(db, CUSTOMERS, "Customers 2024")
    assert r["table"] == "customers_2024"
    decl = {c["name"]: c["declared_type"] for c in r["columns"]}
    assert decl == {"customer_id": "INTEGER", "full_name": "TEXT", "signup_date": "DATE", "tier": "TEXT"}
    assert db.describe("customers_2024")["managed_by_model"] is False
    with pytest.raises(ValueError, match="No model"):
        db.insert("customers_2024", [{"customer_id": 9}])

    a = db.adopt_table("customers_2024", description="Customers", field_descriptions={"tier": "loyalty tier"})
    fields = {f["name"]: f for f in a["model"]["fields"]}
    assert fields["customer_id"]["type"] == "integer" and fields["signup_date"]["type"] == "date"
    assert fields["tier"]["description"] == "loyalty tier" and a["model"]["adopted"]
    assert not any(f["primary_key"] for f in fields.values()) and "id" not in fields  # no phantom id column
    # now writes are validated against the learned model
    assert db.insert("customers_2024", [{"customer_id": 4, "full_name": "Di", "signup_date": "2024-05-01", "tier": "x"}])["inserted"] == 1
    with pytest.raises(ValueError, match="signup_date"):
        db.insert("customers_2024", [{"customer_id": 5, "signup_date": "not a date"}])


def test_adopt_rename_flow_and_overrides(env):
    db, *_ = env
    with db._rw() as c:
        c.execute('CREATE TABLE "Raw Stuff" ("First Name" TEXT, "Qty" INTEGER, "zip" TEXT)')
        c.execute('INSERT INTO "Raw Stuff" VALUES (\'a\', 1, \'01234\'), (\'b\', 2, \'02345\')')
    with pytest.raises(ValueError, match="rename_columns"):
        db.adopt_table("Raw Stuff")
    with pytest.raises(ValueError, match="incompatible"):
        db.adopt_table("Raw Stuff", rename_columns={"First Name": "first_name", "Qty": "qty"},
                       rename_table_to="raw_stuff", type_overrides={"zip": "integer"})
    assert db.table_names() == ["Raw Stuff"] and not db.get_spec("Raw Stuff")  # rejected calls changed nothing
    assert [c["name"] for c in db.pragma_columns("Raw Stuff")] == ["First Name", "Qty", "zip"]
    a = db.adopt_table("Raw Stuff", rename_columns={"First Name": "first_name", "Qty": "qty"}, rename_table_to="raw_stuff")
    assert a["renamed_table"] == "raw_stuff" and db.get_spec("raw_stuff")
    assert {f["name"]: f["type"] for f in a["model"]["fields"]}["zip"] == "string"  # leading zeros preserved
    assert db.query("SELECT first_name FROM raw_stuff ORDER BY 1")["rows"][0]["first_name"] == "a"


def test_profile_relationships_and_annotate(env):
    db, *_ = env
    import_csv(db, CUSTOMERS, "customers")
    import_csv(db, ORDERS, "orders")
    db.adopt_table("customers")
    db.adopt_table("orders")
    p = db.profile_table("customers")
    tier = next(c for c in p["columns"] if c["column"] == "tier")
    assert tier["value_counts"][0] == {"value": "gold", "n": 2}
    assert next(c for c in p["columns"] if c["column"] == "customer_id")["is_unique_key"]

    rels = db.find_relationships("orders")["relationships"]
    top = rels[0]
    assert (top["from"], top["to"], top["confidence"], top["direction"]) == \
        ("orders.customer_id", "customers.customer_id", "high", "outgoing")
    # the same relationship is found from the other side
    assert db.find_relationships("customers")["relationships"][0]["direction"] == "incoming"
    assert not any(r["from"] == "orders.amount" for r in rels)  # no coincidental matches reported

    out = db.annotate_model("orders", "Customer orders", {"customer_id": {"references": "customers.customer_id",
                                                                         "description": "buyer"}})
    assert db.get_spec("orders").description == "Customer orders"
    assert any("logical" in n for n in out["notes"])  # CSV tables have no unique constraint to enforce it
    d = db.describe("customers")
    assert d["referenced_by"] == [{"model": "orders", "field": "customer_id"}]
    with pytest.raises(ValueError, match="no column"):
        db.annotate_model("orders", fields={"customer_id": {"references": "customers.nope"}})


def test_define_model_foreign_key_is_enforced(env):
    db, *_ = env
    db.define_model(ModelSpec(name="author", fields=[{"name": "name"}]))
    spec = ModelSpec(name="book", fields=[{"name": "title"}, {"name": "author_id", "type": "integer", "references": "author.id"}])
    assert "REFERENCES" in db.define_model(spec)["sqlite_ddl"]
    db.insert("author", [{"name": "A"}])
    db.insert("book", [{"title": "ok", "author_id": 1}])
    with pytest.raises(ValueError, match="FOREIGN KEY"):
        db.insert("book", [{"title": "orphan", "author_id": 99}])
    with pytest.raises(ValueError, match="unknown table"):
        db.define_model(ModelSpec(name="x", fields=[{"name": "y", "type": "integer", "references": "nope.id"}]))
    assert "references" in db.export_models("book")[0]["python_source"]


def test_generated_tool_can_read_learned_model(env):
    db, ext, skills, agent = env
    import_csv(db, CUSTOMERS, "customers")
    db.adopt_table("customers", field_descriptions={"tier": "loyalty tier"})
    ext.propose_tool("describe_cols", "Return column descriptions of a model", '''
class Params(BaseModel):
    model: str

def run(params: Params, ctx: ToolContext) -> dict:
    spec = ctx.db.get_spec(params.model)
    return {f.name: f.description for f in spec.fields}
''', "r")
    ext.approve("tool", "describe_cols")
    tools, problems = agent.build_tools()
    assert not problems
    assert agent._execute(tools, "describe_cols", {"model": "customers"})[1].count("loyalty tier") == 1


def test_annotate_tool_flat_shape_and_adopt_references(env):
    db, ext, skills, agent = env
    import_csv(db, CUSTOMERS, "customers")
    import_csv(db, ORDERS, "orders")
    tools, _ = agent.build_tools()
    ok, _ = agent._execute(tools, "adopt_table", {"table": "customers"})
    assert ok
    ok, out = agent._execute(tools, "adopt_table", {"table": "orders", "references": {"customer_id": "customers.customer_id"}})
    assert ok and db.get_spec("orders").fields[1].references == "customers.customer_id"
    ok, out = agent._execute(tools, "annotate_model", {"model": "customers", "description": "People",
                                                       "field_descriptions": {"tier": "loyalty tier"}})
    assert ok and db.get_spec("customers").description == "People"
    ok, out = agent._execute(tools, "adopt_table", {"table": "customers"})
    assert not ok and "already has a model" in out


def test_session_roundtrip_and_traversal(tmp_path):
    from agent_app.sessions import SessionStore
    store = SessionStore(tmp_path)
    s = Session(title="hello")
    s.events.append({"type": "user", "text": "hi"})
    s.messages += [{"role": "user", "content": "hi"},
                   {"role": "assistant", "text": "x", "raw": object(), "tool_calls": [ToolCall("1", "t", {"a": 1})]},
                   {"role": "tool", "results": [{"id": "1", "name": "t", "content": "ok", "is_error": False}]}]
    store.save(s)
    got = store.load(s.id)
    assert got.title == "hello" and got.events == s.events
    assert got.messages[1]["tool_calls"][0] == ToolCall("1", "t", {"a": 1}) and got.messages[1]["raw"] is None
    assert [x["id"] for x in store.list()] == [s.id]
    assert store.load("../../etc/passwd") is None and not store.delete("../x")
    assert store.delete(s.id) and store.list() == []


def test_empty_sessions_not_saved(tmp_path):
    from agent_app.sessions import SessionStore
    store = SessionStore(tmp_path)
    store.save(Session())
    assert store.list() == []


def test_capability_audit_finds_overlap_and_no_builtin_noise(env):
    db, ext, skills, agent = env
    assert not [f for f in __import__("agent_app.introspection", fromlist=["x"]).audit_capabilities(agent.catalog())["findings"]
                if f["kind"].startswith("overlapping")]  # built-ins alone are clean
    for name in ("fetch_prices", "fetch_prices_live"):
        ext.propose_tool(name, "Fetch current prices for a ticker symbol from the market API", GOOD_TOOL, "test")
    cat = agent.catalog()
    assert {t["name"] for t in cat["tools"] if t["status"] == "pending"} == {"fetch_prices", "fetch_prices_live"}
    from agent_app.introspection import audit_capabilities
    f = audit_capabilities(cat)["findings"]
    assert any(x["kind"] == "overlapping-tools" and set(x["subjects"]) == {"fetch_prices", "fetch_prices_live"} for x in f)


def test_introspection_tools_run_through_agent(env):
    db, ext, skills, agent = env
    out = events(agent, "review", Scripted([
        Turn(tool_calls=[ToolCall("1", "list_capabilities", {}), ToolCall("2", "audit_capabilities", {}),
                         ToolCall("3", "audit_database", {}), ToolCall("4", "list_capabilities", {"name": "query"}),
                         ToolCall("5", "list_capabilities", {"name": "self-review"})]),
        Turn(text="done")]))
    res = {e["id"]: e for e in out if e["type"] == "tool_result"}
    assert all(r["ok"] for r in res.values()), res
    assert "self-review" in res["1"]["content"] and '"param_names"' in res["1"]["content"]
    assert "sql" in res["4"]["content"] and "instructions" in res["5"]["content"]


def test_database_audit(env):
    db, *_ = env
    from agent_app.introspection import audit_database
    db.define_model(ModelSpec(name="customer", fields=[{"name": "name", "type": "string"}, {"name": "note", "type": "string", "required": False}]))
    db.define_model(ModelSpec(name="orders", fields=[{"name": "customer_id", "type": "integer", "references": "customer.id"},
                                                     {"name": "total", "type": "number"}]))
    db.insert("customer", [{"name": "a"}])
    db.insert("orders", [{"customer_id": 1, "total": 5}])
    with db._rw() as c:  # raw write behind the app's back: orphan + a table without a model
        c.execute("PRAGMA foreign_keys=OFF")  # enforced FKs make orphans impossible except in adopted tables
        c.execute("INSERT INTO orders(customer_id, total) VALUES (99, 1)")
        c.execute("CREATE TABLE loose (a TEXT, b TEXT, c TEXT)")
    kinds = {(f["kind"], tuple(f["subjects"])) for f in audit_database(db)["findings"]}
    assert ("orphaned-references", ("orders.customer_id",)) in kinds
    assert ("table-without-model", ("loose",)) in kinds and ("empty-table", ("loose",)) in kinds
    assert ("no-primary-key", ("loose",)) in kinds
    assert ("all-null-column", ("customer.note",)) in kinds
    assert db.table_names() == ["customer", "loose", "orders"]  # audit changed nothing


def test_tool_usage_counts(tmp_path):
    from agent_app.sessions import SessionStore
    store = SessionStore(tmp_path)
    s = Session(); s.events += [{"type": "user", "text": "x"}, {"type": "tool_call", "name": "a", "id": "1", "args": {}},
                                {"type": "tool_call", "name": "a", "id": "2", "args": {}}]
    store.save(s)
    assert store.tool_usage() == {"a": 2}


def test_logo_is_served():
    from fastapi.testclient import TestClient
    from agent_app import main
    r = TestClient(main.app).get("/logo.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
