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


def test_query_result_carries_table_for_ui(env):
    db, ext, skills, agent = env
    db.define_model(ModelSpec(**SPEC))
    db.insert("thing", [{"label": "a", "qty": 1}, {"label": "b", "qty": 2}])
    ev = events(agent, "q", Scripted([
        Turn(tool_calls=[ToolCall("1", "query", {"sql": "SELECT label, qty FROM thing ORDER BY label"}),
                         ToolCall("2", "list_tables", {})]), Turn(text="ok")]))
    res = {e["id"]: e for e in ev if e["type"] == "tool_result"}
    assert res["1"]["table"] == {"columns": ["label", "qty"], "rows": [["a", 1], ["b", 2]], "truncated": False}
    assert "table" not in res["2"]  # a list of dicts without columns/rows is not treated as a table
    from agent_app.tables import as_table
    assert as_table({"columns": ["a"], "rows": []}) is None and as_table("x") is None


def test_stale_token_is_rejected_and_page_not_cached():
    from fastapi.testclient import TestClient
    from agent_app import main
    c = TestClient(main.app)
    r = c.post("/api/sessions", headers={"X-Agent-Token": "token-from-a-previous-server-run"})
    assert r.status_code == 403 and "token" in r.json()["detail"].lower()
    page = c.get("/")
    assert page.headers["cache-control"] == "no-store" and main.TOKEN in page.text


# ---- streaming ------------------------------------------------------------------------------------------------

class Streamer(LLMProvider):
    name = "fake"

    def __init__(self, chunks, tool_calls=()):
        super().__init__("m")
        self.chunks, self.tool_calls = chunks, list(tool_calls)

    def generate(self, *a):
        raise AssertionError("agent must use stream()")

    def stream(self, system, messages, tools):
        yield from self.chunks
        yield Turn(text="".join(self.chunks), tool_calls=self.tool_calls)


def test_agent_forwards_deltas_then_authoritative_text(env):
    *_, agent = env
    ev = events(agent, "hi", Streamer(["Hel", "lo ", "there"]))
    assert [e["text"] for e in ev if e["type"] == "text_delta"] == ["Hel", "lo ", "there"]
    assert [e["text"] for e in ev if e["type"] == "text"] == ["Hello there"]
    assert [e["type"] for e in ev].index("text") > [e["type"] for e in ev].index("text_delta")


def test_default_stream_wraps_generate(env):
    *_, agent = env
    ev = events(agent, "hi", Scripted([Turn(text="whole answer")]))  # Scripted only implements generate()
    assert [e["type"] for e in ev] == ["text_delta", "text", "done"]


def test_stream_error_mid_reply_is_reported(env):
    *_, agent = env

    class Boom(Streamer):
        def stream(self, *a):
            yield "partial"
            raise RuntimeError("connection reset")

    ev = events(agent, "hi", Boom([]))
    assert ev[-1]["type"] == "error" and "connection reset" in ev[-1]["text"]


def _mock_ollama(monkeypatch, handler):
    """Make the provider's httpx.Client talk to `handler` instead of a real Ollama."""
    import httpx
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_ollama_stream_parses_ndjson_and_hides_think(monkeypatch):
    import json as _json
    import httpx
    from agent_app.llm.ollama_provider import OllamaProvider
    lines = [{"message": {"content": "<think>plan"}}, {"message": {"content": "ning</think>Hi "}},
             {"message": {"content": "there"}},
             {"message": {"content": "", "tool_calls": [{"function": {"name": "list_tables", "arguments": {}}}]}, "done": True}]

    def handler(req):
        assert _json.loads(req.content)["stream"] is True
        return httpx.Response(200, content="\n".join(_json.dumps(x) for x in lines).encode())

    _mock_ollama(monkeypatch, handler)
    parts = list(OllamaProvider("m").stream("sys", [{"role": "user", "content": "x"}], []))
    assert "".join(p for p in parts if isinstance(p, str)) == "Hi there"
    turn = parts[-1]
    assert turn.text == "Hi there" and [c.name for c in turn.tool_calls] == ["list_tables"]


def test_ollama_stream_error_status(monkeypatch):
    import httpx
    from agent_app.llm.ollama_provider import OllamaProvider
    _mock_ollama(monkeypatch, lambda req: httpx.Response(404, text="model not found"))
    with pytest.raises(RuntimeError, match="404.*model not found"):
        list(OllamaProvider("m").stream("s", [], []))


def test_anthropic_stream(monkeypatch):
    from types import SimpleNamespace as NS
    from agent_app.llm.anthropic_provider import AnthropicProvider
    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model = "m"
    final = NS(content=[NS(type="text", text="Hi there"), NS(type="tool_use", id="t1", name="query", input={"sql": "select 1"})])

    class Ctx:
        text_stream = iter(["Hi ", "there"])
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_final_message(self): return final

    p.client = NS(messages=NS(stream=lambda **kw: Ctx()))
    parts = list(p.stream("sys", [{"role": "user", "content": "x"}], []))
    assert parts[:2] == ["Hi ", "there"] and parts[-1].text == "Hi there"
    assert parts[-1].tool_calls == [ToolCall("t1", "query", {"sql": "select 1"})]


def test_gemini_stream_keeps_parts_for_signatures():
    from types import SimpleNamespace as NS
    from google.genai import types as gt
    from agent_app.llm.gemini_provider import GeminiProvider
    p = GeminiProvider.__new__(GeminiProvider)
    p.model, p.types = "m", gt
    chunk = lambda *parts: NS(candidates=[NS(content=gt.Content(role="model", parts=list(parts)))])
    chunks = [chunk(gt.Part(text="Hel")), chunk(gt.Part(text="lo")),
              chunk(gt.Part(function_call=gt.FunctionCall(name="list_tables", args={})))]
    p.client = NS(models=NS(generate_content_stream=lambda **kw: iter(chunks)))
    parts = list(p.stream("sys", [{"role": "user", "content": "x"}], []))
    assert parts[:2] == ["Hel", "lo"] and parts[-1].text == "Hello" and parts[-1].tool_calls[0].name == "list_tables"
    assert isinstance(parts[-1].raw, gt.Content) and len(parts[-1].raw.parts) == 3


# ---- chat metadata + exports ----------------------------------------------------------------------------------

def test_session_records_provider_model_and_deltas_not_persisted(tmp_path, monkeypatch):
    import re
    from fastapi.testclient import TestClient
    from agent_app import main
    from agent_app.sessions import SessionStore
    monkeypatch.setattr(main, "store", SessionStore(tmp_path))
    monkeypatch.setattr(main, "make_provider", lambda name, model=None: Streamer(["a", "b"]))
    c = TestClient(main.app)
    tok = {"X-Agent-Token": main.TOKEN}
    sid = c.post("/api/sessions", headers=tok).json()["id"]
    body = c.post("/api/chat", headers=tok, json={"session_id": sid, "message": "hi", "provider": "ollama", "model": "qwen3:8b"}).text
    assert body.count('"text_delta"') == 2
    saved = c.get(f"/api/sessions/{sid}").json()
    assert (saved["provider"], saved["model"]) == ("ollama", "m")
    assert [e["type"] for e in saved["events"]] == ["user", "text"]
    assert saved["events"][0]["provider"] == "ollama"
    assert c.get("/api/sessions").json()[0]["provider"] == "ollama"
    assert c.delete(f"/api/sessions/{sid}", headers=tok).json() == {"deleted": True}


def test_table_csv_export(env):
    db, *_ = env
    db.define_model(ModelSpec(**SPEC))
    db.insert("thing", [{"label": "=cmd()", "qty": -3}, {"label": 'a,"b"', "qty": 2}, {"label": "é\nx"}])
    text = "".join(db.export_csv("thing", chunk=2))
    assert text.startswith("﻿id,label,qty\r\n")
    assert "'=cmd(),-3" in text and '"a,""b"""' in text and '"é\nx"' in text
    from fastapi.testclient import TestClient
    from agent_app import main
    r = TestClient(main.app).get("/api/db/does_not_exist/csv")
    assert r.status_code == 404


def test_dotenv_fills_empty_but_respects_nonempty_shell_vars(tmp_path, monkeypatch):
    from agent_app.config import load_env
    f = tmp_path / ".env"
    f.write_text("A_EMPTY_IN_SHELL=from-file\nB_SET_IN_SHELL=from-file\nC_UNSET=from-file\nD_BLANK_IN_FILE=\n")
    monkeypatch.setenv("A_EMPTY_IN_SHELL", "")
    monkeypatch.setenv("B_SET_IN_SHELL", "from-shell")
    for k in ("C_UNSET", "D_BLANK_IN_FILE"):
        monkeypatch.delenv(k, raising=False)
    load_env(f)
    import os
    assert os.environ["A_EMPTY_IN_SHELL"] == "from-file"      # the bug: empty shell export hid the file's value
    assert os.environ["B_SET_IN_SHELL"] == "from-shell"       # a deliberate shell value still wins
    assert os.environ["C_UNSET"] == "from-file" and os.environ["D_BLANK_IN_FILE"] == ""
    monkeypatch.undo()


def test_claude_provider_ignores_shell_redirect_to_other_backend(monkeypatch):
    from agent_app.llm.anthropic_provider import AnthropicProvider
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:11434")   # e.g. Claude Code pointed at Ollama
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ollama")
    c = AnthropicProvider("claude-sonnet-5-5").client
    assert str(c.base_url).startswith("https://api.anthropic.com") and c.api_key == "sk-ant-test" and not c.auth_token
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        AnthropicProvider("m")


# ---- DNS rebinding ---------------------------------------------------------------------------------------------

def _fake_dns(monkeypatch, answers):
    """answers: list of lists of IPs; each getaddrinfo call pops the next answer (simulates a rebinding DNS server)."""
    import socket
    calls = []

    def gai(host, port, *a, **kw):
        calls.append(host)
        ips = answers.pop(0) if len(answers) > 1 else answers[0]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]

    monkeypatch.setattr(socket, "getaddrinfo", gai)
    return calls


def test_http_connects_only_to_the_ip_that_was_checked(monkeypatch):
    import httpx
    from agent_app.http_safe import SafeHttp
    calls = _fake_dns(monkeypatch, [["93.184.216.34"], ["127.0.0.1"]])  # 1st answer public, any later one would be loopback
    seen = []

    def handler(req):
        seen.append((req.url.host, req.headers["host"], req.extensions.get("sni_hostname")))
        return httpx.Response(200, text="ok")

    http = SafeHttp()
    http.transport = httpx.MockTransport(handler)
    assert http.get("https://example.com/path", params={"a": "1"}).text == "ok"
    assert seen == [("93.184.216.34", "example.com", "example.com")]   # pinned IP, original Host + SNI
    assert calls == ["example.com"]                                     # resolved exactly once


@pytest.mark.parametrize("ips", [["127.0.0.1"], ["10.0.0.5"], ["169.254.169.254"], ["93.184.216.34", "192.168.1.1"], ["::1"]])
def test_http_blocks_private_or_mixed_answers(monkeypatch, ips):
    import httpx
    from agent_app.http_safe import BlockedURL, SafeHttp
    _fake_dns(monkeypatch, [ips])
    http = SafeHttp()
    http.transport = httpx.MockTransport(lambda r: pytest.fail("must not connect"))
    with pytest.raises(BlockedURL):
        http.get("http://evil.example/")


def test_http_redirect_to_private_host_is_blocked(monkeypatch):
    import httpx
    from agent_app.http_safe import BlockedURL, SafeHttp
    import socket
    table = {"good.example": "93.184.216.34", "inner.example": "127.0.0.1"}
    monkeypatch.setattr(socket, "getaddrinfo", lambda h, p, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[h], p))])
    http = SafeHttp()
    http.transport = httpx.MockTransport(lambda r: httpx.Response(302, headers={"location": "http://inner.example/admin"}))
    with pytest.raises(BlockedURL):
        http.get("http://good.example/")


def test_http_rejects_non_http_schemes():
    from agent_app.http_safe import BlockedURL, SafeHttp
    for u in ("file:///etc/passwd", "ftp://example.com/x", "gopher://example.com"):
        with pytest.raises(Exception):  # httpx rejects unknown schemes itself, or BlockedURL
            SafeHttp().get(u)


# ---- stop button -----------------------------------------------------------------------------------------------

def test_stop_mid_stream_keeps_partial_text_and_a_valid_history(env):
    *_, agent = env
    closed = []

    class Slow(LLMProvider):
        name = "fake"
        def generate(self, *a): raise AssertionError
        def stream(self, system, messages, tools):
            try:
                for ch in ["one ", "two ", "three ", "four "]:
                    yield ch
            finally:
                closed.append(True)
            yield Turn(text="one two three four")

    s, out = Session(), []
    for ev in agent.run(s, "go", Slow("m")):
        out.append(ev)
        if ev["type"] == "text_delta" and ev["text"] == "two ":
            s.cancel.set()  # the Stop button
    assert [e["type"] for e in out][-2:] == ["text", "cancelled"] and out[-2]["text"] == "one two"
    assert closed == [True]                                   # the provider stream was closed, so its connection is dropped
    last = s.messages[-1]
    assert last["role"] == "assistant" and last["tool_calls"] == [] and last["text"].endswith("[Stopped by the user.]")
    # the conversation can simply continue afterwards
    ev = events(agent, "again", Scripted([Turn(text="ok")]), s)
    assert [e["type"] for e in ev] == ["text_delta", "text", "done"]
    assert [m["role"] for m in s.messages] == ["user", "assistant", "user", "assistant"]


def test_stop_between_tools_still_pairs_every_tool_call_with_a_result(env):
    db, ext, skills, agent = env
    s, out = Session(), []
    calls = [ToolCall(str(i), "list_tables", {}) for i in range(3)]
    for ev in agent.run(s, "go", Scripted([Turn(tool_calls=calls)])):
        out.append(ev)
        if ev["type"] == "tool_result" and ev["id"] == "0":
            s.cancel.set()
    assert out[-1]["type"] == "cancelled"
    tool_msg = next(m for m in s.messages if m["role"] == "tool")
    assert [r["id"] for r in tool_msg["results"]] == ["0", "1", "2"]
    assert not tool_msg["results"][0]["is_error"] and all("Stop" in r["content"] for r in tool_msg["results"][1:])
    assert sum(e["type"] == "tool_call" for e in out) == 1    # the other two never ran
    assert s.messages[-1]["role"] == "assistant" and s.messages[-1]["text"] == "[Stopped by the user.]"
    # every provider format accepts this history
    from agent_app.llm.anthropic_provider import AnthropicProvider
    from agent_app.llm.ollama_provider import OllamaProvider
    assert AnthropicProvider._convert(s.messages) and OllamaProvider._convert("sys", s.messages)


def test_closing_on_cancel_unblocks_a_waiting_read():
    import threading, time
    p = Streamer([])
    p.cancel = threading.Event()
    blocked = threading.Event()
    t0 = time.time()
    threading.Timer(0.3, p.cancel.set).start()
    with p.closing_on_cancel(blocked.set):
        assert blocked.wait(5)                  # stands in for a socket read that only ends when the connection is closed
    assert time.time() - t0 < 2
    q = Streamer([])                            # no cancel event configured: plain no-op
    with q.closing_on_cancel(lambda: pytest.fail("must not be called")):
        pass


def test_stop_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from agent_app import main
    c, tok = TestClient(main.app), {"X-Agent-Token": main.TOKEN}
    assert c.post("/api/sessions/" + "0" * 32 + "/stop", headers=tok).status_code == 404
    assert c.post("/api/sessions/" + "0" * 32 + "/stop").status_code == 403
    sid = c.post("/api/sessions", headers=tok).json()["id"]
    assert c.post(f"/api/sessions/{sid}/stop", headers=tok).json() == {"stopping": False}   # idle: nothing to stop
    s = main.sessions[sid]
    s.lock.acquire()
    try:
        assert c.post(f"/api/sessions/{sid}/stop", headers=tok).json() == {"stopping": True} and s.cancel.is_set()
    finally:
        s.lock.release()


def test_stop_works_even_when_the_model_is_silent_and_blocked(env):
    import threading, time
    *_, agent = env
    release, closed = threading.Event(), threading.Event()

    class Silent(LLMProvider):
        name = "fake"
        def generate(self, *a): raise AssertionError
        def _stream(self, system, messages, tools):
            try:
                release.wait(30)               # a blocking socket read: no chunks, nothing to poll
                yield "late"
            finally:
                closed.set()

    s = Session()
    p = Silent("m"); p.cancel = s.cancel
    threading.Timer(0.3, s.cancel.set).start()
    t0 = time.time()
    out = [e["type"] for e in agent.run(s, "go", p)]
    assert out == ["cancelled"] and time.time() - t0 < 3   # returns promptly although the read is still blocked
    release.set()                                          # the blocked read finally returns...
    assert closed.wait(3)                                  # ...and the helper thread closes the stream (connection dropped)
    assert s.messages[-1]["text"] == "[Stopped by the user.]"


def test_interruptible_passes_items_and_errors_through():
    import threading
    class P(LLMProvider):
        name = "fake"
        def generate(self, *a): raise AssertionError
        def _stream(self, *a):
            yield "a"
            yield Turn(text="a")
    p = P("m"); p.cancel = threading.Event()
    parts = list(p.stream("s", [], []))
    assert parts[0] == "a" and parts[1].text == "a"

    class Bad(P):
        def _stream(self, *a):
            yield "x"
            raise RuntimeError("boom")
    b = Bad("m"); b.cancel = threading.Event()
    it = b.stream("s", [], [])
    assert next(it) == "x"
    with pytest.raises(RuntimeError, match="boom"):
        next(it)
