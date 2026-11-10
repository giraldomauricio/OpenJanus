import json

import pytest

from agent_app import config
from agent_app.agent import Agent
from agent_app.db import Database
from agent_app.extensions import ExtensionManager, ProposalError
from agent_app.sdk import BUILTIN_TOOLS
from agent_app.skills import SkillStore

V1 = '''
class Params(BaseModel):
    n: int = Field(1, description="n")

def run(params: Params, ctx: ToolContext) -> dict:
    return {"version": 1}
'''
V2 = V1.replace('"version": 1', '"version": 2')


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "t.db")
    skills = None
    ext = ExtensionManager(tmp_path / "ext", lambda: set(BUILTIN_TOOLS), lambda: skills.builtin_names())
    skills = SkillStore(config.BUILTIN_SKILLS_DIR, ext)
    return ext, skills, Agent(db, ext, skills), tmp_path / "ext"


def call(agent, name="thing_tool"):
    ok, text, raw = agent._run(agent.build_tools()[0], name, {})
    return ok, raw


def test_retire_takes_a_tool_out_of_use_and_archives_it(env):
    ext, skills, agent, root = env
    ext.propose_tool("thing_tool", "does a thing for tests", V1, "r"); ext.approve("tool", "thing_tool")
    assert call(agent) == (True, {"version": 1})
    assert ext.retire("tool", "thing_tool") == {"retired": "thing_tool", "kind": "tool"}
    assert "thing_tool" not in agent.build_tools()[0] and agent.build_tools()[1] == []   # gone, and no integrity complaint
    assert not (root / "approved/tools/thing_tool.py").exists() and json.loads((root / "approved/manifest.json").read_text())["tool"] == {}
    assert len(list((root / "retired/tools").glob("thing_tool.*.py"))) == 1               # archived, not deleted
    with pytest.raises(KeyError):
        ext.retire("tool", "thing_tool")
    with pytest.raises(KeyError):
        ext.retire("tool", "never_existed")


def test_replacing_an_approved_tool(env):
    ext, skills, agent, root = env
    ext.propose_tool("thing_tool", "does a thing for tests", V1, "r"); ext.approve("tool", "thing_tool")
    r = ext.propose_tool("thing_tool", "does a thing for tests v2", V2, "fix")
    assert r["replaces_approved"] is True
    pending = ext.list_items("pending")
    assert pending[0]["replaces_approved"] is True
    assert call(agent) == (True, {"version": 1})        # the old version keeps working while the proposal waits
    assert ext.approve("tool", "thing_tool")["replaced"] is True
    assert call(agent) == (True, {"version": 2})
    assert len(list((root / "retired/tools").glob("thing_tool.*.py"))) == 1 and ext.list_items("pending") == []
    assert [i["description"] for i in ext.list_items("approved")] == ["does a thing for tests v2"]
    assert ext.integrity_problems() == []


def test_failed_replacement_restores_the_working_version(env, monkeypatch):
    ext, skills, agent, root = env
    ext.propose_tool("thing_tool", "does a thing for tests", V1, "r"); ext.approve("tool", "thing_tool")
    ext.propose_tool("thing_tool", "does a thing for tests v2", V2, "fix")
    import agent_app.extensions.manager as mgr
    real = mgr.shutil.move
    calls = {"n": 0}

    def flaky(src, dst, *a, **k):
        calls["n"] += 1
        if str(src).endswith("pending/tools/thing_tool.py"):
            raise OSError("disk full")
        return real(src, dst, *a, **k)
    monkeypatch.setattr(mgr.shutil, "move", flaky)
    with pytest.raises(OSError):
        ext.approve("tool", "thing_tool")
    monkeypatch.setattr(mgr.shutil, "move", real)
    assert call(agent) == (True, {"version": 1}) and ext.integrity_problems() == []   # still the old, working tool


def test_lint_still_gates_a_replacement(env):
    ext, *_ = env
    ext.propose_tool("thing_tool", "does a thing for tests", V1, "r"); ext.approve("tool", "thing_tool")
    with pytest.raises(ProposalError, match="lint"):
        ext.propose_tool("thing_tool", "does a thing for tests v2", "import os\n" + V2, "x")
    with pytest.raises(ProposalError, match="built-in"):
        ext.propose_tool("query", "shadow the query tool", V1, "r")


def test_skills_can_be_replaced_and_retired(env):
    ext, skills, agent, root = env
    ext.propose_skill("my-skill", "Use when the user wants things", "Step 1: do the thing properly.", ["query"]); ext.approve("skill", "my-skill")
    assert ext.propose_skill("my-skill", "Use when the user wants other things", "Step 1: do it differently now.", [])["replaces_approved"]
    assert "other" not in skills.all()["my-skill"].description
    assert ext.approve("skill", "my-skill")["replaced"] and "other" in skills.all()["my-skill"].description
    ext.retire("skill", "my-skill")
    assert "my-skill" not in skills.all() and len(list((root / "retired/skills").glob("my-skill.*"))) == 1
    with pytest.raises(ProposalError, match="exists"):
        ext.propose_skill("sqlite-database", "shadow a builtin skill ok", "x" * 30, [])   # built-ins stay protected


def test_endpoints_need_the_token_and_report_dependents(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from agent_app import main
    ext = ExtensionManager(tmp_path / "ext", lambda: set(BUILTIN_TOOLS), lambda: main.skills_store.builtin_names())
    monkeypatch.setattr(main, "extensions", ext)
    monkeypatch.setattr(main.skills_store, "extensions", ext)
    main.agent.extensions = ext
    c, H = TestClient(main.app), {"X-Agent-Token": main.TOKEN}
    ext.propose_tool("thing_tool", "does a thing for tests", V1, "r"); ext.approve("tool", "thing_tool")
    ext.propose_skill("uses-thing", "Use when the user wants the thing", "Step 1: call thing_tool.", ["thing_tool"]); ext.approve("skill", "uses-thing")
    assert c.post("/api/extensions/tool/thing_tool/retire").status_code == 403
    listing = c.get("/api/extensions").json()
    assert next(i for i in listing["approved"] if i["name"] == "thing_tool")["used_by_skills"] == ["uses-thing"]
    assert c.post("/api/extensions/tool/nope/retire", headers=H).status_code == 404
    assert c.post("/api/extensions/tool/thing_tool/retire", headers=H).json()["retired"] == "thing_tool"
    assert c.post("/api/extensions/bogus/thing_tool/retire", headers=H).status_code == 422
