import io
import json
import zipfile

import pytest

from agent_app.extensions import ExtensionManager, PackageError
from agent_app.extensions import package as pkg
from agent_app.sdk import BUILTIN_TOOLS

CODE = '''
class Params(BaseModel):
    n: int = Field(1, description="n")

def run(params: Params, ctx: ToolContext) -> dict:
    return {"ok": True}
'''


def mk(tmp_path, name):
    return ExtensionManager(tmp_path / name, lambda: set(BUILTIN_TOOLS), lambda: {"self-review"})


@pytest.fixture
def pair(tmp_path):
    a, b = mk(tmp_path, "a"), mk(tmp_path, "b")
    a.propose_tool("thing_tool", "does a thing for tests", CODE, "r"); a.approve("tool", "thing_tool")
    a.propose_skill("thing-skill", "Use when doing the thing", "Call thing_tool then report the result.", ["thing_tool"])
    a.approve("skill", "thing-skill")
    return a, b


def rebuild(raw, mutate):
    zin = zipfile.ZipFile(io.BytesIO(raw)); files = {n: zin.read(n) for n in zin.namelist()}
    mutate(files)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
    return buf.getvalue()


def test_roundtrip_lands_in_pending_never_approved(pair):
    a, b = pair
    raw, inc = a.export_package()
    assert {(i["kind"], i["name"]) for i in inc} == {("tool", "thing_tool"), ("skill", "thing-skill")}
    r = b.import_package(raw)
    assert len(r["imported"]) == 2 and not r["skipped"]
    assert {i["name"] for i in b.list_items("pending")} == {"thing_tool", "thing-skill"}
    assert b.list_items("approved") == [] and b.approved_tool_files() == []
    t = next(i for i in b.list_items("pending") if i["name"] == "thing_tool")
    assert t["description"] == "does a thing for tests" and "Imported" in t["rationale"]
    b.approve("tool", "thing_tool")
    assert [n for n, _ in b.approved_tool_files()] == ["thing_tool"]


def test_exporting_a_skill_brings_its_tools(pair):
    a, _ = pair
    _, inc = a.export_package([("skill", "thing-skill")])
    assert ("tool", "thing_tool") in {(i["kind"], i["name"]) for i in inc}


def test_export_unknown_or_unapproved_fails(pair):
    a, _ = pair
    with pytest.raises(PackageError):
        a.export_package([("tool", "nope_tool")])
    with pytest.raises(PackageError):
        mk(a.root.parent, "empty").export_package()


def test_tampered_content_is_skipped(pair):
    a, b = pair
    raw, _ = a.export_package([("tool", "thing_tool")])
    bad = rebuild(raw, lambda f: f.update({"tools/thing_tool.py": f["tools/thing_tool.py"] + b"\n# edited"}))
    r = b.import_package(bad)
    assert not r["imported"] and "manifest" in r["skipped"][0]["reason"]


def test_dangerous_tool_is_rejected_by_lint_even_with_valid_hash(pair):
    a, b = pair
    raw, _ = a.export_package([("tool", "thing_tool")])
    evil = b"import os\n" + a.get_source("tool", "thing_tool", "approved").encode()

    def mutate(f):
        m = json.loads(f["manifest.json"]); m["items"][0]["sha256"] = pkg._sha(evil)
        f["manifest.json"] = json.dumps(m).encode(); f["tools/thing_tool.py"] = evil
    r = b.import_package(rebuild(raw, mutate))
    assert not r["imported"] and "lint" in r["skipped"][0]["reason"]
    assert b.list_items("pending") == []


def test_path_traversal_and_junk_entries_rejected(pair):
    a, b = pair
    raw, _ = a.export_package()
    for name in ("../evil.py", "/abs.py", "tools/sub/x.py", "other.txt"):
        with pytest.raises(PackageError):
            b.import_package(rebuild(raw, lambda f, n=name: f.update({n: b"x"})))


def test_not_a_package(pair):
    _, b = pair
    for raw in (b"hello", rebuild(pair[0].export_package()[0], lambda f: f.update({"manifest.json": b'{"format":"x"}'}))):
        with pytest.raises(PackageError):
            b.import_package(raw)


def test_builtin_name_collision_skipped_and_replace_flagged(pair):
    a, b = pair
    raw, _ = a.export_package()
    b.propose_tool("thing_tool", "does a thing for tests", CODE, "r"); b.approve("tool", "thing_tool")
    r = b.import_package(raw)
    assert next(i for i in r["imported"] if i["name"] == "thing_tool")["replaces_approved"] is True
    a.propose_skill("clash-skill", "Use when clashing here", "Long enough instructions here.", [])
    a.approve("skill", "clash-skill")
    c = ExtensionManager(a.root.parent / "c", lambda: set(), lambda: {"clash-skill"})
    r = c.import_package(a.export_package([("skill", "clash-skill")])[0])
    assert r["skipped"] and "built-in" in r["skipped"][0]["reason"]
