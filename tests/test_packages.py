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
    raw, res = a.export_package(); inc = res["items"]
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
    _, res = a.export_package([("skill", "thing-skill")]); inc = res["items"]
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


SECRET_CODE = "SECRETS = {'DEMO_TOKEN': ['api.example.com']}\n" + CODE.replace(
    'return {"ok": True}', 'return {"ok": ctx.http.get("https://api.example.com/x", headers={"Authorization": "Bearer {secret:DEMO_TOKEN}"}).status}')


def test_metadata_roundtrip_and_defaults(pair, monkeypatch):
    a, b = pair
    raw, res = a.export_package(info={"name": "my-pack", "version": "2.1.0", "description": "Demo pack", "author": "Ana"})
    assert res["package"]["name"] == "my-pack" and res["package"]["version"] == "2.1.0"
    r = b.import_package(raw)
    assert r["package"] == {"name": "my-pack", "version": "2.1.0", "description": "Demo pack", "author": "Ana"}
    assert len(r["fingerprint"]) == 16
    t = next(i for i in b.list_items("pending") if i["name"] == "thing_tool")
    assert {k: t["package"][k] for k in ("name", "version", "author")} == {"name": "my-pack", "version": "2.1.0", "author": "Ana"} \
        and t["package"]["fingerprint"] == r["fingerprint"] and "my-pack 2.1.0 by Ana" in t["rationale"]
    b.approve("tool", "thing_tool")  # provenance survives approval
    assert next(i for i in b.list_items("approved") if i["name"] == "thing_tool")["package"]["name"] == "my-pack"
    assert a.export_package([("tool", "thing_tool")])[1]["package"] == {"name": "thing-tool", "version": "1.0.0", "requires_secrets": {}}


@pytest.mark.parametrize("bad", [{"name": "X"}, {"version": "1.0"}, {"description": "a\nb"}, {"author": "x" * 81}])
def test_bad_metadata_rejected_on_export(pair, bad):
    with pytest.raises(PackageError):
        pair[0].export_package(info=bad)


def test_required_secrets_come_from_source_not_manifest(tmp_path, monkeypatch):
    a, b = mk(tmp_path, "a"), mk(tmp_path, "b")
    a.propose_tool("secret_tool", "calls an api with a secret", SECRET_CODE, "r"); a.approve("tool", "secret_tool")
    raw, res = a.export_package()
    assert res["package"]["requires_secrets"] == {"DEMO_TOKEN": ["api.example.com"]}
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    assert b.import_package(raw)["requires_secrets"] == [{"name": "DEMO_TOKEN", "hosts": ["api.example.com"], "configured": False}]
    monkeypatch.setenv("DEMO_TOKEN", "x")
    lie = rebuild(raw, lambda f: f.update({"manifest.json": json.dumps({**json.loads(f["manifest.json"]), "package": {
        "name": "p-pack", "version": "1.0.0", "requires_secrets": {}}}).encode()}))
    assert b.import_package(lie)["requires_secrets"][0]["configured"] is True  # still reported: derived from the code


def test_old_or_malformed_metadata_still_imports(pair):
    a, b = pair
    raw, _ = a.export_package()
    for pk in (None, "junk", {"name": "BAD NAME", "version": "x"}):
        def mutate(f, pk=pk):
            m = json.loads(f["manifest.json"]); m.pop("package", None)
            if pk is not None:
                m["package"] = pk
            f["manifest.json"] = json.dumps(m).encode()
        r = b.import_package(rebuild(raw, mutate))
        assert len(r["imported"]) == 2 and r["package"]["name"] in ("unnamed-package",)


# --- signing -----------------------------------------------------------------------------------------------------------

def test_signed_package_roundtrip_and_trust(pair):
    a, b = pair
    raw, res = a.export_package(sign=True)
    fp = a.signer.own_fingerprint()
    assert fp and res["package"]["signed_by"] == fp
    r = b.import_package(raw)
    assert r["signer"] == {"fingerprint": fp, "trusted": False, "label": None}
    assert next(i for i in b.list_items("pending") if i["name"] == "thing_tool")["package"]["signer_trusted"] is False
    b.signer.trust(fp, "Ana")  # trusting later is reflected in the listing without re-importing
    p = next(i for i in b.list_items("pending") if i["name"] == "thing_tool")["package"]
    assert p["signer_trusted"] is True and p["signer_label"] == "Ana"
    assert b.import_package(raw)["signer"]["trusted"] is True


def test_unsigned_package_has_no_signer(pair):
    a, b = pair
    assert b.import_package(a.export_package()[0])["signer"] is None
    assert a.signer.own_fingerprint() is None  # exporting unsigned never creates a key


def test_modified_signed_package_is_refused(pair):
    a, b = pair
    raw, _ = a.export_package(sign=True)
    for mutate in (lambda f: f.update({"manifest.json": f["manifest.json"].replace(b"1.0.0", b"9.9.9")}),
                   lambda f: f.update({"signature.json": b"{}"}),
                   lambda f: f.update({"signature.json": b"not json"})):
        with pytest.raises(PackageError):
            b.import_package(rebuild(raw, mutate))
    assert b.list_items("pending") == []


def test_resigned_by_someone_else_shows_a_different_signer(pair, tmp_path):
    a, b = pair
    raw, _ = a.export_package(sign=True)
    evil = mk(tmp_path, "evil")
    manifest = zipfile.ZipFile(io.BytesIO(raw)).read("manifest.json")
    r = b.import_package(rebuild(raw, lambda f: f.update({"signature.json": json.dumps(evil.signer.sign(manifest)).encode()})))
    assert r["signer"]["fingerprint"] == evil.signer.own_fingerprint() != a.signer.own_fingerprint()


def test_trust_validation(pair):
    from agent_app.extensions import SignatureError
    a, _ = pair
    for fp, label in (("zz", "x"), ("0" * 16, "  ")):
        with pytest.raises(SignatureError):
            a.signer.trust(fp, label)
    assert a.signer.trusted() == {}


def test_private_key_is_owner_only(pair):
    a, _ = pair
    a.export_package(sign=True)
    assert (a.signer._key_path.stat().st_mode & 0o077) == 0
