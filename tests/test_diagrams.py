import base64
import itertools
import urllib.parse
import zlib

import pytest
from pydantic import ValidationError

from agent_app import config
from agent_app.diagrams.drawio import read_drawio, to_drawio
from agent_app.diagrams.layout import layout
from agent_app.diagrams.spec import DiagramSpec
from agent_app.diagrams.store import DiagramStore, clean_name

SPEC = {
    "title": "Pay",
    "groups": [{"id": "dmz", "label": "DMZ", "boundary": True}, {"id": "int", "label": "Internal"}],
    "nodes": [
        {"id": "user", "label": "Customer <script>", "kind": "actor"},
        {"id": "web", "label": "Web", "kind": "service", "group": "dmz",
         "threats": [{"category": "S", "title": "Session hijack", "severity": "high", "mitigation": "MFA"}]},
        {"id": "api", "label": "API", "kind": "process", "group": "int"},
        {"id": "db", "label": "DB", "kind": "datastore", "group": "int", "notes": "PII"},
    ],
    "edges": [{"source": "user", "target": "web", "label": "HTTPS"}, {"source": "web", "target": "api"},
              {"source": "api", "target": "db", "bidirectional": True}, {"source": "db", "target": "web"}],  # cycle on purpose
}


def test_spec_rejects_inconsistent_input():
    for bad in ({"nodes": [{"id": "a", "label": "A"}, {"id": "a", "label": "B"}]},
                {"nodes": [{"id": "a", "label": "A", "group": "nope"}]},
                {"nodes": [{"id": "a", "label": "A"}], "edges": [{"source": "a", "target": "zzz"}]},
                {"nodes": [{"id": "bad id!", "label": "A"}]}):
        with pytest.raises(ValidationError):
            DiagramSpec.model_validate(bad)


def _overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


@pytest.mark.parametrize("direction", ["LR", "TB"])
def test_layout_groups_never_overlap_and_contain_members(direction):
    spec = DiagramSpec.model_validate({**SPEC, "direction": direction})
    nodes, groups, _ = layout([n.id for n in spec.nodes], {n.id: n.group for n in spec.nodes}, [g.id for g in spec.groups],
                              [(e.source, e.target) for e in spec.edges], direction)
    top = {**{g: b for g, b in groups.items()}, **{n.id: nodes[n.id] for n in spec.nodes if n.group is None}}
    for a, b in itertools.combinations(top.values(), 2):
        assert not _overlap(a, b)
    for n in spec.nodes:
        if n.group:  # relative to the group: must lie inside it
            x, y, w, h = nodes[n.id]
            gx, gy, gw, gh = groups[n.group]
            assert x >= 0 and y >= 0 and x + w <= gw and y + h <= gh
    inner = [nodes[n.id] for n in spec.nodes if n.group == "int"]
    assert not _overlap(*inner)


def test_big_cyclic_graph_does_not_hang():
    nodes = [{"id": f"n{i}", "label": str(i)} for i in range(120)]
    edges = [{"source": f"n{i}", "target": f"n{(i * 7 + 3) % 120}"} for i in range(120)]
    xml, stats = to_drawio(DiagramSpec.model_validate({"nodes": nodes, "edges": edges}))
    assert stats["nodes"] == 120 and xml.startswith("<mxfile")


def test_labels_are_escaped_and_roundtrip():
    xml, stats = to_drawio(DiagramSpec.model_validate(SPEC))
    assert "<script>" not in xml and "&lt;script&gt;" in xml and stats["threats"] == 1
    back = read_drawio(xml.encode())
    assert back["nodes"][0]["label"] == "Customer <script>"
    web = next(n for n in back["nodes"] if n["id"] == "web")
    assert web["group"] == "dmz" and web["threats"][0]["mitigation"] == "MFA"
    assert next(g for g in back["groups"] if g["id"] == "dmz")["boundary"] is True
    assert next(e for e in back["edges"] if e["target"] == "db")["bidirectional"] is True
    DiagramSpec.model_validate({k: back[k] for k in ("title", "groups", "nodes", "edges")})  # feeds straight back into create_diagram


def test_reads_compressed_and_foreign_files():
    inner = ('<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
             '<mxCell id="a" value="&lt;b&gt;Web&lt;/b&gt;&lt;br&gt;app" style="rounded=1;" vertex="1" parent="1"><mxGeometry/></mxCell>'
             '<mxCell id="b" value="Store" style="shape=cylinder3;" vertex="1" parent="1"><mxGeometry/></mxCell>'
             '<mxCell id="c" value="reads" edge="1" parent="1" source="a" target="b"><mxGeometry relative="1"/></mxCell>'
             '<mxCell id="d" edge="1" parent="1" source="a"><mxGeometry relative="1"/></mxCell></root></mxGraphModel>')
    comp = zlib.compressobj(wbits=-15)
    packed = base64.b64encode(comp.compress(urllib.parse.quote(inner).encode()) + comp.flush()).decode()
    r = read_drawio(f'<mxfile><diagram name="P1">{packed}</diagram><diagram name="P2">{packed}</diagram></mxfile>'.encode())
    assert r["pages"] == ["P1", "P2"]
    assert [(n["label"], n["kind"]) for n in r["nodes"]] == [("Web app", "service"), ("Store", "datastore")]
    assert r["edges"] == [{"source": "a", "target": "b", "label": "reads"}] and len(r["warnings"]) == 1
    assert read_drawio(inner.encode())["nodes"]  # bare mxGraphModel too


@pytest.mark.parametrize("data", [
    b'<!DOCTYPE x [<!ENTITY a "aaaa">]><mxfile><diagram>x</diagram></mxfile>',
    b"<html></html>", b"not xml", b"<mxfile></mxfile>", b"<mxfile><diagram>!!notbase64!!</diagram></mxfile>",
])
def test_bad_files_raise_value_error(data):
    with pytest.raises(ValueError):
        read_drawio(data)


def test_decompression_bomb_is_refused():
    comp = zlib.compressobj(wbits=-15)
    bomb = base64.b64encode(comp.compress(b"A" * 30_000_000) + comp.flush()).decode()
    with pytest.raises(ValueError):
        read_drawio(f"<mxfile><diagram>{bomb}</diagram></mxfile>".encode())


@pytest.mark.parametrize("name", ["../x", "a/b", "..", ".hidden", "x" * 80, "", "a b", "evil\\x"])
def test_store_names_are_validated(name):
    with pytest.raises(ValueError):
        clean_name(name)


def test_store_roundtrip(tmp_path):
    st = DiagramStore(tmp_path / "d")
    assert st.list() == [] and st.save("arch", b"<x/>") == "arch.drawio" and st.read("arch") == b"<x/>"
    assert [d["name"] for d in st.list()] == ["arch.drawio"]
    with pytest.raises(ValueError):
        st.read("missing")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DIAGRAMS_DIR", tmp_path / "diagrams")
    from fastapi.testclient import TestClient

    from agent_app import main
    monkeypatch.setattr(main, "diagram_store", DiagramStore(tmp_path / "diagrams"))
    return TestClient(main.app), {"X-Agent-Token": main.TOKEN}, main


def test_tools_and_endpoints_end_to_end(client):
    c, H, main = client
    ok, text, raw = main.agent._run(main.agent.build_tools()[0], "create_diagram", {"name": "demo", **SPEC})
    assert ok, text
    assert raw["download_url"] == "/api/diagrams/demo.drawio" and raw["threats"] == 1
    r = c.get(raw["download_url"])
    assert r.status_code == 200 and r.headers["content-disposition"].startswith("attachment") and b"<mxfile" in r.content
    ok, text, raw = main.agent._run(main.agent.build_tools()[0], "read_diagram", {"name": "demo"})
    assert ok and len(raw["nodes"]) == 4
    ok, text, _ = main.agent._run(main.agent.build_tools()[0], "create_diagram", {"name": "x", "nodes": [{"id": "a", "label": "A", "group": "zz"}]})
    assert not ok and "unknown group" in text
    ok, text, _ = main.agent._run(main.agent.build_tools()[0], "create_diagram", {"name": "../evil", **SPEC})
    assert not ok
    assert c.get("/api/diagrams/..%2Fetc").status_code in (404, 422)
    # upload: token required, must be a real diagram
    body = r.content
    assert c.post("/api/diagrams?name=up", content=body).status_code == 403
    assert c.post("/api/diagrams?name=up", content=body, headers=H).json()["nodes"] == 4
    assert c.post("/api/diagrams?name=bad", content=b"<html/>", headers=H).status_code == 400
    assert {d["name"] for d in c.get("/api/diagrams").json()} == {"demo.drawio", "up.drawio"}
    assert c.get("/api/config").json()["features"]["diagrams"] is True


def test_skill_ships_and_references_real_tools(client):
    *_, main = client
    sk = main.skills_store.all()["threat-model-diagrams"]
    assert sk.origin == "builtin"
    tools = main.agent.build_tools()[0]
    assert all(t in tools for t in sk.tools)


def test_create_diagram_accepts_loose_shapes_from_small_models(client):
    """Real failure seen with a local model: from/to, `type`, no ids, groups as name + member list."""
    c, H, main = client
    loose = {"name": "loose", "direction": "LR", "title": "T",
             "nodes": [{"label": "Customer", "type": "user"}, {"label": "Web App", "type": "component"}, {"label": "Orders DB", "type": "database"},
                       {"label": "Web App", "type": "component"}],
             "groups": [{"name": "Internal network", "nodes": ["Orders DB", "Web App"]}],
             "edges": [{"from": "Customer", "to": "Web App", "label": "HTTPS"}, {"from": "Web App", "to": "Orders DB"}]}
    ok, text, raw = main.agent._run(main.agent.build_tools()[0], "create_diagram", loose)
    assert ok, text
    back = read_drawio(c.get(raw["download_url"]).content)
    kinds = {n["label"]: n["kind"] for n in back["nodes"]}
    assert kinds["Customer"] == "actor" and kinds["Orders DB"] == "datastore" and back["groups"][0]["label"] == "Internal network"
    assert next(n for n in back["nodes"] if n["label"] == "Orders DB")["group"] == back["groups"][0]["id"]
    assert len(back["edges"]) == 2 and len(back["nodes"]) == 4  # duplicate label got its own id
    # still strict where it matters: an edge to something that does not exist is reported, not silently dropped
    bad = {**loose, "name": "bad", "edges": [{"from": "Customer", "to": "Nowhere"}]}
    ok, text, _ = main.agent._run(main.agent.build_tools()[0], "create_diagram", bad)
    assert not ok and "not a node id" in text
    # the schema the model sees is unchanged and strict
    assert "source" in main.BUILTIN_TOOLS["create_diagram"].schema()["$defs"]["Edge"]["required"]


def test_create_diagram_second_real_failure_shape(client):
    """gemma4 sent kind 'System' and boundary 'dashed' (strings); both used to reject the whole call."""
    c, H, main = client
    spec = {"name": "second", "groups": [{"id": "G1", "label": "Internal network", "boundary": "dashed"},
                                         {"id": "G2", "label": "Open zone", "boundary": "solid"}],
            "nodes": [{"id": "N1", "kind": "System", "label": "Customer", "group": "G2"}, {"id": "N2", "kind": "Database", "label": "DB", "group": "G1"}],
            "edges": [{"source": "N1", "target": "N2", "label": "SQL"}]}
    ok, text, raw = main.agent._run(main.agent.build_tools()[0], "create_diagram", spec)
    assert ok, text
    back = read_drawio(c.get(raw["download_url"]).content)
    assert {g["id"]: g["boundary"] for g in back["groups"]} == {"G1": True, "G2": False}
    assert {n["id"]: n["kind"] for n in back["nodes"]} == {"N1": "generic", "N2": "datastore"}
