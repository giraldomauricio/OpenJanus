"""DiagramSpec -> .drawio XML, and .drawio (plain or compressed) -> DiagramSpec-shaped dict."""
from __future__ import annotations

import base64
import html
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
import zlib
from typing import Any

from .layout import layout
from .spec import ID, STRIDE, DiagramSpec, Group, Node, Threat

MAX_FILE_BYTES = 5_000_000
MAX_INFLATED_BYTES = 20_000_000

_KIND_STYLE = {
    "actor": "shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;outlineConnect=0;",
    "process": "ellipse;",
    "service": "rounded=1;arcSize=12;",
    "datastore": "shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=12;",
    "queue": "shape=parallelogram;perimeter=parallelogramPerimeter;",
    "external": "rounded=0;dashed=1;",
    "generic": "rounded=0;",
}
_SEV_RANK = {"low": 1, "medium": 2, "high": 3}
_SEV_COLORS = {3: ("#f8cecc", "#b85450"), 2: ("#fff2cc", "#d6b656"), 1: ("#dae8fc", "#6c8ebf")}


def _worst(threats: list[Threat]) -> int:
    return max((_SEV_RANK[t.severity] for t in threats), default=0)


def _badge(threats: list[Threat]) -> str:
    letters = " ".join(sorted({t.category for t in threats}, key="STRIDE".index))
    return f'<br><font style="font-size:10px">⚠ {letters}</font>' if letters else ""


def _tooltip(notes: str, threats: list[Threat]) -> str:
    lines = [notes] if notes else []
    for t in threats:
        lines.append(f"[{t.category}/{t.severity}] {t.title}" + (f" — Mitigation: {t.mitigation}" if t.mitigation else ""))
    return "\n".join(lines)


def to_drawio(spec: DiagramSpec) -> tuple[str, dict[str, Any]]:
    """Returns (xml, stats)."""
    nboxes, gboxes, (W, H) = layout([n.id for n in spec.nodes], {n.id: n.group for n in spec.nodes},
                                    [g.id for g in spec.groups], [(e.source, e.target) for e in spec.edges], spec.direction)
    root = ET.Element("root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    def cell(parent_el: ET.Element, cid: str, value: str, style: str, box, parent: str, extra: dict[str, str] | None = None,
             attrs: dict[str, str] | None = None, vertex: bool = True) -> None:
        holder = parent_el
        if attrs:  # an <object> wrapper carries tooltip + custom data
            holder = ET.SubElement(parent_el, "object", id=cid, label=value, **attrs)
            c = ET.SubElement(holder, "mxCell", style=style, parent=parent, vertex="1")
        else:
            c = ET.SubElement(holder, "mxCell", id=cid, value=value, style=style, parent=parent, vertex="1")
        x, y, w, h = box
        ET.SubElement(c, "mxGeometry", x=str(round(x)), y=str(round(y)), width=str(round(w)), height=str(round(h))).set("as", "geometry")

    for g in spec.groups:
        style = ("rounded=1;arcSize=4;whiteSpace=wrap;html=1;container=1;collapsible=0;verticalAlign=top;fontStyle=1;spacingTop=4;"
                 + ("dashed=1;dashPattern=8 4;fillColor=none;strokeColor=#b85450;strokeWidth=2;fontColor=#b85450;" if g.boundary
                    else "fillColor=#f5f5f5;strokeColor=#999999;fontColor=#333333;"))
        label = html.escape(g.label) + ("" if not g.boundary else " <i>(trust boundary)</i>")
        cell(root, f"g_{g.id}", label, style, gboxes[g.id], "1", attrs={"janus_kind": "group", "janus_boundary": str(g.boundary).lower(),
                                                                       "janus_label": g.label})
    for n in spec.nodes:
        style = f"whiteSpace=wrap;html=1;{_KIND_STYLE[n.kind]}"
        sev = _worst(n.threats)
        if sev:
            fill, stroke = _SEV_COLORS[sev]
            style += f"fillColor={fill};strokeColor={stroke};"
        parent = f"g_{n.group}" if n.group else "1"
        attrs = {"janus_kind": n.kind, "janus_label": n.label}
        if tip := _tooltip(n.notes, n.threats):
            attrs["tooltip"] = tip
        if n.threats:
            attrs["janus_threats"] = json.dumps([t.model_dump() for t in n.threats], ensure_ascii=False)
        if n.notes:
            attrs["janus_notes"] = n.notes
        cell(root, f"n_{n.id}", html.escape(n.label) + _badge(n.threats), style, nboxes[n.id], parent, attrs=attrs)
    for i, e in enumerate(spec.edges):
        style = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;" + ("startArrow=block;startFill=1;" if e.bidirectional else "")
        if e.dashed:
            style += "dashed=1;"
        sev = _worst(e.threats)
        if sev:
            style += f"strokeColor={_SEV_COLORS[sev][1]};"
        value = html.escape(e.label) + _badge(e.threats)
        attrs = {}
        if tip := _tooltip("", e.threats):
            attrs["tooltip"] = tip
        if e.threats:
            attrs["janus_threats"] = json.dumps([t.model_dump() for t in e.threats], ensure_ascii=False)
        if e.label:
            attrs["janus_label"] = e.label
        if attrs:
            holder = ET.SubElement(root, "object", id=f"e_{i}", label=value, **attrs)
            c = ET.SubElement(holder, "mxCell", style=style, edge="1", parent="1", source=f"n_{e.source}", target=f"n_{e.target}")
        else:
            c = ET.SubElement(root, "mxCell", id=f"e_{i}", value=value, style=style, edge="1", parent="1",
                              source=f"n_{e.source}", target=f"n_{e.target}")
        ET.SubElement(c, "mxGeometry", relative="1").set("as", "geometry")

    used = sorted({t.category for x in [*spec.nodes, *spec.edges] for t in x.threats}, key="STRIDE".index)
    if used:
        text = "<b>STRIDE</b><br>" + "<br>".join(f"<b>{k}</b> {STRIDE[k]}" for k in used) + "<br><br>Red = high · Yellow = medium · Blue = low severity"
        cell(root, "legend", text, "text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;", (0, H + 50, 320, 20 + 18 * (len(used) + 3)), "1")

    model = ET.Element("mxGraphModel", dx="1200", dy="800", grid="1", gridSize="10", guides="1", tooltips="1", connect="1",
                       arrows="1", fold="1", page="1", pageScale="1", pageWidth=str(max(827, round(W) + 100)),
                       pageHeight=str(max(1169, round(H) + 300)), math="0", shadow="0")
    model.append(root)
    file = ET.Element("mxfile", host="OpenJanus", version="1")
    page = ET.SubElement(file, "diagram", id="page-1", name=spec.title[:60] or "Diagram")
    page.append(model)
    ET.indent(file)
    stats = {"nodes": len(spec.nodes), "edges": len(spec.edges), "groups": len(spec.groups),
             "threats": sum(len(x.threats) for x in [*spec.nodes, *spec.edges])}
    return ET.tostring(file, encoding="unicode", xml_declaration=False), stats


# ---------------------------------------------------------------------------------------------------------------- reading
def _check_xml(data: bytes) -> None:
    if re.search(rb"<!\s*(DOCTYPE|ENTITY)", data, re.I):
        raise ValueError("DOCTYPE/ENTITY declarations are not allowed in diagram files")


def _inflate(text: str) -> bytes:
    """draw.io 'compressed' pages: base64( raw-deflate( urlencode(xml) ) )."""
    d = zlib.decompressobj(-15)
    out = d.decompress(base64.b64decode(text.strip()), MAX_INFLATED_BYTES)
    if d.unconsumed_tail:
        raise ValueError("Diagram is too large once decompressed")
    return urllib.parse.unquote(out.decode("utf-8")).encode("utf-8")


def _pages(data: bytes) -> list[tuple[str, ET.Element]]:
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"File too large ({MAX_FILE_BYTES // 1_000_000} MB max)")
    _check_xml(data)
    try:
        top = ET.fromstring(data)
    except ET.ParseError as e:
        raise ValueError(f"Not valid XML: {e}") from e
    if top.tag == "mxGraphModel":
        return [("Page 1", top)]
    if top.tag != "mxfile":
        raise ValueError("Not a draw.io file (expected <mxfile> or <mxGraphModel>)")
    pages = []
    for i, d in enumerate(top.findall("diagram")):
        name = d.get("name") or f"Page {i + 1}"
        if (m := d.find("mxGraphModel")) is not None:
            pages.append((name, m))
        elif d.text and d.text.strip():
            try:
                raw = _inflate(d.text)
                _check_xml(raw)
                pages.append((name, ET.fromstring(raw)))
            except (ValueError, zlib.error, ET.ParseError, UnicodeDecodeError) as e:
                raise ValueError(f"Cannot decode page '{name}': {e}") from e
    if not pages:
        raise ValueError("The file has no diagram pages")
    return pages


def _plain(label: str) -> str:
    t = re.sub(r"<\s*(br|/p|/div|/li)\s*/?>", " ", label, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    return " ".join(html.unescape(t).split())


def _kind(style: str) -> str:
    s = style.lower()
    if "umlactor" in s or "shape=actor" in s:
        return "actor"
    if "cylinder" in s or "database" in s or "datastore" in s:
        return "datastore"
    if s.startswith("ellipse") or ";ellipse" in s or "shape=ellipse" in s:
        return "process"
    if "parallelogram" in s or "queue" in s:
        return "queue"
    if "dashed=1" in s:
        return "external"
    if "rounded=1" in s:
        return "service"
    return "generic"


def _collect(model: ET.Element) -> dict[str, dict[str, Any]]:
    """Every cell by id, with the attributes of <object>/<UserObject> wrappers (label, tooltip, custom data) merged in."""
    cells: dict[str, dict[str, Any]] = {}
    for el in model.iter():
        if el.tag == "mxCell":
            attrs, cid = dict(el.attrib), el.get("id")
            attrs["custom"] = {}
        elif el.tag in ("object", "UserObject") and (inner := el.find("mxCell")) is not None:
            attrs, cid = {**inner.attrib, "value": el.get("label", "")}, el.get("id")
            attrs["custom"] = dict(el.attrib)
        else:
            continue
        if cid and cid not in ("0", "1"):
            attrs["_id"] = cid
            cells[cid] = attrs
    return cells


def _threats(raw: str | None, label: str, warnings: list[str]) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        return [Threat.model_validate(t).model_dump() for t in json.loads(raw)]
    except (ValueError, TypeError):
        warnings.append(f"Could not read stored threats on '{label}'")
        return []


def read_drawio(data: bytes, page: int = 0) -> dict[str, Any]:
    pages = _pages(data)
    if not 0 <= page < len(pages):
        raise ValueError(f"page must be 0..{len(pages) - 1}")
    title, model = pages[page]
    cells = _collect(model)
    parents = {c.get("parent") for c in cells.values()}
    warnings: list[str] = []
    used: set[str] = set()

    def new_id(raw: str) -> str:
        base = re.sub(r"[^A-Za-z0-9_.-]", "_", re.sub(r"^[gne]_", "", raw))[:36] or "x"
        out, k = base, 2
        while out in used:
            out = f"{base}_{k}"; k += 1
        used.add(out)
        return out

    def label_of(c: dict[str, Any]) -> str:
        return c["custom"].get("janus_label") or _plain(c.get("value", ""))

    def is_group(c: dict[str, Any]) -> bool:
        st = c.get("style", "")
        return (c["_id"] in parents or "container=1" in st or st.startswith("swimlane") or ";swimlane" in st
                or c["custom"].get("janus_kind") == "group")

    vertices = [c for c in cells.values() if c.get("vertex") == "1"]
    connected = {x for c in cells.values() if c.get("edge") == "1" for x in (c.get("source"), c.get("target"))}
    groups, nodes, annotations, idmap = [], [], [], {}
    gcells = [c for c in vertices if is_group(c)]
    for c in gcells:  # ids first, so a child can name its parent whatever the order in the file
        idmap[c["_id"]] = new_id(c["_id"])
    for c in vertices:
        if c in gcells or c["_id"] == "legend":
            continue
        if c.get("style", "").startswith("text") and c["_id"] not in connected:  # free text: keep as an annotation, not a node
            if label_of(c):
                annotations.append(label_of(c))
            continue
        idmap[c["_id"]] = new_id(c["_id"])
    for c in gcells:
        st = c.get("style", "")
        boundary = (c["custom"].get("janus_boundary") == "true" if "janus_boundary" in c["custom"]
                    else "dashed=1" in st and "fillColor=none" in st)
        groups.append({"id": idmap[c["_id"]], "label": label_of(c).removesuffix(" (trust boundary)") or idmap[c["_id"]], "boundary": boundary})
    for c in vertices:
        if c in gcells or c["_id"] not in idmap:
            continue
        cu, label = c["custom"], label_of(c)
        n: dict[str, Any] = {"id": idmap[c["_id"]], "label": label or idmap[c["_id"]],
                             "kind": cu["janus_kind"] if cu.get("janus_kind") in _KIND_STYLE else _kind(c.get("style", ""))}
        if (p := c.get("parent")) in idmap and p in {g["_id"] for g in gcells}:
            n["group"] = idmap[p]
        if t := _threats(cu.get("janus_threats"), label, warnings):
            n["threats"] = t
        if cu.get("janus_notes"):
            n["notes"] = cu["janus_notes"]
        nodes.append(n)
    gids = {g["id"] for g in groups}
    edges = []
    for c in cells.values():
        if c.get("edge") != "1":
            continue
        s, t, label = idmap.get(c.get("source", "")), idmap.get(c.get("target", "")), label_of(c)
        if not s or not t or s in gids or t in gids:
            why = "attached to a container" if (s in gids or t in gids) else "not attached to a shape at both ends"
            warnings.append(f"Skipped connector '{label or c['_id']}': {why}")
            continue
        st = c.get("style", "")
        e: dict[str, Any] = {"source": s, "target": t, "label": label}
        if "startArrow=" in st and "startArrow=none" not in st:
            e["bidirectional"] = True
        if "dashed=1" in st:
            e["dashed"] = True
        if th := _threats(c["custom"].get("janus_threats"), label or c["_id"], warnings):
            e["threats"] = th
        edges.append(e)
    out: dict[str, Any] = {"title": title, "page": page, "pages": [n for n, _ in pages], "groups": groups, "nodes": nodes, "edges": edges}
    if annotations:
        out["annotations"] = annotations[:50]
    if warnings:
        out["warnings"] = warnings[:20]
    return out
