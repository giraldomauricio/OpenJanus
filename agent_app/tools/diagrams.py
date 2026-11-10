"""Optional built-in tools: create and read draw.io diagrams (see skills `threat-model-diagrams`)."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .. import config
from ..diagrams.drawio import read_drawio as _read, to_drawio
from ..diagrams.spec import DiagramSpec, Edge, Group, Node
from ..diagrams.store import DiagramStore, clean_name
from ..sdk import AgentContext, tool


def _store() -> DiagramStore:
    return DiagramStore(config.DIAGRAMS_DIR)  # read at call time so tests / env changes apply


_KIND_WORDS = {
    "database": "datastore", "db": "datastore", "storage": "datastore", "store": "datastore", "datastore": "datastore",
    "service": "service", "app": "service", "application": "service", "api": "service", "server": "service", "component": "generic",
    "process": "process", "actor": "actor", "user": "actor", "person": "actor", "client": "actor", "external": "external",
    "third_party": "external", "queue": "queue", "broker": "queue", "stream": "queue", "topic": "queue",
}


_VALID_KINDS = {"actor", "process", "service", "datastore", "queue", "external", "generic"}


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("_.-")[:40] or "x"


def _loose(raw: Any) -> Any:
    """Smaller models often send a plausible but slightly different shape (from/to, type, no ids, groups as name + member list).
    Map the common variants onto the strict schema; anything it cannot map is left for the normal validation to report."""
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    used: set[str] = set()
    by_label: dict[str, str] = {}

    def unique(base: str) -> str:
        i, cand = 2, base
        while cand in used:
            cand, i = f"{base[:36]}_{i}", i + 1
        used.add(cand)
        return cand

    nodes = []
    for n in out.get("nodes") if isinstance(out.get("nodes"), list) else []:
        if not isinstance(n, dict):
            nodes.append(n)
            continue
        n = dict(n)
        label = n.get("label") or n.get("name") or n.get("title")
        if label is not None:
            n["label"] = str(label)
        if not n.get("id") and n.get("label"):
            n["id"] = _slug(n["label"])
        if isinstance(n.get("id"), str):
            n["id"] = unique(n["id"])
            by_label[n.get("label", n["id"])] = n["id"]
        word = n.get("kind") if isinstance(n.get("kind"), str) else n.get("type")
        if isinstance(word, str):  # an unknown word ("System", "component") becomes generic instead of failing the whole call
            word = word.strip().lower().replace(" ", "_")
            n["kind"] = word if word in _VALID_KINDS else _KIND_WORDS.get(word, "generic")
        n.pop("type", None); n.pop("name", None); n.pop("title", None)
        nodes.append(n)
    if nodes:
        out["nodes"] = nodes
    ids = {n["id"] for n in nodes if isinstance(n, dict) and isinstance(n.get("id"), str)}
    ref = lambda v: v if v in ids else by_label.get(v, v)  # noqa: E731  (a name or an id both work)

    groups = []
    for g in out.get("groups") if isinstance(out.get("groups"), list) else []:
        if not isinstance(g, dict):
            groups.append(g)
            continue
        g = dict(g)
        label = g.get("label") or g.get("name") or g.get("title")
        if label is not None:
            g["label"] = str(label)
        if not g.get("id") and g.get("label"):
            g["id"] = _slug(g["label"])
        if isinstance(g.get("boundary"), str):  # "dashed", "yes", "trust boundary" ... all mean a boundary
            g["boundary"] = g["boundary"].strip().lower() not in ("", "false", "no", "none", "0", "n", "solid")
        for member in g.pop("nodes", None) or g.pop("members", None) or []:
            for n in nodes:
                if isinstance(n, dict) and ref(member) == n.get("id") and not n.get("group"):
                    n["group"] = g.get("id")
        g.pop("name", None); g.pop("title", None)
        groups.append(g)
    if groups:
        out["groups"] = groups

    edges = []
    for e in out.get("edges") if isinstance(out.get("edges"), list) else []:
        if isinstance(e, dict):
            e = dict(e)
            for want, alts in (("source", ("from", "src", "start")), ("target", ("to", "dst", "end"))):
                if want not in e:
                    for a in alts:
                        if a in e:
                            e[want] = e.pop(a)
                            break
                if isinstance(e.get(want), str):
                    e[want] = ref(e[want])
        edges.append(e)
    if edges:
        out["edges"] = edges
    return out


class CreateDiagramParams(BaseModel):
    name: str = Field(description="File name without extension, e.g. 'payments-architecture' (letters, digits, . _ -). "
                                  "Re-using a name overwrites that file.")
    title: str = Field("Architecture", max_length=120)
    direction: str = Field("LR", pattern="^(LR|TB)$", description="LR = left to right, TB = top to bottom")
    groups: list[Group] = Field(default_factory=list, description="Zones / trust boundaries that contain nodes")
    nodes: list[Node] = Field(min_length=1)
    edges: list[Edge] = Field(default_factory=list)

    _normalise = model_validator(mode="before")(staticmethod(_loose))


@tool("create_diagram",
      "Create a draw.io (.drawio) architecture or data-flow diagram from a structured description and save it. You describe WHAT "
      "(nodes, edges, groups/trust boundaries, STRIDE threats); the tool does the layout and XML, so never write draw.io XML "
      "yourself. Shape: nodes [{id, label, kind}], edges [{source, target, label}] (source/target = node ids), groups [{id, label, "
      "boundary}] and a node's `group` = a group id. Returns a download link the user can open in draw.io / diagrams.net. Load the 'threat-model-diagrams' skill first.")
def create_diagram(params: CreateDiagramParams, ctx: AgentContext) -> Any:
    spec = DiagramSpec(title=params.title, direction=params.direction, groups=params.groups, nodes=params.nodes, edges=params.edges)
    name = clean_name(params.name)
    xml, stats = to_drawio(spec)
    _store().save(name, xml.encode("utf-8"))
    return {"file": name, "download_url": f"/api/diagrams/{name}", **stats,
            "note": "Saved. Tell the user they can download it from the link, or open it in draw.io. To change it, call "
                    "create_diagram again with the same name and the updated description."}


class ReadDiagramParams(BaseModel):
    name: str = Field(description="File name of an uploaded or created diagram (see list_diagrams)")
    page: int = Field(0, ge=0, description="Page index for multi-page files; the result lists all page names")


@tool("read_diagram",
      "Read a draw.io (.drawio) file (plain or compressed) into the same structure create_diagram accepts: groups, nodes (with a "
      "guessed kind), edges and, for diagrams made by OpenJanus, their STRIDE threats. Geometry/colours are not returned. "
      "Use it to understand, extend or modify an existing diagram, then call create_diagram to write the result.")
def read_diagram(params: ReadDiagramParams, ctx: AgentContext) -> Any:
    return _read(_store().read(params.name), params.page)


class NoParams(BaseModel):
    pass


@tool("list_diagrams", "List the draw.io diagram files available (uploaded by the user or created by you), newest first.")
def list_diagrams(params: NoParams, ctx: AgentContext) -> Any:
    return {"diagrams": [{"file": d["name"], "bytes": d["bytes"]} for d in _store().list()]}


class DraftFromDbParams(BaseModel):
    tables: list[str] = Field(default_factory=list, description="Only these tables; empty = all")
    infer: bool = Field(True, description="Also look for undeclared relationships by value overlap and column names (slower)")


MAX_DRAFT_TABLES = 25


@tool("draft_diagram_from_database",
      "GUESS a data-model diagram from the database itself: one node per table, one edge per relationship (declared references "
      "first, then inferred ones with their evidence and confidence). Does NOT save anything and does not call create_diagram: it "
      "returns `enough_data` (false = too little to guess, so ask the user instead) and a `draft` you can review, amend and pass "
      "to create_diagram. A database shows tables and how they relate, NOT services, networks or trust boundaries: say so, and "
      "never present the draft as the real architecture. If tables DESCRIBE components/flows (rows are systems and connections), "
      "read those rows with `query` and build the diagram from them instead.")
def draft_diagram_from_database(params: DraftFromDbParams, ctx: AgentContext) -> Any:
    db = ctx.db
    infos = {t["table"]: t for t in db.list_tables()}
    names = [t for t in (params.tables or list(infos)) if t in infos]
    unknown = [t for t in params.tables if t not in infos]
    if unknown:
        raise ValueError(f"No such table(s): {unknown}. Existing: {list(infos)}")
    if len(names) > MAX_DRAFT_TABLES:
        raise ValueError(f"{len(names)} tables is too many to draft at once (max {MAX_DRAFT_TABLES}): pass `tables` to pick a subset")
    rels: dict[tuple[str, str, str], dict[str, Any]] = {}
    for spec in db.all_specs():  # relationships the user/agent already declared
        for f in spec.fields:
            if f.references and spec.name in names and f.references.split(".")[0] in names:
                rels[(spec.name, f.name, f.references)] = {"from": f"{spec.name}.{f.name}", "to": f.references, "basis": "declared reference",
                                                           "confidence": "high"}
    if params.infer:
        for t in names:
            for r in db.find_relationships(t)["relationships"]:
                if r["direction"] != "outgoing" or r["to"].split(".")[0] not in names:
                    continue
                src, col = r["from"].split(".")
                rels.setdefault((src, col, r["to"]), {"from": r["from"], "to": r["to"], "confidence": r["confidence"],
                                                      "basis": f"inferred: {round(r['overlap'] * 100)}% of values match" + (", name matches" if r["name_match"] else "")})
    relationships = sorted(rels.values(), key=lambda r: {"high": 0, "medium": 1, "low": 2}[r["confidence"]])
    notes = ["This is the DATA MODEL as the database shows it, not the real system architecture."]
    if len(names) < 2:
        return {"enough_data": False, "reason": "Fewer than 2 tables: nothing to relate. Ask the user to describe the architecture or provide an image.",
                "tables": names, "notes": notes}
    if not relationships:
        return {"enough_data": False, "tables": names, "notes": notes,
                "reason": "No declared or inferable relationships between the tables, so a diagram would be a set of unconnected boxes. "
                          "Ask the user how the components connect, or whether some tables describe components/flows (then read them with `query`)."}
    nodes = [{"id": n, "label": f"{n} ({infos[n]['rows']} rows)", "kind": "datastore"} for n in names]
    edges = [{"source": r["from"].split(".")[0], "target": r["to"].split(".")[0], "label": r["from"].split(".")[1] + " → " + r["to"],
              "dashed": r["confidence"] != "high"} for r in relationships]
    if any(r["confidence"] == "low" for r in relationships):
        notes.append("'low' confidence relationships may be coincidences: confirm them with the user before drawing them.")
    notes.append("Dashed edges in the draft are inferred, not declared. Mention that when you show the diagram.")
    return {"enough_data": True, "tables": names, "relationships": relationships, "notes": notes,
            "draft": {"title": "Data model (inferred from the database)", "direction": "LR", "nodes": nodes, "edges": edges}}
