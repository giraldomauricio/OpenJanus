"""The small JSON description of a diagram that the model produces (and that `read_diagram` returns)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ID = r"^[A-Za-z0-9_.-]{1,40}$"
Kind = Literal["actor", "process", "service", "datastore", "queue", "external", "generic"]
STRIDE = {"S": "Spoofing", "T": "Tampering", "R": "Repudiation", "I": "Information disclosure",
          "D": "Denial of service", "E": "Elevation of privilege"}
Severity = Literal["low", "medium", "high"]


class Threat(BaseModel):
    category: Literal["S", "T", "R", "I", "D", "E"] = Field(description="STRIDE letter: S Spoofing, T Tampering, R Repudiation, "
                                                                        "I Information disclosure, D Denial of service, E Elevation of privilege")
    title: str = Field(max_length=200, description="Short threat statement")
    severity: Severity = "medium"
    mitigation: str = Field("", max_length=400)


class Node(BaseModel):
    id: str = Field(pattern=ID, description="Unique short id, e.g. 'api' (letters, digits, _ . -)")
    label: str = Field(max_length=120)
    kind: Kind = Field("generic", description="actor = person/external entity, process = logic/ellipse, service = app or API, "
                                              "datastore = database/storage, queue = broker/stream, external = third-party system")
    group: str | None = Field(None, description="id of the group (zone / trust boundary) this node sits in")
    threats: list[Threat] = Field(default_factory=list, max_length=20)
    notes: str = Field("", max_length=600, description="Shown as a tooltip in draw.io")


class Edge(BaseModel):
    source: str
    target: str
    label: str = Field("", max_length=120, description="What flows, e.g. 'HTTPS: credentials'")
    bidirectional: bool = False
    dashed: bool = Field(False, description="e.g. asynchronous or out-of-band flows")
    threats: list[Threat] = Field(default_factory=list, max_length=20)


class Group(BaseModel):
    id: str = Field(pattern=ID)
    label: str = Field(max_length=120)
    boundary: bool = Field(False, description="true = trust boundary (drawn dashed red); false = plain zone/grouping")


class DiagramSpec(BaseModel):
    title: str = Field("Architecture", max_length=120)
    direction: Literal["LR", "TB"] = Field("LR", description="LR = left to right, TB = top to bottom")
    groups: list[Group] = Field(default_factory=list, max_length=50)
    nodes: list[Node] = Field(min_length=1, max_length=300)
    edges: list[Edge] = Field(default_factory=list, max_length=600)

    @model_validator(mode="after")
    def _consistent(self) -> "DiagramSpec":
        gids = [g.id for g in self.groups]
        nids = [n.id for n in self.nodes]
        for what, ids in (("group", gids), ("node", nids)):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            if dup:
                raise ValueError(f"duplicate {what} id(s): {', '.join(dup)}")
        if set(gids) & set(nids):
            raise ValueError(f"group and node ids must differ: {', '.join(sorted(set(gids) & set(nids)))}")
        for n in self.nodes:
            if n.group is not None and n.group not in gids:
                raise ValueError(f"node '{n.id}' is in unknown group '{n.group}'")
        for e in self.edges:
            for end in (e.source, e.target):
                if end not in nids:
                    raise ValueError(f"edge {e.source}->{e.target}: '{end}' is not a node id")
        return self
