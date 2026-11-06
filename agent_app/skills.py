"""Skills: folders with a SKILL.md (frontmatter + instructions), loaded on demand.

The system prompt only lists name + description; the agent calls ``load_skill`` to read the body.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .extensions import ExtensionManager


@dataclass
class Skill:
    name: str
    description: str
    body: str
    origin: str  # "builtin" | "generated"
    tools: list[str] = field(default_factory=list)


def parse_skill(text: str, origin: str) -> Skill:
    if not text.startswith("---"):
        raise ValueError("SKILL.md must start with frontmatter")
    _, fm, body = text.split("---", 2)
    meta = {}
    for line in fm.strip().splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    tools = [t.strip() for t in meta.get("tools", "").split(",") if t.strip()]
    return Skill(meta["name"], meta["description"], body.strip(), origin, tools)


class SkillStore:
    def __init__(self, builtin_dir: Path, extensions: ExtensionManager):
        self.builtin_dir = Path(builtin_dir)
        self.extensions = extensions

    def builtin_names(self) -> set[str]:
        return {p.parent.name for p in self.builtin_dir.glob("*/SKILL.md")}

    def all(self) -> dict[str, Skill]:
        out: dict[str, Skill] = {}
        for p in sorted(self.builtin_dir.glob("*/SKILL.md")):
            s = parse_skill(p.read_text(), "builtin")
            out[s.name] = s
        for name, p in self.extensions.approved_skill_files():
            try:
                s = parse_skill(p.read_text(), "generated")
            except Exception:
                continue
            out.setdefault(s.name, s)  # builtin wins on collision
        return out
