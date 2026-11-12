"""Pending -> approved lifecycle for generated tools and skills.

    data/extensions/
      pending/{tools,skills}/    written by the agent (propose_*), NEVER loaded
      approved/{tools,skills}/   moved here only by approve(), called from the UI endpoint
      rejected/{tools,skills}/   audit trail
      approved/manifest.json     sha256 of each approved file; a mismatch at load time = not loaded

The agent has no tool that calls approve()/reject(); only the HTTP layer does.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from .lint import TOOL_NAME, lint_skill, lint_tool_source
from .signing import Signer

log = logging.getLogger("openjanus.ext")
Kind = Literal["tool", "skill"]
Status = Literal["pending", "approved", "rejected", "retired"]

TOOL_TEMPLATE = '''"""Generated tool: {name}

Rationale: {rationale}
"""
from pydantic import BaseModel, Field

from agent_app.sdk import ToolContext

NAME = {name!r}
DESCRIPTION = {description!r}

{code}
'''

SKILL_TEMPLATE = """---
name: {name}
description: {description}
tools: {tools}
---
{instructions}
"""


class ProposalError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExtensionManager:
    def __init__(self, root: Path, reserved_tool_names: Callable[[], set[str]] = lambda: set(),
                 reserved_skill_names: Callable[[], set[str]] = lambda: set()):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._reserved_tools = reserved_tool_names
        self._reserved_skills = reserved_skill_names
        self.signer = Signer(self.root / "trust")
        for st in ("pending", "approved", "rejected", "retired"):
            for k in ("tools", "skills"):
                (self.root / st / k).mkdir(parents=True, exist_ok=True)

    # --- paths -------------------------------------------------------------

    def _content(self, kind: Kind, name: str, status: Status) -> Path:
        base = self.root / status / f"{kind}s"
        return base / f"{name}.py" if kind == "tool" else base / name / "SKILL.md"

    def _meta(self, kind: Kind, name: str, status: Status) -> Path:
        base = self.root / status / f"{kind}s"
        return base / f"{name}.json" if kind == "tool" else base / name / "meta.json"

    def _manifest_path(self) -> Path:
        return self.root / "approved" / "manifest.json"

    def _manifest(self) -> dict[str, dict[str, str]]:
        p = self._manifest_path()
        return json.loads(p.read_text()) if p.exists() else {"tool": {}, "skill": {}}

    def _exists_anywhere(self, kind: Kind, name: str, statuses=("pending", "approved")) -> Status | None:
        return next((s for s in statuses if self._content(kind, name, s).exists()), None)  # type: ignore[return-value]

    # --- proposing (called by the agent) -----------------------------------

    def propose_tool(self, name: str, description: str, code: str, rationale: str) -> dict[str, Any]:
        with self._lock:
            if not TOOL_NAME.match(name):
                raise ProposalError("Tool name must be snake_case: 3-41 chars, start with a letter")
            if name in self._reserved_tools():
                raise ProposalError(f"'{name}' is a built-in tool name")
            if not 10 <= len(description) <= 400:
                raise ProposalError("Description must be 10-400 chars")
            replaces_approved = self._exists_anywhere("tool", name, ("approved",)) is not None  # shown to the reviewer; approving swaps it in
            source = TOOL_TEMPLATE.format(name=name, description=description, rationale=rationale.replace('"""', "'''"),
                                          code=code.strip("\n"))
            lint = lint_tool_source(source)
            if not lint.ok:
                raise ProposalError("Tool rejected by lint: " + " | ".join(lint.errors))
            replaced = self._exists_anywhere("tool", name, ("pending",)) is not None
            path = self._content("tool", name, "pending")
            path.write_text(source)
            self._write_meta("tool", name, "pending", description, rationale, lint.capabilities)
            log.info("tool %s proposed (pending approval); capabilities=%s%s", name, lint.capabilities or "none",
                     "; replaces the approved version" if replaces_approved else "")
            return {"status": "pending_approval", "kind": "tool", "name": name, "replaced_pending": replaced,
                    "replaces_approved": replaces_approved, "capabilities": lint.capabilities}

    def propose_skill(self, name: str, description: str, instructions: str, tools: list[str]) -> dict[str, Any]:
        with self._lock:
            lint = lint_skill(name, description, instructions)
            if not lint.ok:
                raise ProposalError("Skill rejected by lint: " + " | ".join(lint.errors))
            if name in self._reserved_skills():
                raise ProposalError(f"A skill named '{name}' already exists; choose a different name")
            if instructions.lstrip().startswith("---"):
                raise ProposalError("Instructions must not start with '---' (frontmatter is added for you)")
            path = self._content("skill", name, "pending")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(SKILL_TEMPLATE.format(name=name, description=description,
                                                  tools=", ".join(tools), instructions=instructions.strip()))
            self._write_meta("skill", name, "pending", description, "", [])
            log.info("skill %s proposed (pending approval)", name)
            known = self._reserved_tools() | {n for n, _ in self.approved_tool_files()} | {
                i["name"] for i in self.list_items("pending") if i["kind"] == "tool"}
            unknown = [t for t in tools if t not in known]
            return {"status": "pending_approval", "kind": "skill", "name": name,
                    "replaces_approved": self._exists_anywhere("skill", name, ("approved",)) is not None,
                    "warning": f"References unknown tools: {unknown}" if unknown else None}

    def _write_meta(self, kind: Kind, name: str, status: Status, description: str, rationale: str,
                    caps: list[str], package: dict[str, str] | None = None) -> None:
        meta: dict[str, Any] = {
            "description": description, "rationale": rationale, "capabilities": caps,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if package:  # where an imported item came from; shown to the reviewer
            meta["package"] = package
        self._meta(kind, name, status).write_text(json.dumps(meta))

    # --- review (called only from the UI endpoints) ------------------------

    def list_items(self, status: Status) -> list[dict[str, Any]]:
        items = []
        for kind in ("tool", "skill"):
            base = self.root / status / f"{kind}s"
            names = sorted(p.stem for p in base.glob("*.py")) if kind == "tool" else \
                sorted(p.name for p in base.iterdir() if p.is_dir())
            for n in names:
                mp = self._meta(kind, n, status)  # type: ignore[arg-type]
                meta = json.loads(mp.read_text()) if mp.exists() else {}
                item = {"kind": kind, "name": n, "status": status, **meta}
                if (pk := item.get("package")) and pk.get("signer"):  # trust is looked up now, so trusting a publisher later shows up
                    t = self.signer.trusted().get(pk["signer"])
                    item["package"] = {**pk, "signer_trusted": bool(t), "signer_label": t["label"] if t else None}
                if status == "pending":
                    item["replaces_approved"] = self._content(kind, n, "approved").exists()  # type: ignore[arg-type]
                items.append(item)
        return items

    def get_source(self, kind: Kind, name: str, status: Status) -> str:
        p = self._content(kind, name, status)
        if not p.exists():
            raise KeyError(f"No {status} {kind} named {name}")
        return p.read_text()

    def _archive_approved(self, kind: Kind, name: str, stamp: str) -> Path:
        """Move an approved item (and its metadata) to retired/, keeping it for the record. Returns where the content went."""
        dest_dir = self.root / "retired" / f"{kind}s"
        if kind == "tool":
            dest = dest_dir / f"{name}.{stamp}.py"
            shutil.move(self._content("tool", name, "approved"), dest)
            if self._meta("tool", name, "approved").exists():
                shutil.move(self._meta("tool", name, "approved"), dest_dir / f"{name}.{stamp}.json")
        else:
            dest = dest_dir / f"{name}.{stamp}"
            shutil.move(self._content("skill", name, "approved").parent, dest)
        return dest

    def _restore_archived(self, kind: Kind, name: str, dest: Path) -> None:
        if kind == "tool":
            shutil.move(dest, self._content("tool", name, "approved"))
            meta = dest.with_suffix(".json")
            if meta.exists():
                shutil.move(meta, self._meta("tool", name, "approved"))
        else:
            shutil.move(dest, self._content("skill", name, "approved").parent)

    def approve(self, kind: Kind, name: str) -> dict[str, Any]:
        with self._lock:
            src = self._content(kind, name, "pending")
            if not src.exists():
                raise KeyError(f"No pending {kind} named {name}")
            if kind == "tool":  # re-lint at the moment of approval
                lint = lint_tool_source(src.read_text())
                if not lint.ok:
                    raise ProposalError("Lint failed: " + " | ".join(lint.errors))
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            old: Path | None = None
            replaced = self._exists_anywhere(kind, name, ("approved",)) is not None
            if replaced:  # the new version takes over; the old one is kept in retired/
                old = self._archive_approved(kind, name, stamp)
            try:
                if kind == "tool":
                    shutil.move(src, self._content("tool", name, "approved"))
                    shutil.move(self._meta("tool", name, "pending"), self._meta("tool", name, "approved"))
                else:
                    shutil.move(src.parent, self._content("skill", name, "approved").parent)
                m = self._manifest()
                m[kind][name] = _sha(self._content(kind, name, "approved"))
                self._manifest_path().write_text(json.dumps(m, indent=2))
            except Exception:
                if old is not None:  # never leave the user without the version that was working
                    self._restore_archived(kind, name, old)
                raise
            return {"approved": name, "kind": kind, "replaced": replaced}

    def retire(self, kind: Kind, name: str) -> dict[str, Any]:
        """Take an APPROVED tool/skill out of use. It is archived under retired/, never deleted."""
        with self._lock:
            if not self._content(kind, name, "approved").exists():
                raise KeyError(f"No approved {kind} named {name}")
            m = self._manifest()
            m[kind].pop(name, None)
            self._manifest_path().write_text(json.dumps(m, indent=2))  # unlisted first: from here on it is no longer loaded
            self._archive_approved(kind, name, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
            return {"retired": name, "kind": kind}

    def reject(self, kind: Kind, name: str) -> dict[str, Any]:
        with self._lock:
            src = self._content(kind, name, "pending")
            if not src.exists():
                raise KeyError(f"No pending {kind} named {name}")
            dest_dir = self.root / "rejected" / f"{kind}s"
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            if kind == "tool":
                shutil.move(src, dest_dir / f"{name}.{stamp}.py")
                self._meta("tool", name, "pending").unlink(missing_ok=True)
            else:
                shutil.move(src.parent, dest_dir / f"{name}.{stamp}")
            return {"rejected": name, "kind": kind}

    # --- sharing (packages are .janus ZIP files; importing only ever creates PENDING items) -------------

    def export_package(self, items: list[tuple[str, str]] | None = None,
                       info: dict[str, str] | None = None, sign: bool = False) -> tuple[bytes, dict[str, Any]]:
        from .package import export_package
        return export_package(self, items, info, sign)

    def import_package(self, raw: bytes) -> dict[str, Any]:
        from .package import import_package
        return import_package(self, raw)

    # --- loading (used by the agent each turn) -----------------------------

    def _verified(self, kind: Kind) -> tuple[list[tuple[str, Path]], list[str]]:
        ok, problems = [], []
        for name, sha in self._manifest()[kind].items():
            p = self._content(kind, name, "approved")
            if not p.exists():
                problems.append(f"{kind} '{name}' is in the manifest but its file is missing")
            elif _sha(p) != sha:
                problems.append(f"{kind} '{name}' was modified after approval and was NOT loaded")
            else:
                ok.append((name, p))
        return ok, problems

    def approved_tool_files(self) -> list[tuple[str, Path]]:
        return self._verified("tool")[0]

    def approved_skill_files(self) -> list[tuple[str, Path]]:
        return self._verified("skill")[0]

    def integrity_problems(self) -> list[str]:
        return self._verified("tool")[1] + self._verified("skill")[1]
