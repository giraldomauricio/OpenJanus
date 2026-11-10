"""Share approved tools/skills as a single ZIP ("package", extension .janus) and import one from someone else.

    manifest.json                    {"format": "openjanus-package", "version": 1, "items": [{kind, name, description, sha256}]}
    tools/<name>.py                  the approved tool source, exactly as approved
    skills/<name>/SKILL.md           the approved skill

Importing NEVER approves anything: every item is re-linted on this machine and lands in pending/, where the user reads it
and approves it like any tool the agent proposed. The package is untrusted input: the archive is size-limited, entry names
are matched against a strict allow-list (no paths, no zip-slip), and nothing in it is trusted (capabilities and metadata are
recomputed here; the sha256 only detects corruption, it is not a signature).
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .lint import SKILL_NAME, TOOL_NAME, lint_skill, lint_tool_source

if TYPE_CHECKING:
    from .manager import ExtensionManager

FORMAT = "openjanus-package"
VERSION = 1
EXTENSION = ".janus"
MAX_PACKAGE_BYTES = 5_000_000      # compressed upload
MAX_UNPACKED_BYTES = 5_000_000     # sum of declared sizes: guards against zip bombs
MAX_ITEMS = 100

_ENTRY = re.compile(r"^(manifest\.json|tools/[a-z][a-z0-9_]{2,40}\.py|skills/[a-z][a-z0-9-]{2,40}/SKILL\.md)$")


class PackageError(ValueError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _skill_tools(text: str) -> list[str]:
    from ..skills import parse_skill
    return parse_skill(text, "generated").tools


def export_package(mgr: "ExtensionManager", items: list[tuple[str, str]] | None = None) -> tuple[bytes, list[dict[str, str]]]:
    """Zip approved items (all of them when `items` is None). A skill brings the approved generated tools it lists, so the
    recipient gets something that works. Returns (zip bytes, what was included)."""
    with mgr._lock:
        approved = {(i["kind"], i["name"]): i for i in mgr.list_items("approved")}
        wanted = list(approved) if items is None else list(dict.fromkeys(items))
        if not wanted:
            raise PackageError("Nothing approved to export yet")
        for k in wanted:
            if k not in approved:
                raise PackageError(f"No approved {k[0]} named {k[1]}")
        for kind, name in list(wanted):  # pull in the generated tools a skill depends on
            if kind == "skill":
                for t in _skill_tools(mgr.get_source("skill", name, "approved")):
                    if ("tool", t) in approved and ("tool", t) not in wanted:
                        wanted.append(("tool", t))
        manifest_items, files = [], {}
        for kind, name in wanted:
            text = mgr.get_source(kind, name, "approved")  # type: ignore[arg-type]
            data = text.encode()
            path = f"tools/{name}.py" if kind == "tool" else f"skills/{name}/SKILL.md"
            files[path] = data
            manifest_items.append({"kind": kind, "name": name, "description": approved[(kind, name)].get("description", ""),
                                   "sha256": _sha(data)})
    manifest = {"format": FORMAT, "version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "items": manifest_items}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        for path, data in files.items():
            z.writestr(path, data)
    return buf.getvalue(), [{"kind": i["kind"], "name": i["name"]} for i in manifest_items]


def _read_archive(raw: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    if len(raw) > MAX_PACKAGE_BYTES:
        raise PackageError("Package too large (5 MB max)")
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as e:
        raise PackageError("Not a valid package (not a ZIP file)") from e
    infos = [i for i in z.infolist() if not i.is_dir()]
    if len(infos) > MAX_ITEMS * 2 + 1:
        raise PackageError("Package has too many files")
    if sum(i.file_size for i in infos) > MAX_UNPACKED_BYTES:
        raise PackageError("Package is too large once unpacked")
    bad = [i.filename for i in infos if not _ENTRY.match(i.filename)]
    if bad:
        raise PackageError(f"Unexpected entries in package: {bad[:3]}")
    files = {i.filename: z.read(i) for i in infos}
    if "manifest.json" not in files:
        raise PackageError("Package has no manifest.json")
    try:
        manifest = json.loads(files["manifest.json"])
    except ValueError as e:
        raise PackageError("manifest.json is not valid JSON") from e
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise PackageError("This is not an OpenJanus package")
    if manifest.get("version") != VERSION:
        raise PackageError(f"Unsupported package version {manifest.get('version')} (this OpenJanus reads version {VERSION})")
    if not isinstance(manifest.get("items"), list) or not manifest["items"]:
        raise PackageError("Package lists no items")
    return manifest, files


def _tool_name_in_source(source: str) -> str | None:
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NAME" for t in n.targets) \
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            return n.value.value
    return None


def import_package(mgr: "ExtensionManager", raw: bytes) -> dict[str, Any]:
    """Validate a package and put each item in pending/. Items that fail are skipped with a reason; the rest still import."""
    manifest, files = _read_archive(raw)
    stamp_note = f"Imported from a package created {manifest.get('created_at', 'at an unknown time')}"
    imported, skipped = [], []
    with mgr._lock:
        for item in manifest["items"][:MAX_ITEMS]:
            kind, name = (item.get("kind"), item.get("name")) if isinstance(item, dict) else (None, None)
            label = {"kind": kind, "name": name}
            try:
                if kind not in ("tool", "skill") or not isinstance(name, str):
                    raise PackageError("Malformed manifest entry")
                path = f"tools/{name}.py" if kind == "tool" else f"skills/{name}/SKILL.md"
                if path not in files:
                    raise PackageError("Listed in the manifest but missing from the package")
                data = files[path]
                if item.get("sha256") != _sha(data):
                    raise PackageError("Content does not match the manifest (corrupted or edited package)")
                try:
                    text = data.decode()
                except UnicodeDecodeError as e:
                    raise PackageError("File is not UTF-8 text") from e
                res = _import_one(mgr, kind, name, text, stamp_note)
                imported.append({**label, **res})
            except (PackageError, ValueError) as e:
                skipped.append({**label, "reason": str(e)})
    return {"imported": imported, "skipped": skipped}


def _import_one(mgr: "ExtensionManager", kind: str, name: str, text: str, note: str) -> dict[str, Any]:
    from ..skills import parse_skill
    from .manager import ProposalError
    try:
        if kind == "tool":
            if not TOOL_NAME.match(name):
                raise PackageError("Invalid tool name")
            if name in mgr._reserved_tools():
                raise PackageError("Conflicts with a built-in tool name")
            if _tool_name_in_source(text) != name:
                raise PackageError("NAME inside the tool does not match its file name")
            lint = lint_tool_source(text)
            if not lint.ok:
                raise PackageError("Rejected by lint: " + " | ".join(lint.errors))
            desc = next((i for i in re.findall(r"^DESCRIPTION = (.+)$", text, re.M)), "")
            try:
                desc = str(ast.literal_eval(desc))
            except (ValueError, SyntaxError):
                desc = ""
            dest = mgr._content("tool", name, "pending")
            dest.write_text(text)
            mgr._write_meta("tool", name, "pending", desc, note, lint.capabilities)
        else:
            if not SKILL_NAME.match(name):
                raise PackageError("Invalid skill name")
            if name in mgr._reserved_skills():
                raise PackageError("A built-in skill with this name already exists")
            try:
                sk = parse_skill(text, "generated")
            except Exception as e:
                raise PackageError(f"Unreadable SKILL.md: {e}") from e
            if sk.name != name:
                raise PackageError("Skill name inside SKILL.md does not match its folder")
            lint = lint_skill(sk.name, sk.description, sk.body)
            if not lint.ok:
                raise PackageError("Rejected by lint: " + " | ".join(lint.errors))
            dest = mgr._content("skill", name, "pending")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
            mgr._write_meta("skill", name, "pending", sk.description, note, [])
    except ProposalError as e:
        raise PackageError(str(e)) from e
    return {"replaces_approved": mgr._content(kind, name, "approved").exists()}  # type: ignore[arg-type]
