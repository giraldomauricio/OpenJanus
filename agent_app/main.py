from __future__ import annotations

import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import config, images
from .agent import Agent, Session
from .db import Database
from .db import file_import
from .db.csv_import import import_csv
from .extensions import ExtensionManager, PackageError, ProposalError
from .llm import PROVIDERS, make_provider, ollama_models, provider_status
from .introspection import audit_capabilities, audit_database
from .sdk import BUILTIN_TOOLS
from .sessions import SessionStore
from .skills import SkillStore
from .ui_config import load_ui_config
from . import tools as _tools  # noqa: F401

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
db = Database(config.DB_PATH)
skills_store: SkillStore
extensions = ExtensionManager(
    config.EXT_DIR,
    reserved_tool_names=lambda: set(BUILTIN_TOOLS),
    reserved_skill_names=lambda: skills_store.builtin_names(),
)
skills_store = SkillStore(config.BUILTIN_SKILLS_DIR, extensions, config.OPTIONAL_SKILLS_DIRS)
agent = Agent(db, extensions, skills_store)
sessions: dict[str, Session] = {}
store = SessionStore(config.SESSIONS_DIR)
agent.usage = store.tool_usage

# Every state-changing request must carry this token. It is only embedded in the page we serve, so another
# website in the same browser (or an approved tool, which cannot reach localhost) cannot approve anything.
TOKEN = secrets.token_urlsafe(32)

app = FastAPI(title="OpenJanus")
INDEX = (Path(__file__).parent / "static" / "index.html").read_text()


def require_token(x_agent_token: str = Header(default="")) -> None:
    if not secrets.compare_digest(x_agent_token, TOKEN):
        raise HTTPException(403, "Missing or invalid token")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # no-store: a cached or restored page would carry the token of a previous server process
    return HTMLResponse(INDEX.replace("__TOKEN__", TOKEN), headers={"Cache-Control": "no-store"})


@app.get("/logo.png", include_in_schema=False)
def logo() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "logo.png", media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})


def app_version() -> str:
    """Contents of VERSION.TXT (read each time, so the version you bump with a commit shows without a restart); "" if absent/odd."""
    try:
        v = (config.ROOT / "VERSION.TXT").read_text().strip()
    except OSError:
        return ""
    return v if re.fullmatch(r"[0-9A-Za-z._+-]{1,40}", v) else ""


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return {"providers": provider_status(), "ollama_models": ollama_models(), "version": app_version(), "features": {"diagrams": config.DIAGRAMS_ENABLED, "images": config.IMAGES_ENABLED}}


@app.get("/api/ui-config")
def get_ui_config() -> dict[str, Any]:
    return load_ui_config()


@app.post("/api/sessions", dependencies=[Depends(require_token)])
def new_session() -> dict[str, str]:
    s = Session()
    sessions[s.id] = s
    return {"id": s.id}


def get_session(sid: str) -> Session | None:
    s = sessions.get(sid)
    if s is None and (s := store.load(sid)) is not None:
        sessions[sid] = s
    return s


@app.get("/api/sessions")
def list_sessions() -> list[dict[str, Any]]:
    return store.list()


@app.get("/api/sessions/{sid}")
def read_session(sid: str) -> dict[str, Any]:
    s = get_session(sid)
    if s is None:
        raise HTTPException(404, "Unknown session")
    return {"id": s.id, "title": s.title, "provider": s.provider, "model": s.model, "events": s.events}


@app.delete("/api/sessions/{sid}", dependencies=[Depends(require_token)])
def delete_session(sid: str) -> dict[str, bool]:
    s = sessions.get(sid)
    if s is not None and s.lock.locked():
        raise HTTPException(409, "This session is running a request")
    sessions.pop(sid, None)
    return {"deleted": store.delete(sid)}


class RenameRequest(BaseModel):
    title: str


@app.patch("/api/sessions/{sid}", dependencies=[Depends(require_token)])
def rename_session(sid: str, req: RenameRequest) -> dict[str, str]:
    title = " ".join(req.title.split())[:80]  # one line, bounded
    if not title:
        raise HTTPException(400, "Title cannot be empty")
    if not store.rename(sid, title):
        raise HTTPException(404, "Unknown (or not yet saved) session")
    if (s := sessions.get(sid)) is not None:
        s.title = title  # keep the in-memory copy in step, or the next save would restore the old title
    return {"title": title}


@app.post("/api/sessions/{sid}/stop", dependencies=[Depends(require_token)])
def stop_session(sid: str) -> dict[str, bool]:
    s = sessions.get(sid)
    if s is None:
        raise HTTPException(404, "Unknown session")
    running = s.lock.locked()
    if running:
        s.cancel.set()
    return {"stopping": running}


class ChatRequest(BaseModel):
    session_id: str
    message: str
    provider: Literal["claude", "gemini", "ollama"]
    model: str | None = None
    images: list[str] = Field(default_factory=list, max_length=4, description="ids from POST /api/images")


@app.post("/api/chat", dependencies=[Depends(require_token)])
def chat(req: ChatRequest) -> StreamingResponse:
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(404, "Unknown session")
    if req.images and not config.IMAGES_ENABLED:
        raise HTTPException(400, "Image attachments are disabled")
    if any(images.load(i) is None for i in req.images):
        raise HTTPException(400, "An attached image is missing: upload it again")
    if not session.lock.acquire(blocking=False):
        raise HTTPException(409, "This session is already running a request")

    def sse(ev: dict[str, Any]) -> str:
        return f"data: {json.dumps(ev, default=str)}\n\n"

    def stream():
        try:
            try:
                provider = make_provider(req.provider, req.model or None)
            except Exception as e:
                yield sse({"type": "error", "text": f"Could not start {req.provider}: {e}"})
                return
            if not session.title:
                session.title = " ".join(req.message.split())[:60]
            provider.cancel = session.cancel
            session.provider, session.model = req.provider, provider.model
            session.events.append({"type": "user", "text": req.message, "provider": req.provider, "model": provider.model,
                                   **({"images": req.images} if req.images else {})})
            try:
                for ev in agent.run(session, req.message, provider, req.images):
                    if ev["type"] not in ("done", "text_delta"):  # deltas are transient; the final "text" event is what is replayed
                        session.events.append(ev)
                    yield sse(ev)
            finally:  # also runs if the browser disconnects mid-stream
                store.save(session)
        finally:
            session.lock.release()

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/extensions")
def list_extensions() -> dict[str, Any]:
    approved = extensions.list_items("approved")
    users: dict[str, list[str]] = {}
    for sk in skills_store.all().values():
        for t in sk.tools:
            users.setdefault(t, []).append(sk.name)
    for it in approved:  # lets the UI warn before retiring a tool that skills depend on
        if it["kind"] == "tool":
            it["used_by_skills"] = users.get(it["name"], [])
    return {"pending": extensions.list_items("pending"), "approved": approved, "problems": extensions.integrity_problems()}


@app.get("/api/extensions/export")
def export_extensions(items: str = "") -> Response:
    """Download approved tools/skills as a .janus package. `items` = "tool:name,skill:name" (empty = everything approved)."""
    try:
        wanted = [tuple(x.split(":", 1)) for x in items.split(",") if x] or None
        data, included = extensions.export_package(wanted)  # type: ignore[arg-type]
    except (PackageError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    fname = (included[0]["name"] if len(included) == 1 else "openjanus-extensions") + ".janus"
    return Response(data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.post("/api/extensions/import", dependencies=[Depends(require_token)])
async def import_extensions(request: Request) -> dict:
    """Raw package bytes in the body. Everything lands in PENDING; nothing is approved here."""
    raw = await request.body()
    try:
        return await run_in_threadpool(extensions.import_package, raw)
    except PackageError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/extensions/{kind}/{name}/source")
def extension_source(kind: Literal["tool", "skill"], name: str, status: Literal["pending", "approved"] = "pending") -> dict:
    try:
        return {"source": extensions.get_source(kind, name, status)}
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/extensions/{kind}/{name}/{action}", dependencies=[Depends(require_token)])
def review(kind: Literal["tool", "skill"], name: str, action: Literal["approve", "reject", "retire"]) -> dict:
    try:
        return {"approve": extensions.approve, "reject": extensions.reject, "retire": extensions.retire}[action](kind, name)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ProposalError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    return agent.catalog()


@app.get("/api/skills/{name}")
def skill_body(name: str) -> dict[str, Any]:
    sk = skills_store.all().get(name)
    if sk is None:
        raise HTTPException(404, "Unknown (or not yet approved) skill")
    return {"name": sk.name, "origin": sk.origin, "tools": sk.tools, "instructions": sk.body}


@app.get("/api/audit/capabilities")
def audit_caps() -> dict[str, Any]:
    return audit_capabilities(agent.catalog())


@app.get("/api/audit/database")
def audit_db() -> dict[str, Any]:
    return audit_database(db)


@app.get("/api/db")
def db_overview() -> dict[str, Any]:
    return {"tables": db.list_tables(), "models": [s.model_dump() for s in db.all_specs()]}


@app.get("/api/db/{table}/csv")
def db_table_csv(table: str) -> StreamingResponse:
    if table not in db.table_names():
        raise HTTPException(404, f"No such table: {table}")
    return StreamingResponse(db.export_csv(table), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{table}.csv"'})


@app.get("/api/db/{table}")
def db_table(table: str) -> dict[str, Any]:
    try:
        return {"describe": db.describe(table), "preview": db.query(f'SELECT * FROM "{table}"', limit=50)}
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/import/csv", dependencies=[Depends(require_token)])
async def import_csv_endpoint(request: Request, table: str) -> dict[str, Any]:
    """Raw CSV text in the body. Creates a typed table; the agent adopts it as a model afterwards."""
    raw = await request.body()
    if len(raw) > 20_000_000:
        raise HTTPException(413, "File too large (20 MB max)")
    try:
        return await run_in_threadpool(import_csv, db, raw, table)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


if config.DIAGRAMS_ENABLED:  # optional feature pack: draw.io files (tools in agent_app/tools/diagrams.py)
    from .diagrams.drawio import read_drawio
    from .diagrams.store import DiagramStore, clean_name

    diagram_store = DiagramStore(config.DIAGRAMS_DIR)

    @app.get("/api/diagrams")
    def list_diagram_files() -> list[dict[str, Any]]:
        return diagram_store.list()

    @app.get("/api/diagrams/{name}")
    def download_diagram(name: str) -> Response:
        try:
            data = diagram_store.read(name)
        except ValueError as e:
            raise HTTPException(404, str(e)) from e
        # attachment + a non-HTML type: the browser downloads it instead of rendering it
        return Response(data, media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{clean_name(name)}"', "X-Content-Type-Options": "nosniff"})

    @app.post("/api/diagrams", dependencies=[Depends(require_token)])
    async def upload_diagram(request: Request, name: str) -> dict[str, Any]:
        """Raw .drawio text in the body. Only stored if it parses as a draw.io file; the agent reads it with read_diagram."""
        raw = await request.body()
        try:
            summary = await run_in_threadpool(read_drawio, raw)
            saved = diagram_store.save(name, raw)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"file": saved, "pages": summary["pages"], "nodes": len(summary["nodes"]), "edges": len(summary["edges"])}


if config.IMAGES_ENABLED:  # chat image attachments (e.g. an architecture screenshot)
    from fastapi.responses import Response as _Response

    @app.post("/api/images", dependencies=[Depends(require_token)])
    async def upload_image(request: Request) -> dict[str, str]:
        raw = await request.body()
        try:
            return images.save(raw)
        except ValueError as e:
            raise HTTPException(413 if "large" in str(e) else 400, str(e)) from e

    @app.get("/api/images/{iid}")
    def get_image(iid: str) -> _Response:
        got = images.load(iid)
        if got is None:
            raise HTTPException(404, "Unknown image")
        return _Response(got[1], media_type=got[0], headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=86400",
                                                              "Content-Security-Policy": "default-src 'none'"})


# ---- big file import (CSV / JSON): read from a path on this machine, in a background job with progress and Cancel ----------------------
@dataclass
class ImportJob:
    id: str
    path: str
    table: str
    mode: str
    phase: str = "scan"          # scan | load | done | error | cancelled
    rows: int = 0
    bytes_read: int = 0
    total: int = 0
    error: str = ""
    result: dict[str, Any] | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    started: float = field(default_factory=time.time)


_import_jobs: dict[str, ImportJob] = {}
_import_lock = threading.Lock()


class FilePathRequest(BaseModel):
    path: str = Field(max_length=2000)
    records_key: str | None = Field(None, max_length=200, description="Only for an object that holds several arrays: which one has the records")


class FileImportRequest(FilePathRequest):
    table: str = Field(min_length=1, max_length=80)
    mode: Literal["new", "append", "replace"] = "new"


@app.post("/api/import/file/preview", dependencies=[Depends(require_token)])
def preview_file(req: FilePathRequest) -> dict[str, Any]:
    """Look at the first records only: the columns and types the import would create."""
    try:
        path = file_import.resolve_path(req.path)
        info = file_import.scan(path, req.records_key, limit=2000)
    except file_import.ImportFileError as e:
        raise HTTPException(400, str(e)) from e
    return {"path": str(path), "size_bytes": path.stat().st_size, "records_scanned": info["rows"], "complete": info["rows"] < 2000,
            "columns": [{"name": n, "type": file_import.DECLARED[t]} for n, t in zip(info["names"], info["types"])]}


def _run_import(job: ImportJob, path: Any, records_key: str | None) -> None:
    def progress(phase: str, rows: int, read: int, total: int) -> None:
        job.phase, job.rows, job.bytes_read, job.total = phase, rows, read, total

    try:
        job.result = file_import.load(db, path, job.table, job.mode, records_key, progress, job.cancel.is_set)
        job.phase = "done"
        job.rows = job.result["rows"]
    except file_import.Cancelled:
        job.phase = "cancelled"
    except file_import.ImportFileError as e:
        job.phase, job.error = "error", str(e)
    except Exception as e:  # never leave a job stuck in "running"
        job.phase, job.error = "error", f"{type(e).__name__}: {e}"


@app.post("/api/import/file", dependencies=[Depends(require_token)])
def start_file_import(req: FileImportRequest) -> dict[str, str]:
    try:
        path = file_import.resolve_path(req.path)
    except file_import.ImportFileError as e:
        raise HTTPException(400, str(e)) from e
    with _import_lock:
        if any(j.phase in ("scan", "load") for j in _import_jobs.values()):
            raise HTTPException(409, "Another JSON import is still running")
        job = ImportJob(secrets.token_hex(8), str(path), req.table, req.mode, total=path.stat().st_size)
        for old in sorted(_import_jobs.values(), key=lambda j: j.started)[:-9]:  # keep the last few for the UI to read
            _import_jobs.pop(old.id, None)
        _import_jobs[job.id] = job
    threading.Thread(target=_run_import, args=(job, path, req.records_key), daemon=True).start()
    return {"job_id": job.id}


def _job(jid: str) -> ImportJob:
    j = _import_jobs.get(jid)
    if j is None:
        raise HTTPException(404, "Unknown import")
    return j


@app.get("/api/import/file/{jid}")
def file_import_status(jid: str) -> dict[str, Any]:
    j = _job(jid)
    return {"phase": j.phase, "rows": j.rows, "bytes_read": j.bytes_read, "total": j.total, "error": j.error, "result": j.result,
            "table": j.table, "mode": j.mode}


@app.post("/api/import/file/{jid}/cancel", dependencies=[Depends(require_token)])
def cancel_file_import(jid: str) -> dict[str, bool]:
    j = _job(jid)
    running = j.phase in ("scan", "load")
    if running:
        j.cancel.set()
    return {"cancelling": running}
