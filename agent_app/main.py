from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from . import config
from .agent import Agent, Session
from .db import Database
from .db.csv_import import import_csv
from .extensions import ExtensionManager, ProposalError
from .llm import PROVIDERS, make_provider, ollama_models, provider_status
from .introspection import audit_capabilities, audit_database
from .sdk import BUILTIN_TOOLS
from .sessions import SessionStore
from .skills import SkillStore
from . import tools as _tools  # noqa: F401

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
db = Database(config.DB_PATH)
skills_store: SkillStore
extensions = ExtensionManager(
    config.EXT_DIR,
    reserved_tool_names=lambda: set(BUILTIN_TOOLS),
    reserved_skill_names=lambda: skills_store.builtin_names(),
)
skills_store = SkillStore(config.BUILTIN_SKILLS_DIR, extensions)
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
def index() -> str:
    return INDEX.replace("__TOKEN__", TOKEN)


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return {"providers": provider_status(), "ollama_models": ollama_models()}


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
    return {"id": s.id, "title": s.title, "events": s.events}


@app.delete("/api/sessions/{sid}", dependencies=[Depends(require_token)])
def delete_session(sid: str) -> dict[str, bool]:
    s = sessions.get(sid)
    if s is not None and s.lock.locked():
        raise HTTPException(409, "This session is running a request")
    sessions.pop(sid, None)
    return {"deleted": store.delete(sid)}


class ChatRequest(BaseModel):
    session_id: str
    message: str
    provider: Literal["claude", "gemini", "ollama"]
    model: str | None = None


@app.post("/api/chat", dependencies=[Depends(require_token)])
def chat(req: ChatRequest) -> StreamingResponse:
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(404, "Unknown session")
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
            session.events.append({"type": "user", "text": req.message})
            try:
                for ev in agent.run(session, req.message, provider):
                    if ev["type"] != "done":
                        session.events.append(ev)
                    yield sse(ev)
            finally:  # also runs if the browser disconnects mid-stream
                store.save(session)
        finally:
            session.lock.release()

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/extensions")
def list_extensions() -> dict[str, Any]:
    return {"pending": extensions.list_items("pending"), "approved": extensions.list_items("approved"),
            "problems": extensions.integrity_problems()}


@app.get("/api/extensions/{kind}/{name}/source")
def extension_source(kind: Literal["tool", "skill"], name: str, status: Literal["pending", "approved"] = "pending") -> dict:
    try:
        return {"source": extensions.get_source(kind, name, status)}
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/extensions/{kind}/{name}/{action}", dependencies=[Depends(require_token)])
def review(kind: Literal["tool", "skill"], name: str, action: Literal["approve", "reject"]) -> dict:
    try:
        return extensions.approve(kind, name) if action == "approve" else extensions.reject(kind, name)
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
