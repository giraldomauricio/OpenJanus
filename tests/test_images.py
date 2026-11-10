import base64
import struct
import zlib

import pytest

from agent_app import config, images
from agent_app.llm.anthropic_provider import AnthropicProvider
from agent_app.llm.base import LLMProvider, Turn
from agent_app.llm.ollama_provider import OllamaProvider


def png(w=2, h=2, extra=b""):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"") + extra


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    return tmp_path


def test_sniff_and_limits(uploads):
    assert images.sniff(png()) == "image/png" and images.sniff(b"\xff\xd8\xff\xe0abc") == "image/jpeg"
    assert images.sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp" and images.sniff(b"GIF89a....") == "image/gif"
    for bad in (b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>", b"<html>", b"MZ\x90\x00"):
        with pytest.raises(ValueError):
            images.save(bad)
    with pytest.raises(ValueError):
        images.save(b"")
    with pytest.raises(ValueError, match="too large"):
        images.save(png(extra=b"0" * images.MAX_BYTES))


def test_save_load_resolve(uploads):
    a = images.save(png())
    assert images.save(png()) == a  # content-addressed
    assert images.load(a["id"])[0] == "image/png" and images.load("../../etc/passwd") is None and images.load("0" * 32) is None
    found, missing = images.resolve({"images": [a["id"], "f" * 32]})
    assert len(found) == 1 and missing == 1


def test_providers_send_the_image(uploads):
    iid = images.save(png())["id"]
    msgs = [{"role": "user", "content": "what is this", "images": [iid]}, {"role": "user", "content": "plain"}]
    out = AnthropicProvider._convert(msgs)
    assert out[0]["content"][0]["type"] == "image" and out[0]["content"][0]["source"]["media_type"] == "image/png"
    assert base64.b64decode(out[0]["content"][0]["source"]["data"]).startswith(b"\x89PNG") and out[0]["content"][1]["text"] == "what is this"
    assert out[1] == {"role": "user", "content": "plain"}
    o = OllamaProvider._convert("sys", msgs)
    assert len(o[1]["images"]) == 1 and "images" not in o[2]
    # a deleted file is reported to the model instead of breaking the chat
    gone = AnthropicProvider._convert([{"role": "user", "content": "x", "images": ["e" * 32]}])
    assert "no longer available" in gone[0]["content"][-1]["text"]


def test_ollama_refuses_images_for_text_only_models(monkeypatch):
    class R:
        def __init__(self, caps): self.caps = caps
        def json(self): return {"capabilities": self.caps}
    p = OllamaProvider("qwen3:8b")
    msgs = [{"role": "user", "content": "x", "images": ["a" * 32]}]
    monkeypatch.setattr("httpx.post", lambda *a, **k: R(["completion", "tools"]))
    with pytest.raises(RuntimeError, match="cannot read images"):
        p._require_vision(msgs)
    monkeypatch.setattr("httpx.post", lambda *a, **k: R(["completion", "vision"]))
    p._require_vision(msgs)
    monkeypatch.setattr("httpx.post", lambda *a, **k: R(["completion"]))
    p._require_vision([{"role": "user", "content": "text only"}])  # no images: never asks


class Echo(LLMProvider):
    name = "echo"
    def __init__(self): super().__init__("echo"); self.seen = None
    def generate(self, system, messages, tools):
        self.seen = [dict(m) for m in messages]
        return Turn(text="ok")


def test_chat_endpoint_with_images(uploads, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from agent_app import main
    fake = Echo()
    monkeypatch.setattr(main, "make_provider", lambda *a, **k: fake)
    monkeypatch.setattr(main.store, "dir", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()
    c, H = TestClient(main.app), {"X-Agent-Token": main.TOKEN}
    assert c.post("/api/images", content=png()).status_code == 403
    assert c.post("/api/images", content=b"<svg/>", headers=H).status_code == 400
    up = c.post("/api/images", content=png(), headers=H).json()
    got = c.get(f"/api/images/{up['id']}")
    assert got.status_code == 200 and got.headers["content-type"] == "image/png" and got.headers["x-content-type-options"] == "nosniff"
    assert c.get("/api/images/zzz").status_code == 404
    sid = c.post("/api/sessions", headers=H).json()["id"]
    bad = c.post("/api/chat", headers=H, json={"session_id": sid, "message": "hi", "provider": "ollama", "images": ["a" * 32]})
    assert bad.status_code == 400
    r = c.post("/api/chat", headers=H, json={"session_id": sid, "message": "hi", "provider": "ollama", "images": [up["id"]]})
    assert r.status_code == 200 and fake.seen[0]["images"] == [up["id"]]
    ev = c.get(f"/api/sessions/{sid}").json()["events"][0]
    assert ev["type"] == "user" and ev["images"] == [up["id"]]  # replayed with the thumbnail when the chat is reopened
    assert c.get("/api/config").json()["features"]["images"] is True


def test_draft_from_database(tmp_path):
    from agent_app import main
    from agent_app.db import Database, ModelSpec
    db = Database(tmp_path / "t.db")
    ctx = main.agent.agent_ctx.__class__(db, main.agent.agent_ctx.http, main.extensions, main.skills_store)
    tool = main.BUILTIN_TOOLS["draft_diagram_from_database"]
    db.define_model(ModelSpec.model_validate({"name": "customers", "fields": [{"name": "name", "type": "string"}]}))
    r = tool.call({}, ctx)
    assert r["enough_data"] is False and "Fewer than 2" in r["reason"]
    db.define_model(ModelSpec.model_validate({"name": "orders", "fields": [{"name": "customer_id", "type": "integer", "references": "customers.id"}]}))
    r = tool.call({"infer": False}, ctx)
    assert r["enough_data"] is True and r["relationships"][0]["basis"] == "declared reference"
    assert {n["id"] for n in r["draft"]["nodes"]} == {"customers", "orders"} and r["draft"]["edges"][0]["source"] == "orders"
    from agent_app.diagrams.spec import DiagramSpec
    DiagramSpec.model_validate(r["draft"])  # the draft is directly usable by create_diagram
    db.define_model(ModelSpec.model_validate({"name": "tags", "fields": [{"name": "label", "type": "string"}]}))
    assert tool.call({"tables": ["tags", "customers"], "infer": False}, ctx)["enough_data"] is False  # no relationship between them
    with pytest.raises(ValueError):
        tool.call({"tables": ["nope"]}, ctx)
