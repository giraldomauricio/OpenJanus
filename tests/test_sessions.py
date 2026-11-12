import json

from agent_app import main
from agent_app.agent import Session
from agent_app.sessions import SessionStore


def saved(store, title, tools=()):
    s = Session(title=title)
    s.events = [{"type": "user", "text": title}] + [{"type": "tool_call", "name": t, "id": "1", "args": {}} for t in tools]
    s.messages = [{"role": "user", "content": title}]
    store.save(s)
    return s


def test_list_and_usage_use_the_cache_and_notice_changes(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    a = saved(store, "first", ["query", "query"]); saved(store, "second", ["query", "list_tables"])
    assert store.tool_usage() == {"query": 3, "list_tables": 1}
    assert {x["title"] for x in store.list()} == {"first", "second"}

    reads = []
    orig = type(tmp_path).read_text
    monkeypatch.setattr(type(tmp_path), "read_text", lambda self, *a, **k: (reads.append(self.name), orig(self, *a, **k))[1])
    store.list(); store.tool_usage()
    assert reads == []  # nothing changed: no chat file was opened again

    assert store.rename(a.id, "renamed")
    reads.clear()  # (rename itself reads the file once)
    assert "renamed" in {x["title"] for x in store.list()} and reads == [a.id + ".json"]  # only the changed chat is re-read
    store.delete(a.id)
    assert store.tool_usage() == {"query": 1, "list_tables": 1}


def test_corrupt_chat_is_skipped(tmp_path):
    store = SessionStore(tmp_path)
    saved(store, "ok")
    (tmp_path / ("0" * 32 + ".json")).write_text("{not json")
    assert [x["title"] for x in store.list()] == ["ok"]


def test_trim_keeps_running_and_unsaved_sessions(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    monkeypatch.setattr(main, "store", store); monkeypatch.setattr(main, "sessions", {}); monkeypatch.setattr(main, "MAX_LIVE_SESSIONS", 3)
    old = [saved(store, f"c{i}") for i in range(4)]
    for i, s in enumerate(old):
        s.updated = i; main.sessions[s.id] = s
    running = old[0]; running.lock.acquire()
    fresh = Session(); main.sessions[fresh.id] = fresh  # never saved
    main.trim_sessions()
    assert running.id in main.sessions and fresh.id in main.sessions  # protected
    assert len(main.sessions) == 3
    assert main.get_session(old[1].id) is not None  # an evicted chat reloads from disk
