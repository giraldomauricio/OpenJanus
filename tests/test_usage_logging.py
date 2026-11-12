"""Token/cache accounting is the same for every provider, the Claude cache marker is sent, and the log never holds your data."""
import json
import logging
import socket
from types import SimpleNamespace

import httpx
import pytest

from agent_app import config, log as applog
from agent_app.agent import Agent, Session
from agent_app.db import Database
from agent_app.extensions import ExtensionManager
from agent_app.http_safe import SafeHttp
from agent_app.llm import ToolCall, Turn, Usage
from agent_app.llm.anthropic_provider import AnthropicProvider
from agent_app.llm.base import LLMProvider
from agent_app.llm.gemini_provider import GeminiProvider
from agent_app.sdk import BUILTIN_TOOLS
from agent_app.skills import SkillStore
from agent_app.tool_secrets import SecretResolver


# --- per-provider usage -------------------------------------------------------------------------------------------------

def test_usage_adds_and_keeps_unreported_as_none():
    assert Usage(10, 2, None, None) + Usage(5, 1, None, None) == Usage(15, 3, None, None)
    assert Usage(10, 2, 4, 0) + Usage(5, 1, None, None) == Usage(15, 3, 4, 0)


def test_claude_usage_folds_cache_into_one_prompt_figure():
    u = AnthropicProvider._usage(SimpleNamespace(input_tokens=50, output_tokens=7, cache_read_input_tokens=900, cache_creation_input_tokens=100))
    assert u == Usage(1050, 7, 900, 100)
    assert AnthropicProvider._usage(SimpleNamespace(input_tokens=5, output_tokens=1, cache_read_input_tokens=None,
                                                    cache_creation_input_tokens=None)) == Usage(5, 1, 0, 0)


def test_gemini_usage_reads_implicit_cache_counts():
    m = SimpleNamespace(prompt_token_count=2000, candidates_token_count=30, thoughts_token_count=20, cached_content_token_count=1500)
    assert GeminiProvider._usage(m) == Usage(2000, 50, 1500, None)


def test_ollama_usage_comes_from_the_done_line(monkeypatch):
    from agent_app.llm import ollama_provider as op
    body = b'{"message":{"content":"hi"}}\n{"message":{"content":""},"done":true,"prompt_eval_count":120,"eval_count":9}\n'
    real = httpx.Client
    monkeypatch.setattr(op.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)), **kw))
    turn = op.OllamaProvider("m").generate("sys", [{"role": "user", "content": "x"}], [])
    assert turn.text == "hi" and turn.usage == Usage(120, 9)


# --- the Claude cache marker ----------------------------------------------------------------------------------------------

class FakeStream:
    text_stream = iter(["ok"])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")],
                               usage=SimpleNamespace(input_tokens=3, output_tokens=1, cache_read_input_tokens=7, cache_creation_input_tokens=0))


@pytest.mark.parametrize("enabled", [True, False])
def test_claude_request_carries_the_cache_marker_only_when_enabled(monkeypatch, enabled):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(config, "PROMPT_CACHE", enabled)
    sent = {}
    p = AnthropicProvider("claude-sonnet-5-5")
    p.client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: (sent.update(kw), FakeStream())[1]), close=lambda: None)
    turn = p.generate("sys", [{"role": "user", "content": "x"}], [])
    assert ("cache_control" in sent) is enabled and (not enabled or sent["cache_control"] == {"type": "ephemeral"})
    assert turn.usage == Usage(10, 1, 7, 0)


# --- the agent sums a whole tool loop into one usage event ------------------------------------------------------------------

class Scripted(LLMProvider):
    name = "fake"

    def __init__(self, turns):
        super().__init__("fake-model")
        self.turns = list(turns)

    def generate(self, system, messages, tools):
        return self.turns.pop(0)


@pytest.fixture
def agent(tmp_path):
    skills = None
    ext = ExtensionManager(tmp_path / "ext", lambda: set(BUILTIN_TOOLS), lambda: skills.builtin_names())
    skills = SkillStore(config.BUILTIN_SKILLS_DIR, ext)
    return Agent(Database(tmp_path / "t.db"), ext, skills)


def test_one_usage_event_sums_the_loop_and_comes_before_done(agent):
    turns = [Turn(tool_calls=[ToolCall("1", "list_tables", {})], usage=Usage(1000, 20, 0, 900)),
             Turn(text="done!", usage=Usage(1100, 30, 900, 0))]
    ev = list(agent.run(Session(), "go", Scripted(turns)))
    assert [e["type"] for e in ev][-2:] == ["usage", "done"]
    u = ev[-2]
    assert (u["calls"], u["input"], u["output"], u["cache_read"], u["cache_write"]) == (2, 2100, 50, 900, 900)


def test_no_usage_event_when_the_provider_reports_nothing(agent):
    assert "usage" not in [e["type"] for e in agent.run(Session(), "go", Scripted([Turn(text="hi")]))]


# --- logging ----------------------------------------------------------------------------------------------------------------

@pytest.fixture
def logfile(tmp_path, monkeypatch):
    names = ("openjanus", "uvicorn.error", "uvicorn.access")
    saved = {n: list(logging.getLogger(n).handlers) for n in names}
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    yield tmp_path / "logs" / "openjanus.log"
    root = logging.getLogger("openjanus")
    for n in names:
        lg = logging.getLogger(n)
        for h in list(lg.handlers):
            if h not in saved[n]:
                lg.removeHandler(h)
                h.close()
    root.propagate = True
    if hasattr(root, "_configured"):
        del root._configured


def test_setup_writes_a_file_and_is_idempotent(logfile):
    applog.setup(level="INFO")
    applog.setup(level="INFO")
    logging.getLogger("openjanus.tools").info("hello")
    text = logfile.read_text()
    assert text.count("hello") == 1 and "INFO" in text and "tools" in text


def test_log_never_contains_messages_results_or_args_at_info(agent, logfile):
    applog.setup(level="INFO")
    turns = [Turn(tool_calls=[ToolCall("1", "query", {"sql": "select 'SECRET_ARG_AND_RESULT' as x"})], usage=Usage(10, 1)),
             Turn(text="ANSWER_TEXT_MARKER", usage=Usage(10, 1))]
    list(agent.run(Session(), "USER_TEXT_MARKER", Scripted(turns)))
    text = logfile.read_text()
    for forbidden in ("USER_TEXT_MARKER", "ANSWER_TEXT_MARKER", "SECRET_ARG_AND_RESULT"):
        assert forbidden not in text
    assert "query ok" in text and "message start" in text and "message done" in text and "step 1/" in text


def test_debug_level_adds_tool_arguments(agent, logfile):
    applog.setup(level="DEBUG")
    list(agent.run(Session(), "x", Scripted([Turn(tool_calls=[ToolCall("1", "query", {"sql": "select 'ARG_AT_DEBUG'"})]), Turn(text="ok")])))
    assert "ARG_AT_DEBUG" in logfile.read_text()


def test_failed_tool_and_unknown_tool_are_logged_as_warnings(agent, caplog):
    with caplog.at_level(logging.INFO, logger="openjanus"):
        list(agent.run(Session(), "x", Scripted([Turn(tool_calls=[ToolCall("1", "nope_tool", {}), ToolCall("2", "describe_table", {"table": "missing"})]),
                                                 Turn(text="ok")])))
    msgs = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert any(l == "WARNING" and "unknown tool nope_tool" in m for l, m in msgs)
    assert any(l == "WARNING" and "describe_table FAILED" in m for l, m in msgs)


def test_outbound_http_logs_host_and_path_but_not_the_query(monkeypatch, caplog):
    monkeypatch.setattr(socket, "getaddrinfo", lambda h, p, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", p))])
    http = SafeHttp()
    http.transport = httpx.MockTransport(lambda r: httpx.Response(200, text="ok"))
    with caplog.at_level(logging.INFO, logger="openjanus"):
        http.get("https://api.example.com/v1/items", params={"api_key": "QUERY_SECRET"})
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "api.example.com/v1/items" in text and "200" in text and "QUERY_SECRET" not in text


def test_blocked_address_is_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(socket, "getaddrinfo", lambda h, p, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", p))])
    http = SafeHttp()
    http.transport = httpx.MockTransport(lambda r: httpx.Response(200))
    with caplog.at_level(logging.INFO, logger="openjanus"), pytest.raises(Exception):
        http.get("https://localhost.example.com/")
    assert any(r.levelname == "WARNING" and "BLOCKED" in r.getMessage() for r in caplog.records)


def test_secret_use_logs_the_name_and_host_never_the_value(monkeypatch, caplog):
    monkeypatch.setenv("DEMO_TOKEN", "VALUE_THAT_MUST_NOT_LEAK")
    with caplog.at_level(logging.INFO, logger="openjanus"):
        SecretResolver({"DEMO_TOKEN": ["api.example.com"]}).headers_for("https://api.example.com/x", {"Authorization": "Bearer {secret:DEMO_TOKEN}"})
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "DEMO_TOKEN" in text and "api.example.com" in text and "VALUE_THAT_MUST_NOT_LEAK" not in text


def test_bad_token_is_logged_as_a_security_warning(caplog):
    from fastapi.testclient import TestClient
    from agent_app import main
    with caplog.at_level(logging.INFO, logger="openjanus"):
        assert TestClient(main.app).post("/api/sessions").status_code == 403
    assert any(r.name == "openjanus.security" and "403" in r.getMessage() for r in caplog.records)


def test_server_startup_logs_a_summary_to_the_file(logfile):
    from fastapi.testclient import TestClient
    from agent_app import main
    with TestClient(main.app):
        pass
    text = logfile.read_text()
    assert "starting" in text and "providers configured" in text and "prompt_cache=" in text and "tools:" in text
