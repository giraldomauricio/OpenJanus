import base64
import socket

import httpx
import pytest

from agent_app.extensions.lint import lint_tool_source
from agent_app.http_safe import SafeHttp
from agent_app.tool_secrets import SecretError, validate_declaration

TOKEN = "tok_live_abc123XYZ"

TOOL = '''
from pydantic import BaseModel, Field
from agent_app.sdk import ToolContext
SECRETS = {"BITBUCKET_ACCESS_TOKEN": ["api.bitbucket.org"]}
class Params(BaseModel):
    workspace: str
def run(params: Params, ctx: ToolContext) -> dict:
    r = ctx.http.get("https://api.bitbucket.org/2.0/repositories/" + params.workspace,
                     headers={"Authorization": "Bearer {secret:BITBUCKET_ACCESS_TOKEN}"})
    return {"status": r.status, "body": r.text}
'''


@pytest.fixture
def net(monkeypatch):
    """Fake DNS (everything public) + a recording transport."""
    monkeypatch.setattr(socket, "getaddrinfo", lambda h, p, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", p))])
    monkeypatch.setenv("BITBUCKET_ACCESS_TOKEN", TOKEN)
    seen: list[httpx.Request] = []
    box = {"handler": lambda r: httpx.Response(200, text="ok")}

    def handler(req):
        seen.append(req)
        return box["handler"](req)
    http = SafeHttp()
    http.transport = httpx.MockTransport(handler)
    return http, seen, box


def test_placeholder_is_filled_in_only_at_send_time(net):
    http, seen, _ = net
    scoped = http.scoped({"BITBUCKET_ACCESS_TOKEN": ["api.bitbucket.org"]})
    assert scoped.get("https://api.bitbucket.org/x", headers={"Authorization": "Bearer {secret:BITBUCKET_ACCESS_TOKEN}"}).status == 200
    assert seen[0].headers["authorization"] == f"Bearer {TOKEN}"


def test_basic_auth_placeholder(net):
    http, seen, _ = net
    http.scoped({"BITBUCKET_ACCESS_TOKEN": ["api.bitbucket.org"]}).get(
        "https://api.bitbucket.org/x", headers={"Authorization": "Basic {basic:alice:BITBUCKET_ACCESS_TOKEN}"})
    assert base64.b64decode(seen[0].headers["authorization"].split()[1]).decode() == f"alice:{TOKEN}"


@pytest.mark.parametrize("url,why", [("https://evil.example/x", "may only be sent to"), ("http://api.bitbucket.org/x", "only sent over https")])
def test_secret_is_not_sent_to_other_hosts_or_plain_http(net, url, why):
    http, seen, _ = net
    with pytest.raises(SecretError, match=why):
        http.scoped({"BITBUCKET_ACCESS_TOKEN": ["api.bitbucket.org"]}).get(url, headers={"Authorization": "Bearer {secret:BITBUCKET_ACCESS_TOKEN}"})
    assert seen == []  # nothing was sent


def test_redirect_to_another_host_does_not_leak_the_secret(net):
    http, seen, box = net
    box["handler"] = lambda r: httpx.Response(302, headers={"location": "https://evil.example/steal"})
    with pytest.raises(SecretError):
        http.scoped({"BITBUCKET_ACCESS_TOKEN": ["api.bitbucket.org"]}).get(
            "https://api.bitbucket.org/x", headers={"Authorization": "Bearer {secret:BITBUCKET_ACCESS_TOKEN}"})
    assert [r.url.host for r in seen] == ["93.184.216.34"]  # only the first hop was made


def test_undeclared_or_unset_secret_fails_clearly_without_leaking(net, monkeypatch):
    http, _, _ = net
    with pytest.raises(SecretError, match="did not declare"):
        http.scoped({"OTHER_SECRET": ["api.bitbucket.org"]}).get("https://api.bitbucket.org/x", headers={"A": "{secret:BITBUCKET_ACCESS_TOKEN}"})
    monkeypatch.setenv("BITBUCKET_ACCESS_TOKEN", "")
    with pytest.raises(SecretError, match="not set") as e:
        http.scoped({"BITBUCKET_ACCESS_TOKEN": ["api.bitbucket.org"]}).get("https://api.bitbucket.org/x", headers={"A": "{secret:BITBUCKET_ACCESS_TOKEN}"})
    assert TOKEN not in str(e.value)


def test_echoed_secret_is_redacted_from_responses(net):
    http, _, box = net
    box["handler"] = lambda r: httpx.Response(401, text=f"bad credentials: {r.headers['authorization']}")
    r = http.scoped({"BITBUCKET_ACCESS_TOKEN": ["api.bitbucket.org"]}).get(
        "https://api.bitbucket.org/x", headers={"Authorization": "Bearer {secret:BITBUCKET_ACCESS_TOKEN}"})
    assert TOKEN not in r.text and "[REDACTED]" in r.text


def test_unscoped_http_never_substitutes(net):
    http, seen, _ = net
    http.get("https://api.bitbucket.org/x", headers={"Authorization": "Bearer {secret:BITBUCKET_ACCESS_TOKEN}"})
    assert seen[0].headers["authorization"] == "Bearer {secret:BITBUCKET_ACCESS_TOKEN}"


@pytest.mark.parametrize("bad", [{}, [], {"lower": ["a.example.com"]}, {"TOKEN_X": []}, {"TOKEN_X": ["https://a.example.com"]},
                                 {"TOKEN_X": ["a.example.com/path"]}, {"TOKEN_X": ["*.example.com"]}, {"TOKEN_X": ["localhost"]},
                                 {"TOKEN_X": ["a.example.com:8080"]}, {"TOKEN_X": "a.example.com"}])
def test_declaration_validation(bad):
    with pytest.raises(SecretError):
        validate_declaration(bad)


def test_lint_shows_declared_secrets_and_requires_declaration():
    ok = lint_tool_source(TOOL)
    assert ok.ok and "secret:BITBUCKET_ACCESS_TOKEN->api.bitbucket.org" in ok.capabilities and "network" in ok.capabilities
    undeclared = lint_tool_source(TOOL.replace('SECRETS = {"BITBUCKET_ACCESS_TOKEN": ["api.bitbucket.org"]}', ""))
    assert not undeclared.ok and "not declared in SECRETS" in undeclared.errors[0]
    assert not lint_tool_source(TOOL.replace("api.bitbucket.org\"]}", "https://api.bitbucket.org\"]}")).ok
    assert not lint_tool_source(TOOL.replace("import os", "") + "\nimport os\n").ok  # still no os access to read the environment


def test_generated_tool_end_to_end_never_sees_the_token(tmp_path, monkeypatch, net):
    from agent_app import main
    http, seen, box = net
    box["handler"] = lambda r: httpx.Response(200, text='{"values": []}')
    main.agent.http.transport = http.transport
    path = tmp_path / "fetch_bb.py"
    path.write_text(TOOL.replace("from pydantic", "NAME = 'fetch_bb'\nDESCRIPTION = 'x'\nfrom pydantic", 1))
    from agent_app.agent import _load_generated
    tool = _load_generated("fetch_bb", path)
    assert tool.secrets == {"BITBUCKET_ACCESS_TOKEN": ["api.bitbucket.org"]}
    ok, text, raw = main.agent._run({"fetch_bb": tool}, "fetch_bb", {"workspace": "acme"})
    assert ok, text
    assert seen[-1].headers["authorization"] == f"Bearer {TOKEN}"   # the API got the real token...
    assert TOKEN not in text and TOKEN not in str(raw)                # ...the tool result (and so the chat) never did
    assert "token" not in tool.schema()["properties"]               # and it is not a parameter the model must fill in
