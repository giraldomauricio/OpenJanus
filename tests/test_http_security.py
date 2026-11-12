"""Every route that changes state must refuse a request without the page's token; reads stay open."""
import pytest
from fastapi.testclient import TestClient
from fastapi.routing import APIRoute

from agent_app import main

client = TestClient(main.app)
STATE_CHANGING = {"POST", "PUT", "PATCH", "DELETE"}


def routes():
    for r in main.app.routes:
        if isinstance(r, APIRoute):
            for m in r.methods & STATE_CHANGING:
                yield m, r.path


def concrete(path):
    import re
    return re.sub(r"\{[^}]+\}", "x", path)


@pytest.mark.parametrize("method,path", sorted(routes()))
def test_state_changing_route_requires_the_token(method, path):
    r = client.request(method, concrete(path), content=b"{}")
    assert r.status_code == 403, f"{method} {path} answered {r.status_code} without a token"


def test_a_wrong_token_is_refused():
    assert client.post("/api/publishers/trust", json={"fingerprint": "0" * 16, "label": "x"}, headers={"X-Agent-Token": "nope"}).status_code == 403


def test_review_actions_cannot_be_called_without_token():
    for action in ("approve", "reject", "retire"):
        assert client.post(f"/api/extensions/tool/some_tool/{action}").status_code == 403
