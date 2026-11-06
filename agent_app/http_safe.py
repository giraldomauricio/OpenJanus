"""HTTP client handed to *generated* tools as ``ctx.http``.

Generated tools may not import network libraries. They go through this wrapper,
which refuses loopback/private/link-local targets, so an approved tool cannot
reach this app's own API (or other local services). Redirects are re-checked.

Known limit: DNS is resolved once for the check and again by httpx, so a hostile
DNS server could still rebind. Treat network-capable tools with care at review time.
"""
from __future__ import annotations

import ipaddress
import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

MAX_BYTES = 2_000_000


class BlockedURL(ValueError):
    pass


@dataclass
class HttpResponse:
    status: int
    text: str

    def json(self) -> Any:
        return json.loads(self.text)


def _check(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise BlockedURL("Only absolute http(s) URLs are allowed")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror as e:
        raise BlockedURL(f"Cannot resolve {p.hostname}: {e}") from e
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise BlockedURL(f"Blocked non-public address for {p.hostname}")


class SafeHttp:
    def _request(self, method: str, url: str, **kw: Any) -> HttpResponse:
        with httpx.Client(follow_redirects=False, timeout=20) as client:
            for _ in range(4):
                _check(url)
                r = client.request(method, url, **kw)
                if r.is_redirect and "location" in r.headers:
                    url = urljoin(url, r.headers["location"])
                    continue
                return HttpResponse(r.status_code, r.text[:MAX_BYTES])
        raise BlockedURL("Too many redirects")

    def get(self, url: str, params: dict | None = None, headers: dict | None = None) -> HttpResponse:
        return self._request("GET", url, params=params, headers=headers)

    def post(self, url: str, json: Any = None, headers: dict | None = None) -> HttpResponse:  # noqa: A002
        return self._request("POST", url, json=json, headers=headers)
