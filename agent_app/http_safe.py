"""HTTP client handed to *generated* tools as ``ctx.http``.

Generated tools may not import network libraries. They go through this wrapper, which refuses
loopback/private/link-local targets, so an approved tool cannot reach this app's own API (or other
local services). Redirects are re-checked.

DNS rebinding: the hostname is resolved ONCE per request, every address is checked, and the connection is
then pinned to that exact IP (the URL host is replaced by the IP; the Host header and TLS SNI/certificate
check keep the original hostname). A hostile DNS server therefore cannot answer "public" for the check and
"127.0.0.1" for the connection.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from .tool_secrets import SecretResolver

MAX_BYTES = 2_000_000
log = logging.getLogger("openjanus.http")


class BlockedURL(ValueError):
    pass


@dataclass
class HttpResponse:
    status: int
    text: str

    def json(self) -> Any:
        return json.loads(self.text)


def resolve_public(host: str, port: int) -> str:
    """Resolve once; refuse if ANY address is non-public (a mixed answer is a rebinding signal); return the IP to connect to."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise BlockedURL(f"Cannot resolve {host}: {e}") from e
    ips = [i[4][0].split("%")[0] for i in infos]
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        log.warning("BLOCKED outbound request to %s: resolves to a non-public address", host)
        raise BlockedURL(f"Blocked non-public address for {host}")
    return ips[0]


def pin(request: httpx.Request) -> httpx.Request:
    """Validate the request's host and rewrite it to connect to the checked IP only."""
    url = request.url
    if url.scheme not in ("http", "https") or not url.host:
        raise BlockedURL("Only absolute http(s) URLs are allowed")
    host = url.host
    ip = resolve_public(host, url.port or (443 if url.scheme == "https" else 80))
    request.headers["Host"] = url.netloc.decode()           # keep the original authority for the server
    request.extensions = {**request.extensions, "sni_hostname": host}  # TLS: SNI + certificate verified against the hostname
    request.url = url.copy_with(host=ip)
    return request


class SafeHttp:
    transport: httpx.BaseTransport | None = None  # tests inject a fake transport here
    resolver: SecretResolver | None = None        # set on the scoped copy a tool with SECRETS receives

    def scoped(self, declared: dict[str, list[str]]) -> "SafeHttp":
        """A copy for one generated tool: header placeholders for ITS declared secrets are filled in (see tool_secrets.py)."""
        s = SafeHttp()
        s.transport, s.resolver = self.transport, SecretResolver(declared)
        return s

    def _request(self, method: str, url: str, **kw: Any) -> HttpResponse:
        headers = kw.pop("headers", None)
        started, first_url = time.monotonic(), url
        with httpx.Client(follow_redirects=False, timeout=20, transport=self.transport) as client:
            for _ in range(4):
                # per hop: a redirect to a host that may not receive the secret fails here instead of leaking it
                hop_headers = self.resolver.headers_for(url, headers) if self.resolver else headers
                r = client.send(pin(client.build_request(method, url, headers=hop_headers, **kw)))
                if r.is_redirect and "location" in r.headers:
                    url = urljoin(url, r.headers["location"])
                    continue
                text = r.text[:MAX_BYTES]
                h = urlsplit(url)  # host and path only: query strings can carry tokens
                log.info("%s %s%s -> %d (%.0fms, %d bytes)", method, h.hostname, h.path[:80], r.status_code,
                         (time.monotonic() - started) * 1000, len(r.content))
                return HttpResponse(r.status_code, self.resolver.redact(text) if self.resolver else text)
        log.warning("too many redirects starting at %s", urlsplit(first_url).hostname)
        raise BlockedURL("Too many redirects")

    def get(self, url: str, params: dict | None = None, headers: dict | None = None) -> HttpResponse:
        return self._request("GET", url, params=params, headers=headers)

    def post(self, url: str, json: Any = None, headers: dict | None = None) -> HttpResponse:  # noqa: A002
        return self._request("POST", url, json=json, headers=headers)
