"""Secrets for generated tools, without the tool (or the chat) ever holding the value.

A tool declares what it needs, at module level, in its own source (so the reviewer sees it and the integrity hash covers it):

    SECRETS = {"BITBUCKET_ACCESS_TOKEN": ["api.bitbucket.org"]}     # env var name -> hosts it may be sent to

and writes placeholders in request HEADERS only:

    headers={"Authorization": "Bearer {secret:BITBUCKET_ACCESS_TOKEN}"}
    headers={"Authorization": "Basic {basic:some-user:BITBUCKET_APP_PASSWORD}"}     # base64("user:value")

The HTTP layer substitutes the real value from the server's environment (.env or shell) at send time, only over https and only to a
declared host, on every redirect hop too. The value never enters tool code, tool results, the database or the chat; if a server
echoes it back it is redacted from the response text.
"""
from __future__ import annotations

import base64
import logging
import os
import re
from typing import Any
from urllib.parse import urlsplit

log = logging.getLogger("openjanus.security")
NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,60}$")
HOST_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")  # exact public hostname: no scheme, port, path, wildcard
SECRET_PH = re.compile(r"\{secret:([A-Z][A-Z0-9_]{2,60})\}")
BASIC_PH = re.compile(r"\{basic:([^{}:\s]{1,100}):([A-Z][A-Z0-9_]{2,60})\}")
ANY_PH = re.compile(r"\{(?:secret|basic):[^{}]*\}")


class SecretError(ValueError):
    pass


def placeholder_names(text: str) -> set[str]:
    return set(SECRET_PH.findall(text)) | {m[1] for m in BASIC_PH.findall(text)}


def validate_declaration(obj: Any) -> dict[str, list[str]]:
    """The SECRETS constant -> {NAME: [host, ...]} or SecretError describing what is wrong."""
    if not isinstance(obj, dict) or not obj:
        raise SecretError('SECRETS must be a non-empty dict like {"BITBUCKET_ACCESS_TOKEN": ["api.bitbucket.org"]}')
    out: dict[str, list[str]] = {}
    for name, hosts in obj.items():
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise SecretError(f"Secret name {name!r} must be an UPPER_SNAKE environment variable name (3-61 chars)")
        if not isinstance(hosts, (list, tuple)) or not hosts or len(hosts) > 5:
            raise SecretError(f"{name}: give 1-5 allowed hosts, e.g. [\"api.example.com\"]")
        for h in hosts:
            if not isinstance(h, str) or not HOST_RE.match(h):
                raise SecretError(f"{name}: {h!r} is not an exact hostname (no scheme, port, path or wildcard)")
        out[name] = [h.lower() for h in hosts]
    return out


class SecretResolver:
    def __init__(self, declared: dict[str, list[str]]):
        self.declared = declared
        self._used: set[str] = set()

    def _value(self, name: str) -> str:
        if name not in self.declared:
            raise SecretError(f"This tool did not declare secret {name} in SECRETS")
        v = os.environ.get(name, "")
        if not v:
            raise SecretError(f"Secret {name} is not set. Add {name}=... to the .env file (or the server's environment) and restart OpenJanus.")
        self._used.add(name)
        return v

    def headers_for(self, url: str, headers: dict[str, str] | None) -> dict[str, str] | None:
        """Headers with placeholders filled in for THIS url, or SecretError if the url may not receive a placeholder's secret."""
        if not headers:
            return headers
        out = {}
        parts = urlsplit(url)
        for k, v in headers.items():
            if isinstance(v, str) and ANY_PH.search(v):
                for name in placeholder_names(v):
                    if name not in self.declared:
                        raise SecretError(f"This tool did not declare secret {name} in SECRETS")
                    if parts.scheme != "https":
                        raise SecretError(f"Secret {name} is only sent over https")
                    if (parts.hostname or "").lower() not in self.declared[name]:
                        raise SecretError(f"Secret {name} may only be sent to {', '.join(self.declared[name])}, not {parts.hostname}")
                for name in sorted(placeholder_names(v)):
                    log.info("secret %s sent to %s", name, parts.hostname)  # the name only, never the value
                v = SECRET_PH.sub(lambda m: self._value(m.group(1)), v)
                v = BASIC_PH.sub(lambda m: base64.b64encode(f"{m.group(1)}:{self._value(m.group(2))}".encode()).decode(), v)
            out[k] = v
        return out

    def redact(self, text: str) -> str:
        for name in self._used:
            v = os.environ.get(name, "")
            if len(v) >= 4:
                text = text.replace(v, "[REDACTED]").replace(base64.b64encode(v.encode()).decode(), "[REDACTED]")
        return text
