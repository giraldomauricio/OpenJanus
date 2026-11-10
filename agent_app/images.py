"""Chat image attachments: raster images only, content-addressed files under data/uploads/, never embedded in the chat JSON.

A user message references its images by id: {"role": "user", "content": str, "images": [id, ...]}.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import tempfile
from typing import Any

from . import config

MAX_BYTES = 4_500_000          # under the 5 MB per-image limit of the Claude API
MAX_PER_MESSAGE = 4
_ID = re.compile(r"^[0-9a-f]{32}$")


def sniff(data: bytes) -> str | None:
    """MIME type from the file's own bytes (the client's Content-Type is not trusted). SVG is deliberately NOT accepted."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def save(data: bytes) -> dict[str, str]:
    if not data:
        raise ValueError("Empty file")
    if len(data) > MAX_BYTES:
        raise ValueError(f"Image too large ({MAX_BYTES // 1_000_000} MB max)")
    mime = sniff(data)
    if mime is None:
        raise ValueError("Unsupported image: use PNG, JPEG, GIF or WebP")
    iid = hashlib.sha256(data).hexdigest()[:32]
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOADS_DIR / f"{iid}.img"
    if not path.exists():
        fd, tmp = tempfile.mkstemp(dir=config.UPLOADS_DIR, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    return {"id": iid, "mime": mime}


def load(iid: str) -> tuple[str, bytes] | None:
    if not _ID.match(iid):
        return None
    try:
        data = (config.UPLOADS_DIR / f"{iid}.img").read_bytes()
    except OSError:
        return None
    mime = sniff(data)
    return (mime, data) if mime else None


def resolve(message: dict[str, Any]) -> tuple[list[tuple[str, bytes]], int]:
    """(images, number missing) for a user message. A deleted file must not break the whole conversation."""
    found, missing = [], 0
    for iid in message.get("images") or []:
        got = load(iid)
        if got:
            found.append(got)
        else:
            missing += 1
    return found, missing


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def missing_note(n: int) -> str:
    return f"[{n} attached image(s) are no longer available]" if n else ""
