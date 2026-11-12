"""Optional publisher signatures for .janus packages (Ed25519).

A signature proves two things, and only these: the package was not changed after it was signed, and it was signed by whoever holds
the private key with a given FINGERPRINT. It says nothing about whether that person is trustworthy or their code is safe: that is
what the reviewer's approval is for. A key becomes "trusted" only when you add its fingerprint (after checking it with the
sender by some other channel); trusted packages are labelled as such in Approvals, and nothing is ever approved automatically.

Files (under data/extensions/trust/): publisher.key (this installation's private key, owner-only) and publishers.json.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    AVAILABLE = True
except ImportError:  # signing is optional: without the library packages simply stay unsigned
    AVAILABLE = False

FP_RE = re.compile(r"^[0-9a-f]{16}$")


class SignatureError(ValueError):
    pass


def fingerprint(public_raw: bytes) -> str:
    return hashlib.sha256(public_raw).hexdigest()[:16]


def verify(data: bytes, sig: Any) -> str:
    """Check `sig` ({algorithm, public_key, signature}, base64) against `data`. Returns the signer's fingerprint."""
    if not AVAILABLE:
        raise SignatureError("This package is signed but signatures cannot be checked here (the 'cryptography' package is missing)")
    try:
        if sig["algorithm"] != "ed25519":
            raise SignatureError(f"Unsupported signature algorithm {sig['algorithm']!r}")
        pub = base64.b64decode(sig["public_key"], validate=True)
        Ed25519PublicKey.from_public_bytes(pub).verify(base64.b64decode(sig["signature"], validate=True), data)
    except InvalidSignature as e:
        raise SignatureError("The signature does not match: the package was changed after it was signed") from e
    except (KeyError, TypeError, ValueError) as e:
        if isinstance(e, SignatureError):
            raise
        raise SignatureError("Malformed signature") from e
    return fingerprint(pub)


class Signer:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    @property
    def _key_path(self) -> Path:
        return self.dir / "publisher.key"

    def _private(self) -> "Ed25519PrivateKey":
        if not AVAILABLE:
            raise SignatureError("Signing needs the 'cryptography' package (pip install cryptography)")
        if self._key_path.exists():
            return serialization.load_pem_private_key(self._key_path.read_bytes(), password=None)  # type: ignore[return-value]
        self.dir.mkdir(parents=True, exist_ok=True)
        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        fd = os.open(self._key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # never readable by other users
        with os.fdopen(fd, "wb") as f:
            f.write(pem)
        return key

    def _public_raw(self) -> bytes:
        return self._private().public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def own_fingerprint(self, create: bool = False) -> str | None:
        """This installation's fingerprint. Without `create`, None if it has never signed anything (no key is made just to look)."""
        if not AVAILABLE or (not create and not self._key_path.exists()):
            return None
        return fingerprint(self._public_raw())

    def sign(self, data: bytes) -> dict[str, str]:
        key = self._private()
        pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return {"algorithm": "ed25519", "public_key": base64.b64encode(pub).decode(),
                "signature": base64.b64encode(key.sign(data)).decode()}

    # --- trusted publishers ---------------------------------------------------

    @property
    def _trust_path(self) -> Path:
        return self.dir / "publishers.json"

    def trusted(self) -> dict[str, dict[str, str]]:
        try:
            data = json.loads(self._trust_path.read_text())
        except (OSError, ValueError):
            data = {}
        out = {k: v for k, v in data.items() if FP_RE.match(k) and isinstance(v, dict)} if isinstance(data, dict) else {}
        own = self.own_fingerprint()
        if own:
            out.setdefault(own, {"label": "this installation"})
        return out

    def trust(self, fp: str, label: str) -> None:
        label = " ".join(label.split())[:60]
        if not FP_RE.match(fp):
            raise SignatureError("A fingerprint is 16 lowercase hex characters")
        if not label:
            raise SignatureError("Give the publisher a label, e.g. their name")
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(self._trust_path.read_text())
        except (OSError, ValueError):
            data = {}
        data[fp] = {"label": label, "added": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self._trust_path.write_text(json.dumps(data, indent=2))
