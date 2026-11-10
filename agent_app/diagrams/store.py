"""Where diagram files live (data/diagrams/): validated names only, atomic writes."""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,60}$")
EXTS = (".drawio", ".xml")


def clean_name(name: str, default_ext: str = ".drawio") -> str:
    name = name.strip()
    if not name.lower().endswith(EXTS):
        name += default_ext
    if not _NAME.match(name) or ".." in name:
        raise ValueError("Invalid file name: use letters, digits, '.', '_' or '-' (max 60), e.g. 'payments-architecture'")
    return name


class DiagramStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def path(self, name: str) -> Path:
        return self.dir / clean_name(name)

    def save(self, name: str, data: bytes) -> str:
        name = clean_name(name)
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, self.dir / name)
        return name

    def read(self, name: str) -> bytes:
        p = self.path(name)
        if not p.is_file():
            raise ValueError(f"No diagram file named '{p.name}'. Use list_diagrams to see what exists.")
        return p.read_bytes()

    def list(self) -> list[dict[str, object]]:
        if not self.dir.is_dir():
            return []
        out = [{"name": p.name, "bytes": p.stat().st_size, "modified": p.stat().st_mtime}
               for p in self.dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS and _NAME.match(p.name)]
        return sorted(out, key=lambda x: x["modified"], reverse=True)  # type: ignore[arg-type,return-value]
