"""Optional UI customisation from a JSON file (`ui.json` in the project root, or $OPENJANUS_UI_CONFIG).

The file is read on every request and *sanitised*: only whitelisted keys are kept and every value is validated, because
the values end up in CSS / DOM properties in the browser. Problems never raise; they are returned as `warnings`.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from . import config

UI_CONFIG_PATH = Path(os.getenv("OPENJANUS_UI_CONFIG") or config.ROOT / "ui.json")

THEMES = ("dark", "light", "hacker")
TOPBAR_ITEMS = ("logo", "title", "provider", "model", "theme", "history", "import", "new", "rename", "delete")
TABS = ("approvals", "capabilities", "database")  # "approvals" can never be hidden: it is how tool proposals get reviewed
COLOR_KEYS = ("bg", "panel", "line", "text", "muted", "accent", "ok", "bad", "warn", "code_bg", "user_bg", "on_accent")

_COLOR = re.compile(r"^(#[0-9a-fA-F]{3,8}|(rgb|hsl)a?\([0-9\s.,%/-]{1,40}\)|[a-zA-Z]{3,24})$")
_FONT = re.compile(r"^[\w\s,\"'.\-]{1,200}$")


def _num(v: Any, lo: float, hi: float) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        return None
    return v


def _names(v: Any, allowed: tuple[str, ...], where: str, warnings: list[str]) -> list[str] | None:
    if not isinstance(v, list):
        warnings.append(f"{where}: expected a list")
        return None
    out: list[str] = []
    for x in v:
        if x in allowed:
            if x not in out:
                out.append(x)
        else:
            warnings.append(f"{where}: unknown item {x!r} (allowed: {', '.join(allowed)})")
    return out


def sanitize(raw: Any) -> tuple[dict[str, Any], list[str]]:
    w: list[str] = []
    if not isinstance(raw, dict):
        return {}, ["ui.json must contain a JSON object"]
    out: dict[str, Any] = {}

    if "theme" in raw:
        if raw["theme"] in THEMES:
            out["theme"] = raw["theme"]
        else:
            w.append(f"theme: must be one of {', '.join(THEMES)}")

    if isinstance(c := raw.get("colors"), dict):
        colors = {}
        for k, v in c.items():
            if k not in COLOR_KEYS:
                w.append(f"colors.{k}: unknown key (allowed: {', '.join(COLOR_KEYS)})")
            elif not isinstance(v, str) or not _COLOR.match(v.strip()):
                w.append(f"colors.{k}: not a valid color")
            else:
                colors[k] = v.strip()
        out["colors"] = colors
    elif "colors" in raw:
        w.append("colors: expected an object")

    if isinstance(f := raw.get("fonts"), dict):
        fonts: dict[str, Any] = {}
        for k in ("body", "mono"):
            if k in f:
                if isinstance(f[k], str) and _FONT.match(f[k]):
                    fonts[k] = f[k]
                else:
                    w.append(f"fonts.{k}: invalid font-family list")
        if "size" in f:
            if (n := _num(f["size"], 10, 24)) is not None:
                fonts["size"] = n
            else:
                w.append("fonts.size: number between 10 and 24 (px)")
        out["fonts"] = fonts
    elif "fonts" in raw:
        w.append("fonts: expected an object")

    if "radius" in raw:
        if (n := _num(raw["radius"], 0, 24)) is not None:
            out["radius"] = n
        else:
            w.append("radius: number between 0 and 24 (px)")

    if isinstance(t := raw.get("topbar"), dict):
        top: dict[str, Any] = {}
        if isinstance(t.get("visible"), bool):
            top["visible"] = t["visible"]
        if isinstance(t.get("title"), str) and t["title"].strip():
            top["title"] = t["title"].strip()[:40]
        if "show" in t and (s := _names(t["show"], TOPBAR_ITEMS, "topbar.show", w)) is not None:
            top["show"] = s  # order in the list = order on screen
        out["topbar"] = top
    elif "topbar" in raw:
        w.append("topbar: expected an object")

    if isinstance(ch := raw.get("chat"), dict):
        chat: dict[str, Any] = {}
        if isinstance(ch.get("placeholder"), str):
            chat["placeholder"] = ch["placeholder"][:200]
        out["chat"] = chat
    elif "chat" in raw:
        w.append("chat: expected an object")

    if isinstance(sb := raw.get("sidebar"), dict):
        side: dict[str, Any] = {}
        if "position" in sb:
            if sb["position"] in ("left", "right"):
                side["position"] = sb["position"]
            else:
                w.append("sidebar.position: 'left' or 'right'")
        if "width" in sb:
            if (n := _num(sb["width"], 240, 900)) is not None:
                side["width"] = n
            else:
                w.append("sidebar.width: number between 240 and 900 (px)")
        if "tabs" in sb and (tabs := _names(sb["tabs"], TABS, "sidebar.tabs", w)) is not None:
            if "approvals" not in tabs:
                tabs.insert(0, "approvals")
                w.append("sidebar.tabs: 'approvals' cannot be hidden, so it was kept")
            side["tabs"] = tabs
        if "default_tab" in sb:
            if sb["default_tab"] in TABS:
                side["default_tab"] = sb["default_tab"]
            else:
                w.append(f"sidebar.default_tab: one of {', '.join(TABS)}")
        if "default_tab" in side and "tabs" in side and side["default_tab"] not in side["tabs"]:
            w.append("sidebar.default_tab is not in sidebar.tabs; ignored")
            del side["default_tab"]
        out["sidebar"] = side
    elif "sidebar" in raw:
        w.append("sidebar: expected an object")

    for k in raw:
        if k not in ("theme", "colors", "fonts", "radius", "topbar", "chat", "sidebar") and not k.startswith("_"):
            w.append(f"{k}: unknown key ignored")
    return out, w


def load_ui_config() -> dict[str, Any]:
    """Always returns {"config": {...}, "warnings": [...]}; a missing file is simply an empty config."""
    try:
        text = UI_CONFIG_PATH.read_text()
    except FileNotFoundError:
        return {"config": {}, "warnings": []}
    except OSError as e:
        return {"config": {}, "warnings": [f"Cannot read {UI_CONFIG_PATH.name}: {e}"]}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        return {"config": {}, "warnings": [f"{UI_CONFIG_PATH.name} is not valid JSON ({e}); using defaults"]}
    cfg, warnings = sanitize(raw)
    return {"config": cfg, "warnings": warnings}
