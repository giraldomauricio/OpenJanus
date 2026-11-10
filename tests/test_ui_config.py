import json

from agent_app import ui_config
from agent_app.ui_config import sanitize


def test_valid_config_passes_through():
    cfg, w = sanitize({"theme": "hacker", "colors": {"accent": "#0f0", "bg": "rgb(1, 2, 3)"}, "radius": 4,
                       "sidebar": {"position": "left", "tabs": ["database", "approvals"], "default_tab": "database"}})
    assert w == [] and cfg["theme"] == "hacker" and cfg["colors"]["bg"] == "rgb(1, 2, 3)"
    assert cfg["sidebar"]["tabs"] == ["database", "approvals"]


def test_css_injection_is_rejected():
    cfg, w = sanitize({"colors": {"accent": "red; background:url(http://x)", "evil": "#fff"},
                       "fonts": {"body": "x}</style><script>", "size": 999}})
    assert cfg["colors"] == {} and cfg["fonts"] == {} and len(w) == 4


def test_approvals_tab_cannot_be_hidden():
    cfg, w = sanitize({"sidebar": {"tabs": ["database"]}})
    assert cfg["sidebar"]["tabs"] == ["approvals", "database"] and w


def test_bad_types_never_raise():
    assert sanitize([1])[0] == {}
    cfg, w = sanitize({"topbar": "x", "sidebar": {"width": "wide"}, "bogus": 1})
    assert len(w) == 3


def test_file_loading(tmp_path, monkeypatch):
    p = tmp_path / "ui.json"
    monkeypatch.setattr(ui_config, "UI_CONFIG_PATH", p)
    assert ui_config.load_ui_config() == {"config": {}, "warnings": []}
    p.write_text("{nope")
    assert "not valid JSON" in ui_config.load_ui_config()["warnings"][0]
    p.write_text(json.dumps({"theme": "light"}))
    assert ui_config.load_ui_config()["config"] == {"theme": "light"}


def test_rename_chat(tmp_path):
    from agent_app.agent import Session
    from agent_app.sessions import SessionStore
    st = SessionStore(tmp_path)
    s = Session(); s.title = "old"; s.events.append({"type": "user", "text": "hi"})
    st.save(s)
    before = st.list()[0]["updated"]
    assert st.rename(s.id, "new name") and not st.rename("../x", "t") and not st.rename("0" * 32, "t")
    assert st.list()[0]["title"] == "new name" and st.list()[0]["updated"] == before
    assert st.load(s.id).title == "new name" and st.load(s.id).events == s.events


def test_version_comes_from_version_txt(tmp_path, monkeypatch):
    from agent_app import config, main
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert main.app_version() == ""                       # no file: nothing shown
    (tmp_path / "VERSION.TXT").write_text("1.0.3.9742273\n")
    assert main.app_version() == "1.0.3.9742273"
    (tmp_path / "VERSION.TXT").write_text("1.0.4.abc1234\n")  # bumped by a commit: picked up without a restart
    assert main.app_version() == "1.0.4.abc1234"
    (tmp_path / "VERSION.TXT").write_text("<script>alert(1)</script>")
    assert main.app_version() == ""                       # anything odd is dropped, never injected
