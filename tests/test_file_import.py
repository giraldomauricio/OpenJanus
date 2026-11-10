import json

import pytest

from agent_app.db import Database
from agent_app.db import file_import as ji
from agent_app.db.file_import import Cancelled, ImportFileError, load, resolve_path, scan

REC = [
    {"id": 1, "sev": "High", "score": 7.5, "seen": "2026-01-02", "ok": True, "meta": {"a": 1, "b": {"c": "x"}}, "tags": ["a", "b"], "note": None},
    {"id": 2, "sev": "Low", "score": 3, "seen": "2026-02-03", "ok": False, "meta": {"a": 2, "b": {"c": "y"}}, "tags": [], "note": "hello"},
]


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "t.db")


def write(tmp_path, text, name="f.json", enc="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(enc))
    return p


def rows(db, table):
    return db.query(f'SELECT * FROM "{table}" ORDER BY 1', limit=1000)["rows"]


@pytest.mark.parametrize("chunk", [1 << 20, 7, 64])  # tiny chunks put every possible boundary inside a token, number or string
@pytest.mark.parametrize("shape", ["array", "pretty", "lines", "wrapper"])
def test_all_shapes_and_chunk_boundaries(tmp_path, db, monkeypatch, chunk, shape):
    monkeypatch.setattr(ji, "CHUNK", chunk)
    text = {"array": json.dumps(REC), "pretty": json.dumps(REC, indent=2), "lines": "\n".join(json.dumps(r) for r in REC),
            "wrapper": json.dumps({"total": 2, "tags": ["x"], "results": REC, "after": 1}, indent=1)}[shape]
    r = load(db, write(tmp_path, text), "findings", "new")
    assert r["rows"] == 2
    got = rows(db, "findings")
    assert got[0]["meta_b_c"] == "x" and got[1]["meta_a"] == 2 and got[0]["tags"] == '["a","b"]' and got[0]["note"] is None
    assert got[0]["ok"] == 1 and got[1]["ok"] == 0                       # booleans become 0/1
    types = {c["name"]: c["declared_type"] for c in r["columns"]}
    assert types["id"] == "INTEGER" and types["score"] == "REAL" and types["seen"] == "DATE" and types["sev"] == "TEXT"


def test_types_and_awkward_values(tmp_path, db):
    data = [{"big": 2 ** 70, "mix": 1, "code": "007", "flag": True, "empty": ""}, {"big": 5, "mix": "abc", "code": "012", "flag": "yes", "empty": ""}]
    r = load(db, write(tmp_path, json.dumps(data)), "t", "new")
    types = {c["name"]: c["declared_type"] for c in r["columns"]}
    assert types["big"] == "TEXT" and types["mix"] == "TEXT" and types["code"] == "TEXT" and types["flag"] == "TEXT"
    got = rows(db, "t")
    assert got[0]["big"] == str(2 ** 70) and got[0]["code"] == "007" and got[0]["flag"] == "true" and got[1]["flag"] == "yes"


def test_keys_that_vary_between_records_and_names_are_cleaned(tmp_path, db):
    data = [{"First Name": "a"}, {"First Name": "b", "extraField": 1}, {"First Name": "c", "extra_field": 2}]
    r = load(db, write(tmp_path, json.dumps(data)), "t", "new")
    assert [c["name"] for c in r["columns"]] == ["first_name", "extra_field", "extra_field_2"]
    assert [x["extra_field"] for x in rows(db, "t")] == [None, 1, None]


def test_modes(tmp_path, db):
    p1 = write(tmp_path, json.dumps(REC))
    load(db, p1, "f", "new")
    with pytest.raises(ImportFileError, match="already exists"):
        load(db, p1, "f", "new")
    with pytest.raises(ImportFileError, match="no table"):
        load(db, p1, "nope", "append")
    assert load(db, p1, "f", "append")["rows"] == 2 and len(rows(db, "f")) == 4
    p2 = write(tmp_path, json.dumps(REC[:1]), "g.json")
    r = load(db, p2, "f", "replace")
    assert r["replaced_rows"] == 4 and len(rows(db, "f")) == 1
    p3 = write(tmp_path, json.dumps([{"id": 9, "brand_new_field": 1}]), "h.json")
    with pytest.raises(ImportFileError, match="brand_new_field"):
        load(db, p3, "f", "append")
    assert len(rows(db, "f")) == 1                                       # the failed import changed nothing


def test_failure_and_cancel_roll_everything_back(tmp_path, db, monkeypatch):
    monkeypatch.setattr(ji, "BATCH", 2)
    ok = [{"n": i} for i in range(10)]
    # fails in the middle of the file (invalid JSON at the end): no half-filled table is left behind
    p = write(tmp_path, json.dumps(ok)[:-1] + ',{"n": oops}]')
    with pytest.raises(ImportFileError):
        load(db, p, "t", "new")
    assert "t" not in db.table_names()
    # cancel during the load
    p = write(tmp_path, json.dumps(ok), "ok.json")
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2
    with pytest.raises(Cancelled):
        load(db, p, "t", "new", cancelled=stop)
    assert "t" not in db.table_names()
    # replace that fails keeps the old rows
    load(db, p, "t", "new")
    bad = write(tmp_path, json.dumps([{"n": "abc"}]), "bad.json")
    with pytest.raises(ImportFileError, match="cannot store"):
        load(db, bad, "t", "replace")
    assert len(rows(db, "t")) == 10


@pytest.mark.parametrize("text,msg", [("", "empty"), ("   ", "no records"), ("[]", "no records"), ("[1, 2]", "JSON object"), ("42", "must start"),
                                      ('{"a": 1}', "No array of objects"), ('[{"a": 1},', "Invalid JSON"), ('[{"a": 1} {"a": 2}]', "expected"),
                                      ('{"a":[{"x":1}]}', None)])
def test_bad_input_gives_clear_errors(tmp_path, db, text, msg):
    p = tmp_path / "x.json"
    p.write_text(text)
    if msg is None:
        assert load(db, p, "t", "new")["rows"] == 1
        return
    with pytest.raises(ImportFileError, match="(?i)" + msg):
        if text == "":
            resolve_path(str(p))
        else:
            load(db, p, "t", "new")


def test_utf8_bom_and_non_utf8(tmp_path, db):
    assert load(db, write(tmp_path, json.dumps([{"n": "é"}], ensure_ascii=False), enc="utf-8-sig"), "t", "new")["rows"] == 1
    assert rows(db, "t")[0]["n"] == "é"
    with pytest.raises(ImportFileError, match="UTF-8"):
        load(db, write(tmp_path, json.dumps([{"n": "é"}], ensure_ascii=False), "l.json", "latin-1"), "u", "new")


def test_path_rules(tmp_path):
    for bad in ("relative/file.json", str(tmp_path / "x.txt"), str(tmp_path / "missing.json")):
        with pytest.raises(ImportFileError):
            resolve_path(bad)
    (tmp_path / "a.json").write_text("[]")
    assert resolve_path(str(tmp_path / "a.json")).name == "a.json"


def test_too_many_columns(tmp_path):
    with pytest.raises(ImportFileError, match="columns"):
        scan(write(tmp_path, json.dumps([{f"k{i}": 1 for i in range(ji.MAX_COLUMNS + 1)}])))


# ---- endpoints / background job -------------------------------------------------------------------------------
@pytest.fixture
def api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from agent_app import main
    monkeypatch.setattr(main, "db", Database(tmp_path / "api.db"))   # never the real database
    main._import_jobs.clear()
    return TestClient(main.app), {"X-Agent-Token": main.TOKEN}, main


def wait(c, jid, tries=200):
    import time
    for _ in range(tries):
        s = c.get(f"/api/import/file/{jid}").json()
        if s["phase"] not in ("scan", "load"):
            return s
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_import_endpoints_end_to_end(api, tmp_path):
    c, H, main = api
    p = write(tmp_path, json.dumps(REC))
    body = {"path": str(p), "table": "Findings 2026", "mode": "new"}
    assert c.post("/api/import/file", json=body).status_code == 403                       # token required
    assert c.post("/api/import/file/preview", json={"path": str(p)}).status_code == 403
    pv = c.post("/api/import/file/preview", json={"path": str(p)}, headers=H).json()
    assert pv["complete"] and pv["records_scanned"] == 2 and {"name": "meta_b_c", "type": "TEXT"} in pv["columns"]
    jid = c.post("/api/import/file", json=body, headers=H).json()["job_id"]
    s = wait(c, jid)
    assert s["phase"] == "done" and s["result"]["table"] == "findings_2026" and s["result"]["rows"] == 2
    assert main.db.query("select count(*) n from findings_2026")["rows"][0]["n"] == 2
    again = wait(c, c.post("/api/import/file", json=body, headers=H).json()["job_id"])
    assert again["phase"] == "error" and "already exists" in again["error"]               # errors come back as data, not a stuck job
    rep = wait(c, c.post("/api/import/file", json={**body, "table": "findings_2026", "mode": "replace"}, headers=H).json()["job_id"])
    assert rep["phase"] == "done" and rep["result"]["replaced_rows"] == 2


def test_import_endpoint_rejects_bad_paths_and_runs_one_job_at_a_time(api, tmp_path):
    c, H, main = api
    for bad in ("relative.json", str(tmp_path / "nope.json"), str(tmp_path / "x.txt"), "/etc/passwd"):
        r = c.post("/api/import/file", json={"path": bad, "table": "t"}, headers=H)
        assert r.status_code == 400, bad
    assert c.post("/api/import/file", json={"path": str(write(tmp_path, "[]")), "table": "", "mode": "new"}, headers=H).status_code == 422
    assert c.get("/api/import/file/unknown").status_code == 404
    main._import_jobs["busy"] = main.ImportJob("busy", "x", "t", "new")                    # a job that is still running
    p = write(tmp_path, json.dumps(REC))
    assert c.post("/api/import/file", json={"path": str(p), "table": "t"}, headers=H).status_code == 409


def test_cancel_endpoint_stops_a_running_import(api, tmp_path, monkeypatch):
    import threading
    c, H, main = api
    monkeypatch.setattr(ji, "BATCH", 1)
    gate = threading.Event()
    real = ji.Records.__iter__

    def slow(self):
        for i, rec in enumerate(real(self)):
            if i == 3:
                gate.wait(5)                                                            # hold the import mid-way until we cancel
            yield rec
    monkeypatch.setattr(ji.Records, "__iter__", slow)
    p = write(tmp_path, json.dumps([{"n": i} for i in range(50)]))
    jid = c.post("/api/import/file", json={"path": str(p), "table": "t"}, headers=H).json()["job_id"]
    import time
    time.sleep(0.3)
    assert c.post(f"/api/import/file/{jid}/cancel").status_code == 403
    assert c.post(f"/api/import/file/{jid}/cancel", headers=H).json() == {"cancelling": True}
    gate.set()
    assert wait(c, jid)["phase"] == "cancelled" and "t" not in main.db.table_names()


# ---- CSV ------------------------------------------------------------------------------------------------------
CSV = 'id,name,score,seen,zip,note\n1,"Smith, J",7.5,2026-01-02,007,"multi\nline"\n2,Lee,3,2026-02-03,012,\n\n3,Kim,,2026-03-04,345,x\n'


def test_csv_types_blanks_quotes_and_multiline(tmp_path, db):
    r = load(db, write(tmp_path, CSV, "t.csv"), "people", "new")
    assert r["rows"] == 3                                                 # the blank line is skipped
    types = {c["name"]: c["declared_type"] for c in r["columns"]}
    assert types == {"id": "INTEGER", "name": "TEXT", "score": "REAL", "seen": "DATE", "zip": "TEXT", "note": "TEXT"}
    got = rows(db, "people")
    assert got[0]["name"] == "Smith, J" and got[0]["note"] == "multi\nline" and got[0]["zip"] == "007"
    assert got[1]["note"] is None and got[2]["score"] is None             # blank cell = NULL


@pytest.mark.parametrize("sep", [",", ";", "\t", "|"])
@pytest.mark.parametrize("chunk", [1 << 20, 5])
def test_csv_separators_bom_and_crlf(tmp_path, db, monkeypatch, sep, chunk):
    monkeypatch.setattr(ji, "CHUNK", chunk)
    text = "\r\n".join([sep.join(['"Last, Name"', "age"]), sep.join(["a", "1"]), sep.join(["b", "2"])]) + "\r\n"
    r = load(db, write(tmp_path, text, "x.csv", "utf-8-sig"), "t", "new")
    assert [c["name"] for c in r["columns"]] == ["last_name", "age"] and [x["age"] for x in rows(db, "t")] == [1, 2]


def test_csv_matches_the_small_csv_import(tmp_path, db):
    """Same names and types as the existing in-browser CSV import, so both routes give the same table."""
    from agent_app.db.csv_import import parse_csv
    text = 'First Name,First Name,,Qty,Price,When\nA,B,c,1,1.5,2026-01-01T10:00:00\nD,E,f,2,2,2026-01-02T11:30:00\n'
    _, names, _, declared = parse_csv(text.encode())
    r = load(db, write(tmp_path, text, "x.csv"), "t", "new")
    assert [c["name"] for c in r["columns"]] == names and [c["declared_type"] for c in r["columns"]] == declared


def test_csv_errors_are_clear(tmp_path, db):
    with pytest.raises(ImportFileError, match="Line 3 has 3 fields, expected 2"):
        load(db, write(tmp_path, "a,b\n1,2\n3,4,5\n", "x.csv"), "t", "new")
    assert "t" not in db.table_names()
    with pytest.raises(ImportFileError, match="no data rows"):
        load(db, write(tmp_path, "a,b\n", "h.csv"), "t", "new")
    with pytest.raises(ImportFileError, match="UTF-8"):
        load(db, write(tmp_path, "a\né\n", "l.csv", "latin-1"), "t", "new")


def test_csv_huge_field_and_modes(tmp_path, db):
    big = "x" * 300_000                                                   # larger than Python's default 128 KB csv field limit
    p = write(tmp_path, f"id,body\n1,{big}\n2,short\n", "b.csv")
    assert load(db, p, "docs", "new")["rows"] == 2 and len(rows(db, "docs")[0]["body"]) == 300_000
    assert load(db, p, "docs", "append")["rows"] == 2 and len(rows(db, "docs")) == 4
    r = load(db, write(tmp_path, "id,body\n9,only\n", "c.csv"), "docs", "replace")
    assert r["replaced_rows"] == 4 and [x["id"] for x in rows(db, "docs")] == [9]
    with pytest.raises(ImportFileError, match="does not have"):
        load(db, write(tmp_path, "id,other\n1,2\n", "d.csv"), "docs", "append")
    assert [x["id"] for x in rows(db, "docs")] == [9]


def test_tsv_and_extension_rules(tmp_path):
    assert resolve_path(str(write(tmp_path, "a\tb\n1\t2\n", "x.tsv"))).suffix == ".tsv"
    with pytest.raises(ImportFileError):
        resolve_path(str(write(tmp_path, "a,b", "x.txt")))
