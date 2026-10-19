"""Tests for the local frontend: checkpoint pick, suggest logic, HTTP API."""
import glob
import json
import sys
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server as S

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


@pytest.fixture(scope="module")
def app():
    mapping, names, champs = S.load_champ_data()
    ckpt, score = S.find_best_ckpt()
    pred = S.Predictor(ckpt, device="cpu")
    S.APP.update(mapping=mapping, names_by_idx=names, champions=champs,
                 predictor=pred)
    return S.APP


def test_find_best_ckpt_picks_run1(app):
    ckpt = Path(app["predictor"].ckpt_path)
    assert "run_" in str(ckpt)
    assert app["predictor"].val_top5 >= 0.28, "expected run1's 0.281 score"


def test_suggest_empty_draft(app):
    picks = [None] * 10
    out = app["predictor"].suggest(picks, set(), app["mapping"])
    assert len(out["roles"]) == 10
    assert all(r == [] for r in out["roles"])   # nothing filled yet
    for team in ("blue", "red"):
        assert set(out[team]) == set(ROLES)
        for role in ROLES:
            cands = out[team][role]
            assert 1 <= len(cands) <= 5
            idxs = [c["idx"] for c in cands]
            assert len(set(idxs)) == len(idxs)
            probs = [c["prob"] for c in cands]
            assert probs == sorted(probs, reverse=True)
            assert all(0.0 <= p <= 1.0 for p in probs)
            assert all(isinstance(c.get("flex", False), bool)
                       for c in cands)
            assert all(app["names_by_idx"][i] for i in idxs)


def test_suggest_roles_and_exclusions(app):
    mapping = app["mapping"]
    riot_a = next(iter(mapping))               # pick into blue slot 0
    riot_b = next(r for r in mapping if r != riot_a)   # ban another
    picks = [None] * 10
    picks[0] = riot_a
    out = app["predictor"].suggest(picks, {mapping[riot_b]}, mapping)
    # filled slot -> likely-role distribution over the 5 roles
    rp = out["roles"][0]
    assert len(rp) == 5
    assert sorted(r["role"] for r in rp) == [0, 1, 2, 3, 4]
    probs = [r["prob"] for r in rp]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) == pytest.approx(1.0, abs=1e-3)
    assert out["roles"][1] == []               # empty slots: no role preds
    # picked/banned champs are never suggested anywhere
    for team in ("blue", "red"):
        for role in ROLES:
            idxs = {c["idx"] for c in out[team][role]}
            assert mapping[riot_a] not in idxs, \
                "picked champ leaked (%s %s)" % (team, role)
            assert mapping[riot_b] not in idxs, \
                "banned champ leaked (%s %s)" % (team, role)


def test_outputs_are_only_real_classes(app):
    """[MASK]/[PAD]/FLEX are INPUT-only vocabulary (model/README.md
    Sec.2): suggestions must always be real champions (< num_champs)
    and role predictions must always be one of the 5 real roles -
    the model never 'guesses' a special token as an output class."""
    pred = app["predictor"]
    assert pred.model.champ_head.out_features == pred.num_champs  # 173
    assert pred.model.role_head.out_features == S.NUM_ROLES       # 5, no FLEX
    mapping = app["mapping"]
    r0 = next(iter(mapping))
    r1 = next(r for r in mapping if r != r0)
    r2 = next(r for r in mapping if r not in (r0, r1))
    bans = {mapping[r2]}
    for picks in ([None] * 10,
                  [r0, None, r1, None, None, None, None, None, None, None]):
        out = pred.suggest(picks, bans, mapping)
        for s, entries in enumerate(out["roles"]):
            for e in entries:
                assert 0 <= e["role"] < S.NUM_ROLES        # never FLEX
            if picks[s] is not None:                       # filled slot:
                assert sorted(e["role"] for e in entries) == \
                    [0, 1, 2, 3, 4]
        for team in ("blue", "red"):
            for role in ROLES:
                for c in out[team][role]:
                    assert 0 <= c["idx"] < pred.num_champs  # never MASK/PAD


def test_http_api(app):
    httpd = S.ThreadingHTTPServer(("127.0.0.1", 0), S.Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/api/champions")
        r = conn.getresponse()
        assert r.status == 200
        data = json.loads(r.read())
        assert len(data["champions"]) == 173
        assert data["model"]["val_top5"] >= 0.28
        conn.request("GET", "/")
        r = conn.getresponse()
        assert r.status == 200 and b"Draft Assistant" in r.read()  # index.html
        body = json.dumps({"picks": [None] * 10, "bans": []})
        conn.request("POST", "/api/suggest", body=body,
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        assert r.status == 200
        sug = json.loads(r.read())
        for key in ("blue", "red"):
            assert set(sug[key]) == set(ROLES)
            for role in ROLES:
                assert 1 <= len(sug[key][role]) <= 5
        # bad payload -> 400
        conn.request("POST", "/api/suggest", body='{"picks": [1]}',
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        assert r.status == 400
        r.read()
        conn.close()
    finally:
        httpd.shutdown()


# --- finetune checkpoints + runtime model switching ---------------------

FINETUNE_GLOB = str(S.ROOT / "model" / "finetune" / "ckpt" / "run_*" / "*.pt")


def _finetune_paths():
    return sorted(glob.glob(FINETUNE_GLOB))


@pytest.fixture()
def httpd(app):
    """Live server for endpoint tests; predictor restored on teardown."""
    srv = S.ThreadingHTTPServer(("127.0.0.1", 0), S.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    orig_pred = app["predictor"]
    yield srv
    app["predictor"] = orig_pred
    srv.shutdown()


def _post(srv, route, body, ctype="application/json"):
    conn = HTTPConnection("127.0.0.1", srv.server_address[1], timeout=120)
    conn.request("POST", route, body=body, headers={"Content-Type": ctype})
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    return r.status, (json.loads(raw) if raw else None)


def test_predictor_loads_finetune_ckpt():
    paths = _finetune_paths()
    if not paths:
        pytest.skip("no finetune checkpoints on disk")
    mapping, _, _ = S.load_champ_data()
    pred = S.Predictor(paths[-1], device="cpu")
    assert pred.kind == "finetune"
    assert pred.stage is not None
    assert 0.0 <= pred.val_top5 <= 1.0
    out = pred.suggest([None] * 10, set(), mapping)
    # finetuned encoder still answers every role query
    assert len(out["roles"]) == 10 and all(r == [] for r in out["roles"])
    for team in ("blue", "red"):
        assert set(out[team]) == set(ROLES)
        for role in ROLES:
            cands = out[team][role]
            assert 1 <= len(cands) <= 5
            probs = [c["prob"] for c in cands]
            assert probs == sorted(probs, reverse=True)
            assert all(0.0 <= p <= 1.0 for p in probs)


def test_list_checkpoints_covers_both_kinds():
    cps = S.list_checkpoints()
    kinds = {c["kind"] for c in cps}
    assert {"pretrain", "finetune"} <= kinds
    assert all(not Path(c["path"]).is_absolute() for c in cps)
    assert all(c["label"].startswith("[" + c["kind"] + "]") for c in cps)


def test_resolve_ckpt_path_validation():
    with pytest.raises(ValueError):
        S.resolve_ckpt_path("../outside.pt")        # escapes the project
    with pytest.raises(ValueError):
        S.resolve_ckpt_path("README.md")            # not a checkpoint
    with pytest.raises(FileNotFoundError):
        S.resolve_ckpt_path("model/does/not/exist.pt")
    # CLI-style (confine=False) may point outside, but must still exist
    with pytest.raises(FileNotFoundError):
        S.resolve_ckpt_path(str(Path.home() / "nope.pt"), confine=False)


def test_load_endpoint_switches_model(app, httpd):
    paths = _finetune_paths()
    if not paths:
        pytest.skip("no finetune checkpoints on disk")
    rel = str(Path(paths[-1]).relative_to(S.ROOT))
    status, d = _post(httpd, "/api/load", json.dumps({"path": rel}))
    assert status == 200, d
    assert d["model"]["kind"] == "finetune"
    assert app["predictor"].kind == "finetune"
    # suggestions keep flowing through the API with the new weights
    status, sug = _post(httpd, "/api/suggest",
                        json.dumps({"picks": [None] * 10, "bans": []}))
    assert status == 200, sug
    assert sug["model"]["kind"] == "finetune"
    for key in ("blue", "red"):
        assert set(sug[key]) == set(ROLES)
    # guard rails: path escape -> 400, missing file -> 404
    status, _ = _post(httpd, "/api/load", json.dumps({"path": "../evil.pt"}))
    assert status == 400
    status, _ = _post(httpd, "/api/load",
                      json.dumps({"path": "model/nope/missing.pt"}))
    assert status == 404


def test_upload_endpoint(app, httpd):
    # wrong extension is rejected (body consumed, nothing written)
    status, _ = _post(httpd, "/api/upload?name=evil.txt", b"x")
    assert status == 400
    # garbage bytes -> 400 and the file does not survive
    status, d = _post(httpd, "/api/upload?name=unit_garbage.pt",
                      b"not a checkpoint")
    assert status == 400, d
    assert not (S.UPLOAD_DIR / "unit_garbage.pt").exists()
    # real round trip with the smallest finetune checkpoint on disk
    paths = sorted(_finetune_paths(), key=lambda p: Path(p).stat().st_size)
    if not paths:
        pytest.skip("no finetune checkpoints on disk")
    dest = S.UPLOAD_DIR / "unit_roundtrip.pt"
    try:
        status, d = _post(httpd, "/api/upload?name=unit_roundtrip.pt",
                          Path(paths[0]).read_bytes())
        assert status == 200, d
        assert d["model"]["kind"] == "finetune"
        assert d["model"]["path"] == str(dest.relative_to(S.ROOT))
        assert dest.exists()
        assert app["predictor"].ckpt_path == str(dest)
        # shows up in the dropdown listing afterwards
        listed = {c["path"] for c in S.list_checkpoints()}
        assert str(dest.relative_to(S.ROOT)) in listed
    finally:
        dest.unlink(missing_ok=True)


# --- dDragon art + real-draft walkthrough --------------------------------

def _get(srv, path):
    conn = HTTPConnection("127.0.0.1", srv.server_address[1], timeout=120)
    conn.request("GET", path)
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    return r.status, (json.loads(raw) if raw else None)


def test_champion_images_are_ddragon(app):
    champs = app["champions"]
    assert len(champs) == 173
    prefix = ("https://ddragon.leagueoflegends.com/cdn/"
              + S.DDRAGON_VERSION + "/img/champion/")
    for c in champs:
        assert c["img"].startswith(prefix) and c["img"].endswith(".png")
    by_name = {c["name"]: c for c in champs}
    # manifest and offline-fallback must agree on the tricky renames
    assert by_name["Aatrox"]["img"].endswith("/Aatrox.png")
    assert by_name["LeBlanc"]["img"].endswith("/Leblanc.png")
    assert by_name["Wukong"]["img"].endswith("/MonkeyKing.png")
    assert by_name["Nunu & Willump"]["img"].endswith("/Nunu.png")
    assert by_name["Renata Glasc"]["img"].endswith("/Renata.png")


def test_drafts_endpoints(app, httpd):
    total, drafts = S.list_drafts(limit=5)
    assert total >= 40000 and len(drafts) == 5
    dates = [d["date"] for d in drafts]
    assert dates == sorted(dates, reverse=True)      # newest first
    one = drafts[0]
    for key in ("i", "date", "league", "blue", "red", "game_number",
                "patch", "first_pick", "winner"):
        assert key in one
    # filter by league / team
    total2, hits = S.list_drafts(q=one["league"], limit=50)
    assert total2 >= 1 and any(h["i"] == one["i"] for h in hits)
    # full game: 20 moves, 6 bans -> 6 picks -> 4 bans -> 4 picks
    d0 = S.get_draft(one["i"])
    kinds = [m["kind"] for m in d0["moves"]]
    assert kinds == ["ban"] * 6 + ["pick"] * 6 + ["ban"] * 4 + ["pick"] * 4
    for m in d0["moves"]:
        assert m["side"] in (0, 1)
        assert m["idx"] in set(app["mapping"].values())
        assert m["riot"] in app["mapping"]
        assert m["name"] == app["names_by_idx"][m["idx"]]
        if m["kind"] == "pick":
            assert m["role"] in range(5)
        else:
            assert m["role"] is None
    with pytest.raises(FileNotFoundError):
        S.get_draft(10 ** 9)
    # HTTP layer
    status, d = _get(httpd, "/api/drafts?limit=3")
    assert status == 200 and len(d["drafts"]) == 3 and d["total"] >= 40000
    status, d = _get(httpd, "/api/draft?i=%d" % one["i"])
    assert status == 200 and d["i"] == one["i"] and len(d["moves"]) == 20
    status, d = _get(httpd, "/api/drafts?random=1&limit=1")
    assert status == 200 and len(d["drafts"]) == 1
    status, _ = _get(httpd, "/api/draft?i=abc")
    assert status == 404
