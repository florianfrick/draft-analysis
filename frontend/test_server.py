"""Tests for the local frontend: checkpoint pick, suggest logic, HTTP API."""
import glob
import json
import re
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


def _best_available_score():
    """The champ_top5 find_best_ckpt is supposed to pick, from the files."""
    import torch
    best = 0.0
    pattern = str(S.ROOT / "model" / "pretraining" / "ckpt" / "run_*" / "best.pt")
    for path in glob.glob(pattern):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001  unreadable ckpt is not this test
            continue
        best = max(best, float(payload.get("metrics", {})
                               .get("champ_top5", -1.0)))
    return best


def test_find_best_ckpt_picks_the_best_available(app):
    """The default must be the highest val champ_top5 present, not run1.

    Hard-coding 0.281 tied this test to one checkpoint that has since been
    deleted; the contract is "max over ckpt/", which survives churn.
    """
    ckpt = Path(app["predictor"].ckpt_path)
    assert "run_" in str(ckpt)
    expected = _best_available_score()
    assert expected > 0.0, "no readable pretraining checkpoint under ckpt/"
    assert app["predictor"].val_top5 == pytest.approx(expected, abs=1e-6)


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


def test_role_assignment_is_one_to_one(app):
    """The role board puts each pick in exactly ONE role per team.

    The role head is per-slot (`Linear(D, 5)`), so two picks can argmax the
    same role. `_assign_roles` solves an injective assignment and the UI reads
    `assignment` for both the pick badges and the role board - a collision
    would draw two champions in one slot and leave another role empty.
    """
    mapping = app["mapping"]
    riot = list(mapping)[:5]                    # a full blue team
    picks = riot + [None] * 5
    out = app["predictor"].suggest(picks, set(), mapping)
    asg = out["assignment"]
    assert len(asg) == 10
    assert sorted(asg[:5]) == [0, 1, 2, 3, 4]   # full team -> every role once
    assert asg[5:] == [None] * 5                # red untouched
    for s in range(5):                          # assigned role was predicted
        assert asg[s] in {r["role"] for r in out["roles"][s]}


def test_assign_roles_breaks_collisions():
    """Two picks whose argmax is the same role must still land in two
    different slots. The assignment maximises the SUM of role probs over
    injective assignments, so the weaker pick takes its runner-up role
    instead of duplicating the stronger one's argmax."""
    def dist(pairs):
        return [{"role": r, "prob": p} for r, p in pairs]
    role_dists = [None] * 10
    filled = [False] * 10
    role_dists[0] = dist([(0, 0.60), (1, 0.30), (2, 0.10)])   # argmax TOP
    role_dists[1] = dist([(0, 0.55), (1, 0.40), (2, 0.05)])   # argmax TOP too
    filled[0] = filled[1] = True
    out = S._assign_roles(role_dists, filled)
    assert out[0] != out[1], "collision: both picks assigned the same role"
    assert sorted(out[:2]) == [0, 1]            # 0.60+0.40 beats 0.30+0.55
    assert out[0] == 0 and out[1] == 1          # stronger pick keeps its argmax
    assert out[2:] == [None] * 8                # empty slots get no role


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
        assert data["model"]["val_top5"] == pytest.approx(
            _best_available_score(), abs=1e-6)
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


# --- next-move policy models --------------------------------------------

POLICY_GLOB = str(S.ROOT / "model" / "policy" / "ckpt"
                  / "run_*" / "best_policy.pt")


@pytest.fixture()
def policy_ckpt():
    hits = sorted(glob.glob(POLICY_GLOB))
    if not hits:
        pytest.skip("no policy checkpoints on disk")
    return hits[-1]


def _empty_board(first_pick=0):
    return json.dumps({"picks": [None] * 10, "bans": [],
                       "first_pick": first_pick})


def test_policy_checkpoint_switches_kind(app, policy_ckpt):
    """A best_policy.pt must load as a next-move model, not an encoder."""
    pred = S.load_predictor(Path(policy_ckpt))
    assert pred.kind == "policy"
    info = S.model_info(pred)
    assert info["kind"] == "policy"
    assert info["metric"] == "next-move top5"
    # policy checkpoints report val next-move top5, not champ_top5
    assert info["val_top5"] is None or 0.0 <= info["val_top5"] <= 1.0
    assert info["ckpt"].endswith("best_policy.pt")


def test_next_move_endpoint_returns_a_ban_decision(app, httpd, policy_ckpt):
    S.load_predictor(Path(policy_ckpt))
    status, d = _post(httpd, "/api/next-move", _empty_board())
    assert status == 200, d
    st = d["state"]
    assert (st["move_idx"], st["kind"], st["side"]) == (0, "ban", 0)
    assert st["phase"] == "bans1" and st["complete"] is False
    # suggestions: real champions with names, descending, legal
    assert d["suggestions"], "expected next-move suggestions"
    probs = [c["prob"] for c in d["suggestions"]]
    assert probs == sorted(probs, reverse=True)
    for c in d["suggestions"]:
        assert 0 <= c["idx"] < 173 and c["name"]
    # roles stay in the shape the masked panel already renders
    assert len(d["roles"]) == 10
    assert d["roles"][0] == []                       # nothing picked yet
    # ban panel: denial-ranked candidates from the rollout
    bans = d["bans"]
    assert bans and bans["candidates"] and bans["ranked_by"] == "denial"
    assert [c["rank"] for c in bans["candidates"]] == \
        list(range(1, len(bans["candidates"]) + 1))
    denials = [c["denial"] for c in bans["candidates"]]
    assert denials == sorted(denials, reverse=True)
    for c in bans["candidates"]:
        assert 0 <= c["idx"] < 173 and c["name"]
        assert c["win_prob"] is None or 0.0 <= c["win_prob"] <= 1.0
    for c in bans["opponent_picks"]:
        assert 0 <= c["idx"] < 173 and c["name"]


def test_next_move_reports_a_pick_move_without_a_ban_panel(app, httpd,
                                                           policy_ckpt):
    S.load_predictor(Path(policy_ckpt))
    body = json.dumps({"picks": [None] * 10,
                       "bans": [11, 12, 13, 14, 15, 16],
                       "first_pick": 0})
    status, d = _post(httpd, "/api/next-move", body)
    assert status == 200, d
    assert d["state"]["kind"] == "pick"
    assert d["state"]["move_idx"] == 6
    assert d["bans"] is None
    # the ban panel can be skipped entirely for speed
    status, d = _post(httpd, "/api/next-move",
                      json.dumps({"picks": [None] * 10, "bans": [],
                                  "rollout": False}))
    assert status == 200 and d["bans"] is None


def test_next_move_honours_first_pick(app, httpd, policy_ckpt):
    S.load_predictor(Path(policy_ckpt))
    status, d = _post(httpd, "/api/next-move", _empty_board(first_pick=1))
    assert status == 200, d
    assert d["state"]["side"] == 1 and d["state"]["first_pick"] == 1


def test_next_move_rejects_a_masked_model(app, httpd):
    S.load_predictor(S.find_best_ckpt()[0])
    status, d = _post(httpd, "/api/next-move", _empty_board())
    assert status == 400 and "not a next-move policy" in d["error"]


def test_suggest_rejects_a_policy_model(app, httpd, policy_ckpt):
    S.load_predictor(Path(policy_ckpt))
    status, d = _post(httpd, "/api/suggest",
                      json.dumps({"picks": [None] * 10, "bans": []}))
    assert status == 400 and "next-move" in d["error"]


def test_next_move_rejects_an_illegal_board(app, httpd, policy_ckpt):
    S.load_predictor(Path(policy_ckpt))
    # same champion picked twice
    picks = [None] * 10
    picks[0] = picks[5] = next(iter(app["mapping"]))
    status, d = _post(httpd, "/api/next-move",
                      json.dumps({"picks": picks, "bans": []}))
    assert status == 400 and "twice" in d["error"]


def _banned_board(app, n=6):
    """A legal board at move n: n bans, nothing picked."""
    bans = list(app["mapping"])[:n]
    return json.dumps({"picks": [None] * 10, "bans": bans, "first_pick": 0})


def test_explain_returns_evidence_without_an_llm_key(app, httpd, policy_ckpt,
                                                     monkeypatch):
    """No key must degrade to retrieval, not to an error: the evidence is the
    part that has to be right, and it is useful on its own."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    S.load_predictor(Path(policy_ckpt))
    status, d = _post(httpd, "/api/explain", _banned_board(app))
    assert status == 200, d
    assert d["llm"]["configured"] is False
    assert d["explanation"] is None
    assert "LLM_API_KEY" in d["llm"]["reason"]
    ret = d["retrieval"]
    assert ret["n_indexed"] > 1000
    assert ret["games"], "expected comparable boards from the pro corpus"
    for g in ret["games"]:
        assert g["gameid"] and g["next_kind"] in ("ban", "pick")
        assert isinstance(g["next_champ"], str) and g["next_champ"]
        assert 0.0 <= g["overlap"] <= 1.0
    overlaps = [g["overlap"] for g in ret["games"]]
    assert overlaps == sorted(overlaps, reverse=True)
    # the board echo is what the prompt is built from, so it must be complete
    assert d["board"]["move_idx"] == 6
    assert len(d["board"]["bans"]) == 6


class _StubClient:
    """A client in the shape `llm.explain` expects: callable + `.model`."""
    model = "stub-1"

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, messages):
        return self.fn(messages)


def test_explain_returns_the_reasoning_the_ui_renders(app, httpd, policy_ckpt,
                                                      monkeypatch, tmp_path):
    """The panel shows `explanation.reasoning` plus the evidence it resolves
    against, so both have to survive the HTTP round trip."""
    S.load_predictor(Path(policy_ckpt))
    monkeypatch.setattr(S.llm, "CACHE_DIR", tmp_path / "cache")

    def fake_client(messages):
        # Cite the first game the server actually retrieved and recommend its
        # own top candidate, so the answer passes the validator: the test is
        # about transport, not leniency. Read the *first* user turn - a retry
        # appends the rejection note, which has neither list in it.
        text = messages[1]["content"]
        gid = re.search(r"^  - (\S+) \| patch", text, re.M).group(1)
        cand = re.search(r"^  - (.+?)  [\d.]+%$", text, re.M).group(1)
        return json.dumps({
            "reasoning": ["Their board is light on engage.",
                          "%s answered it with a pick-first support." % gid],
            "recommend": cand, "why": "x", "their_answer": "Azir",
            "risk": "Falls off.", "evidence": [{"type": "game", "id": gid}]})

    monkeypatch.setattr(S.llm, "client_from_env",
                        lambda env=None: _StubClient(fake_client))
    status, d = _post(httpd, "/api/explain", _banned_board(app))
    assert status == 200, d
    exp = d["explanation"]
    assert exp["ok"] is True, exp.get("problems")
    assert len(exp["reasoning"]) == 2
    assert exp["reasoning"][1].endswith("support.")
    cited = exp["evidence"][0]["id"]
    assert cited in {g["gameid"] for g in d["retrieval"]["games"]}
    assert d["llm"]["configured"] is True


def test_explain_rejects_a_masked_model(app, httpd):
    ckpt, _score = S.find_best_ckpt()
    S.APP["predictor"] = S.Predictor(ckpt, device="cpu")
    status, d = _post(httpd, "/api/explain", _empty_board())
    assert status == 400 and "policy" in d["error"]
