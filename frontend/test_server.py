"""Tests for the local frontend: checkpoint pick, suggest logic, HTTP API."""
import glob
import json
import re
import shutil
import sys
import tempfile
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
                 masked=pred)
    S.APP.pop("policy", None)      # role board only unless a test loads one
    return S.APP


def _best_available_score():
    """The champ_top5 find_best_ckpt is supposed to pick, from the files."""
    import torch
    best = 0.0
    for pattern in S.CKPT_GLOB:
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
    deleted; the contract is "max over both checkpoint stores", which survives
    churn.
    """
    ckpt = Path(app["masked"].ckpt_path)
    assert "checkpoints" in ckpt.as_posix()
    expected = _best_available_score()
    assert expected > 0.0, "no readable checkpoint under either store"
    assert app["masked"].val_top5 == pytest.approx(expected, abs=1e-6)


def test_suggest_empty_draft(app):
    picks = [None] * 10
    out = app["masked"].suggest(picks, set(), app["mapping"])
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
    out = app["masked"].suggest(picks, {mapping[riot_b]}, mapping)
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
    out = app["masked"].suggest(picks, set(), mapping)
    asg = out["assignment"]
    assert len(asg) == 10
    assert sorted(asg[:5]) == [0, 1, 2, 3, 4]   # full team -> every role once
    assert asg[5:] == [None] * 5                # red untouched
    for s in range(5):                          # assigned role was predicted
        assert asg[s] in {r["role"] for r in out["roles"][s]}


def test_assign_roles_breaks_collisions():
    """Two picks whose argmax is the same role must still land in two
    different slots - but the NEWEST pick wins the contested role.

    This is the "locking in" contract. A pure sum-of-probabilities optimum
    would give slot 0 TOP (0.60 + 0.40 > 0.30 + 0.55), so the pick the user
    just placed would slide to its runner-up and the role it was picked for
    would keep showing its old occupant. Pinning `newest` first is what
    makes a fresh pick land where the model thinks it belongs.
    """
    def dist(pairs):
        return [{"role": r, "prob": p} for r, p in pairs]
    role_dists = [None] * 10
    filled = [False] * 10
    role_dists[0] = dist([(0, 0.60), (1, 0.30), (2, 0.10)])   # argmax TOP
    role_dists[1] = dist([(0, 0.55), (1, 0.40), (2, 0.05)])   # argmax TOP too
    filled[0] = filled[1] = True

    out = S._assign_roles(role_dists, filled)                 # no hint
    assert out[0] != out[1], "collision: both picks assigned the same role"
    assert sorted(out[:2]) == [0, 1]
    assert out[2:] == [None] * 8                # empty slots get no role

    out = S._assign_roles(role_dists, filled, newest=1)       # slot 1 just picked
    assert out[1] == 0, "the newest pick must keep its own argmax"
    assert out[0] == 1, "the older pick moves to its runner-up"
    assert out[2:] == [None] * 8


def test_newest_pick_takes_the_role_it_was_picked_for():
    """Pantheon JUNGLE .60 / TOP .30, then Vi JUNGLE .97: the board must put
    Vi in JUNGLE, not keep Pantheon there because it was picked first."""
    def dist(pairs):
        return [{"role": r, "prob": p} for r, p in pairs]
    role_dists = [None] * 10
    filled = [False] * 10
    role_dists[5] = dist([(1, 0.60), (0, 0.30), (2, 0.10)])   # Pantheon
    role_dists[6] = dist([(1, 0.97), (0, 0.02), (3, 0.01)])   # Vi
    filled[5] = filled[6] = True
    out = S._assign_roles(role_dists, filled, newest=6)
    assert out[6] == 1, "Vi was just picked for JUNGLE"
    assert out[5] == 0, "Pantheon moves to TOP, its runner-up"


def test_last_slot_wins_its_argmax(app):
    """`last_slot` (the slot the user just filled) is pinned to its own
    argmax, whatever the rest of the team would prefer jointly."""
    mapping = app["mapping"]
    riot = list(mapping)[:3]
    picks = riot + [None] * 7
    for slot in (0, 1, 2):
        out = app["masked"].suggest(picks, set(), mapping, last_slot=slot)
        asg = out["assignment"]
        assert asg[slot] == out["roles"][slot][0]["role"], \
            "slot %d did not keep its argmax" % slot
        assert sorted(x for x in asg[:3] if x is not None) == \
            sorted(set(asg[:3])), "assignment collided"


def test_suggest_rejects_a_bad_last_slot(app, httpd):
    body = json.dumps({"picks": [None] * 10, "bans": [], "last_slot": 12})
    status, d = _post(httpd, "/api/suggest", body)
    assert status == 400 and "last_slot" in d["error"]


def test_outputs_are_only_real_classes(app):
    """[MASK]/[PAD]/FLEX are INPUT-only vocabulary (solo_queue/model/README.md
    Sec.2): suggestions must always be real champions (< num_champs)
    and role predictions must always be one of the 5 real roles -
    the model never 'guesses' a special token as an output class."""
    pred = app["masked"]
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


# --- draft order + the next-move highlight ------------------------------

def test_champions_endpoint_publishes_the_move_plan(app, httpd):
    """The green highlight is computed on the client from /api/champions'
    move_plan. It must be exactly infer.move_plan for both first-pick sides,
    so the highlight can never disagree with the server's next-move state.
    """
    status, d = _get(httpd, "/api/champions")
    assert status == 200
    plan = d["move_plan"]
    assert set(plan) == {"0", "1"}
    for fp in (0, 1):
        expected = [["ban" if kind == S.infer.MOVE_BAN else "pick", int(team)]
                    for kind, team in S.infer.move_plan(fp)]
        assert plan[str(fp)] == expected
        kinds = [m[0] for m in plan[str(fp)]]
        assert kinds == ["ban"] * 6 + ["pick"] * 6 \
            + ["ban"] * 4 + ["pick"] * 4
        # phase 1 starts with first pick, phase 2 with the other side
        assert plan[str(fp)][0][1] == fp
        assert plan[str(fp)][6][1] == fp
        assert plan[str(fp)][12][1] == 1 - fp
        assert plan[str(fp)][16][1] == 1 - fp


def test_turn_highlight_is_board_derived():
    """renderTurn must read the synchronous board-derived turn, never the
    async policy answer: that answer belongs to the previous board, which
    is what made the highlight lag one move.
    """
    src = (Path(__file__).resolve().parent / "static" / "app.js").read_text(
        encoding="utf-8")
    start = src.index("function renderTurn()")
    end = src.find("\nfunction ", start + 1)
    body = src[start:end if end > 0 else len(src)]
    assert "presp.state" not in body, \
        "renderTurn must not read the previous board's next-move state"
    assert "nextTurn(currentFirstPick())" in body


# --- finetune checkpoints + runtime model switching ---------------------

# The finetune stage lives in the solo-queue track; the split moved it out of
# `model/checkpoints/`. Both globs in this file named the old location and so
# skipped instead of failing - see the note on the policy_ckpt fixture.
FINETUNE_GLOB = str(S.SOLOQ_CKPT_DIR / "masking-finetune" / "*.pt")


def _finetune_paths():
    return sorted(glob.glob(FINETUNE_GLOB))


@pytest.fixture()
def httpd(app):
    """Live server for endpoint tests; predictor restored on teardown."""
    srv = S.ThreadingHTTPServer(("127.0.0.1", 0), S.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    orig = {slot: app.get(slot) for slot in ("masked", "policy")}
    yield srv
    for slot, pred in orig.items():
        if pred is None:
            app.pop(slot, None)
        else:
            app[slot] = pred
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
    assert "checkpoints" in kinds
    assert all(not Path(c["path"]).is_absolute() for c in cps)
    # the kept artefacts are labelled by directory name, which is the name
    # that states the inference; uploads keep the family tag
    assert all(c["label"].startswith("[%s]" % c["kind"])
               for c in cps if c["kind"] != "checkpoints")
    assert all(c["label"].startswith(Path(c["path"]).parent.name + "/")
               for c in cps if c["kind"] == "checkpoints")
    assert all(set(c["roles"]) <= {"masked", "policy"} for c in cps)


def test_ckpt_roles_follow_the_name():
    """A dropdown must not offer a policy checkpoint under "role" (or the
    reverse): loading it would fail, and most of the list would fail.

    The kept artefacts are named for the inference they serve, so the name is
    the whole rule. Paths from before the rename (or an upload) fall back to
    the family rules, which is why those cases are still asserted.
    """
    # the name decides
    assert S._ckpt_roles("pro/model/checkpoints/masking-proonly/best.pt",
                         "checkpoints") == ["masked"]
    assert S._ckpt_roles("solo_queue/model/checkpoints/masking-finetune/best_mask.pt",
                         "checkpoints") == ["masked"]
    assert S._ckpt_roles("solo_queue/model/checkpoints/masking-soloqueue/best.pt",
                         "checkpoints") == ["masked"]
    assert S._ckpt_roles("solo_queue/model/checkpoints/nextpick-policy/best_policy.pt",
                         "checkpoints") == ["policy"]
    assert S._ckpt_roles("pro/model/checkpoints/nextpick-proonly/best.pt",
                         "checkpoints") == ["policy"]
    # a value artefact fills neither slot
    assert S._ckpt_roles("pro/model/checkpoints/value-x/best.pt",
                         "checkpoints") == []
    # legacy families, still reachable by uploading an old checkpoint
    assert S._ckpt_roles("solo_queue/model/pretraining/ckpt/run_1/best.pt",
                         "pretrain") == ["masked"]
    assert S._ckpt_roles("solo_queue/model/finetune/ckpt/run_1/best_mask.pt",
                         "finetune") == ["masked"]
    assert S._ckpt_roles("solo_queue/model/policy/ckpt/run_1/best_policy.pt",
                         "policy") == ["policy"]
    assert S._ckpt_roles(
        "pro/model/proonly/ckpt/run_20261101_182415_m-patch-s0/best.pt",
        "proonly") == ["masked"]
    assert S._ckpt_roles(
        "pro/model/proonly/ckpt/run_20261101_183519_p-patch-s0/best.pt",
        "proonly") == ["policy"]
    # unknown (an upload) is offered in both and validated on load
    assert S._ckpt_roles("frontend/uploaded/x.pt", "uploaded") == \
        ["masked", "policy"]


def test_resolve_ckpt_path_validation():
    with pytest.raises(ValueError):
        S.resolve_ckpt_path("../outside.pt")        # escapes the project
    with pytest.raises(ValueError):
        S.resolve_ckpt_path("README.md")            # not a checkpoint
    with pytest.raises(FileNotFoundError):
        S.resolve_ckpt_path("pro/model/does/not/exist.pt")
    # CLI-style (confine=False) may point outside, but must still exist
    with pytest.raises(FileNotFoundError):
        S.resolve_ckpt_path(str(Path.home() / "nope.pt"), confine=False)


# --------------------------------------------------------------- ensembles
def test_ensemble_manifest_is_still_servable():
    """The proto ensemble is no longer the default, but must stay loadable.

    Its members alone are no better than the checkpoint it replaced, so the
    manifest is the only thing carrying that measurement; a future cleanup that
    drops or flattens it would lose the +1.33 pp claim. Re-run
    `pro/model/proto/compare_shipped.py` before deleting anything here.

    The checkpoint stores are git-ignored (weights), so only the code-level half
    is asserted unconditionally; the on-disk half skips on a fresh clone, like
    the other checkpoint tests.
    """
    assert S.ENSEMBLE_CKPT.suffix == ".json"
    if not S.ENSEMBLE_CKPT.is_file():
        pytest.skip("ensemble manifest not on disk (checkpoints are ignored)")
    members, ev, _ = S.read_ensemble(S.ENSEMBLE_CKPT)
    assert len(members) >= 2
    assert all(m.is_file() and m.suffix == ".pt" for m in members)
    assert ev.get("top5") and ev.get("n_drafts")


def test_policy_default_is_the_ban_aware_arm():
    """A tripwire, not a law: this pins the default the project chose.

    Two ways to get the default wrong and both look fine at runtime - point it
    at a ban-blind checkpoint, or pass the flags at construction time instead of
    reading them from the checkpoint (which serves a model whose weights were
    trained with different modules present). So the assertion is on the *built*
    model, which is where infer.py's config read lands.

    If a better next-move model ships, update this test and the three docs that
    name the default (`README.md`, `STATUS.md` §7, `frontend/README.md`)
    together. Failing here means the artefact and the documentation disagree,
    which is the thing worth catching.
    """
    assert S.POLICY_DEFAULT.suffix == ".pt", "the default is a checkpoint now"
    assert "bans-aware" in S.POLICY_DEFAULT.parent.name
    if not S.POLICY_DEFAULT.is_file():
        pytest.skip("checkpoint not on disk (checkpoints are ignored)")
    model = S.make_predictor(S.POLICY_DEFAULT, device="cpu").models[0]
    assert model.ban_side is True
    assert model.ban_fearless is True
    assert model.counter_rank == 32
    assert hasattr(model, "counter") and hasattr(model, "ban_context")


def test_ensemble_predictor_averages_member_logits():
    """next_move must be the MEAN OF LOGITS, the quantity that was measured.

    Averaging probabilities, or voting, is a different model; this pins the
    contract so a refactor cannot quietly change what the numbers describe.
    """
    import torch
    if not S.ENSEMBLE_CKPT.is_file():
        pytest.skip("ensemble manifest absent")
    pred = S.make_predictor(S.ENSEMBLE_CKPT, device="cpu")
    assert pred.kind == "policy"
    assert len(pred.models) == len(S.read_ensemble(S.ENSEMBLE_CKPT)[0])

    picks = [1, 2, None, None, None, None, None, None, None, None]
    bans = [11, 12, 13, 14, 15, 16]
    out = pred.next_move(picks, bans, first_pick=0, k=173)
    # rebuild the identical board state (next_move fills roles first)
    board = S.infer.build_board(picks, bans, pred.num_champs, first_pick=0,
                                device="cpu")
    pred._fill_roles(board)
    mv, _, _ = S.infer.resolve_position(int(board["n_bans"][0]),
                                        int((~board["pad_mask"][0]).sum()), 0)
    kind, team = S.infer.move_plan(0)[mv]
    batch = S.infer._policy_batch(board, mv, kind, team,
                                  patch_id=S.infer.PATCH_UNK_ID)
    with torch.no_grad():
        logits = torch.stack([m(batch)["policy_logits"][0]
                              for m in pred.models]).mean(0)
    ref = torch.softmax(logits.float(), -1)
    got = {s["idx"]: s["prob"] for s in out["suggestions"]}
    assert max(abs(float(ref[i]) - p) for i, p in got.items()) < 1e-6


def test_read_ensemble_rejects_bad_manifests():
    """Members resolve relative to the manifest and must stay in the project.

    The manifest is written INSIDE the project because that is the only place
    a servable manifest can live - the containment rule is part of what is
    being tested, so a tmp_path outside the repo would pass for the wrong
    reason.
    """
    good = S.read_ensemble(S.ENSEMBLE_CKPT)[0][0]
    work = Path(tempfile.mkdtemp(prefix=".pytest_ens_", dir=str(S.ROOT)))
    try:
        man = work / "ensemble.json"
        man.write_text(json.dumps({"members": []}), "utf-8")
        with pytest.raises(ValueError):
            S.read_ensemble(man)

        man.write_text(json.dumps({"members": ["nope.pt"]}), "utf-8")
        with pytest.raises(FileNotFoundError):
            S.read_ensemble(man)

        man.write_text(json.dumps({"members": ["../../etc/passwd"]}), "utf-8")
        with pytest.raises(ValueError):
            S.read_ensemble(man)

        # a relative member resolves against the manifest's own directory
        link = work / "member.pt"
        try:
            link.symlink_to(good)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks need SeCreateSymbolicLinkPrivilege on Windows")
        man.write_text(json.dumps({"members": ["member.pt"]}), "utf-8")
        assert S.read_ensemble(man)[0] == [good]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_load_endpoint_switches_model(app, httpd):
    paths = _finetune_paths()
    if not paths:
        pytest.skip("no finetune checkpoints on disk")
    rel = str(Path(paths[-1]).relative_to(S.ROOT))
    status, d = _post(httpd, "/api/load", json.dumps({"path": rel}))
    assert status == 200, d
    assert d["model"]["kind"] == "finetune"
    assert app["masked"].kind == "finetune"
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
                      json.dumps({"path": "pro/model/nope/missing.pt"}))
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
        assert app["masked"].ckpt_path == str(dest)
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

POLICY_GLOB = str(S.PRO_CKPT_DIR / "nextpick-*" / "best.pt")


@pytest.fixture()
def policy_ckpt():
    """A working next-move model: the shipped default if present, else any.

    This fixture used to name one hand-picked path under `model/checkpoints/`.
    The two-track split removed that path, and because the fixture called
    pytest.skip instead of failing, 14 next-move tests quietly stopped running
    while the suite still reported green. Preferring POLICY_DEFAULT also means
    these tests exercise the model that actually ships.
    """
    if S.POLICY_DEFAULT.is_file():
        return str(S.POLICY_DEFAULT)
    hits = sorted(glob.glob(POLICY_GLOB))
    if not hits:
        pytest.skip("no policy checkpoints on disk")
    return hits[-1]


def _empty_board(first_pick=0):
    return json.dumps({"picks": [None] * 10, "bans": [],
                       "first_pick": first_pick})


def test_policy_checkpoint_switches_kind(app, policy_ckpt):
    """A `nextpick-*` checkpoint must load as a next-move model, not an encoder."""
    pred = S.load_predictor(Path(policy_ckpt))
    assert pred.kind == "policy"
    info = S.model_info(pred)
    assert info["kind"] == "policy"
    assert info["metric"] == "next-move top5"
    # policy checkpoints report val next-move top5, not champ_top5
    assert info["val_top5"] is None or 0.0 <= info["val_top5"] <= 1.0
    # The contract is the `nextpick-` prefix, not a literal filename: the
    # stores name artefacts `<stage>-<recipe>-through-<series>/best.pt`, and
    # pinning `best_policy.pt` here failed the moment the default moved.
    assert info["ckpt"].startswith("nextpick-")


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
    # one answer per board: the removed ban rollout must not come back
    assert "bans" not in d
    # ... and the answer says which patch row it conditioned on
    status, ch = _get(httpd, "/api/champions")
    want = next(p for p in ch["patches"] if p["key"] == S.infer.DEFAULT_PATCH)
    assert d["patch_id"] == want["id"]
    assert d["patch"] == S.infer.DEFAULT_PATCH


def test_next_move_reports_a_pick_move(app, httpd, policy_ckpt):
    S.load_predictor(Path(policy_ckpt))
    body = json.dumps({"picks": [None] * 10,
                       "bans": [11, 12, 13, 14, 15, 16],
                       "first_pick": 0})
    status, d = _post(httpd, "/api/next-move", body)
    assert status == 200, d
    assert d["state"]["kind"] == "pick"
    assert d["state"]["move_idx"] == 6
    assert "bans" not in d


class _StubReranker:
    """A reranker as the panel sees it: the head's list plus a choice.

    Only the shape matters here - the real one needs CUDA and Qwen, and what
    this file can test is whether the endpoint carries its two extra fields.
    """

    kind = "policy"
    accepts_lane = True
    val_top5 = None
    epoch = 1
    stage = None
    metric = "LLM k-way"
    rerank_eval = {"k": 5, "kway": 0.3531, "exact": 0.3465, "n": 439}
    ckpt_path = str(S.PRO_CKPT_DIR / "llm-lora-rerank5" / "projector.pt")

    def __init__(self):
        self.lanes = []

    def default_patch_id(self):
        return 106

    def patch_key_for(self, pid):
        return "16.20"

    def next_move(self, picks, bans, first_pick=0, k=8, unavailable=None,
                  patch_id=None, lane=None):
        self.lanes.append(lane)
        return {
            "move_idx": 6, "kind": "pick", "side": 0, "phase": "picks1",
            "first_pick": first_pick, "warning": None, "complete": False,
            "n_unavailable": 0, "patch_id": patch_id, "roles": [[]] * 10,
            "suggestions": [
                {"idx": 10, "prob": 0.42, "name": "A", "llm_prob": 0.55,
                 "llm_pick": False},
                {"idx": 11, "prob": 0.30, "name": "B", "llm_prob": 0.45,
                 "llm_pick": True},
                {"idx": 12, "prob": 0.10, "name": "C"},
            ],
            "rerank": {"k": 2, "chosen": "B", "chosen_idx": 11, "lane": lane,
                       "lane_name": "Bot", "lane_source": "user",
                       "lane_assumed": False, "lanes_open": [3, 4],
                       "head_top1_agrees": False, "candidates": []},
        }


def test_next_move_carries_the_rerank_block_and_the_lane(app, httpd):
    """The panel's contract: lane in, `rerank` + `llm_prob` out.

    Both halves are easy to lose in a refactor - `lane` is the only field the
    reranker reads that the head does not, and `rerank` is the only place the
    UI learns which candidate the LLM chose. Without them the panel silently
    renders the head's ranking with no LLM badge and no lane selector, which is
    exactly the state this test was written after finding.
    """
    stub = _StubReranker()
    app["policy"] = stub
    body = json.dumps({"picks": [None] * 10,
                       "bans": [11, 12, 13, 14, 15, 16],
                       "first_pick": 0, "lane": 3, "k": 5})
    status, d = _post(httpd, "/api/next-move", body)
    assert status == 200, d
    assert stub.lanes == [3], "the lane the client chose must reach the model"
    assert d["rerank"]["lane"] == 3
    assert d["rerank"]["chosen"] == "B"
    assert d["rerank"]["lane_source"] == "user"
    # the header reports the k-way number, never a borrowed "val top-5"
    assert d["model"]["family"] == "llm-rerank"
    assert d["model"]["val_top5"] is None
    chips = {c["idx"]: c for c in d["suggestions"]}
    assert chips[11]["llm_pick"] is True
    assert chips[10]["llm_prob"] == 0.55
    assert "llm_prob" not in chips[12], "outside the shortlist: head's number"


def test_a_head_never_receives_the_lane(app, httpd, policy_ckpt):
    """The panel sends `lane` unconditionally; a head raises on the keyword."""
    S.load_predictor(Path(policy_ckpt))
    body = json.dumps({"picks": [None] * 10,
                       "bans": [11, 12, 13, 14, 15, 16],
                       "first_pick": 0, "lane": 3})
    status, d = _post(httpd, "/api/next-move", body)
    assert status == 200, d
    assert d["rerank"] is None


def test_the_head_can_be_constrained_to_a_lane(policy_ckpt):
    """`allowed` masks the distribution, it does not filter the result.

    The difference is the whole fix: filtering an unconstrained top-5 leaves an
    empty list whenever the head's five favourites do not play the lane (22 % of
    pick moves), while masking produces a real ranking inside the pool.
    """
    import roles as ROLES
    pred = S.load_predictor(Path(policy_ckpt))
    lane = 3                                     # BOTTOM
    allowed = ROLES.legal_champions(lane)
    picks = [None] * 10
    bans = [11, 12, 13, 14, 15, 16]

    plain = pred.next_move(picks, bans, first_pick=0, k=8)
    constrained = pred.next_move(picks, bans, first_pick=0, k=8, allowed=allowed)

    ids = [c["idx"] for c in constrained["suggestions"]]
    assert ids, "masking to a lane must not empty the list"
    assert set(ids) <= set(allowed)
    assert len(ids) == 8
    # A real distribution over the pool, not a renormalised tail: the top
    # champion keeps a healthy share, and the order is still the head's.
    assert constrained["suggestions"][0]["prob"] > 0.05
    probs = [c["prob"] for c in constrained["suggestions"]]
    assert probs == sorted(probs, reverse=True)
    # ... and it is a different answer than the unconstrained one, or the lane
    # would be doing nothing.
    assert ids != [c["idx"] for c in plain["suggestions"]][:len(ids)] or \
        set(ids) != set(c["idx"] for c in plain["suggestions"])

    with pytest.raises(ValueError):
        pred.next_move(picks, bans, first_pick=0, k=8, allowed=[])


# ---------------------------------------------------------------- patch id
def test_champions_publishes_the_checkpoint_patch_table(app, httpd,
                                                        policy_ckpt):
    """The selector is built from the model's own vocabulary, not a list.

    A patch id is an embedding row, so the client must be offered exactly
    the rows this checkpoint was trained with.
    """
    S.load_predictor(Path(policy_ckpt))
    status, d = _get(httpd, "/api/champions")
    assert status == 200, d
    patches = d["patches"]
    assert patches == S.APP["policy"].patch_options()
    keys = [p["key"] for p in patches]
    assert keys[0] == S.infer.DEFAULT_PATCH        # newest first
    assert [p["id"] for p in patches] == list(range(len(keys), 0, -1))
    assert d["default_patch"] == S.infer.DEFAULT_PATCH
    assert d["default_patch_id"] == patches[0]["id"]


def test_next_move_honours_and_clamps_patch_id(app, httpd, policy_ckpt):
    S.load_predictor(Path(policy_ckpt))
    status, ch = _get(httpd, "/api/champions")
    older = ch["patches"][2]                       # a patch that is not 16.20
    body = json.dumps({"picks": [None] * 10, "bans": [],
                       "patch_id": older["id"]})
    status, d = _post(httpd, "/api/next-move", body)
    assert status == 200, d
    assert d["patch_id"] == older["id"] and d["patch"] == older["key"]
    # an id the checkpoint cannot embed becomes UNK, not a 400
    body = json.dumps({"picks": [None] * 10, "bans": [], "patch_id": 99999})
    status, d = _post(httpd, "/api/next-move", body)
    assert status == 200, d
    assert d["patch_id"] == S.infer.PATCH_UNK_ID and d["patch"] is None


def test_next_move_honours_first_pick(app, httpd, policy_ckpt):
    S.load_predictor(Path(policy_ckpt))
    status, d = _post(httpd, "/api/next-move", _empty_board(first_pick=1))
    assert status == 200, d
    assert d["state"]["side"] == 1 and d["state"]["first_pick"] == 1


# ------------------------------------------------------------ fearless
# `fearless` is a list of *riot ids* (the same space as `picks`/`bans`), so
# the client can send its pool without knowing the champion-index mapping.
def _riot_of(app, idx):
    return {v: k for k, v in app["mapping"].items()}[idx]


def test_next_move_excludes_the_fearless_pool_on_a_pick(app, httpd,
                                                        policy_ckpt):
    """A champion picked in an earlier game must drop out of a pick answer.

    The assertion is differential on purpose: it takes whatever the model
    actually ranked first, adds *that* to the pool, and checks it is gone.
    A fixed champion would pass or fail depending on the weights.
    """
    S.load_predictor(Path(policy_ckpt))
    body = {"picks": [None] * 10, "bans": [11, 12, 13, 14, 15, 16],
            "first_pick": 0}
    status, d = _post(httpd, "/api/next-move", json.dumps(body))
    assert status == 200, d
    assert d["state"]["kind"] == "pick"
    assert d["state"]["n_unavailable"] == 0
    top = d["suggestions"][0]["idx"]

    body["fearless"] = [_riot_of(app, top)]
    status, d2 = _post(httpd, "/api/next-move", json.dumps(body))
    assert status == 200, d2
    assert d2["state"]["n_unavailable"] == 1
    assert top not in [c["idx"] for c in d2["suggestions"]]


def test_next_move_excludes_fearless_champions_from_bans(app, httpd,
                                                         policy_ckpt):
    """The pool is masked on bans too, not only on picks.

    Differential on purpose: take whatever the model ranked first for the
    opening ban, add *that* to the pool, and require it to be gone. A fixed
    champion would pass or fail depending on the weights.
    """
    S.load_predictor(Path(policy_ckpt))
    status, d = _post(httpd, "/api/next-move", _empty_board())
    assert status == 200, d
    top = d["suggestions"][0]["idx"]
    body = json.dumps({"picks": [None] * 10, "bans": [], "first_pick": 0,
                       "fearless": [_riot_of(app, top)]})
    status, d2 = _post(httpd, "/api/next-move", body)
    assert status == 200, d2
    assert d2["state"]["kind"] == "ban"
    assert d2["state"]["n_unavailable"] == 1
    assert top not in [c["idx"] for c in d2["suggestions"]]


def test_a_board_that_bans_a_fearless_champion_still_loads(app, httpd,
                                                           policy_ckpt):
    """Technically legal, never done in practice - so no 400.

    The corpus contains 8 such bans, which is why the board keeps the pool
    separate from `taken`. The champion is already taken, so it cannot be
    suggested regardless.
    """
    S.load_predictor(Path(policy_ckpt))
    riot = _riot_of(app, 7)
    body = json.dumps({"picks": [None] * 10, "bans": [riot],
                       "first_pick": 0, "fearless": [riot]})
    status, d = _post(httpd, "/api/next-move", body)
    assert status == 200, d
    assert d["state"]["kind"] == "ban"
    assert d["state"]["n_unavailable"] == 1
    assert 7 not in [c["idx"] for c in d["suggestions"]]


def test_suggest_filters_the_fearless_pool(app, httpd):
    """The role model cannot consume the pool, so the server filters it.

    `fearless_filtered` is what the UI turns into its caveat, so it has to
    be non-zero here or the warning would never appear.
    """
    body = {"picks": [None] * 10, "bans": []}
    status, d = _post(httpd, "/api/suggest", json.dumps(body))
    assert status == 200, d
    assert d["fearless_filtered"] == 0
    cand = d["blue"]["TOP"][0]

    body["fearless"] = [_riot_of(app, cand["idx"])]
    status, d2 = _post(httpd, "/api/suggest", json.dumps(body))
    assert status == 200, d2
    assert d2["fearless_filtered"] == 1
    for team in ("blue", "red"):
        for _role, lst in d2[team].items():
            assert cand["idx"] not in [c["idx"] for c in lst]


def test_draft_endpoint_exposes_the_real_fearless_pool(app, httpd):
    """A real G2+ fearless game carries its own pool; the walkthrough seeds
    from it, so the endpoint has to publish it in the client's id space."""
    status, d = _get(httpd, "/api/drafts?limit=500")
    assert status == 200, d
    fear = [x for x in d["drafts"] if x.get("n_unavailable")]
    if not fear:
        pytest.skip("no fearless game on the first page of drafts")
    game = fear[0]
    status, g = _get(httpd, "/api/draft?i=%d" % game["i"])
    assert status == 200, g
    assert g["fearless"] is True
    assert len(g["unavailable"]) == game["n_unavailable"]
    assert g["unavailable"], "fearless game with an empty pool"
    for riot in g["unavailable"]:
        assert riot in app["mapping"], "pool must be in the client's id space"
    # nothing in the pool may also be picked in this game
    picked = [m["riot"] for m in g["moves"] if m["kind"] == "pick"]
    assert not (set(picked) & set(g["unavailable"]))


def test_next_move_without_a_policy_model_is_an_error(app, httpd):
    """The two slots are independent: loading only a role model must not
    silently make the next-move panel answer with it."""
    app.pop("policy", None)
    status, d = _post(httpd, "/api/next-move", _empty_board())
    assert status == 400 and "no next-move policy" in d["error"]


def test_both_models_answer_the_same_board(app, httpd, policy_ckpt):
    """The point of the two slots: one board, two answers.

    The role model keeps answering the role board while a policy model is
    loaded, and the policy keeps answering the next move while the role
    model is loaded - neither load evicts the other.
    """
    assert app["masked"].kind != "policy"
    S.load_predictor(Path(policy_ckpt))            # lands in the policy slot
    assert app["masked"].kind != "policy"
    assert app["policy"].kind == "policy"
    status, sug = _post(httpd, "/api/suggest",
                        json.dumps({"picks": [None] * 10, "bans": []}))
    assert status == 200, sug
    assert sug["model"]["kind"] != "policy"
    assert len(sug["roles"]) == 10 and sug["assignment"] == [None] * 10
    status, nm = _post(httpd, "/api/next-move", _empty_board())
    assert status == 200, nm
    assert nm["model"]["kind"] == "policy" and nm["state"]["kind"] == "ban"


def test_suggest_uses_the_head_when_the_next_move_is_a_pick(app, httpd,
                                                            policy_ckpt):
    """The role board is the next-move head with the pool masked to the lane.

    Two things have to hold at once: the answer says which model produced it,
    and every candidate is a champion that actually plays that lane - the mask
    is applied to the logits, so an unconstrained favourite cannot leak in.
    """
    from roles import legal_champions

    S.load_predictor(Path(policy_ckpt))
    # 10 bans, then three picks a side: the next move is a pick.
    mapping = app["mapping"]
    names = app["names_by_idx"]
    by_name = {n: i for i, n in enumerate(names)}
    inv = {i: r for r, i in mapping.items()}
    blue = [by_name[n] for n in ("Aatrox", "Vi", "Ahri")]
    red = [by_name[n] for n in ("Gnar", "Wukong", "Syndra")]
    picks = ([inv[i] for i in blue] + [None, None]
             + [inv[i] for i in red] + [None, None])
    bans = [inv[i] for i in range(10)]
    status, d = _post(httpd, "/api/suggest",
                      json.dumps({"picks": picks, "bans": bans,
                                  "first_pick": 0}))
    assert status == 200, d
    assert d["method"] == "head"
    assert d["model"]["kind"] == "policy"
    assert d["blue"] and d["red"], "both sides have open lanes"
    for side in ("blue", "red"):
        for role, cands in d[side].items():
            lane = ROLES.index(role)
            legal = set(legal_champions(lane))
            assert cands, "%s %s is empty" % (side, role)
            assert all(c["idx"] in legal for c in cands), (side, role)
            probs = [c["prob"] for c in cands]
            assert probs == sorted(probs, reverse=True)


def test_suggest_falls_back_to_the_masked_board_in_the_ban_phase(app, httpd,
                                                                policy_ckpt):
    """A ban-phase board has no in-distribution form of the head's question,
    so the role board's own checkpoint answers it - and says so.

    Six bans and six picks is the state where the next move is a ban again
    (moves 12-15); six bans with no picks is *not*, because move 6 is the
    first pick and the head answers it.
    """
    S.load_predictor(Path(policy_ckpt))
    mapping = app["mapping"]
    names = app["names_by_idx"]
    by_name = {n: i for i, n in enumerate(names)}
    inv = {i: r for r, i in mapping.items()}
    picks = ([inv[by_name[n]] for n in ("Aatrox", "Vi", "Ahri")]
             + [None, None]
             + [inv[by_name[n]] for n in ("Gnar", "Wukong", "Syndra")]
             + [None, None])
    status, d = _post(httpd, "/api/suggest", json.dumps({
        "picks": picks, "bans": [inv[i] for i in range(6)],
        "first_pick": 0}))
    assert status == 200, d
    assert d["method"] == "masked"
    # the masking checkpoint's kind is "pretrain" (it is the encoder + MASK
    # queries); what matters is that the *policy* did not answer
    assert d["model"]["kind"] != "policy"


def test_draft_score_reports_both_sides_and_their_picks(app, httpd,
                                                        policy_ckpt):
    """The panel needs a verdict, a per-side summary and one row per pick, in
    draft order - and the rows have to agree with the board it was given."""
    S.load_predictor(Path(policy_ckpt))
    names = app["names_by_idx"]
    mapping = app["mapping"]
    by_name = {n: i for i, n in enumerate(names)}
    inv = {i: r for r, i in mapping.items()}
    blue = [by_name[n] for n in ("Aatrox", "Vi", "Ahri", "Jinx", "Rakan")]
    red = [by_name[n] for n in ("Gnar", "Wukong", "Syndra", "Kai'Sa",
                                "Nautilus")]
    status, d = _post(httpd, "/api/draft-score", json.dumps({
        "picks": [inv[i] for i in blue + red],
        "bans_blue": [inv[i] for i in range(0, 5)],
        "bans_red": [inv[i] for i in range(5, 10)],
        "first_pick": 0}))
    assert status == 200, d
    assert d["verdict"] in ("blue", "red", "even")
    assert d["n_picks"] == 10
    assert d["blue"]["n"] == 5 and d["red"]["n"] == 5
    assert [p["move_idx"] for p in d["picks"]] == sorted(
        p["move_idx"] for p in d["picks"])
    assert {p["name"] for p in d["picks"]} == set(
        names[i] for i in blue + red)
    for p in d["picks"]:
        assert 0.0 <= p["share"] <= 1.0
        assert p["rank"] is None or p["rank"] >= 1


def test_suggest_without_a_role_model_is_an_error(app, httpd):
    """Neither slot loaded: the board cannot be filled at all, and the message
    has to name both checkpoints because either one can answer it now."""
    app.pop("masked", None)
    app.pop("policy", None)
    status, d = _post(httpd, "/api/suggest",
                      json.dumps({"picks": [None] * 10, "bans": []}))
    assert status == 400 and "no role board" in d["error"]
    assert "best_policy" in d["error"] and "best.pt" in d["error"]


def test_load_files_a_checkpoint_into_the_right_slot(app, httpd, policy_ckpt):
    """`role` is honoured, and a mismatched role is rejected rather than
    quietly filing a policy checkpoint under the role board."""
    rel = str(Path(policy_ckpt).relative_to(S.ROOT))
    status, d = _post(httpd, "/api/load",
                      json.dumps({"path": rel, "role": "masked"}))
    assert status == 400 and "not a masked one" in d["error"]
    status, d = _post(httpd, "/api/load",
                      json.dumps({"path": rel, "role": "policy"}))
    assert status == 200 and d["role"] == "policy", d
    assert app["policy"].kind == "policy"
    # unloading the next-move model leaves the role board alone
    status, d = _post(httpd, "/api/load",
                      json.dumps({"path": None, "role": "policy"}))
    assert status == 200 and d["model"] is None
    assert "policy" not in app and app["masked"] is not None


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
    """No key is the shipped path: the explanation is composed locally, so the
    panel still gets reasoning + citations - it just says where they came from.
    """
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    S.load_predictor(Path(policy_ckpt))
    status, d = _post(httpd, "/api/explain", _banned_board(app))
    assert status == 200, d
    assert d["llm"]["configured"] is False
    exp = d["explanation"]
    assert exp is not None and exp["local"] is True
    assert exp["recommend"] in {s["name"] for s in d["suggestions"]}
    assert exp["reasoning"], "the panel renders these as the answer"
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
    against, so both have to survive the HTTP round trip.

    The stub client used to answer here; the handler composes the explanation
    locally now, so the test asks the real model and checks the same contract -
    every rendered field present, and every citation resolvable against the
    retrieved set (the panel looks the id up, and shows "not in the retrieved
    set" when it cannot).
    """
    S.load_predictor(Path(policy_ckpt))
    monkeypatch.setattr(S.llm, "CACHE_DIR", tmp_path / "cache")
    status, d = _post(httpd, "/api/explain", _banned_board(app))
    assert status == 200, d
    exp = d["explanation"]
    assert exp["ok"] is True, exp.get("problems")
    assert exp["local"] is True
    for key in ("recommend", "why", "their_answer", "risk", "reasoning"):
        assert exp[key], "the panel renders %r" % key
    assert d["llm"]["configured"] is False       # no key, and none needed
    cited = exp["evidence"][0]["id"]
    assert cited in {g["gameid"] for g in d["retrieval"]["games"]}


def test_explain_without_a_policy_model_is_an_error(app, httpd):
    app.pop("policy", None)
    status, d = _post(httpd, "/api/explain", _empty_board())
    assert status == 400 and "policy" in d["error"]


# ---------------------------------------------------------------- state service
@pytest.fixture()
def fresh_store():
    """A store per test: `real` is process-wide, so a stale one leaks a board."""
    S._STATE_STORE = None
    yield
    S._STATE_STORE = None


def _state_post(srv, body):
    return _post(srv, "/api/state", json.dumps(body))


def test_suggest_publishes_the_board_the_user_is_looking_at(app, httpd,
                                                            fresh_store):
    """The one thing the frontend owes the agent: the live board, kept current.

    `/api/suggest` is what the UI calls on every click, so publishing there is
    what makes the state never more than one request stale.
    """
    bans = list(app["mapping"])[:6]
    status, d = _post(httpd, "/api/suggest",
                      json.dumps({"picks": [None] * 10, "bans": bans,
                                  "first_pick": 0}))
    assert status == 200, d
    status, st = _get(httpd, "/api/state")
    assert status == 200, st
    real = st["real"]
    assert real["move_idx"] == 6 and real["state"] == "real"
    assert real["kind"] == "pick" and real["side"] == "blue"
    assert real["readonly"] is True
    # six bans are on the state, in the order the schedule deals them
    assert real["rendered"].count(",") >= 5
    assert st["error"] is None


def test_a_sandbox_is_explored_over_http_without_touching_the_live_board(
        app, httpd, fresh_store):
    """The workflow the assistant is given, driven through the API."""
    bans = list(app["mapping"])[:6]
    _post(httpd, "/api/suggest", json.dumps({"picks": [None] * 10, "bans": bans,
                                             "first_pick": 0}))
    status, d = _state_post(httpd, {"op": "clone", "name": "alistar-line"})
    assert status == 200, d
    sid = d["state"]
    assert sid == "alistar-line"

    name = app["names_by_idx"][next(i for i in range(173)
                                    if i not in {app["mapping"][b] for b in bans})]
    status, d = _state_post(httpd, {"op": "apply", "state": sid,
                                    "champion": name})
    assert status == 200, d
    assert d["move_idx"] == 7
    assert d["real"]["move_idx"] == 6          # the live board did not move

    status, d = _state_post(httpd, {"op": "undo", "state": sid})
    assert status == 200 and d["move_idx"] == 6

    status, d = _state_post(httpd, {"op": "drop", "state": sid})
    assert status == 200 and d["dropped"] == sid
    assert [r["state"] for r in d["states"]] == ["real"]


def test_the_live_board_cannot_be_edited_over_http(app, httpd, fresh_store):
    _post(httpd, "/api/suggest", _empty_board())
    name = app["names_by_idx"][next(iter(app["mapping"].values()))]
    status, d = _state_post(httpd, {"op": "apply", "state": "real",
                                    "champion": name})
    assert status == 400 and "read-only" in d["error"]
    status, d = _state_post(httpd, {"op": "nonsense"})
    assert status == 400 and "unknown op" in d["error"]


def test_a_board_the_state_refuses_is_reported_not_fatal(app, httpd,
                                                         fresh_store):
    """`/api/suggest` must still answer; the panel shows why the state is behind."""
    riot = list(app["mapping"])[0]
    _post(httpd, "/api/suggest", json.dumps({"picks": [None] * 10,
                                             "bans": [riot], "first_pick": 0}))
    # The same champion banned twice cannot be a draft.
    status, d = _post(httpd, "/api/suggest",
                      json.dumps({"picks": [None] * 10, "bans": [riot, riot],
                                  "first_pick": 0}))
    assert status == 200, d
    status, st = _get(httpd, "/api/state")
    assert st["error"] and "twice" in st["error"]
    assert st["real"]["move_idx"] == 1         # the last good board survived


def test_the_agent_route_says_why_it_cannot_answer(app, httpd, policy_ckpt,
                                                   fresh_store, monkeypatch):
    """No key is a sentence, not a traceback - and the wiring still runs."""
    pytest.importorskip("langgraph")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    S.load_predictor(Path(policy_ckpt))
    status, d = _post(httpd, "/api/agent",
                      json.dumps({"message": "what do we pick here?",
                                  "picks": [None] * 10, "bans": [],
                                  "first_pick": 0}))
    assert status == 200, d
    assert d["stop"] == "no client"
    assert "LLM_API_KEY" in d["error"]
    # the tools it would have offered, and the state it would have read
    assert "get_state" in d["tools_offered"]
    assert "predict_next_move" in d["tools_offered"]
    assert d["state"]["move_idx"] == 0


def test_the_agent_route_needs_a_policy(app, httpd, fresh_store):
    pytest.importorskip("langgraph")
    app.pop("policy", None)
    status, d = _post(httpd, "/api/agent",
                      json.dumps({"message": "what next?", "picks": [None] * 10,
                                  "bans": [], "first_pick": 0}))
    assert status == 400 and "policy" in d["error"]


def test_the_agent_turn_reads_the_live_board_and_answers_over_http(
        app, httpd, policy_ckpt, fresh_store, monkeypatch):
    """The whole route, end to end, with a scripted model instead of a key.

    What this pins is the wiring, which is the part a key cannot fix: the board
    the handler publishes is the board the tool answers about, and the answer
    comes back with the calls that produced it.
    """
    loop = pytest.importorskip("pro.model.agent.loop")
    S.load_predictor(Path(policy_ckpt))
    monkeypatch.setitem(S.APP, "agent_client",
                        loop.scripted_client(calls=("get_state",
                                                    "predict_next_move")))
    bans = list(app["mapping"])[:6]
    status, d = _post(httpd, "/api/agent",
                      json.dumps({"message": "what do we pick here?",
                                  "picks": [None] * 10, "bans": bans,
                                  "first_pick": 0}))
    assert status == 200, d
    assert d["stop"] == "answered", d
    assert d["model"] == "scripted"
    assert [c["name"] for c in d["calls"]] == ["get_state",
                                               "predict_next_move"]
    assert all(c["ok"] for c in d["calls"])
    # the tool ran against the board the handler published, not a stale one
    assert d["state"]["move_idx"] == 6
    assert d["summary"]["calls"] == 2 and d["summary"]["sql_share"] == 0.0
    # ... and the loop never retyped the board, so nothing was mismatched
    assert d["transcription"]["mismatched"] == []


def test_the_agent_can_explore_a_sandbox_and_the_live_board_survives(
        app, httpd, policy_ckpt, fresh_store, monkeypatch):
    """The hypothetical the prompt describes, driven by the model's own calls."""
    loop = pytest.importorskip("pro.model.agent.loop")
    S.load_predictor(Path(policy_ckpt))
    banned = {app["mapping"][b] for b in list(app["mapping"])[:6]}
    idx = next(i for i in range(173) if i not in banned)
    champ = app["names_by_idx"][idx]
    S.APP["agent_client"] = None
    monkeypatch.setitem(S.APP, "agent_client", _explore_client(loop, champ))

    bans = list(app["mapping"])[:6]
    status, d = _post(httpd, "/api/agent",
                      json.dumps({"message": "and if they take %s?" % champ,
                                  "picks": [None] * 10, "bans": bans,
                                  "first_pick": 0}))
    assert status == 200, d
    assert [c["name"] for c in d["calls"]] == [
        "clone_state", "apply_move", "predict_next_move", "undo_move"]
    assert all(c["ok"] for c in d["calls"])
    assert d["state"]["move_idx"] == 6          # the live board never moved
    assert [r["state"] for r in d["states"]] == ["real", "hypo"]


def _explore_client(loop, champ):
    """clone -> apply -> predict -> undo, then answer. Order is the assertion."""
    steps = [
        ("clone_state", {"from_state": "real", "name": "hypo"}),
        ("apply_move", {"state": "hypo", "champion": champ}),
        ("predict_next_move", {"state": "hypo"}),
        ("undo_move", {"state": "hypo"}),
    ]
    queue = [{"id": "c%d" % i, "name": n, "args": a}
             for i, (n, a) in enumerate(steps)]

    def call(messages, tools=None):
        if queue:
            c = queue.pop(0)
            return {"role": "assistant", "content": "", "tool_calls": [c],
                    "model": "scripted"}
        return {"role": "assistant", "model": "scripted",
                "content": "Taking %s there changes nothing material." % champ,
                "tool_calls": []}
    return call


def test_the_published_state_names_the_patch_the_head_will_use(app, httpd,
                                                               policy_ckpt,
                                                               fresh_store):
    """The UI sends a `patch_id`; the head silently defaults when it does not.

    A state saying "patch: not given" while the head answered 16.20 would let the
    agent attribute a patch-specific ranking to no patch at all.
    """
    S.load_predictor(Path(policy_ckpt))
    _post(httpd, "/api/suggest", _empty_board())
    status, st = _get(httpd, "/api/state")
    assert status == 200, st
    assert st["real"]["patch"] == S.infer.DEFAULT_PATCH
    assert S.infer.DEFAULT_PATCH in st["real"]["rendered"]
