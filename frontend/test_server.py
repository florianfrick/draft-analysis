"""Tests for the local frontend: checkpoint pick, role board, head, state, agent.

What is left to test is what the server still does. The chat composer, the
breakdown page, the draft-score comparison and the local reranker are gone
(`pro/model/agent/` replaced them), so their tests are gone with them; the
surviving surface is the board, the two model slots, the state service and the
agent route, and that is what this file covers.

The three model-facing routes all read one object - the live `DraftState` - so
most tests here go through HTTP and assert on the payload the UI renders, not on
the internals of a call.
"""
import glob
import json
import sys
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server as S

# The UI's role keys. The model's own vocabulary is lowercase and ends in
# `support`, not `utility`; `_row_key` is the one edge that converts between
# them, so this list has to agree with `server.ROLE_ROWS` and `app.js`.
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]

# The role board's keys are the UI's vocabulary and the model's is lowercase;
# `_row_key` is the one edge that converts, so it is pinned here rather than
# assumed by every test.
assert S.model_tools.ROLES == ("top", "jungle", "mid", "bot", "support")


# --------------------------------------------------------------- fixtures
# The module's baseline: one masker, no head. Every test starts from this and
# every test is put back to it, because `APP` is process-wide and a head left
# behind by one test silently changes what the next one is asking.
_BASE = {"masked": None, "policy": None}


@pytest.fixture(scope="module")
def app():
    mapping, names, champs = S.load_champ_data()
    ckpt, _score = S.find_best_ckpt()
    pred = S.Predictor(ckpt, device="cpu")
    S.APP.update(mapping=mapping, names_by_idx=names, champions=champs,
                 masked=pred, agent_calls=12)
    S.APP.pop("policy", None)      # role board only unless a test loads one
    S.APP["agent_client"] = None
    _BASE["masked"], _BASE["policy"] = pred, None
    return S.APP


def _restore(app):
    for slot in ("masked", "policy"):
        if _BASE[slot] is None:
            app.pop(slot, None)
        else:
            app[slot] = _BASE[slot]


@pytest.fixture()
def httpd(app):
    """Live server for endpoint tests; slots and the store restored after."""
    S._STATE_STORE = None
    _restore(app)
    srv = S.ThreadingHTTPServer(("127.0.0.1", 0), S.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    _restore(app)
    app["agent_client"] = None
    S._STATE_STORE = None
    srv.shutdown()


@pytest.fixture()
def fresh_store():
    """A store per test: `real` is process-wide, so a stale one leaks a board."""
    S._STATE_STORE = None
    yield
    S._STATE_STORE = None


def _get(srv, route):
    conn = HTTPConnection("127.0.0.1", srv.server_address[1], timeout=120)
    conn.request("GET", route)
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw


def _post(srv, route, body, ctype="application/json"):
    conn = HTTPConnection("127.0.0.1", srv.server_address[1], timeout=120)
    conn.request("POST", route, body=body, headers={"Content-Type": ctype})
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    return r.status, (json.loads(raw) if raw else None)


def _riot_of(app, idx):
    return {v: k for k, v in app["mapping"].items()}[idx]


def _name_of(app, idx):
    return app["names_by_idx"][idx]


def _board(picks=None, bans=(), first_pick=0, fearless=(), k=None):
    body = {"picks": list(picks or [None] * 10),
            "bans": list(bans), "first_pick": first_pick,
            "fearless": list(fearless)}
    if k is not None:
        body["k"] = k
    return json.dumps(body)


def _empty_board(first_pick=0):
    return _board(first_pick=first_pick)


def _phase2(app, first_pick=0):
    """A board where the head may be asked a role question.

    Ten bans and six picks: the ban block is complete and the move on the clock
    is a pick, which are the two conditions `can_answer_in_lane` measures. The
    six picks are the first six corpus champions, minus whatever the bans took,
    so this does not depend on any model's opinion.
    """
    banned = list(app["mapping"])[:10]
    taken = set(banned)
    picks, used = [None] * 10, 0
    for riot in app["mapping"]:
        if used == 6:
            break
        if riot in taken:
            continue
        picks[used if used < 3 else 5 + (used - 3)] = riot
        used += 1
    return _board(picks=picks, bans=banned, first_pick=first_pick)


# ------------------------------------------------------- checkpoint choice
def _best_available_score():
    """The champ_top5 find_best_ckpt is supposed to pick, from the files."""
    import torch
    best = 0.0
    for pattern in S.CKPT_GLOB:
        for path in glob.glob(pattern):
            try:
                payload = torch.load(path, map_location="cpu",
                                     weights_only=False)
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


def test_ckpt_roles_follow_the_name():
    """The prefix states the inference, so the dropdown cannot offer a
    policy checkpoint to the role slot. `llm-*` is the reranker: a directory
    that fills the next-move slot, never the role board."""
    base = "/repo/pro/model/checkpoints"
    assert S._ckpt_roles(base + "/masking-x/best.pt", "checkpoints") == ["masked"]
    assert S._ckpt_roles(base + "/nextpick-x/best.pt", "checkpoints") == ["policy"]
    assert S._ckpt_roles(base + "/value-x/best.pt", "checkpoints") == []
    assert S._ckpt_roles(base + "/llm-lora-rerank5", "checkpoints") == ["policy"]
    # the projector is one piece of the reranker, not a model to offer
    assert S._ckpt_roles(base + "/llm-lora-rerank5/projector.pt",
                         "checkpoints") == []
    # an upload has no directory name to read, so the payload kind decides
    assert S._ckpt_roles("/tmp/uploaded/mine.pt", "policy") == ["policy"]


def test_slot_of_puts_every_next_move_model_in_the_policy_slot():
    """The two one-line rules that were tried here each broke one family.

    `Predictor.kind` is "pretrain"/"finetune" - the *stage*, not the inference -
    so no masking checkpoint ever reports "masked". A rule keyed on that filed
    every real masker into the next-move slot and `load_predictor(role="masked")`
    raised at startup; a rule keyed on `kind == "policy"` filed the reranker
    into the role slot, where it would answer the ten lane questions with a
    shortlist of picks.
    """
    class P:
        def __init__(self, kind):
            self.kind = kind

    assert S.slot_of(P("masked")) == "masked"
    # what `Predictor` actually sets for a masking checkpoint
    assert S.slot_of(P("finetune")) == "masked"
    assert S.slot_of(P("pretrain")) == "masked"
    assert S.slot_of(P("policy")) == "policy"
    assert S.slot_of(P("rerank")) == "policy"


def test_resolve_ckpt_path_validation(tmp_path):
    """HTTP may not escape the project, and a directory is not a checkpoint -
    except the one artefact that *is* a directory: an llm-* reranker."""
    outside = tmp_path / "elsewhere.pt"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="outside project"):
        S.resolve_ckpt_path(str(outside))
    with pytest.raises(ValueError, match="directory"):
        S.resolve_ckpt_path("pro/model/archive/checkpoints/nextpick-proto-ensemble")
    with pytest.raises(ValueError, match="expected a .pt"):
        S.resolve_ckpt_path("frontend/README.md")
    with pytest.raises(FileNotFoundError):
        S.resolve_ckpt_path("pro/model/archive/checkpoints/nope/best.pt")
    if S.RERANK_DEFAULT.is_dir():
        got = S.resolve_ckpt_path(str(S.RERANK_DEFAULT))
        assert got.is_dir() and got.name.startswith("llm-")


def test_list_checkpoints_covers_both_kinds():
    rows = S.list_checkpoints()
    roles = {r for row in rows for r in row["roles"]}
    assert "masked" in roles and "policy" in roles
    # the retired reranker's entry point is a folder, not something this server
    # can answer from, so it is never offered
    assert not [r for r in rows if r["path"].endswith("projector.pt")]
    assert all(r["label"] for r in rows)


def test_default_models_are_the_nov13_pair():
    """Both shipped slots are the `-nov13-` pair, and nothing else is nearby.

    `pro/model/checkpoints/` holds exactly these two directories and
    `pro/model/archive/checkpoints/` holds everything else, so the test pins
    both halves: the shipped paths, and the fact that the reranker - opt-in and
    measured at +2.4 pp over its own proposer - is *not* sitting in the default
    store where a later edit could promote it by accident.

    The pair is load-bearing: the policy's encoder *is* the masking checkpoint
    above it, so the two names move together.
    """
    assert S.MASKED_DEFAULT.parent.name == (
        "masking-proonly-nov13-through-lcp-df-sw")
    assert S.POLICY_DEFAULT.parent.name == (
        "nextpick-naivetransformer-role-d512-l8-h8-k5-through-lcp-df-sw")
    assert S.MASKED_DEFAULT.name == S.POLICY_DEFAULT.name == "best.pt"
    assert S.MASKED_DEFAULT.parent.parent == S.POLICY_DEFAULT.parent.parent
    assert S.ENSEMBLE_CKPT.suffix == ".json"
    # the reranker is a directory and its name must stay `llm-*`, because that
    # prefix is what `resolve_ckpt_path` and `_ckpt_roles` key on
    assert S.RERANK_DEFAULT.name.startswith("llm-")
    # archived, not shipped: same slot, different store
    assert S.RERANK_DEFAULT.parent == S.ARCHIVE_CKPT_DIR
    assert S.ENSEMBLE_CKPT.parent.parent == S.ARCHIVE_CKPT_DIR
    assert S.ARCHIVE_CKPT_DIR != S.POLICY_DEFAULT.parent.parent


def test_the_reranker_is_opt_in_and_its_failure_is_fatal():
    """The 4B is one flag away, and asking for it is a promise.

    It is not the default: the number that justified that (34.27 %) is measured
    with the *true lane* in the task line, and the panel's question has no lane
    (20.85 % - frontend/README.md, "What the reranker is worth"). So the head
    ships and `--rerank` opts in. Once asked for, a build failure must be fatal:
    a server that starts and silently serves a different model than the flag
    named is worse than one that refuses to start.
    """
    src = (S.FRONTEND_DIR / "server.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--rerank"' in src
    assert "raise SystemExit(\"cannot serve the reranker" in src
    # no automatic preference: the startup block must not reach for
    # RERANK_DEFAULT on its own
    assert "elif not args.no_rerank and RERANK_DEFAULT.is_dir()" not in src
    # the four measured arms the decision rests on are named where the constant is
    for n in ("34.27", "20.85", "18.25", "24.90"):
        assert n in src, n


# -------------------------------------------------------------- ensembles
def test_ensemble_manifest_is_still_servable():
    if not S.ENSEMBLE_CKPT.is_file():
        pytest.skip("no ensemble manifest on disk")
    members, ev, _ = S.read_ensemble(S.ENSEMBLE_CKPT)
    assert len(members) >= 2
    assert all(m.is_file() for m in members)
    assert all(m.suffix == ".pt" for m in members)
    assert isinstance(ev, dict)


def test_read_ensemble_rejects_bad_manifests(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"members": []}))
    with pytest.raises(ValueError, match="lists no members"):
        S.read_ensemble(empty)

    # Members are resolved relative to the manifest and must stay inside the
    # project, so a manifest outside it is refused before the member is looked
    # for. Both refusals are the point: a manifest is a file a user can write.
    outside = tmp_path / "missing.json"
    outside.write_text(json.dumps({"members": ["gone.pt"]}))
    with pytest.raises(ValueError, match="outside project"):
        S.read_ensemble(outside)

    inside = S.UPLOAD_DIR / "test-ensemble.json"
    S.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        inside.write_text(json.dumps({"members": ["gone.pt"]}))
        with pytest.raises(FileNotFoundError, match="member missing"):
            S.read_ensemble(inside)
    finally:
        inside.unlink(missing_ok=True)


# ------------------------------------------------- champions + draft order
def test_http_api(app, httpd):
    status, data = _get(httpd, "/api/champions")
    assert status == 200
    assert len(data["champions"]) == 173
    assert data["model"]["val_top5"] == pytest.approx(
        _best_available_score(), abs=1e-6)
    status, page = _get(httpd, "/")
    assert status == 200 and b"Draft Assistant" in page
    # The walkthrough opens on Worlds, not on every league's newest 100 games.
    assert b'id="wt-filter" type="text" value="WLDs"' in page
    # The header no longer narrates the loaded checkpoints, and the model
    # glossary at the bottom of the page is gone.
    assert b'id="model-info"' not in page
    assert b"<footer" not in page
    # an unknown endpoint is a 404, not the static handler's idea of a path
    status, _ = _get(httpd, "/api/nope")
    assert status == 404


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
    async head's answer: that answer belongs to the previous board, which
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


def test_champion_images_are_ddragon(app):
    for c in app["champions"][:5]:
        assert c["img"].startswith("https://ddragon.leagueoflegends.com/cdn/")
        assert c["name"] and isinstance(c["idx"], int)


# ------------------------------------------------------------- the role board
def test_suggest_empty_draft(app, httpd, fresh_store):
    """Ten cells, all open, all answered by the masker on an empty board.

    An empty board is move 0, a ban: the head cannot be asked a role question
    there, so every cell comes from the masking checkpoint and the payload says
    so. This is also the shape the UI renders before any pick is made.
    """
    status, sug = _post(httpd, "/api/suggest", _empty_board())
    assert status == 200, sug
    for team in ("blue", "red"):
        assert set(sug[team]) == set(ROLES)
        for role in ROLES:
            assert 1 <= len(sug[team][role]) <= 3
            cell = sug[team][role][0]
            assert set(cell) >= {"name", "idx", "prob"}
            assert 0.0 <= cell["prob"] <= 1.0
    assert sug["method"] == "masker"
    assert set(sug["methods"].values()) == {"masker"}
    assert sug["assignment"] == [None] * 10
    assert sug["conf"] == [None] * 10
    assert sug["open_roles"] == {"blue": ROLES, "red": ROLES}
    assert sug["complete"] is False
    assert sug["notes"] == {}


def test_the_role_board_only_fills_the_roles_still_open(app, httpd,
                                                       fresh_store):
    """A filled role is not a question. Six picks close six roles, so the
    board carries four open cells and the rest of the assignment."""
    status, sug = _post(httpd, "/api/suggest", _phase2(app))
    assert status == 200, sug
    cells = [(team, role) for team in ("blue", "red") for role in ROLES
             if role in sug[team]]
    assert len(cells) == 4, cells
    filled = [i for i, r in enumerate(sug["assignment"]) if r is not None]
    assert len(filled) == 6
    # one-to-one per side: six distinct roles across two teams of three
    for base in (0, 5):
        got = [sug["assignment"][s] for s in range(base, base + 5)
               if sug["assignment"][s] is not None]
        assert len(got) == len(set(got)), "two picks claim one role"
    assert all(0.0 <= c <= 1.0 for c in sug["conf"] if c is not None)


def test_the_assignment_uses_the_role_names_the_client_renders(app, httpd,
                                                                fresh_store):
    """Regression for `undefined` under a picked champion and its empty cell.

    `/api/suggest` keys every role by display name (`TOP`..`SUPPORT`); app.js
    used to index its `ROLES` array with that value as if it were a number, so
    `ROLES["SUPPORT"]` printed `undefined` and `roleSlot["SUPPORT"]` was never
    found by the numeric loop. The server now publishes the vocabulary through
    `/api/champions` and the client normalises both shapes with `roleIndex`.
    """
    status, sug = _post(httpd, "/api/suggest", _phase2(app))
    assert status == 200, sug
    keys = [r for r in sug["assignment"] if r is not None]
    assert keys and set(keys) <= set(ROLES)
    assert "UTILITY" not in ROLES
    status, ch = _get(httpd, "/api/champions")
    assert status == 200 and ch["roles"] == ROLES
    assert S.model_tools.ROLES[-1] == "support"
    from solo_queue.model.policy import lanes
    assert lanes.table()["roles"] == ROLES
    assert lanes.lane_name(4) == "SUPPORT"
    src = (Path(__file__).resolve().parent / "static" / "app.js").read_text(
        encoding="utf-8")
    assert "function roleIndex(" in src
    assert "ROLES[ar]" not in src, "do not index ROLES with a raw role key"
    assert "roleIndex(assign[s])" in src
    assert "d.roles" in src, "the server should own the role vocabulary"


def test_the_assignment_is_the_table_not_the_role_head(app, httpd,
                                                       fresh_store):
    """`p_lane` is the champion's own share of its picks in that role.

    A dedicated champion must come back high and a flex one lower - if this
    were a per-slot argmax from a head, every value would be a softmax over
    five roles and Vi (jungle 99.8 %) would not be distinguishable from a
    champion the table is unsure about.
    """
    _post(httpd, "/api/suggest", _phase2(app))
    status, st = _get(httpd, "/api/state")
    assert status == 200, st
    roles = st["real"]["roles"]
    assert len(roles) == 6
    assert all(0.0 < r["p_lane"] <= 1.0 for r in roles)
    assert {r["side"] for r in roles} == {"blue", "red"}
    # the assignment travels with the state, so `get_state` needs no second call
    assert st["real"]["open_roles"]["blue"] is not None
    for side in ("blue", "red"):
        taken = {r["role"] for r in roles if r["side"] == side}
        assert sorted(st["real"]["open_roles"][side]) == sorted(
            set(S.model_tools.ROLES) - taken)


def test_suggest_uses_the_head_once_the_bans_are_done(app, httpd,
                                                      fresh_store,
                                                      policy_ckpt):
    """With the ban block complete and a pick on the clock, the head answers.

    And it answers for *both* sides: the side not on the clock is a
    hypothetical, which the result flags rather than hiding - the two sides come
    out nearly identical, because the pick counts already determine whose turn
    it is.
    """
    S.load_predictor(Path(policy_ckpt))
    status, sug = _post(httpd, "/api/suggest", _phase2(app))
    assert status == 200, sug
    assert sug["method"] == "head", sug["notes"]
    assert set(sug["methods"].values()) == {"head"}
    for team in ("blue", "red"):
        for role in sug[team]:
            # p_in_lane is a share of the champions that play the role, so the
            # cell sums to <= 1 and never to "share of the whole pool"
            assert sum(c["prob"] for c in sug[team][role]) <= 1.0001
    assert sug["model"]["kind"] == "policy"


def test_suggest_reports_a_refused_cell_instead_of_failing_the_board(
        app, httpd, fresh_store, policy_ckpt):
    """A ban-phase role question needs a masker. Without one every cell is
    refused, with its reason - and the board still answers 200.

    The alternative - a 400 - would blank a panel because one of its ten cells
    could not be filled, and the other nine were fine.
    """
    S.load_predictor(Path(policy_ckpt))
    app.pop("masked", None)
    status, sug = _post(httpd, "/api/suggest", _empty_board())
    assert status == 200, sug
    assert sug["method"] == "none"
    assert len(sug["notes"]) == 10
    assert all("masking checkpoint" in v for v in sug["notes"].values())
    assert sug["blue"] == {} and sug["red"] == {}
    # the assignment still stands: it is a table lookup, not a model answer
    assert sug["open_roles"] == {"blue": ROLES, "red": ROLES}


def test_suggest_needs_a_model(app, httpd, fresh_store):
    app.pop("masked", None)
    app.pop("policy", None)
    status, d = _post(httpd, "/api/suggest", _empty_board())
    assert status == 400 and "no model loaded" in d["error"]


def test_suggest_filters_the_fearless_pool(app, httpd, fresh_store):
    """The pool is a condition on the head's path and a filter on the masker's,
    and the payload counts it either way so the UI can warn."""
    status, first = _post(httpd, "/api/suggest", _empty_board())
    assert status == 200, first
    top = first["blue"][ROLES[0]][0]["idx"]
    status, sug = _post(httpd, "/api/suggest", _board(fearless=[_riot_of(app, top)]))
    assert status == 200, sug
    assert sug["fearless_filtered"] == 1
    names = {c["name"] for role in sug["blue"].values() for c in role}
    assert _name_of(app, top) not in names


def test_suggest_rejects_a_board_the_store_cannot_build(app, httpd,
                                                        fresh_store):
    """The same champion twice is not a board the state can hold.

    The store records the refusal instead of raising, and `/api/state` publishes
    it, so the panel can say what is wrong with the board rather than showing a
    traceback for a click.

    Reaching the duplicate takes a board the schedule can replay to it. Blue
    first, six bans: the picks are consumed in the order blue 0, red 5, red 6,
    blue 1 - so the duplicate goes in blue's first two slots and red's are
    filled, or the log stops at an empty slot before it ever gets there.
    """
    pool = list(app["mapping"])
    bans, riot, other, third = pool[:6], pool[6], pool[7], pool[8]
    body = json.dumps({"picks": [riot, riot, None, None, None, other, third,
                                 None, None, None], "bans": bans})
    status, sug = _post(httpd, "/api/suggest", body)
    assert status == 200, sug
    status, st = _get(httpd, "/api/state")
    assert st["error"], "the refused board should be reported"
    assert "twice" in st["error"]


# ------------------------------------------------------------- the next move
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


def test_policy_checkpoint_switches_kind(policy_ckpt):
    """A `nextpick-*` checkpoint must load as a next-move model, not an encoder."""
    pred = S.load_predictor(Path(policy_ckpt))
    assert pred.kind == "policy"
    info = S.model_info(pred)
    assert info["kind"] == "policy"
    assert info["metric"] == "next-move top5"
    assert info["val_top5"] is None or 0.0 <= info["val_top5"] <= 1.0
    assert info["ckpt"].startswith("nextpick-")
    # no rerank block: the retired local reranker used to publish one here
    assert "rerank_eval" not in info and "family" not in info


def test_next_move_endpoint_returns_a_ban_decision(app, httpd, policy_ckpt,
                                                   fresh_store):
    S.load_predictor(Path(policy_ckpt))
    status, d = _post(httpd, "/api/next-move", _empty_board())
    assert status == 200, d
    assert d["state"]["kind"] == "ban" and d["state"]["side"] == 0
    assert d["state"]["complete"] is False
    assert 1 <= len(d["suggestions"]) <= 20
    probs = [c["prob"] for c in d["suggestions"]]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 < p <= 1.0 for p in probs)
    # The note no longer carries the "p_head is a share of the legal pool" tag
    # (that context moved to the agent's SYSTEM prompt); for a known/omitted
    # patch it is empty.
    assert "p_head" not in d["note"]
    assert d["model"]["kind"] == "policy"


def test_next_move_reports_a_pick_move(app, httpd, policy_ckpt, fresh_store):
    S.load_predictor(Path(policy_ckpt))
    bans = list(app["mapping"])[:6]
    status, d = _post(httpd, "/api/next-move", _board(bans=bans))
    assert status == 200, d
    assert d["state"]["kind"] == "pick" and d["state"]["side"] == 0
    assert d["state"]["move_idx"] == 6


def test_next_move_honours_first_pick(app, httpd, policy_ckpt, fresh_store):
    S.load_predictor(Path(policy_ckpt))
    status, d = _post(httpd, "/api/next-move", _empty_board(first_pick=1))
    assert status == 200, d
    assert d["state"]["side"] == 1 and d["state"]["first_pick"] == 1


def test_next_move_honours_and_clamps_patch_id(app, httpd, policy_ckpt,
                                               fresh_store):
    """A patch id the checkpoint cannot embed becomes UNK, not a 400."""
    S.load_predictor(Path(policy_ckpt))
    status, ch = _get(httpd, "/api/champions")
    older = ch["patches"][2]                       # a patch that is not 16.20
    status, d = _post(httpd, "/api/next-move",
                      json.dumps({"picks": [None] * 10, "bans": [],
                                  "patch_id": older["id"]}))
    assert status == 200, d
    assert d["patch"] == older["key"]
    status, d = _post(httpd, "/api/next-move",
                      json.dumps({"picks": [None] * 10, "bans": [],
                                  "patch_id": 99999}))
    assert status == 200, d
    assert d["patch"] is None


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
    assert keys == sorted(keys, reverse=True)      # newest first
    assert [p["id"] for p in patches] == list(range(len(keys), 0, -1))
    assert d["default_patch"] == S.infer.DEFAULT_PATCH
    # The default is a preference, not "the newest row": the selector must open
    # on the named patch whenever the checkpoint's own table has it.
    assert S.infer.DEFAULT_PATCH in keys
    assert d["default_patch_id"] == next(
        p["id"] for p in patches if p["key"] == S.infer.DEFAULT_PATCH)


def test_next_move_excludes_the_fearless_pool(app, httpd, policy_ckpt,
                                              fresh_store):
    """A champion picked in an earlier game must drop out of a pick answer.

    The assertion is differential on purpose: it takes whatever the model
    actually ranked first, adds *that* to the pool, and checks it is gone.
    A fixed champion would pass or fail depending on the weights.
    """
    S.load_predictor(Path(policy_ckpt))
    bans = list(app["mapping"])[:6]
    status, d = _post(httpd, "/api/next-move", _board(bans=bans))
    assert status == 200, d
    assert d["state"]["kind"] == "pick"
    assert d["state"]["n_unavailable"] == 0
    top = d["suggestions"][0]["idx"]

    status, d2 = _post(httpd, "/api/next-move",
                       _board(bans=bans, fearless=[_riot_of(app, top)]))
    assert status == 200, d2
    assert d2["state"]["n_unavailable"] == 1
    assert top not in [c["idx"] for c in d2["suggestions"]]


def test_next_move_without_a_policy_model_is_an_error(app, httpd, fresh_store):
    app.pop("policy", None)
    status, d = _post(httpd, "/api/next-move", _empty_board())
    assert status == 400 and "no next-move model" in d["error"]


def test_next_move_rejects_an_illegal_board(app, httpd, policy_ckpt,
                                            fresh_store):
    """The same champion twice, past the ban block: the store refuses it.

    A distribution computed for an illegal board is worse than no answer - it is
    indistinguishable from a good one - so this is a 400, not a list. This is
    the one route that cannot fall back to the last good board: `/api/suggest`
    may answer about the previous state and let the panel show `state_error`,
    but a ranking is a claim about *this* board.
    """
    S.load_predictor(Path(policy_ckpt))
    pool = list(app["mapping"])
    bans, riot, other, third = pool[:6], pool[6], pool[7], pool[8]
    status, d = _post(httpd, "/api/next-move",
                      json.dumps({"picks": [riot, riot, None, None, None, other,
                                            third, None, None, None],
                                  "bans": bans}))
    assert status == 400, d
    assert "twice" in d["error"]


def test_a_complete_draft_has_no_next_move(app, httpd, policy_ckpt,
                                           fresh_store):
    """Twenty moves: the panel says "complete" instead of ranking a 21st."""
    S.load_predictor(Path(policy_ckpt))
    schedule = S.infer.move_plan(0)
    pool = list(app["mapping"])
    picks, bans, used = [None] * 10, [], {"blue": 0, "red": 0}
    for kind, side in schedule:
        riot = pool[len(picks and []) + len(bans) + used["blue"]
                    + used["red"]]
        if kind == S.infer.MOVE_BAN:
            bans.append(riot)
        else:
            slot = used["blue"] if side == 0 else 5 + used["red"]
            picks[slot] = riot
            used["blue" if side == 0 else "red"] += 1
    status, d = _post(httpd, "/api/next-move", _board(picks=picks, bans=bans))
    assert status == 200, d
    assert d["state"]["complete"] is True
    assert d["suggestions"] == []


# ------------------------------------------------------------- state service
def _state_post(srv, body):
    return _post(srv, "/api/state", json.dumps(body))


def test_suggest_publishes_the_board_the_user_is_looking_at(app, httpd,
                                                            fresh_store):
    """The one thing the frontend owes the agent: the live board, kept current.

    `/api/suggest` is what the UI calls on every click, so publishing there is
    what makes the state never more than one request stale.
    """
    bans = list(app["mapping"])[:6]
    status, d = _post(httpd, "/api/suggest", _board(bans=bans))
    assert status == 200, d
    status, st = _get(httpd, "/api/state")
    assert status == 200, st
    real = st["real"]
    assert real["move_idx"] == 6 and real["state"] == "real"
    assert real["kind"] == "pick" and real["side"] == "blue"
    assert real["readonly"] is True
    assert real["rendered"].count(",") >= 5
    assert st["error"] is None


def test_a_sandbox_is_explored_over_http_without_touching_the_live_board(
        app, httpd, fresh_store):
    """The workflow the assistant is given, driven through the API."""
    bans = list(app["mapping"])[:6]
    _post(httpd, "/api/suggest", _board(bans=bans))
    status, d = _state_post(httpd, {"op": "clone", "name": "alistar-line"})
    assert status == 200, d
    sid = d["state"]
    assert sid == "alistar-line"

    name = _name_of(app, next(i for i in range(173)
                              if i not in {app["mapping"][b] for b in bans}))
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
    name = _name_of(app, next(iter(app["mapping"].values())))
    status, d = _state_post(httpd, {"op": "apply", "state": "real",
                                    "champion": name})
    assert status == 400 and "read-only" in d["error"]
    status, d = _state_post(httpd, {"op": "nonsense"})
    assert status == 400 and "unknown op" in d["error"]


def test_a_sandbox_carries_the_roles_its_moves_imply(app, httpd, fresh_store):
    """`apply` recomputes the assignment, so a clone never shows a stale one.

    A ban has no role, so the first six moves leave the assignment empty - which
    is the correct answer, not a missing one. The seventh is a pick, and it
    arrives with its role already attached.
    """
    _post(httpd, "/api/suggest", _empty_board())
    status, d = _state_post(httpd, {"op": "clone", "name": "s1"})
    assert status == 200, d
    for i in range(6):                     # phase 1 bans, dealt by the schedule
        status, d = _state_post(httpd, {"op": "apply", "state": "s1",
                                        "champion": _name_of(app, i)})
        assert status == 200, d
    box = next(r for r in d["states"] if r["state"] == "s1")
    assert box["kind"] == "pick" and box["roles"] == []

    status, d = _state_post(httpd, {"op": "apply", "state": "s1",
                                    "champion": _name_of(app, 6)})
    assert status == 200, d
    box = next(r for r in d["states"] if r["state"] == "s1")
    assert len(box["roles"]) == 1
    assert box["roles"][0]["role"] in S.model_tools.ROLES
    assert 0.0 < box["roles"][0]["p_lane"] <= 1.0
    # the live board is untouched: a sandbox is a clone, not a mode
    assert d["real"]["move_idx"] == 0


def test_the_published_state_names_the_patch_the_head_will_use(app, httpd,
                                                              fresh_store,
                                                              policy_ckpt):
    """The state carries the patch *label*, resolved from the id the UI sends.

    A state that said "patch: not given" while the head answered 16.20 would
    make the agent attribute a patch-specific ranking to no patch at all.
    """
    S.load_predictor(Path(policy_ckpt))
    status, ch = _get(httpd, "/api/champions")
    pid = ch["patches"][1]["id"]
    status, d = _post(httpd, "/api/suggest",
                      json.dumps({"picks": [None] * 10, "bans": [],
                                  "patch_id": pid}))
    assert status == 200, d
    status, st = _get(httpd, "/api/state")
    assert st["real"]["patch"] == ch["patches"][1]["key"]


def test_the_fearless_pool_reaches_the_published_state(app, httpd,
                                                       fresh_store):
    """The pool is published in the state's own key space, which is the corpus
    spelling - the same one every tool result renders."""
    idxs = list(app["mapping"].values())[:3]
    pool = [_riot_of(app, i) for i in idxs]
    status, d = _post(httpd, "/api/suggest",
                      json.dumps({"picks": [None] * 10, "bans": [],
                                  "fearless": pool}))
    assert status == 200, d
    status, st = _get(httpd, "/api/state")
    assert st["real"]["fearless"] == [S.draft_state.key(_name_of(app, i))
                                      for i in idxs]


def test_an_absent_fearless_pool_keeps_the_live_one(app, httpd, fresh_store):
    """Absent is not empty. A payload that says nothing about the pool must not
    clear one, or a caller forwarding a board would silently unblock ten
    champions. `[]` is the explicit "this series has no pool"."""
    idxs = list(app["mapping"].values())[:2]
    _post(httpd, "/api/suggest",
          json.dumps({"picks": [None] * 10, "bans": [],
                      "fearless": [_riot_of(app, i) for i in idxs]}))
    # no `fearless` key at all
    _post(httpd, "/api/suggest",
          json.dumps({"picks": [None] * 10, "bans": [], "first_pick": 0}))
    status, st = _get(httpd, "/api/state")
    assert len(st["real"]["fearless"]) == 2
    # and the explicit empty list does clear it
    _post(httpd, "/api/suggest", _empty_board())
    status, st = _get(httpd, "/api/state")
    assert st["real"]["fearless"] == []


# --------------------------------------------------------------- model switch
def test_load_endpoint_switches_model(app, httpd, fresh_store):
    rel = S._rel(S.POLICY_DEFAULT) if S.POLICY_DEFAULT.is_file() else None
    if rel is None:
        pytest.skip("no policy checkpoint on disk")
    status, d = _post(httpd, "/api/load", json.dumps({"path": rel}))
    assert status == 200, d
    assert d["role"] == "policy" and d["model"]["kind"] == "policy"
    status, cps = _get(httpd, "/api/checkpoints")
    assert cps["current"]["policy"] == rel
    # a null path unloads the slot
    status, d = _post(httpd, "/api/load",
                      json.dumps({"path": None, "role": "policy"}))
    assert status == 200 and d["model"] is None
    assert app.get("policy") is None


def test_load_files_a_checkpoint_into_the_right_slot(app, httpd, policy_ckpt):
    """`role` is a claim the server checks against the payload's kind."""
    rel = S._rel(Path(policy_ckpt))
    status, d = _post(httpd, "/api/load",
                      json.dumps({"path": rel, "role": "masked"}))
    assert status == 400 and "not a masked one" in d["error"]


def test_loading_a_model_rebinds_the_open_board(app, httpd, fresh_store,
                                                policy_ckpt):
    """A checkpoint loaded while a board is open must answer about that board.

    The engine is swapped under every state in the store, not rebuilt: an
    exploration made under the old model would otherwise keep answering with
    it, and `/api/suggest` would name a model the header no longer shows.
    """
    bans = list(app["mapping"])[:10]
    status, d = _post(httpd, "/api/suggest", _phase2(app))
    assert status == 200, d
    assert d["method"] == "masker"
    S.load_predictor(Path(policy_ckpt))
    status, d = _post(httpd, "/api/suggest", _phase2(app))
    assert status == 200, d
    assert d["method"] == "head", d["notes"]
    assert bans  # the board survived the swap


def test_upload_endpoint(app, httpd, tmp_path):
    """Raw bytes with a name -> frontend/uploaded/, then loaded."""
    if not S.POLICY_DEFAULT.is_file():
        pytest.skip("no checkpoint to upload")
    data = S.POLICY_DEFAULT.read_bytes()
    conn = HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=120)
    conn.request("POST", "/api/upload?name=test-upload.pt", body=data,
                 headers={"Content-Type": "application/octet-stream"})
    r = conn.getresponse()
    out = json.loads(r.read())
    conn.close()
    dest = S.UPLOAD_DIR / "test-upload.pt"
    try:
        assert r.status == 200, out
        assert out["role"] == "policy"
        assert dest.is_file()
    finally:
        dest.unlink(missing_ok=True)
    # a name that is not a .pt is refused before anything is written
    conn = HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=30)
    conn.request("POST", "/api/upload?name=evil.sh", body=b"x",
                 headers={"Content-Type": "application/octet-stream"})
    r = conn.getresponse()
    assert r.status == 400
    r.read()
    conn.close()


# ------------------------------------------------------------------- the agent
def test_the_agent_route_says_why_it_cannot_answer(app, httpd, policy_ckpt,
                                                   fresh_store, monkeypatch):
    """No key is a sentence, not a traceback: the panel opens and says so.

    The key has to be *removed*, not just unset on the app: `agent_client=None`
    means "fall back to the environment", so on a machine with LLM_API_KEY in
    `.env` this test answered for real (and spent tokens) instead of exercising
    the no-client path.
    """
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    S.load_predictor(Path(policy_ckpt))
    app["agent_client"] = None
    status, d = _post(httpd, "/api/agent",
                      json.dumps({"message": "what now?",
                                  "picks": [None] * 10, "bans": []}))
    assert status == 200, d
    assert d["answer"] is None
    assert d["stop"] == "no client"
    assert "LLM_API_KEY" in d["error"]
    # and the tools it *would* have had are published, so the failure is
    # legible: this is a missing key, not a missing model
    assert "predict_next_move" in d["tools_offered"]


def test_the_agent_turn_reads_the_live_board_and_answers_over_http(
        app, httpd, policy_ckpt, fresh_store):
    """One turn through the graph, with a scripted client standing in for the
    model. It proves the wiring - board published, tools offered, state read -
    not the model's judgement, which is not this file's to test."""
    from pro.model.agent import loop
    S.load_predictor(Path(policy_ckpt))
    app["agent_client"] = loop.scripted_client(
        calls=("get_state", "predict_roles"), answer="Blue is fine.")
    bans = list(app["mapping"])[:6]
    status, d = _post(httpd, "/api/agent",
                      json.dumps({"message": "how is blue?",
                                  "picks": [None] * 10, "bans": bans}))
    assert status == 200, d
    assert d["answer"] == "Blue is fine."
    assert d["stop"] == "answered"
    assert [c["name"] for c in d["calls"]] == ["get_state", "predict_roles"]
    assert all(c["ok"] for c in d["calls"])
    # the answer is about the board that was sent, and the state says so
    assert d["state"]["move_idx"] == 6
    assert d["summary"]["sql_share"] == 0.0


def test_the_agent_can_explore_a_sandbox_and_the_live_board_survives(
        app, httpd, policy_ckpt, fresh_store):
    """A hypothetical is a clone, and `real` stays read-only even for the agent.

    The scripted client names `real` in every `apply_move`, which is exactly the
    mistake a real model makes; the tool layer refuses it, the turn still
    answers, and the live board is where it was. That refusal is the guarantee
    the panel relies on, so it is asserted rather than avoided.
    """
    from pro.model.agent import loop
    S.load_predictor(Path(policy_ckpt))
    bans = list(app["mapping"])[:6]
    app["agent_client"] = loop.scripted_client(
        calls=("clone_state", "apply_move"), champion=_name_of(app, 6),
        answer="Tried it on a clone.")
    status, d = _post(httpd, "/api/agent",
                      json.dumps({"message": "what if we take it?",
                                  "picks": [None] * 10, "bans": bans}))
    assert status == 200, d
    calls = {c["name"]: c for c in d["calls"]}
    assert calls["clone_state"]["ok"] is True
    assert calls["apply_move"]["ok"] is False
    assert "read-only" in calls["apply_move"]["error"]
    # the clone exists, the live board did not move, and the answer still came
    assert d["state"]["move_idx"] == 6
    assert "scripted" in [r["state"] for r in d["states"]]
    assert d["answer"] == "Tried it on a clone."


def test_the_agent_without_a_state_has_nothing_to_answer(app, httpd,
                                                         fresh_store):
    """A session with no store offers no state tools and says why, rather than
    answering about a board it does not have."""
    from pro.model.agent.session import Session
    sess = Session(con=None, store=None)
    # No store means no state or model tools; the soloQ matchup tool is offered
    # only when its store is on disk, which is independent of this session.
    board_tools = {"get_state", "clone_state", "apply_move", "undo_move",
                   "list_states", "predict_next_move", "predict_roles"}
    assert not (set(sess.names()) & board_tools)
    out = __import__("pro.model.agent.loop", fromlist=["run"]).run(
        sess, "anything?", client=lambda *a, **k: {})
    assert out["stop"] == "no state"
    assert "Store" in out["error"]


# --------------------------------- the agent's one-click suggestion (rerank)
def _series(game=3, of=5):
    """The series bar, as the browser sends it: numbers, never prose."""
    return {"game": game, "of": of,
            "games": [{"game": i + 1, "picks": 10, "fearless": 0,
                       "closed": i + 1 < game} for i in range(of)]}


def _suggest_board(bans=(), series=None, **extra):
    body = {"picks": [None] * 10, "bans": list(bans), "first_pick": 0,
            "fearless": [], "series": _series() if series is None else series}
    body.update(extra)
    return json.dumps(body)


def _legal_name(app, bans):
    """A champion name that is on neither side of this board."""
    used = {app["mapping"][r] for r in bans}
    return next(app["names_by_idx"][i]
                for i in range(len(app["names_by_idx"])) if i not in used)


def test_agent_suggest_without_a_key_is_an_outcome_not_a_crash(
        app, httpd, policy_ckpt, fresh_store, monkeypatch):
    """A key is a capability. The panel still gets the board, the config and
    the head's #1, plus a sentence saying what is missing."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    S.load_predictor(Path(policy_ckpt))
    app["agent_client"] = None
    status, d = _post(httpd, "/api/agent-suggest",
                      _suggest_board(list(app["mapping"])[:6]))
    assert status == 200, d
    assert d["stop"] == "no client" and d["answer"] is None
    assert "LLM_API_KEY" in d["error"]
    assert d["suggestion"]["champion"] is None
    assert d["suggestion"]["parse"] == "empty"
    assert d["config"]["ablate"] == ["soloq_matchup", "sql"] \
        and d["config"]["max_steps"] == 5
    # The rerank is a rerank of the head, not a data dive: both free-form data
    # tools are ablated from the one-click suggestion.
    assert "sql" not in d["tools_offered"]
    assert "soloq_matchup" not in d["tools_offered"]
    assert d["head"] and d["head"]["champion"]
    assert "series: game 3 of 5" in d["state"]["rendered"]


def test_agent_suggest_reads_the_live_board_and_returns_one_parsed_move(
        app, httpd, policy_ckpt, fresh_store):
    """The wiring, not the model's judgement: state published, tools called,
    one champion parsed, and fidelity measured against the head's own #1."""
    from pro.model.agent import loop
    S.load_predictor(Path(policy_ckpt))
    bans = list(app["mapping"])[:6]
    status, nm = _post(httpd, "/api/next-move", _board(bans=bans))
    assert status == 200, nm
    head_name = nm["suggestions"][0]["name"]
    app["agent_client"] = loop.scripted_client(
        calls=("get_state", "predict_roles", "predict_next_move"),
        answer="%s is the strongest lane-legal move here.\nPICK: %s"
               % (head_name, head_name))
    status, d = _post(httpd, "/api/agent-suggest", _suggest_board(bans))
    assert status == 200, d
    assert d["stop"] == "answered"
    sug = d["suggestion"]
    assert sug["champion"] == head_name and sug["legal"] is True
    assert sug["kind"] == "pick" and sug["idx"] is not None
    assert sug["parse"] in ("marker", "first-sentence")
    assert d["head"]["champion"] == head_name and d["head"]["matches"] is True
    assert [c["name"] for c in d["calls"]] == [
        "get_state", "predict_roles", "predict_next_move"]
    assert all(c["ok"] for c in d["calls"])
    assert d["state"]["move_idx"] == 6
    assert d["summary"]["sql_share"] == 0.0


def test_agent_suggest_reports_an_overrule_of_the_head(
        app, httpd, policy_ckpt, fresh_store):
    """Fidelity is the component's metric: an overrule is a result, not a
    failure, and the chip has to say which of the two happened."""
    from pro.model.agent import loop
    S.load_predictor(Path(policy_ckpt))
    bans = list(app["mapping"])[:6]
    status, nm = _post(httpd, "/api/next-move", _board(bans=bans))
    first, second = nm["suggestions"][0]["name"], nm["suggestions"][1]["name"]
    app["agent_client"] = loop.scripted_client(
        answer="%s fits the open lane better.\nPICK: %s" % (second, second))
    status, d = _post(httpd, "/api/agent-suggest", _suggest_board(bans))
    assert status == 200, d
    assert d["suggestion"]["champion"] == second
    assert d["head"]["champion"] == first
    assert d["head"]["matches"] is False


def test_agent_suggest_publishes_the_series_and_a_later_board_keeps_it(
        app, httpd, fresh_store):
    """The agent reads the game on screen, not a board retyped into a prompt.
    Absent `series` means "unchanged", exactly like an absent fearless pool."""
    from pro.model.agent import loop
    app["agent_client"] = loop.scripted_client(answer="The list is flat.")
    bans = list(app["mapping"])[:6]
    status, d = _post(httpd, "/api/agent-suggest",
                      _suggest_board(bans, series=_series(3, 5)))
    assert status == 200, d
    status, st = _get(httpd, "/api/state")
    assert status == 200, st
    assert "series: game 3 of 5" in st["real"]["rendered"]
    assert "G2 10/10 (closed)" in st["real"]["rendered"]
    assert st["real"]["series"][0] == "series: game 3 of 5"
    # `/api/suggest` carries no series field; clearing the lines there would
    # make the next agent turn describe game 1 of a game-3 draft.
    status, _ = _post(httpd, "/api/suggest", _board(bans=bans))
    assert status == 200
    status, st = _get(httpd, "/api/state")
    assert "series: game 3 of 5" in st["real"]["rendered"]


def test_agent_suggest_refuses_a_board_the_store_cannot_build(
        app, httpd, fresh_store):
    """A recommendation is a claim about this board. A stale one is a 400."""
    pool = list(app["mapping"])
    bans, riot, other, third = pool[:6], pool[6], pool[7], pool[8]
    body = _suggest_board(
        bans, picks=[riot, riot, None, None, None, other, third,
                     None, None, None])
    status, d = _post(httpd, "/api/agent-suggest", body)
    assert status == 400, d
    assert "twice" in d["error"]


def test_a_complete_draft_has_no_agent_suggestion(app, httpd, fresh_store):
    """Twenty moves: no LLM call, no parsed champion, an explicit reason."""
    schedule = S.infer.move_plan(0)
    pool = list(app["mapping"])
    picks, bans, used = [None] * 10, [], {"blue": 0, "red": 0}
    for kind, side in schedule:
        riot = pool[len(bans) + used["blue"] + used["red"]]
        if kind == S.infer.MOVE_BAN:
            bans.append(riot)
        else:
            slot = used["blue"] if side == 0 else 5 + used["red"]
            picks[slot] = riot
            used["blue" if side == 0 else "red"] += 1
    status, d = _post(httpd, "/api/agent-suggest",
                      _suggest_board(bans, picks=picks))
    assert status == 200, d
    assert d["complete"] is True and d["stop"] == "draft complete"
    assert d["answer"] is None and d["head"] is None
    assert d["suggestion"]["champion"] is None
    assert d["state"]["complete"] is True


def test_agent_rank_without_a_key_is_an_outcome_not_a_crash(
        app, httpd, policy_ckpt, fresh_store, monkeypatch):
    """The ranking arm degrades like the single-move arm: a 200 with the board,
    the config and a head-fallback list, plus a sentence about the missing key -
    never a traceback."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    S.load_predictor(Path(policy_ckpt))
    app["agent_client"] = None
    status, d = _post(httpd, "/api/agent-rank",
                      _suggest_board(list(app["mapping"])[:6]))
    assert status == 200, d
    assert d["stop"] == "no client" and d["answer"] is None
    assert "LLM_API_KEY" in d["error"]
    # Without a key the parser falls back to the head's own legal shortlist.
    assert d["suggestions"], "head fallback should still rank the board"
    assert len(d["suggestions"]) == 8
    assert all(s["source"] == "head" for s in d["suggestions"])
    assert d["config"]["ablate"] == ["soloq_matchup", "sql"]
    assert d["config"]["k"] == 8 and d["config"]["max_steps"] == 5
    # The rerank is a rerank of the head, not a data dive: both free-form data
    # tools are ablated from this arm too.
    assert "sql" not in d["tools_offered"]
    assert "soloq_matchup" not in d["tools_offered"]
    assert d["head"] and d["head"]["champion"]
    assert "series: game 3 of 5" in d["state"]["rendered"]


def test_agent_rank_reads_the_board_and_returns_a_ranking(
        app, httpd, policy_ckpt, fresh_store):
    """The wiring: state published, tools called, a top-k list parsed, each with
    a confidence and a source, and the head fills whatever the agent left open."""
    from pro.model.agent import loop
    S.load_predictor(Path(policy_ckpt))
    bans = list(app["mapping"])[:6]
    status, nm = _post(httpd, "/api/next-move", _board(bans=bans, k=20))
    assert status == 200, nm
    names = [s["name"] for s in nm["suggestions"]]
    answer = "\n".join("%d. %s (%d%%)" % (i + 1, n, 80 - i * 7)
                       for i, n in enumerate(names[:4]))
    app["agent_client"] = loop.scripted_client(
        calls=("get_state", "predict_roles", "predict_next_move"), answer=answer)
    status, d = _post(httpd, "/api/agent-rank", _suggest_board(bans))
    assert status == 200, d
    assert d["stop"] == "answered" and d["complete"] is False
    sugs = d["suggestions"]
    assert len(sugs) == 8
    # The agent's own four keep their order and are model-sourced.
    assert [s["name"] for s in sugs[:4]] == names[:4]
    assert all(s["source"] == "model" for s in sugs[:4])
    assert round(sugs[0]["conf"], 2) == 0.80
    assert all(s["legal"] for s in sugs[:4])
    # The rest are filled from the head (head-sourced) and the list is honest:
    # every name is one the head listed, with no duplicates.
    assert all(s["source"] == "head" for s in sugs[4:])
    assert set(s["name"] for s in sugs).issubset(set(names))
    assert len(set(s["name"] for s in sugs)) == 8
    assert d["config"]["k"] == 8


def test_agent_rank_seeds_the_rerank_from_the_conversation(
        app, httpd, policy_ckpt, fresh_store):
    """The chat can seed the rerank: the seed messages are prepended, so a
    conversation that changed the agent's mind changes this list too."""
    from pro.model.agent import loop
    S.load_predictor(Path(policy_ckpt))
    bans = list(app["mapping"])[:6]
    status, nm = _post(httpd, "/api/next-move", _board(bans=bans))
    names = [s["name"] for s in nm["suggestions"]]
    app["agent_client"] = loop.scripted_client(
        calls=("get_state", "predict_next_move"),
        answer="%s (70%%)" % names[0])
    seed = [{"role": "user", "content": "what about ahri?"},
            {"role": "assistant", "content": "Ahri is strong mid."}]
    status, d = _post(httpd, "/api/agent-rank",
                      _suggest_board(bans, messages=seed))
    assert status == 200, d
    assert d["messages"][0]["role"] == "user"
    assert "what about ahri?" in d["messages"][0]["content"]
    assert d["suggestions"][0]["name"] == names[0]


def test_agent_rank_refuses_a_board_the_store_cannot_build(
        app, httpd, fresh_store):
    pool = list(app["mapping"])
    bans, riot, other, third = pool[:6], pool[6], pool[7], pool[8]
    body = _suggest_board(
        bans, picks=[riot, riot, None, None, None, other, third,
                     None, None, None])
    status, d = _post(httpd, "/api/agent-rank", body)
    assert status == 400, d
    assert "twice" in d["error"]


def test_a_complete_draft_has_no_agent_ranking(app, httpd, fresh_store):
    schedule = S.infer.move_plan(0)
    pool = list(app["mapping"])
    picks, bans, used = [None] * 10, [], {"blue": 0, "red": 0}
    for kind, side in schedule:
        riot = pool[len(bans) + used["blue"] + used["red"]]
        if kind == S.infer.MOVE_BAN:
            bans.append(riot)
        else:
            slot = used["blue"] if side == 0 else 5 + used["red"]
            picks[slot] = riot
            used["blue" if side == 0 else "red"] += 1
    status, d = _post(httpd, "/api/agent-rank",
                      _suggest_board(bans, picks=picks))
    assert status == 200, d
    assert d["complete"] is True and d["stop"] == "draft complete"
    assert d["answer"] is None and d["head"] is None
    assert d["suggestions"] == []


# ------------------------------------------------------- walkthrough of pros
def test_drafts_endpoints(app, httpd):
    status, d = _get(httpd, "/api/drafts?limit=5")
    if status == 500:
        pytest.skip("pro_drafts.pt not available")
    assert status == 200 and d["total"] > 0
    assert len(d["drafts"]) == 5
    status, one = _get(httpd, "/api/draft?i=%d" % d["drafts"][0]["i"])
    assert status == 200
    assert len(one["moves"]) == 20
    assert one["blue"] and one["red"]


def test_the_walkthrough_reaches_past_the_corpus_freeze(app, httpd):
    """Browsing is not a measurement: the dropdown shows the newest games.

    `pro_drafts.pt` is frozen at CORPUS_MAX_DATE so no recorded number moves;
    the walkthrough reads the unfrozen superset when it is on disk, so the
    Worlds final and the first 16.22 series are in the list.
    """
    if not S.PRO_DRAFTS_FULL_PT.exists():
        pytest.skip("pro_drafts_full.pt not available")
    status, d = _get(httpd, "/api/drafts?limit=1")
    if status == 500:
        pytest.skip("no pro draft cache")
    newest = d["drafts"][0]["date"]
    assert newest > "2026-11-08", newest


def test_draft_endpoint_exposes_the_real_fearless_pool(app, httpd):
    status, d = _get(httpd, "/api/drafts?limit=200")
    if status == 500:
        pytest.skip("pro_drafts.pt not available")
    rows = [r for r in d["drafts"] if r.get("fearless")]
    if not rows:
        pytest.skip("no fearless row in the sample")
    status, one = _get(httpd, "/api/draft?i=%d" % rows[0]["i"])
    assert status == 200
    assert len(one["unavailable"]) >= 1


# ------------------------------------------------- the retired panels are gone
def test_the_frontend_does_not_call_a_retired_route():
    """No page or script may still name an endpoint the server dropped.

    The panels were deleted together, so a leftover fetch would be a 404 in the
    console rather than a wrong number - but a 404 nobody reads is how a
    feature stays broken for a week.
    """
    static = Path(__file__).resolve().parent / "static"
    retired = ("/api/chat", "/api/tool", "/api/breakdown", "/api/draft-score",
               "/api/explain", "chat.js", "breakdown.js", "breakdown.html")
    for name in ("index.html", "app.js", "agent.js", "draft-order.js"):
        text = (static / name).read_text(encoding="utf-8")
        for route in retired:
            assert route not in text, "%s still references %s" % (name, route)


def test_the_browser_has_no_model_picker():
    """Models are chosen at startup. No dropdown, Load button or file upload.

    The HTTP `/api/load` and `/api/upload` routes remain for local scripts and
    are tested above, but no page can reach them: swapping weights mid-draft
    would leave answers in flight describing the previous checkpoint.
    """
    static = Path(__file__).resolve().parent / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "app.js").read_text(encoding="utf-8")
    css = (static / "style.css").read_text(encoding="utf-8")
    gone = ("ckpt-select", "policy-select", "btn-load-ckpt",
            "btn-load-policy", "btn-upload", "file-input", "model-pick",
            "refreshCheckpoints", "switchModel", "refreshPatches")
    for control in gone:
        for name, text in (("index.html", html), ("app.js", js),
                           ("style.css", css)):
            assert control not in text, "%s still has %s" % (name, control)
    # the CLI is still the supported way to choose them
    server = (static.parent / "server.py").read_text(encoding="utf-8")
    for flag in ("--ckpt", "--policy-ckpt", "--no-policy"):
        assert flag in server


def test_the_board_controls_and_agent_button_are_where_the_board_is():
    """First pick / reset live in <main>; the head's suggestion panel is
    followed by the agent's reranked-suggestions column, then the agent chat
    (shown by default now, no `hidden`); the Real draft walkthrough starts
    collapsed."""
    html = (Path(__file__).resolve().parent / "static" / "index.html").read_text(
        encoding="utf-8")
    main = html[html.index("<main>"):html.index("</main>")]
    assert 'id="first-pick"' in main and 'id="btn-reset"' in main
    assert 'id="first-pick"' not in html[:html.index("<main>")]
    # The head's own list, then the agent's rerank column, then the chat it
    # opens - in that order.
    assert (html.index('id="nextmove"') < html.index('id="agentsuggest"')
            < html.index('id="agent"'))
    # The agent chat is shown by default (the feature toggle can hide it).
    assert 'id="agent" class="hidden"' not in html
    # The agent-features toggle lives in the header.
    assert 'id="agent-features"' in html
    assert 'id="wt-bar" class="hidden"' in html
    assert 'id="btn-wt"' in html
    js = (Path(__file__).resolve().parent / "static" / "app.js").read_text(
        encoding="utf-8")
    assert "function setWalkthrough(" in js
    assert "walkthroughVisible()" in js
    css = (Path(__file__).resolve().parent / "static" / "style.css").read_text(
        encoding="utf-8")
    assert "#wt-bar.hidden" in css, ".hidden alone cannot beat an ID selector"
