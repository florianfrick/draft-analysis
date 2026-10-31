"""Tests for frontend/llm.py - retrieval, grounding, cache, degradation.

No network anywhere: the LLM is a function that returns a string. What is
worth testing here is not the prose (nobody can assert on that) but the three
mechanical guarantees:

  * retrieval compares the board *as it was at the query move*, not the final
    draft - otherwise a row that shares champions with us later looks like
    evidence about a decision made earlier;
  * every citation resolves, and unresolvable ones are dropped and retried;
  * no key means evidence without prose, never an error.
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import llm  # noqa: E402

PAD_FROM = 160          # filler champion ids, below the real 173


def make_row(gameid, prefix, patch="16.20", first_pick=0, blue_win=1,
             date_ordinal=1, league="LCK"):
    """A pro row with a 20-move list padded to length by unique fillers."""
    moves = list(prefix)
    filler = PAD_FROM
    while len(moves) < llm.N_MOVES:
        moves.append((1, len(moves) % 2, filler, 0))
        filler -= 1
    return {"gameid": gameid, "patch": patch, "first_pick": first_pick,
            "blue_win": blue_win, "date_ordinal": date_ordinal,
            "league": league, "date": "2026-09-01 00:00:00",
            "blue_team": "T1", "red_team": "GEN", "moves": moves}


def bans(*champs):
    return [(0, i % 2, c, -100) for i, c in enumerate(champs)]


def picks(*champs):
    return [(1, i % 2, c, i % 5) for i, c in enumerate(champs)]


BOARD_CHAMPS = [1, 2, 3, 4, 5, 6]


def test_retrieval_ranks_the_shared_board_first():
    close = make_row("close", bans(1, 2, 3) + picks(4, 5, 6) + picks(7, 8))
    far = make_row("far", bans(1, 2, 90) + picks(91, 92, 93) + picks(7, 8))
    index = llm.ProIndex([far, close])
    games, relaxed = index.similar(BOARD_CHAMPS, move_idx=6, first_pick=0, k=2)
    assert [g["gameid"] for g in games][0] == "close"
    assert games[0]["overlap"] > games[1]["overlap"]
    assert relaxed is False


def test_retrieval_compares_the_prefix_not_the_final_draft():
    """The trap: a row whose final draft contains our champions but whose
    board at move 6 does not. It must not be shown as evidence."""
    leaky = make_row("leaky", picks(80, 81, 82, 83, 84, 85)
                     + picks(*BOARD_CHAMPS))
    honest = make_row("honest", bans(1, 2, 3) + picks(4, 5, 6) + picks(70, 71))
    index = llm.ProIndex([leaky, honest])
    games, _ = index.similar(BOARD_CHAMPS, move_idx=6, first_pick=0, k=2)
    assert [g["gameid"] for g in games][0] == "honest"
    by_id = {g["gameid"]: g for g in games}
    assert by_id["leaky"]["overlap"] == 0.0


def test_retrieval_prefers_the_same_first_pick_side():
    same = make_row("same", bans(1, 2, 3) + picks(4, 5, 6), first_pick=0)
    other = make_row("other", bans(1, 2, 3) + picks(4, 5, 6), first_pick=1)
    index = llm.ProIndex([other, same])
    games, relaxed = index.similar(BOARD_CHAMPS, move_idx=6, first_pick=0, k=1)
    assert games[0]["gameid"] == "same"
    assert relaxed is False


def test_retrieval_says_so_when_it_has_to_relax_the_side():
    only_other = make_row("other", bans(1, 2, 3) + picks(4, 5, 6), first_pick=1)
    index = llm.ProIndex([only_other])
    games, relaxed = index.similar(BOARD_CHAMPS, move_idx=6, first_pick=0, k=1)
    assert relaxed is True and len(games) == 1


def test_retrieval_reports_the_move_that_was_played():
    row = make_row("g1", bans(1, 2, 3) + picks(4, 5, 6) + [(1, 1, 42, 3)])
    index = llm.ProIndex([row])
    games, _ = index.similar(BOARD_CHAMPS, move_idx=6, first_pick=0, k=1)
    assert games[0]["next_kind"] == "pick"
    assert games[0]["next_champ_idx"] == 42
    assert games[0]["next_role"] == 3


def test_empty_corpus_returns_nothing_instead_of_raising():
    games, relaxed = llm.ProIndex([]).similar(BOARD_CHAMPS, 6, 0, k=3)
    assert games == [] and relaxed is False


# ------------------------------------------------------------- patch notes
NOTES = """# Patch 16.20 notes

### Champions

#### Ahri

Charm duration increased.

- Charm Duration : 1.5 seconds ⇒ 1.8 seconds

#### Azir

Base attack damage reduced.

- Base AD : 56 ⇒ 53

#### Zed

W cooldown reduced.
"""


@pytest.fixture
def notes_dir(tmp_path, monkeypatch):
    (tmp_path / "patch_16_20.md").write_text(NOTES, encoding="utf-8")
    monkeypatch.setattr(llm, "PATCH_DIR", tmp_path)
    return tmp_path


def test_bullets_reads_only_the_named_champions(notes_dir):
    found, missing = llm.bullets("16.20", ["Ahri", "Zed", "Notachampion"])
    assert set(found) == {"Ahri", "Zed"}
    assert "Charm Duration" in found["Ahri"]
    assert "Charm Duration" not in found["Zed"]
    assert missing == ["Notachampion"]


def test_bullets_truncate_long_sections(notes_dir):
    found, _ = llm.bullets("16.20", ["Ahri"], max_chars=40)
    assert found["Ahri"].endswith("...")
    assert len(found["Ahri"]) <= 43


def test_bullets_survive_a_patch_with_no_notes(notes_dir):
    found, missing = llm.bullets("99.99", ["Ahri"])
    assert found == {} and missing == ["Ahri"]


def test_prompt_separates_unchanged_from_unknown(notes_dir):
    """"Not changed this patch" and "no notes for this patch" are different
    claims, and the model should not be allowed to confuse them."""
    board = {"move_idx": 8, "side": "blue", "first_pick": 0, "patch": "16.20",
             "bans": [], "blue": ["Ahri"], "red": []}
    sugg = [("Azir", 0.3), ("Ahri", 0.2)]
    # Notes exist for 16.20, but neither candidate was touched.
    msg = llm.build_prompt(board, sugg, [], {}, notes_available=True)[1]
    assert "no champion-level changes" in msg["content"]
    assert "do not claim anything about patch changes" not in msg["content"]
    # Same empty dict, but the patch itself is unknown to the corpus.
    msg = llm.build_prompt(board, sugg, [], {}, notes_available=False)[1]
    assert "no notes for this patch" in msg["content"]


def test_explain_reports_whether_the_patch_was_found(notes_dir):
    index = llm.ProIndex([make_row("g", picks(10, 11))])
    board = {"move_idx": 2, "side": "blue", "first_pick": 0, "patch": "99.99",
             "bans": [], "blue": ["Ahri"], "red": [], "champ_ids": [10, 11]}
    out = llm.explain(board, [("Ahri", 0.4)], index, client=None)
    assert out["patch"]["available"] is False
    board["patch"] = "16.20"
    out = llm.explain(board, [("Ahri", 0.4)], index, client=None)
    assert out["patch"]["available"] is True
    assert out["patch"]["notes"]["Ahri"]  # the fixture notes have Ahri


def test_notes_are_pulled_for_every_candidate_not_just_the_top(notes_dir):
    """Otherwise the candidate that happens to be in this patch wins the
    argument by default."""
    index = llm.ProIndex([make_row("g", picks(10, 11))])
    board = {"move_idx": 2, "side": "blue", "first_pick": 0, "patch": "16.20",
             "bans": [], "blue": ["Ahri"], "red": [], "champ_ids": [10, 11]}
    cands = [("A%d" % i, 0.1) for i in range(6)] + [("Ahri", 0.05), ("Zed", 0.01)]
    out = llm.explain(board, cands, index, client=None)
    assert set(out["patch"]["notes"]) == {"Ahri", "Zed"}
    assert "Zed" not in out["patch"]["missing"]


def test_the_corpus_can_be_absent_entirely(tmp_path, monkeypatch):
    """A clean clone has no `data/patch notes/` at all, not just no file for
    this patch. The endpoint is supposed to keep working."""
    monkeypatch.setattr(llm, "PATCH_DIR", tmp_path / "not-checked-out")
    found, missing = llm.bullets("16.20", ["Ahri", "Zed"])
    assert found == {} and missing == ["Ahri", "Zed"]
    index = llm.ProIndex([make_row("g", picks(10, 11))])
    board = {"move_idx": 2, "side": "blue", "first_pick": 0, "patch": "16.20",
             "bans": [], "blue": ["Ahri"], "red": [], "champ_ids": [10, 11]}
    out = llm.explain(board, [("Ahri", 0.4)], index, client=None)
    assert out["patch"]["available"] is False
    assert out["retrieval"]["games"]  # evidence that does not depend on it
    msg = llm.build_prompt(board, [("Ahri", 0.4)], out["retrieval"]["games"],
                           {}, notes_available=out["patch"]["available"])[1]
    assert "do not claim anything about patch changes" in msg["content"]


# -------------------------------------------------------------- validation
def games_fixture():
    return [{"gameid": "G-1", "patch": "16.20", "league": "LCK",
             "date": "2026-09-01", "blue_team": "T1", "red_team": "GEN",
             "blue_win": 1, "overlap": 0.5, "next_kind": "pick",
             "next_side": "blue", "next_champ": "Ahri", "next_role": 2}]


def test_validate_keeps_resolvable_evidence():
    answer = {"reasoning": ["Mid priority matters here."],
              "recommend": "Ahri", "why": "Wins the mid lane.",
              "their_answer": "Azir", "risk": "Falls off.",
              "evidence": [{"type": "game", "id": "G-1"},
                           {"type": "patch", "line": "Charm Duration : "
                                                     "1.5 seconds ⇒ 1.8 "
                                                     "seconds"}]}
    clean, problems = llm.validate(answer, games_fixture(),
                                   {"Ahri": "Charm Duration : 1.5 seconds ⇒ "
                                            "1.8 seconds"},
                                   ["Ahri", "Azir"])
    assert problems == []
    assert len(clean["evidence"]) == 2
    assert clean["unverified"] == []


def test_validate_keeps_reasoning_in_order():
    answer = {"reasoning": ["First.", "Second.", "Third."],
              "recommend": "Ahri", "why": "x", "evidence": []}
    clean, problems = llm.validate(answer, games_fixture(), {}, ["Ahri"])
    assert clean["reasoning"] == ["First.", "Second.", "Third."]
    assert problems == []


def test_reasoning_is_capped_and_never_mixed_into_evidence():
    """The steps are the model's own account. They must not be presented as
    citations, and one runaway answer must not fill the panel."""
    answer = {"reasoning": ["step %d" % i for i in range(20)],
              "recommend": "Ahri", "why": "x", "evidence": []}
    clean, _ = llm.validate(answer, games_fixture(), {}, ["Ahri"])
    assert len(clean["reasoning"]) == llm.MAX_STEPS
    assert clean["evidence"] == [] and clean["unverified"] == []


def test_a_single_string_reasoning_is_accepted():
    clean, problems = llm.validate(
        {"reasoning": "Only one step.", "recommend": "Ahri", "why": "x",
         "evidence": []}, games_fixture(), {}, ["Ahri"])
    assert clean["reasoning"] == ["Only one step."]
    assert problems == []


def test_missing_reasoning_is_a_problem():
    clean, problems = llm.validate(
        {"recommend": "Ahri", "why": "x", "evidence": []},
        games_fixture(), {}, ["Ahri"])
    assert clean["reasoning"] == []
    assert any("no reasoning steps" in p for p in problems)


def test_validate_drops_a_fabricated_game():
    answer = {"recommend": "Ahri", "why": "x",
              "evidence": [{"type": "game", "id": "G-1"},
                           {"type": "game", "id": "G-999"}]}
    clean, problems = llm.validate(answer, games_fixture(), {}, ["Ahri"])
    assert [e["id"] for e in clean["evidence"]] == ["G-1"]
    assert len(clean["unverified"]) == 1
    assert any("not in the retrieved set" in p for p in problems)


def test_validate_drops_a_fabricated_patch_line():
    answer = {"recommend": "Ahri", "why": "x",
              "evidence": [{"type": "patch", "line": "Charm now stuns for "
                                                     "two seconds"}]}
    clean, problems = llm.validate(answer, [], {"Ahri": "Charm duration "
                                                        "increased."}, ["Ahri"])
    assert clean["evidence"] == []
    assert len(clean["unverified"]) == 1
    assert any("not found in the notes" in p for p in problems)


def test_validate_rejects_a_recommendation_outside_the_candidates():
    clean, problems = llm.validate({"recommend": "Teemo", "why": "x"},
                                   [], {}, ["Ahri"])
    assert clean["recommend"] is None
    assert any("not one of the candidates" in p for p in problems)


def test_validate_accepts_a_case_different_candidate_name():
    clean, _ = llm.validate({"recommend": "ahri", "why": "x"}, [], {},
                            ["Ahri"])
    assert clean["recommend"] == "Ahri"


# ---------------------------------------------------------------- explain
def board(**kw):
    base = {"move_idx": 6, "side": "blue", "first_pick": 0,
            "first_pick_side": "blue", "patch": "16.20",
            "bans": ["Zed", "Azir", "Ahri"], "blue": ["Sion"],
            "red": ["Jinx"], "champ_ids": BOARD_CHAMPS}
    base.update(kw)
    return base


def good_answer(**kw):
    ans = {"reasoning": ["The board has no engage yet.",
                         "G-1 answered it with a pick-first support."],
           "recommend": "Ahri", "why": "Charm gives the pick a follow-up.",
           "their_answer": "Azir", "risk": "Low damage if it stalls.",
           "evidence": [{"type": "game", "id": "G-1"}]}
    ans.update(kw)
    return json.dumps(ans)


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.model = "fake-1"

    def __call__(self, messages):
        self.calls += 1
        return self.replies.pop(0)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


def index_fixture():
    row = make_row("G-1", bans(1, 2, 3) + picks(4, 5, 6)
                   + [(1, 0, 42, 2)])
    return llm.ProIndex([row])


def test_explain_without_a_key_returns_evidence_and_a_reason(cache_dir):
    out = llm.explain(board(), [("Ahri", 0.4)], index_fixture(), client=None)
    assert out["explanation"] is None
    assert out["llm"]["configured"] is False
    assert "LLM_API_KEY" in out["llm"]["reason"]
    assert out["retrieval"]["games"][0]["gameid"] == "G-1"


def test_explain_caches_and_does_not_call_twice(cache_dir):
    client = FakeClient([good_answer()])
    first = llm.explain(board(), [("Ahri", 0.4)], index_fixture(),
                        client=client, names=["Ahri"] + [""] * 200)
    assert client.calls == 1
    assert first["explanation"]["cached"] is False
    second = llm.explain(board(), [("Ahri", 0.4)], index_fixture(),
                         client=client, names=["Ahri"] + [""] * 200)
    assert client.calls == 1
    assert second["explanation"]["cached"] is True


def test_explain_retries_once_when_evidence_does_not_resolve(cache_dir):
    bad = good_answer(evidence=[{"type": "game", "id": "MADE-UP"}])
    client = FakeClient([bad, good_answer()])
    out = llm.explain(board(), [("Ahri", 0.4)], index_fixture(), client=client,
                      names=["Ahri"] + [""] * 200)
    assert client.calls == 2
    assert out["explanation"]["ok"] is True
    assert out["explanation"]["evidence"] == [{"type": "game", "id": "G-1"}]


def test_explain_returns_the_answer_with_unverified_marked(cache_dir):
    bad = good_answer(evidence=[{"type": "game", "id": "MADE-UP"}])
    client = FakeClient([bad, bad])
    out = llm.explain(board(), [("Ahri", 0.4)], index_fixture(), client=client,
                      names=["Ahri"] + [""] * 200)
    assert client.calls == 2
    assert out["explanation"]["ok"] is False
    assert out["explanation"]["unverified"]


def test_explain_survives_a_broken_response(cache_dir):
    client = FakeClient(["not json at all", "still not json"])
    out = llm.explain(board(), [("Ahri", 0.4)], index_fixture(), client=client,
                      names=["Ahri"] + [""] * 200)
    assert out["explanation"] is None
    assert "unparseable" in out["llm"]["error"]


def test_explain_reports_a_client_failure_without_losing_retrieval(cache_dir):
    def boom(messages):
        raise OSError("connection refused")

    out = llm.explain(board(), [("Ahri", 0.4)], index_fixture(), client=boom)
    assert out["explanation"] is None
    assert "connection refused" in out["llm"]["error"]
    assert out["retrieval"]["games"]


def test_prompt_carries_the_board_and_the_rules(cache_dir):
    messages = llm.build_prompt(board(), [("Ahri", 0.4)],
                                games_fixture(), {"Ahri": "Charm up."})
    assert messages[0]["role"] == "system"
    assert "Never invent" in messages[0]["content"]
    user = messages[1]["content"]
    assert "MOVE: 6 of 20 (picks1)" in user
    assert "Ahri  40.0%" in user
    assert "G-1" in user
    assert "Charm up." in user


# ------------------------------------------------------------------ misc
def test_strip_fences_handles_markdown_wrapping():
    assert llm._strip_fences('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm._strip_fences('{"a": 1}') == {"a": 1}
    assert llm._strip_fences("nothing here") is None


def test_client_from_env_needs_a_key():
    assert llm.client_from_env({}) is None
    client = llm.client_from_env({"LLM_API_KEY": "k", "LLM_MODEL": "m"})
    assert callable(client) and client.model == "m"


def test_patch_key_orders_patch_strings():
    assert llm.patch_key("16.20") > llm.patch_key("9.10")
    assert llm.patch_key("?") is None


def test_retrieval_breaks_overlap_ties_by_patch_proximity():
    """Equal overlap: a recent board is better evidence than an old one."""
    old = make_row("old", bans(1, 2, 3) + picks(4, 5, 6), patch="12.01",
                   date_ordinal=999)
    recent = make_row("recent", bans(1, 2, 3) + picks(4, 5, 6),
                      patch="16.19", date_ordinal=1)
    index = llm.ProIndex([old, recent])
    games, _ = index.similar(BOARD_CHAMPS, move_idx=6, first_pick=0, k=2,
                             patch="16.20")
    assert [g["gameid"] for g in games] == ["recent", "old"]


def test_retrieval_still_lets_overlap_beat_patch_proximity():
    """A much better board match on an older patch must still win."""
    exact_old = make_row("exact", bans(1, 2, 3) + picks(4, 5, 6),
                         patch="12.01")
    weak_new = make_row("weak", bans(1, 2, 90) + picks(91, 92, 93),
                        patch="16.19")
    index = llm.ProIndex([exact_old, weak_new])
    games, _ = index.similar(BOARD_CHAMPS, move_idx=6, first_pick=0, k=2,
                             patch="16.20")
    assert games[0]["gameid"] == "exact"
