"""Tests for the draft-relevance filter.

Every "kept" case here is a bug the filter actually had at some point -
a denylist's failure mode is quietly deleting real balance changes, and
these are the four it got wrong on the real corpus:

  * "Role Quests" filed as meta -> it is the lane-phase quest system,
    the biggest Summoner's Rift draft change of the 2025 season.
  * `\\bwards?\\b` filed as cosmetic -> also swallowed "Support Control
    Wards" and "Ward Experience", both real SR changes.
  * "Unsealed Spellbook" filed as altmode -> that is a *rune*; only
    "Ultimate Spellbook" is a game mode.
  * "Champion Buffs" under an ARAM heading -> inheritance has to drop
    the child, not keep it because the child's own title is clean.

Run with:  python -m pytest "data/patch notes/test_clean.py" -q
"""

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


clean = _load("clean")


def verdict(title, level="standard"):
    return clean.classify(title, level)[0]


# ------------------------------------------------- things that must survive

@pytest.mark.parametrize("title", [
    "Role Quests",
    "Support Control Wards",
    "Ward Experience",
    "Unsealed Spellbook",
    "Champions",
    "Items",
    "Runes",
    "Summoner Spells",
    "Terrain Changes",
    "Lane Swaps",
    "Epic Objective Tuning",
    "Early Game Adjustments",
    "Snowballing Adjustments",
    "Champion Radii (Radiuses?)",
    "Atakhan, Blood Roses, Feats, and Kills",
    "Faelights and Vision Changes",
    "Turrets",
    "Minions",
    "Bounty Adjustments",
    "Ornn Item Upgrades",
    "Jungle Adjustments",
    "Teleport",
    "Smite",
    "Stat Shard Changes",
    "Trinket Changes",
])
def test_sr_gameplay_is_kept(title):
    assert verdict(title) == "keep"


@pytest.mark.parametrize("title", [
    "Bot Lane Changes",   # `\\bbots?\\b` must not eat this
    "Ward Item Tuning",   # only *ward skins* are cosmetic
])
def test_near_misses_are_kept(title):
    assert verdict(title) == "keep"


def test_unknown_headings_default_to_keep():
    """Conservative: a new section is kept and reported, never guessed away."""
    assert verdict("Some Brand New Section") == "keep"


# ------------------------------------------------- things that must go

@pytest.mark.parametrize("title,family", [
    ("ARAM", "altmode"),
    ("ARAM Balance Changes", "altmode"),
    ("Arena", "altmode"),
    ("Arena Balance Adjustments", "altmode"),
    ("Ultimate Spellbook", "altmode"),
    ("Doom Bots", "altmode"),
    ("Clash - Noxus Cup", "altmode"),
    ("New Cosmetics", "cosmetic"),
    ("Upcoming Skins & Chromas", "cosmetic"),
    ("Mythic Shop Rotation", "cosmetic"),
    ("Patch Highlights", "summary"),
    ("Competitive", "esports"),
    ("Client", "client"),
    ("Ranked Season 2024", "meta"),
    ("Blue Essence Emporium", "cosmetic"),
])
def test_off_topic_is_dropped(title, family):
    assert clean.classify(title, "standard") == ("drop", family)


# ------------------------------------------------------------- inheritance

def test_child_inherits_drop_from_parent():
    body = (
        "### ARAM\n\nintro\n\n"
        "#### Champion Buffs\n\n- Viego: 5% => 10%\n\n"
        "### Champions\n\n#### Diana\n\n- AP Ratio : 40% => 50%\n"
    )
    out, log = clean.filter_text(body, "standard")
    assert "Champion Buffs" not in out
    assert "Viego" not in out
    assert "Diana" in out and "40% => 50%" in out
    buckets = {r["title"]: r["bucket"] for r in log}
    assert buckets["ARAM"] == "drop"
    assert buckets["Champion Buffs"] == "drop"


def test_child_of_kept_parent_is_still_dropped_if_it_matches():
    body = ("### League of Legends V26.20\n\n#### Champions\n\n- x\n\n"
            "#### ARAM\n\n- y\n")
    out, _ = clean.filter_text(body, "standard")
    assert "#### Champions" in out
    assert "#### ARAM" not in out


# ------------------------------------------------------------------ levels

def test_core_drops_bugfixes_standard_keeps_them():
    assert clean.classify("Bugfixes & QoL Changes", "core") == ("drop", "bugfix")
    assert clean.classify("Bugfixes & QoL Changes", "standard")[0] == "keep"


def test_full_keeps_everything():
    for title in ("ARAM", "New Cosmetics", "Bugfixes & QoL Changes"):
        assert clean.classify(title, "full") == ("keep", None)


# ---------------------------------------------------------------- integrity

def test_output_is_a_subsequence_of_input():
    """The filter may only delete whole sections - never rewrite a line.

    This is the property that makes the cleaned corpus trustworthy: any
    surviving line is byte-identical to the source line.
    """
    body = (
        "# Patch 1.0 notes\n\n> intro\n\n"
        "### Champions\n\n#### Diana\n\n- AP Ratio : 40% => 50%\n\n"
        "### ARAM\n\n- something irrelevant\n\n"
        "### Items\n\n#### Bloodthirster\n\n- Life Steal : 15% => 18%\n"
    )
    out, _ = clean.filter_text(body, "standard")
    src, dst = body.splitlines(), out.splitlines()
    i = 0
    for line in dst:
        while i < len(src) and src[i] != line:
            i += 1
        assert i < len(src), "line not found in source: %r" % line
        i += 1
    assert "irrelevant" not in out


def test_ability_headings_do_not_split_sections():
    """##### is an ability under an entity; splitting there would
    misclassify every ability line as a top-level section."""
    body = "### Champions\n\n#### Diana\n\n##### Passive - Moonsilver Blade\n\n- x\n"
    _, log = clean.split_sections(body)
    assert [r["title"] for r in log] == ["Champions", "Diana"]


def test_section_split_handles_no_headings():
    pre, sections = clean.split_sections("just prose\nno headings\n")
    assert sections == []
    assert pre == ["just prose", "no headings"]
