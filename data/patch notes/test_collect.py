"""Regression tests for the patch-notes extractor.

The bug these guard: Riot's CMS before 12.19 marked each stat delta as a
``div.attribute-change`` of spans rather than an ``li``. A block walker
that only knew headings/lists/paragraphs therefore kept the champion
heading and dropped every number under it, so patch_12_01.md through
patch_12_18.md came out as ~100 chars of prose with no values at all -
and nothing raised, because the file was still valid markdown.

Run with:  python -m pytest "data/patch notes/test_collect.py" -q
"""

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("collect", HERE / "collect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


collect = _load()


def _riot(body):
    """Wrap a body fragment the way Riot's article container does."""
    return (
        '<html><body><h1>Patch 12.10 notes</h1>'
        '<div class="width-small"><div id="patch-notes-container">'
        + body
        + "</div></div></body></html>"
    )


# ---------------------------------------------------------------- change divs

def test_attribute_change_div_is_rendered():
    """The 12.01-12.18 markup: before/after live in sibling spans."""
    md, _ = collect.extract_riot(_riot(
        '<h3 class="change-title">Diana</h3>'
        '<h4 class="change-detail-title ability-title">'
        "Passive - Moonsilver Blade</h4>"
        '<div class="attribute-change">'
        '<span class="attribute">AP Ratio</span> '
        '<span class="attribute-before">40%</span> '
        '<span class="change-indicator">\u21d2</span> '
        '<span class="attribute-after">50%</span>'
        "</div>"
    ))
    assert "- AP Ratio : 40% \u21d2 50%" in md
    # the entity and ability headings must survive around it
    assert "#### Diana" in md
    assert "##### Passive - Moonsilver Blade" in md


def test_change_div_without_before_keeps_the_value():
    """New effects have an after value and a "new" badge, no before."""
    md, _ = collect.extract_riot(_riot(
        '<div class="attribute-change">'
        '<span class="attribute">Unique Passive - Engorge</span> '
        '<span class="new">new</span> '
        '<span class="attribute-after">10-40 Attack Damage</span>'
        "</div>"
    ))
    assert "10-40 Attack Damage" in md
    assert md.count("- ") >= 1


def test_change_div_keeps_trailing_note():
    md, _ = collect.extract_riot(_riot(
        '<div class="attribute-change">'
        '<span class="attribute">Total Cost</span> '
        '<span class="attribute-after">3200</span> '
        "<em>(Note: Unchanged)</em>"
        "</div>"
    ))
    assert "- Total Cost : 3200 (Note: Unchanged)" in md


def test_multi_value_change_div():
    md, _ = collect.extract_riot(_riot(
        '<div class="attribute-change">'
        '<span class="attribute">Mana Cost</span> '
        '<span class="attribute-before">60/55/50/45/40</span> '
        '<span class="change-indicator">\u21d2</span> '
        '<span class="attribute-after">55/50/45/40/35</span>'
        "</div>"
    ))
    assert "60/55/50/45/40 \u21d2 55/50/45/40/35" in md


def test_change_div_is_not_duplicated_by_its_spans():
    md, _ = collect.extract_riot(_riot(
        '<div class="attribute-change">'
        '<span class="attribute">AP Ratio</span> '
        '<span class="attribute-before">40%</span> '
        '<span class="change-indicator">\u21d2</span> '
        '<span class="attribute-after">50%</span>'
        "</div>"
    ))
    assert md.count("40%") == 1
    assert md.count("\u21d2") == 1


# ------------------------------------------------------------- 13.x+ markup

def test_list_form_still_works():
    """12.19+ moved the same content into <li>, which must keep working."""
    md, _ = collect.extract_riot(_riot(
        "<h3 class=\"change-title\">Bloodthirster</h3>"
        "<ul><li><strong>Life Steal</strong>: 15% \u21d2 18%</li></ul>"
    ))
    assert "- Life Steal : 15% \u21d2 18%" in md


# --------------------------------------------------------------------- audit

def test_count_deltas_recognises_both_phrasings():
    riot_style = "- AP Ratio : 40% \u21d2 50%"
    wiki_style = "- Base mana reduced to 480 from 450."
    assert collect.count_deltas(riot_style) >= 1
    assert collect.count_deltas(wiki_style) >= 1


def test_count_deltas_is_zero_for_prose_only():
    assert collect.count_deltas("Diana\n\nPassive AP ratio increased.") == 0


@pytest.mark.parametrize("dataset,riot", [
    ("16.20", "26.20"),   # 2026 season drops its leading digit
    ("15.20", "25.20"),
    ("12.01", "12.01"),
    ("13.10", "13.10"),
])
def test_patch_number_mapping(dataset, riot):
    assert collect.riot_patch(dataset) == riot
