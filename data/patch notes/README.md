# Patch notes corpus

Balance/patch notes for every patch in `data/pro play/pro_drafts.pt`,
as markdown, collected by `collect.py`.

## Contents

- `patch_<major>_<minor>.md` — one file per dataset patch (106 files).
- `index.json` — source, URL, publish date and size for each patch.
- `collect.py` — the collector. Re-runnable; skips files that exist
  unless you pass `--refresh`.
- `clean.py` — derives `clean/`, a draft-relevant subset. See below.
- `test_collect.py`, `test_clean.py` — regression tests (no network).

Each file starts with front-matter:

```
---
dataset_patch: 16.20
riot_patch: 26.20
source: wiki
url: https://wiki.leagueoflegends.com/en-us/V26.20
published: 2026-10-07
chars: 24244
---
```

## The patch numbering trap

The pro dataset drops the leading digit for the 2025/2026 seasons:

| dataset | Riot / wiki |
|---|---|
| `15.20` | `25.20` |
| `16.20` | `26.20` |
| `12.01`–`14.24` | unchanged |

Verified against release dates across all 106 patches: every file's
`published` date is monotonically increasing with patch order, and
Riot 25.20 = 2025-10-07 matches the dataset's first `15.20` game on
2025-10-11.

## Sources, in priority order

1. **Riot official notes** — `leagueoflegends.com/en-us/news/
   game-updates/patch-<major>-<minor>-notes/`. 86 of 106 patches.
   Keeps the `⇒` before/after arrows.
2. **LoL Wiki** — `wiki.leagueoflegends.com/en-us/V<major>.<minor>`.
   The other 20, which is every 2026 patch (Riot's news site 404s on
   the whole 26.x line). Champion names arrive as bare `<a>` elements
   rather than headings, so `collect.py` promotes them to `#### Name`;
   without that the change lists lose their subject.

The wiki half also needed three truncations. Its article container
carries `navbox-wrapper` (footer link farm: "Patch History", "Riot
Games", "Microsoft Windows, macOS") and `hidden-metadata` (champion
ability-tooltip tables). `navbox-wrapper` does not match the `.navbox`
selector — that one-word gap was 59% of the article text, 267KB of
645KB, arriving as if it were patch notes. Bare links above the first
`<h2>` are the "Patch chronology" nav and are not promoted. Net effect
on `patch_16_20.md`: 24,244 → 11,381 chars with every real section
intact.

## Numeric deltas: two markup eras

Riot changed its CMS mid-2022, and the two eras need different handling:

| era | markup | rendered as |
|---|---|---|
| 12.01–12.18 | `<div class="attribute-change">` with `span.attribute` / `attribute-before` / `attribute-after` | `- AP Ratio : 40% ⇒ 50%` |
| 12.19+ | plain `<li>` with inline text | `- Life Steal : 15% ⇒ 18%` |

The pre-12.19 form is a `<div>`, not a `<li>`. An extractor that only
walks headings, lists, paragraphs and tables keeps the champion name
and silently drops every number under it — which is exactly what
happened on the first pass: 17 patches (12.01–12.16, 12.18) came out as
bare headings, 1,911 stat deltas missing, and nothing errored because
the files were still valid markdown. `render_change_div()` handles the
div form; `test_collect.py` pins both eras.

Because the source splits before/after into separate spans, the
pre-12.19 files are actually the *most* machine-readable of the set —
old and new values never have to be teased out of prose.

## Auditing the corpus

```
python "data/patch notes/collect.py" --audit
```

Prints per-file chars / `⇒` count / numeric-delta count and exits
non-zero if any patch carries no numbers. Use it after a refetch; it is
the cheap check that would have caught the missing-delta bug.

## Draft-relevance filter (`clean.py`)

`clean/` is a derived subset: the same 106 patches with the sections
that cannot change a Summoner's Rift draft removed — skins, chromas,
Mythic Shop rotations, ARAM, Arena, URF, Ultimate Spellbook, Clash, TFT,
ranked/queue/monetisation changes, esports, client/tech notes.

```
python "data/patch notes/clean.py"                 # -> clean/
python "data/patch notes/clean.py" --level core    # also drop bugfix/QoL
python "data/patch notes/clean.py" --level full    # unfiltered copy
python "data/patch notes/clean.py" --report        # every heading + verdict
```

On the current corpus: **3,072,490 → 2,272,603 chars (74.0% kept)**.
Dropped by family: altmode 14.7%, cosmetic 4.5%, meta 2.8%, client 2.0%,
summary 0.7%, esports 0.5%, event 0.2%. `--level core` drops a further
8.3% of bugfix/QoL text.

**Why a denylist, not an allowlist.** Riot does not file balance under a
few predictable headings. Alongside "Champions" and "Items" there are
"terrain changes", "lane swaps", "epic objective tuning", "early game
adjustments", "snowballing adjustments", "atakhan, blood roses, feats,
and kills", "champion radii". An allowlist of {champions, items, runes}
would silently delete all of it. So the filter drops families it can
name confidently and keeps everything else.

**Why section level only.** Cosmetic bullets inside `### Champions`
("Pengu Garen skin re-enabled") are 0.3% of 4,595 bullets — not worth
sentence-level rules, which is where automatic cleaning starts making
expensive mistakes.

**The guarantee.** The cleaned body is a strict ordered subsequence of
the raw body (verified 106/106): the filter only deletes whole sections,
never rewrites or reorders a line. Any line you read in `clean/` is
byte-identical to the source.

**The failure mode, and how it is caught.** A denylist's risk is
deleting real balance, so `--report` lists every top-level heading that
matched no rule. Four real errors were caught this way and are pinned
by `test_clean.py`:

| rule as written | what it wrongly deleted |
|---|---|
| `role quests` → meta | **Role Quests**, the lane-phase quest system and the biggest SR draft change of 2025 |
| `\bwards?\b` → cosmetic | **Support Control Wards**, **Ward Experience** — both real SR changes |
| `spellbook` → altmode | **Unsealed Spellbook**, which is a *rune*, not the Ultimate Spellbook mode |
| `\bbots?\b` → client | would have eaten a **Bot Lane** section |

Run `--report` after each new patch: the cost is skimming a list, not
hand-editing 106 files.

## Known caveats

- Wiki-sourced files express some changes in prose ("reduced to X from
  Y") where Riot uses arrows. Content is equivalent; `⇒` counts are
  therefore not comparable between sources, which is why `--audit`
  counts both forms.
- The raw files are faithful, not clean: cosmetics, ARAM and shop
  rotations are all still there. Use `clean/` for analysis.
- `--level core` drops bugfix/QoL wholesale. That is a real trade-off:
  it removes 8.3% of the corpus, including champion bug fixes that do
  change power ("Fixed a bug where X's E did not apply on-hit"). Default
  is `standard`, which keeps them.
- This is scraped public data. Riot's and Fandom's terms apply; do not
  redistribute as a standalone corpus.

## Rebuild

```
python "data/patch notes/collect.py"              # fetch what's missing
python "data/patch notes/collect.py" --refresh    # refetch everything
python "data/patch notes/collect.py" --only 16.20 15.20
python "data/patch notes/collect.py" --audit      # check numeric coverage
python "data/patch notes/clean.py"                # rebuild clean/
python -m pytest "data/patch notes/test_collect.py" \
                 "data/patch notes/test_clean.py" -q
```
