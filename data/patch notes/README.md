# Patch notes corpus

Balance/patch notes for every patch in `data/pro play/pro_drafts.pt`,
as markdown, collected by `collect.py`.

## Contents

- `patch_<major>_<minor>.md` — one file per dataset patch (106 files).
- `index.json` — source, URL, publish date and size for each patch.
- `collect.py` — the collector. Re-runnable; skips files that exist
  unless you pass `--refresh`.
- `test_collect.py` — extractor regression tests (no network).

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

## Known caveats

- Wiki-sourced files express some changes in prose ("reduced to X from
  Y") where Riot uses arrows. Content is equivalent; `⇒` counts are
  therefore not comparable between sources, which is why `--audit`
  counts both forms.
- Cosmetic/skin sections are retained as-is. Filter to the
  `Champions` / `Items` sections if you only want balance changes.
- This is scraped public data. Riot's and Fandom's terms apply; do not
  redistribute as a standalone corpus.

## Rebuild

```
python "data/patch notes/collect.py"              # fetch what's missing
python "data/patch notes/collect.py" --refresh    # refetch everything
python "data/patch notes/collect.py" --only 16.20 15.20
python "data/patch notes/collect.py" --audit      # check numeric coverage
python -m pytest "data/patch notes/test_collect.py" -q
```
