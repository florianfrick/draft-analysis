"""Filter the patch-notes corpus down to what affects drafting.

The raw files in this folder are faithful: everything Riot and the wiki
published, including skins, ARAM, TFT, Clash schedules and shop
rotations. This script derives a smaller corpus from them, dropping the
sections that cannot change a Summoner's Rift draft.

    python "data/patch notes/clean.py"                  # -> clean/
    python "data/patch notes/clean.py" --report         # show every heading
    python "data/patch notes/clean.py" --level full     # everything
    python "data/patch notes/clean.py" --with-bugfixes  # + bugfix sections

Design notes
------------
*Denylist, not allowlist.* Riot does not put balance under a handful of
predictable headings. Alongside "Champions" and "Items" there are
"terrain changes", "lane swaps", "epic objective tuning", "early game
adjustments", "snowballing adjustments", "atakhan, blood roses, feats,
and kills", "champion radii". An allowlist of {champions, items, runes}
would silently delete all of that. So: drop the families we can name
confidently, keep everything else, and report what fell through.

*Section level only.* Within "Champions", cosmetic bullets ("Pengu Garen
skin re-enabled") are 0.3% of 4,595 bullets - not worth sentence-level
rules, which is where automatic cleaning starts making expensive
mistakes.

*Nothing is destroyed.* The raw corpus stays authoritative and
git-ignored; `clean/` is regenerable. Re-running after a rule change
needs no refetch.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect  # noqa: E402  (same folder, reuses read_header/count_deltas)

OUT = HERE / "clean"

# Families that cannot affect a Summoner's Rift draft. Matched against
# the lowercased heading; first match wins, so put narrow rules first.
#
# Two of these rules were wrong on the first pass and are worth naming,
# because they are the failure mode of any denylist:
#   * "role quests" was filed under meta - but Role Quests are the
#     lane-phase quest system added in 25.S1, i.e. core SR gameplay and
#     the single biggest draft change of that season.
#   * `\bwards?\b` was filed under cosmetic - which also swallowed
#     "Support Control Wards" and "Ward Experience", both real SR
#     changes. Only *ward skins* are cosmetic.
# Both were caught by `--report`, not by reading the output.
RULES = [
    # -- cosmetics / monetisation ------------------------------------
    ("cosmetic", r"\bskins?\b|\bchromas?\b|cosmetic|prestige|\bemotes?\b"
                 r"|ward skins?|splash|bundle|vfx|sfx|icon rarity"
                 r"|illustration|your shop|personalization|battle ?pass"
                 r"|event pass|blue essence|refund token|\bloot\b"
                 r"|mythic (shop|content)|hall of legends|sanctum"),
    # -- alternate game modes ----------------------------------------
    ("altmode", r"\baram\b|\barurf\b|\burf\b|one for all|nexus blitz"
                r"|ultimate spellbook|swiftplay|doom bots|\bbrawl\b"
                r"|\bswarm\b"
                r"|\bclash\b|teamfight tactics|\btft\b|rotating game mode"
                r"|mayhem|\barena\b|\bclassic\b|april fools|snowdown"
                r"|ascension|co-op vs|\bpve\b"),
    # -- queues, progression, anti-cheat -----------------------------
    ("meta", r"ranked|apex mmr|queue declining|ready check|\bhonor\b"
             r"|\bchallenges?\b|career stats|champion mastery"
             r"|champion pricing|boosting|anti-cheat|vanguard"
             r"|behavioral|matchmaking|progression|event missions"
             r"|season \d|in-game quests|position binding|autofill"
             r"|champion select|quickplay|refund|player days"
             r"|hextech chest|player behavior|hostage"),
    # -- esports ------------------------------------------------------
    ("esports", r"competitive|esports|world'?s|\bmsi\b|tournament"
                r"|championship|\bcup\b"),
    # -- client / platform -------------------------------------------
    # NB: no bare `\bbots?\b` - that would drop a "Bot Lane" section.
    ("client", r"\bclient\b|32.bit|min spec|\bping\b|dynamic music"
               r"|death screen|accessibility|game info tracking"
               r"|custom game|bot ai|intro bot|\bbots\b|practice tool"
               r"|gameplay quality|ability previews|hud|scoreboard"),
    # -- one-off events / metagames ----------------------------------
    ("event", r"anniversary|demacia rising|festival|metagame"
              r"|narrative|activity center|soul fighter"),
    # -- editorial / summaries ---------------------------------------
    ("summary", r"patch highlights|thoughts on|designer|dev (corner|blog)"
                r"|\bfaqs?\b|look ahead|state of"),
]

# Sections that are balance but not usually wanted for draft analysis.
# Kept by default at --level standard, dropped at --level core.
CHAFF_RULES = [
    ("bugfix", r"bug ?fix|bugfix|qol|quality of life"),
]

LEVELS = ("core", "standard", "full")


def classify(title, level_name):
    """Return (bucket, family) for a heading."""
    if level_name == "full":
        return "keep", None
    low = title.lower().strip()
    for family, pattern in RULES:
        if re.search(pattern, low):
            return "drop", family
    if level_name == "core":
        for family, pattern in CHAFF_RULES:
            if re.search(pattern, low):
                return "drop", family
    return "keep", None


def split_front(text):
    if not text.startswith("---"):
        return {}, text
    _, _, rest = text.partition("---\n")
    head, _, body = rest.partition("---\n")
    meta = {}
    for line in head.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, body.lstrip("\n")


def split_sections(body):
    """Split on ### and #### only; ##### ability headings stay in the body.

    Both sources use ### for top-level sections. They differ underneath:
    Riot's #### is an entity (Diana), the wiki's #### is sometimes a
    sub-section (Champions, ARAM) and sometimes an entity, because wiki
    champion names are promoted from bare links. Classifying by heading
    and inheriting downwards handles both without special cases.
    """
    pre, sections, cur = [], [], None
    for line in body.splitlines():
        m = re.match(r"^(#{2,4}) (.+)$", line)
        if m:
            if cur is not None:
                sections.append(cur)
            cur = {"level": len(m.group(1)), "title": m.group(2).strip(),
                   "lines": [], "parent": None}
        elif cur is None:
            pre.append(line)
        else:
            cur["lines"].append(line)
    if cur is not None:
        sections.append(cur)
    return pre, sections


def filter_text(body, level_name):
    """Return (kept_text, log) where log records every heading decision."""
    pre, sections = split_sections(body)
    stack, log = [], []
    for sec in sections:
        while stack and stack[-1]["level"] >= sec["level"]:
            stack.pop()
        sec["parent"] = stack[-1] if stack else None
        bucket, family = classify(sec["title"], level_name)
        if bucket == "keep" and sec["parent"] is not None \
                and sec["parent"]["bucket"] == "drop":
            bucket, family = "drop", sec["parent"]["family"]
        sec["bucket"], sec["family"] = bucket, family
        stack.append(sec)
        log.append({"level": sec["level"], "title": sec["title"],
                    "bucket": bucket, "family": family,
                    "chars": len("\n".join(sec["lines"]))})

    out = list(pre)
    for sec in sections:
        if sec["bucket"] == "drop":
            continue
        out.append("#" * sec["level"] + " " + sec["title"])
        out.extend(sec["lines"])
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n", log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=LEVELS, default="standard",
                    help="core drops bugfix/QoL too; standard keeps them; "
                         "full copies everything (default: standard)")
    ap.add_argument("--with-bugfixes", action="store_true",
                    help="keep bugfix/QoL sections even at --level core")
    ap.add_argument("--report", action="store_true",
                    help="print the heading classification and exit")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    args = ap.parse_args()

    level = args.level
    if args.with_bugfixes and level == "core":
        level = "standard"

    files = sorted(HERE.glob("patch_*.md"))
    if not files:
        print("no patch_*.md found; run collect.py first")
        return 1

    if args.report:
        agg, unclass = {}, {}
        for f in files:
            _, body = split_front(f.read_text(encoding="utf-8"))
            _, log = filter_text(body, level)
            for row in log:
                key = (row["title"].lower(), row["family"], row["bucket"])
                e = agg.setdefault(key, {"n": 0, "chars": 0})
                e["n"] += 1
                e["chars"] += row["chars"]
                # Only top-level (###) headings matter for review: they
                # can hide a whole irrelevant section. A #### that keeps
                # by default is just a champion or item name.
                if row["bucket"] == "keep" and row["family"] is None \
                        and row["level"] == 3:
                    unclass[row["title"].lower()] = row["chars"]
        print("%-46s %5s %8s  %s" % ("heading", "n", "chars", "decision"))
        for (title, family, bucket), e in sorted(
                agg.items(), key=lambda kv: -kv[1]["chars"]):
            print("%-46s %5d %8d  %s%s" % (
                title[:46], e["n"], e["chars"], bucket,
                " (%s)" % family if family else ""))
        print("\n%d top-level headings kept with no matching rule - skim "
              "these when a new patch adds a mode or shop:"
              % len(unclass))
        for t, c in sorted(unclass.items(), key=lambda kv: -kv[1])[:40]:
            print("   %-44s %6d chars" % (t[:44], c))
        return 0

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    report, tot_raw, tot_kept, dropped = [], 0, 0, {}
    for f in files:
        meta, body = split_front(f.read_text(encoding="utf-8"))
        kept, log = filter_text(body, level)
        name = f.name
        (outdir / name).write_text(
            "---\n"
            "dataset_patch: %s\nriot_patch: %s\nsource: %s\n"
            "published: %s\nfilter: %s\nchars_raw: %d\nchars_kept: %d\n"
            "---\n\n%s" % (
                meta.get("dataset_patch", "?"), meta.get("riot_patch", "?"),
                meta.get("source", "?"), meta.get("published", "?"),
                level, len(body), len(kept), kept),
            encoding="utf-8")
        for row in log:
            if row["bucket"] == "drop":
                d = dropped.setdefault(row["family"], [0, 0])
                d[0] += 1
                d[1] += row["chars"]
        tot_raw += len(body)
        tot_kept += len(kept)
        report.append({"file": name, "chars_raw": len(body),
                       "chars_kept": len(kept), "sections": log})

    (outdir / "report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")

    print("level=%s  %d files -> %s" % (level, len(files), outdir))
    print("chars %d -> %d  (kept %.1f%%)"
          % (tot_raw, tot_kept, 100.0 * tot_kept / max(tot_raw, 1)))
    print("\ndropped by family:")
    for fam, (n, c) in sorted(dropped.items(), key=lambda kv: -kv[1][1]):
        print("  %-10s %4d sections %9d chars  (%.1f%%)"
              % (fam, n, c, 100.0 * c / max(tot_raw, 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
