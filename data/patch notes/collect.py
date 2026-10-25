"""Collect League of Legends patch notes for every patch in the pro dataset.

Two sources, tried in order:

  1. Riot official notes - https://www.leagueoflegends.com/en-us/news/
     game-updates/patch-<major>-<minor>-notes/ . Authoritative wording,
     and the before/after arrows are preserved as text.
  2. League of Legends Wiki - https://wiki.leagueoflegends.com/en-us/
     V<major>.<minor> . Covers patches Riot's news site does not (the
     whole 26.x line, i.e. dataset 16.x), and states the release date.

Patch numbering: the dataset drops the leading "2" from the 2025/2026
scheme, so dataset 15.20 is Riot 25.20 and dataset 16.20 is Riot 26.20.
Verified against release dates (Riot 25.20 = 2025-10-07, dataset 15.20
games start 2025-10-11; wiki V26.20 = 2026-10-07, dataset 16.20 games
start 2026-10-09). 12.x-14.x are unchanged.

Output: one markdown file per dataset patch, plus index.json describing
where each came from and which patches could not be resolved.

    python "data/patch notes/collect.py"            # all dataset patches
    python "data/patch notes/collect.py" --only 16.20 15.20
    python "data/patch notes/collect.py" --refresh  # refetch existing
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
PRO_DRAFTS = HERE.parent / "pro play" / "pro_drafts.pt"

RIOT_URL = ("https://www.leagueoflegends.com/en-us/news/game-updates/"
            "patch-{m}-{n}-notes/")
WIKI_URL = "https://wiki.leagueoflegends.com/en-us/V{m}.{n}"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

BLOCK = ("h1", "h2", "h3", "h4", "h5", "p", "li", "table", "blockquote")
SKIP_CLASSES = {"mw-editsection", "toc", "navbox", "noprint", "reference"}


SEASON_MAP = {"15": "25", "16": "26"}


def riot_patch(dataset_patch):
    """dataset '16.20' -> '26.20'; '15.20' -> '25.20'; '12.01' -> '12.01'."""
    major, minor = dataset_patch.split(".")
    return "%s.%s" % (SEASON_MAP.get(major, major), minor)


def fetch(url, timeout=40):
    """Fetch via curl using curl's own User-Agent.

    The wiki sits behind Cloudflare and returns 403 to python-requests
    (it fingerprints the TLS handshake, not just the UA), and also 403s
    curl when a browser UA is spoofed on top of curl's TLS fingerprint.
    curl's default UA is accepted by both sources, so use exactly that.
    """
    try:
        proc = subprocess.run(
            ["curl", "-s", "-L", "--max-time", str(timeout),
             "-w", "\n__HTTP__%{http_code}", url],
            capture_output=True, timeout=timeout + 15)
        raw = proc.stdout.decode("utf-8", "ignore")
        body, _, code = raw.rpartition("\n__HTTP__")
        code = int(code.strip() or 0)
        return code, (body if code == 200 else "")
    except (OSError, ValueError):
        return 0, ""


DECIMAL_SPLIT = re.compile(r"(?<=\d)\.\s+(?=\d)")


def tidy(text):
    """Collapse whitespace and repair decimals split across elements.

    Wiki markup wraps the integer and fractional parts of a number in
    separate spans, so get_text(" ") yields "3. 5". Rejoin those.
    """
    return DECIMAL_SPLIT.sub(".", " ".join(text.split()))


def render_change_div(node):
    """Render one Riot ``div.attribute-change`` as a change bullet.

    Riot's CMS wraps each individual stat delta in a div of spans::

        <div class="attribute-change">
          <span class="attribute">AP Ratio</span>
          <span class="attribute-before">40%</span>
          <span class="change-indicator">⇒</span>
          <span class="attribute-after">50%</span>
        </div>

    That is a <div>, not a <li>, so a block walker that only knows
    headings and lists drops it entirely. Every numeric delta in
    12.01-12.18 was lost that way, leaving champion headings with
    nothing underneath. The before/after split here is more reliable
    than anything a prose parser could recover, so emit it as
    "- AP Ratio : 40% ⇒ 50%".
    """
    text = tidy(node.get_text(" ", strip=True))
    if not text:
        return ""
    label = node.select_one("span.attribute")
    label_text = tidy(label.get_text(" ", strip=True)) if label else ""
    # Slice the label off the front of the full text instead of
    # rebuilding from the before/after spans, so trailing notes
    # ("(Note: Unchanged)"), "new"/"updated" badges and the
    # before-less ("only an after value") case all survive intact.
    if label_text and text.startswith(label_text):
        rest = text[len(label_text):].lstrip(" :")
        if rest:
            text = "%s : %s" % (label_text, rest)
    return "- " + text


def clean_node(node, depth=0):
    """Render one block element as markdown-ish text."""
    name = node.name
    if name == "div":
        return render_change_div(node)
    text = tidy(node.get_text(" ", strip=True))
    if not text:
        return ""
    if name.startswith("h") and len(name) == 2 and name[1].isdigit():
        return "#" * min(int(name[1]) + 1, 6) + " " + text
    if name == "li":
        return "  " * depth + "- " + text
    if name == "blockquote":
        return "> " + text
    if name == "table":
        rows = []
        for tr in node.find_all("tr"):
            cells = [tidy(td.get_text(" ", strip=True))
                     for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)
    return text


def html_to_md(body):
    """Walk the content container in document order, keeping block breaks."""
    for junk in body.select("script, style, noscript, .mw-editsection, "
                            ".toc, .navbox, .noprint, .reference, "
                            "sup.reference"):
        junk.decompose()
    out, emitted = [], set()
    for node in body.find_all(BLOCK + ("div",)):
        # Only div.attribute-change carries content the other blocks miss.
        if node.name == "div" and "attribute-change" not in (
                node.get("class") or []):
            continue
        # Skip a block already emitted as part of an ancestor (a <li> or
        # <p> inside a table we rendered whole). Track ancestors by id():
        # hashing/equality on a bs4 Tag renders and compares the entire
        # subtree, which turns this loop quadratic and took ~80s/page.
        if any(id(p) in emitted for p in node.parents):
            continue
        line = clean_node(node)
        if not line:
            continue
        emitted.add(id(node))
        # A change div owns its spans; mark the whole subtree so nothing
        # inside is emitted a second time.
        for child in node.find_all(True):
            emitted.add(id(child))
        out.append(line)
    # consecutive list items join with a single newline, other blocks
    # with a blank line, so lists stay lists in the rendered markdown
    text = ""
    for line in out:
        if text and line.lstrip().startswith("- ") \
                and text.rsplit("\n", 1)[-1].lstrip().startswith("- "):
            text += "\n" + line
        elif text:
            text += "\n\n" + line
        else:
            text = line
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def wiki_to_md(body):
    """Wiki variant: champion names are bare <a> elements that sit just
    before each change list, not headings. Without promoting them the
    balance changes lose their subject ("Base damage reduced to 5-25")
    and the file is useless for retrieval.
    """
    for junk in body.select("script, style, noscript, .mw-editsection, "
                            ".toc, .navbox, .noprint, .reference, "
                            "sup.reference"):
        junk.decompose()
    SKIP_HEAD = {"contents", "references", "see also", "navigation",
                 "patch chronology"}
    out, emitted = [], set()
    for node in body.find_all(["h2", "h3", "h4", "a", "li", "p", "table"]):
        if any(id(p) in emitted for p in node.parents):
            continue
        if node.name == "a":
            # Only bare links (a champion or item name), never inline
            # links inside prose or list items.
            if node.find_parent(["p", "li", "table", "h2", "h3", "h4"]):
                continue
            text = tidy(node.get_text(" ", strip=True))
            if not text or len(text) > 40:
                continue
            out.append("#### " + text)
            emitted.add(id(node))
            continue
        if node.name in ("h2", "h3", "h4"):
            text = tidy(node.get_text(" ", strip=True))
            if not text or text.lower() in SKIP_HEAD:
                emitted.add(id(node))
                continue
        line = clean_node(node)
        if not line:
            continue
        emitted.add(id(node))
        out.append(line)
    text = ""
    for line in out:
        if text and line.lstrip().startswith("- ") \
                and text.rsplit("\n", 1)[-1].lstrip().startswith("- "):
            text += "\n" + line
        elif text:
            text += "\n\n" + line
        else:
            text = line
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_riot(html):
    s = BeautifulSoup(html, "lxml")
    body = s.select_one("div.width-small") or s.find("main")
    if body is None:
        return "", None
    title = s.find("h1")
    published = None
    m = re.search(r'"datePublished":"([^"]+)"', html)
    if m:
        published = m.group(1)[:10]
    md = html_to_md(body)
    if title:
        md = "# %s\n\n%s" % (title.get_text(" ", strip=True), md)
    return md, published


def extract_wiki(html):
    s = BeautifulSoup(html, "lxml")
    body = s.select_one("div.mw-parser-output")
    if body is None:
        return "", None
    md = wiki_to_md(body)
    # Parse the release date from the rendered TEXT, not the raw HTML:
    # the day suffix is split across tags ("October 7" </span> "th"), so
    # a regex over the markup misses it.
    published = None
    MONTHS = ("January|February|March|April|May|June|July|August|"
              "September|October|November|December")
    head = body.get_text("\n", strip=True)[:600]
    m = re.search(r"(%s)\s+(\d{1,2})\s*(?:st|nd|rd|th)?\s*,?\s*(\d{4})"
                  % MONTHS, head)
    if m:
        month, day, year = m.group(1), m.group(2), m.group(3)
        try:
            import datetime
            dt = datetime.datetime.strptime(
                "%s %s %s" % (month[:3], day, year), "%b %d %Y")
            published = dt.strftime("%Y-%m-%d")
        except ValueError:
            published = "%s %s %s" % (month, day, year)
    return md, published


def resolve(dataset_patch, timeout=20):
    """Try Riot then wiki. Returns dict with md, source, url, published."""
    rp = riot_patch(dataset_patch)
    major, minor = rp.split(".")

    attempts = []
    for n in dict.fromkeys([minor, str(int(minor))]):
        attempts.append(("riot", RIOT_URL.format(m=major, n=n), extract_riot))
    for n in dict.fromkeys([minor, str(int(minor))]):
        attempts.append(("wiki", WIKI_URL.format(m=major, n=n), extract_wiki))

    for source, url, extractor in attempts:
        try:
            code, html = fetch(url, timeout)
        except Exception:
            continue
        if code != 200 or len(html) < 5000:
            continue
        md, published = extractor(html)
        if len(md) < 1500:
            continue
        return {"md": md, "source": source, "url": url,
                "published": published, "riot_patch": rp}
    return None


def read_header(path):
    """Parse the YAML-ish front-matter we wrote, so cached files can
    repopulate the index without refetching."""
    meta = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return meta
    if not text.startswith("---"):
        return meta
    _, _, rest = text.partition("---\n")
    head, _, _ = rest.partition("---\n")
    for line in head.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            v = v.strip()
            if k.strip() == "chars" and v.isdigit():
                meta["chars"] = int(v)
            else:
                meta[k.strip()] = v
    return meta


DELTA_PATTERNS = (
    re.compile(r"\d[^\n]{0,80}?⇒"),          # Riot arrow form
    re.compile(r"(?:increased|reduced|changed|decreased)\s+to\s+"
               r"[-\d]", re.I),               # wiki prose form
)


def count_deltas(text):
    """How many numeric before/after deltas a markdown file contains.

    Cheap proxy for "did the extractor actually keep the numbers". Both
    sources express a delta as either "40% ⇒ 50%" or "reduced to 5 from
    8", so count both and report the sum.
    """
    return sum(len(p.findall(text)) for p in DELTA_PATTERNS)


def audit(paths):
    """Report per-file numeric-delta coverage; return the bad ones."""
    bad = []
    print("%-16s %6s %7s %7s  %s" % ("file", "chars", "arrows", "deltas",
                                     "source"))
    for p in sorted(paths):
        text = p.read_text(encoding="utf-8")
        meta = read_header(p)
        arrows = text.count("⇒")
        deltas = count_deltas(text)
        flag = ""
        if deltas == 0:
            flag = "  <-- NO NUMBERS"
            bad.append(p.name)
        print("%-16s %6d %7d %7d  %-4s%s" %
              (p.name, meta.get("chars", 0), arrows, deltas,
               meta.get("source", "?"), flag))
    return bad


def dataset_patches():
    import torch
    blob = torch.load(str(PRO_DRAFTS), map_location="cpu",
                      weights_only=False)
    return sorted({str(r["patch"]) for r in blob["rows"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="dataset patches to fetch, e.g. 16.20 15.20")
    ap.add_argument("--refresh", action="store_true",
                    help="refetch even if the output file exists")
    ap.add_argument("--workers", type=int, default=5,
                    help="parallel fetches (default 5)")
    ap.add_argument("--timeout", type=int, default=20,
                    help="per-request curl timeout in seconds")
    ap.add_argument("--audit", action="store_true",
                    help="only check the existing files for numeric deltas")
    args = ap.parse_args()

    if args.audit:
        bad = audit(sorted(HERE.glob("patch_*.md")))
        print("\n%d/%d files carry no numeric delta"
              % (len(bad), len(list(HERE.glob("patch_*.md")))))
        return 1 if bad else 0

    targets = args.only or dataset_patches()
    index_path = HERE / "index.json"
    index = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))

    todo, skipped = [], 0
    for dp in targets:
        out = HERE / ("patch_%s.md" % dp.replace(".", "_"))
        if out.exists() and not args.refresh:
            # Rebuild the index entry from the file's own front-matter.
            # Skipping this leaves stale "missing" entries behind when a
            # patch was fetched by an earlier, partially-failed run.
            index[dp] = dict(read_header(out), status="ok", file=out.name)
            skipped += 1
        else:
            todo.append(dp)

    ok = missing = 0
    total = len(todo)

    def work(dp):
        return dp, resolve(dp, args.timeout)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, dp): dp for dp in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            dp = futures[fut]
            try:
                _, res = fut.result()
            except Exception as e:
                print("[%3d/%3d] %-6s ERROR %s" % (i, total, dp, e))
                index[dp] = {"status": "error", "riot_patch": riot_patch(dp)}
                missing += 1
                continue
            if res is None:
                print("[%3d/%3d] %-6s MISSING" % (i, total, dp))
                index[dp] = {"status": "missing",
                             "riot_patch": riot_patch(dp)}
                missing += 1
                continue
            out = HERE / ("patch_%s.md" % dp.replace(".", "_"))
            header = (
                "---\n"
                "dataset_patch: %s\n"
                "riot_patch: %s\n"
                "source: %s\n"
                "url: %s\n"
                "published: %s\n"
                "chars: %d\n"
                "---\n\n" % (dp, res["riot_patch"], res["source"],
                             res["url"], res["published"] or "unknown",
                             len(res["md"]))
            )
            out.write_text(header + res["md"] + "\n", encoding="utf-8")
            index[dp] = {"status": "ok", "riot_patch": res["riot_patch"],
                         "source": res["source"], "url": res["url"],
                         "published": res["published"],
                         "chars": len(res["md"]),
                         "file": out.name}
            print("[%3d/%3d] %-6s %-4s %6d chars  %s" %
                  (i, total, dp, res["source"], len(res["md"]),
                   res["published"] or "?"))
            ok += 1

    index_path.write_text(json.dumps(dict(sorted(index.items())), indent=1),
                          encoding="utf-8")
    print("\nfetched %d, cached %d, missing %d | index -> %s" %
          (ok, skipped, missing, index_path.name))
    if missing:
        print("unresolved: %s" % ", ".join(
            k for k, v in sorted(index.items()) if v["status"] != "ok"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
