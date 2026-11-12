"""Build the queryable corpus, then run one query against it.

    python pro/model/agent/build_db.py --known-patches-from \\
        pro/model/checkpoints/<name>/best_policy.pt

The `--known-patches-from` step is not optional in practice. A patch the
checkpoint never saw becomes UNK at inference and the head answers
patch-agnostically without saying so. For the shipped shared-split head every
corpus patch is in the vocabulary, so the flag currently marks nothing - which
is the honest answer, and the reason it is recorded rather than assumed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import schema  # noqa: E402
import sql_tool  # noqa: E402


def known_patches_from_ckpt(path):
    """Pull the patch vocabulary out of a checkpoint, tolerantly.

    The key has moved between runs, so this looks for the first list of
    patch-shaped strings anywhere in the blob rather than hardcoding a key.
    Returns None when nothing is found - the caller then marks everything
    scored and says so, instead of silently inventing a vocabulary.
    """
    import torch
    blob = torch.load(str(path), weights_only=False, map_location="cpu")
    seen = []

    def walk(obj, depth=0):
        if depth > 4 or len(seen) > 4000:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k) in ("patch_vocab", "patches", "patch_keys"):
                    walk(v, depth + 1)
                else:
                    walk(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            strs = [x for x in obj if isinstance(x, str)]
            if len(strs) >= 10 and all(len(s) <= 8 and "." in s for s in strs):
                seen.extend(strs)
            else:
                for x in obj[:8]:
                    walk(x, depth + 1)

    walk(blob)
    return set(seen) or None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(schema.DB_PATH))
    ap.add_argument("--known-patches-from", default=None,
                    help="a checkpoint whose patch table decides is_scored")
    ap.add_argument("--patches", default=None,
                    help="comma-separated patch ids, if you have no checkpoint")
    ap.add_argument("--query", default=None, help="run one query and print it")
    args = ap.parse_args(argv)

    known = None
    if args.known_patches_from:
        known = known_patches_from_ckpt(args.known_patches_from)
        print("known patches from %s: %s"
              % (Path(args.known_patches_from).name,
                 "%d found" % len(known) if known else "NONE FOUND"))
    elif args.patches:
        known = {p.strip() for p in args.patches.split(",") if p.strip()}

    rows = schema.read_rows()
    stats = schema.build(rows, args.out, known_patches=known)
    print("wrote %(db)s: %(games)d games, %(moves)d moves, %(patches)d patches"
          % stats)
    print("split:", stats["split"])
    con = schema.connect(args.out)
    unscored = [r[0] for r in con.execute(
        "SELECT patch FROM patch WHERE is_scored = 0 ORDER BY patch_rank")]
    print("unscored patches:", unscored or "none")
    if args.query:
        print(sql_tool.run(con, args.query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
