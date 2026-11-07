"""Extract per-team ban lists for one patch from the raw soloQ shards.

The DuckDB store (`store.py`) throws `teams[side].bans` away - no cube or
table keeps it - so any ban-rate question has to go back to the shards. The
whole crawl is 1.1 GB gz, and a full `json.loads` per line costs minutes, so
the cheap filter runs first: the version string is the third field of every
record, so a substring test on the raw line rejects ~98 % of the corpus
without parsing it.

Output: long form (match_id, side, ban). An empty ban slot is stored as 0 by
some regions and -1 by others, so anything <= 0 is "no ban" and must be
filtered by the caller; no champion id is <= 0.

Usage
-----
  python extract_bans.py --patch 16.20 [--out store/bans_16.20.parquet]
"""
import argparse
import glob
import gzip
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MATCHDATA = os.path.join(ROOT, "data", "solo queue", "matchdata")

SCHEMA = pa.schema([("match_id", pa.string()), ("side", pa.string()),
                    ("ban", pa.int16())])


def shards():
    """Archive snapshot first, then the live crawl (same order as store.py)."""
    out = []
    for f in sorted(glob.glob(os.path.join(MATCHDATA, "archive_pre_snapshots",
                                           "*", "*.jsonl.gz"))):
        out.append(f)
    for f in sorted(glob.glob(os.path.join(MATCHDATA, "*", "shard_*.jsonl*"))):
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="16.20")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "store", "bans_%s.parquet" % a.patch)
    needle = '"version":"%s.' % a.patch

    w = pq.ParquetWriter(out, SCHEMA, compression="zstd")
    buf, games, seen = [], 0, set()
    for path in shards():
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if needle not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mid = rec.get("match_id")
                if not mid or mid in seen or not rec.get("players"):
                    continue
                seen.add(mid)
                games += 1
                for side in ("blue", "red"):
                    for ban in ((rec.get("teams") or {}).get(side) or {}).get("bans") or []:
                        # dicts, not tuples: pa.Table.from_pylist silently maps
                        # positional tuples to NULL for every column
                        buf.append({"match_id": mid, "side": side,
                                    "ban": int(ban)})
                if len(buf) >= 200_000:
                    w.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
                    buf = []
        print("  %s  games=%d" % (os.path.basename(path), games), flush=True)
    if buf:
        w.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
    w.close()
    print("wrote %s  (%d games, %d ban rows)" % (out, games, games * 10))


if __name__ == "__main__":
    main()
