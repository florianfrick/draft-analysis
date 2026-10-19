"""SoloQ data pipeline: raw compact match records -> model batch dict.

Raw records in `../matchdata/<routing>/shard_*.jsonl[.gz]` (see
`../collect_match_data.py::extract`): version, queue_id, duration,
surrender flags, teams, players[{team, champ (Riot id), pos, ...}].

Stages (streaming, no full-corpus RAM load):
  1. `build_champ_mapping` — sparse Riot championId -> 0..C-1 for one patch.
  2. `iter_records` — stream shards (gzip-aware).
  3. `record_to_slots` — quality filters + canonical slot order
     [B_TOP,B_JGL,B_MID,B_ADC,B_SUP,R_TOP,...].
  4. `SoloQDraftDataset` — masking (p~0.15 champ, 90/10 role, >=1 champ mask/draft, plus p~0.10 role-only with champ visible).
  5. `soloq_collate` — stack to model/README.md Sec.3 batch.

Role map: TOP->0, JUNGLE->1, MIDDLE->2, BOTTOM->3, UTILITY->4.
Batch keys: champ_ids, role_ids, role_masked, is_masked, champ_labels,
role_labels (-100), pad_mask (all-False). No team_ids, no aux targets.
"""
import gzip
import json
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset

N_SLOTS = 10
TEAM_SIZE = 5
IGNORE_INDEX = -100

ROLE_TO_ID = {"TOP": 0, "JUNGLE": 1, "MIDDLE": 2, "BOTTOM": 3, "UTILITY": 4}

RANKED_QUEUES = (420, 440)
MIN_DURATION_S = 300

BASE_DIR = Path(__file__).resolve().parent
MATCHDATA_DIR = BASE_DIR.parent / "matchdata"
MAPPING_PATH = BASE_DIR / "champ_mapping.json"
CHAMP_NAMES_PATH = BASE_DIR / "champ_names.json"
DDRAGON_VERSION = "16.20.1"  # must match the training patch
DDRAGON_URL = ("https://ddragon.leagueoflegends.com/cdn/{v}/data/en_US/"
               "champion.json")


def open_maybe_gzip(path):
    """Open plain or .gz shard for text reading."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def list_shards(matchdata_dir=MATCHDATA_DIR, regions=None):
    """Sorted shard paths; optionally restrict to routing regions."""
    paths = sorted(glob(str(Path(matchdata_dir) / "*" / "shard_*.jsonl*")))
    if regions:
        want = {r.lower() for r in regions}
        paths = [p for p in paths if Path(p).parent.name.lower() in want]
    return paths


def iter_records(shard_paths):
    """Yield (shard_path, line_no, record), skipping blank/corrupt lines.

    Safe to read while collect_match_data.py is still appending: a truncated
    plain-jsonl line is skipped, and an unfinished gzip stream or mid-write
    read error (EOFError / OSError / zlib.error) just ends that shard early
    instead of crashing the scan.
    """
    import zlib
    for sp in shard_paths:
        try:
            with open_maybe_gzip(sp) as f:
                for i, line in enumerate(f):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        yield sp, i, json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue  # truncated tail from a killed crawl
        except (EOFError, OSError, zlib.error):
            continue  # shard actively being written (unfinished gz stream)


def build_champ_mapping(shard_paths, patch_prefix, queue_ids=RANKED_QUEUES,
                        min_duration_s=MIN_DURATION_S, save_path=MAPPING_PATH):
    """Scan shards for the champion set on `patch_prefix` -> {riot_id: idx}.

    Idx 0..C-1 sorted by Riot id. Saves JSON unless save_path is None.
    """
    seen = set()
    for _, _, rec in iter_records(shard_paths):
        if not str(rec.get("version", "")).startswith(patch_prefix):
            continue
        if rec.get("queue_id") not in tuple(queue_ids):
            continue
        if int(rec.get("duration") or 0) < min_duration_s:
            continue
        for p in rec.get("players") or []:
            if p.get("champ") is not None:
                seen.add(int(p["champ"]))
    mapping = {str(r): i for i, r in enumerate(sorted(seen))}
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"patch": patch_prefix, "num_champs": len(mapping),
                       "riot_to_idx": mapping}, f, indent=1)
    return {int(k): v for k, v in mapping.items()}


def load_champ_mapping(path=MAPPING_PATH):
    """Load saved mapping -> (riot_to_idx {int: int}, num_champs, patch)."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    riot_to_idx = {int(k): int(v) for k, v in payload["riot_to_idx"].items()}
    return riot_to_idx, int(payload["num_champs"]), payload.get("patch")


def fetch_champ_names(version=DDRAGON_VERSION, save_path=CHAMP_NAMES_PATH,
                      timeout_s=30):
    """Download RiotId -> display name from Data Dragon, cache as JSON.

    Returns {riot_id: name}. Cached file: {"version": v, "names": {id: name}}.
    Falls back to the cache on network failure; raises if neither exists.
    """
    import urllib.request
    url = DDRAGON_URL.format(v=version)
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            data = json.load(resp)
        names = {int(v["key"]): v["name"] for v in data["data"].values()}
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"version": version,
                       "names": {str(k): v for k, v in names.items()}},
                      f, indent=1)
        return names
    except Exception:
        if Path(save_path).exists():
            with open(save_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return {int(k): v for k, v in payload["names"].items()}
        raise


def load_champ_names(path=CHAMP_NAMES_PATH):
    """Load cached {riot_id: name} without network."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(k): v for k, v in payload["names"].items()}


def idx_names(riot_to_idx, riot_to_name):
    """Contiguous idx 0..C-1 -> display name list (for plots/eval).

    Unknown Riot ids fall back to "Riot<id>" so a stale cache never crashes.
    """
    inv = {v: k for k, v in riot_to_idx.items()}
    return [riot_to_name.get(inv[i], f"Riot{inv[i]}")
            for i in range(len(riot_to_idx))]


def name_of(idx, names):
    """Display name for a contiguous champ idx (or '?' if out of range)."""
    return names[idx] if 0 <= idx < len(names) else "?"


def record_to_slots(record, riot_to_idx, patch_prefix=None,
                    queue_ids=RANKED_QUEUES, min_duration_s=MIN_DURATION_S,
                    drop_surrender=False):
    """Filters + canonical slots. Returns (champ_idx[10], role[10]) or None.

    None on: wrong patch/queue, short/remake game, surrender (if dropped),
    != 10 players, missing/doubled position (incl. "Invalid"), unknown
    champion, duplicate champions.
    """
    if patch_prefix and not str(record.get("version", "")).startswith(patch_prefix):
        return None
    if record.get("queue_id") not in tuple(queue_ids):
        return None
    if int(record.get("duration") or 0) < min_duration_s:
        return None
    if drop_surrender and (record.get("surrender") or record.get("early_surrender")):
        return None
    players = record.get("players") or []
    if len(players) != N_SLOTS:
        return None
    blue = [p for p in players if p.get("team") == 100]
    red = [p for p in players if p.get("team") == 200]
    if len(blue) != TEAM_SIZE or len(red) != TEAM_SIZE:
        return None
    champ_idx, role_ids = [], []
    for team in (blue, red):
        by_pos = {p.get("pos"): p for p in team}
        for pos in ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"):
            p = by_pos.get(pos)
            if p is None:
                return None
            riot = p.get("champ")
            if riot is None or int(riot) not in riot_to_idx:
                return None
            champ_idx.append(riot_to_idx[int(riot)])
            role_ids.append(ROLE_TO_ID[pos])
    if len(set(champ_idx)) != N_SLOTS:
        return None
    return champ_idx, role_ids


def scan_and_index(shard_paths, riot_to_idx, patch_prefix,
                   queue_ids=RANKED_QUEUES, min_duration_s=MIN_DURATION_S,
                   drop_surrender=False, limit=None):
    """One streaming pass -> kept/skipped counts + slot rows + provenance.

    Rows fit RAM easily (~20 bytes/draft -> ~2 MB per 100k drafts); the raw
    shards are never fully loaded.
    """
    rows, index = [], []
    kept = skipped = 0
    for sp, ln, rec in iter_records(shard_paths):
        out = record_to_slots(rec, riot_to_idx, patch_prefix, queue_ids,
                              min_duration_s, drop_surrender)
        if out is None:
            skipped += 1
        else:
            kept += 1
            rows.append(out)
            index.append((str(sp), ln))
        if limit is not None and kept + skipped >= limit:
            break
    return {"kept": kept, "skipped": skipped, "rows": rows, "index": index}


def apply_masking(champ, role, num_champs, mask_prob=0.15, role_keep_prob=0.1,
                  role_only_prob=0.10, generator=None):
    """Train masking -> (champ_ids, role_ids, role_masked, is_masked, clab, rlab).

    Champ masked w.p. mask_prob, then >= 1 champ mask forced per draft.
    Role masked on 90% of champ masks (FLEX input), true role kept on 10%.
    PLUS role-only masks: on slots where the champ stays VISIBLE, hide the
    role w.p. role_only_prob (role input -> FLEX, champ input stays truth).
    Those slots carry a role label but NO champ label, so the role head must
    learn P(role | champ + context) - i.e. "x champion is most likely y role".
    Without this the role head only ever trains with the champ hidden and can
    never learn champion->role affinities.

    Labels: champ truth on is_masked slots else IGNORE_INDEX; role truth on
    (is_masked | role-only) slots else IGNORE_INDEX. pad_mask by collate.
    """
    champ = torch.as_tensor(champ, dtype=torch.long)
    role = torch.as_tensor(role, dtype=torch.long)

    def _rand(n):
        if generator is None:
            return torch.rand(n)
        return torch.rand(n, generator=generator)

    def _randint(n, k=1):
        if generator is None:
            return torch.randint(n, (k,))
        return torch.randint(n, (k,), generator=generator)
    is_m = _rand(N_SLOTS) < mask_prob
    if not bool(is_m.any()):
        is_m[_randint(N_SLOTS, 1)] = True
    keep = (_rand(N_SLOTS) < role_keep_prob) & is_m
    role_masked_champ = is_m & ~keep
    # role-only: champ visible, role hidden -> teaches P(role | champ)
    role_only = (_rand(N_SLOTS) < role_only_prob) & (~is_m)
    role_masked = role_masked_champ | role_only
    cids = champ.clone()
    rids = role.clone()
    cids[is_m] = num_champs  # CHAMP_MASK_ID
    rids[role_masked] = 5  # ROLE_FLEX_ID
    clab = torch.full((N_SLOTS,), IGNORE_INDEX, dtype=torch.long)
    rlab = torch.full((N_SLOTS,), IGNORE_INDEX, dtype=torch.long)
    clab[is_m] = champ[is_m]
    rlab[is_m] = role[is_m]  # incl. role-visible 10%: copy-path supervision
    rlab[role_only] = role[role_only]  # champ-visible -> true P(role | champ)
    return cids, rids, role_masked, is_m, clab, rlab


class SoloQDraftDataset(Dataset):
    """Map-style dataset over pre-scanned (champ[10], role[10]) rows.

    Train mode applies fresh random masking per access (generator seeded by
    index + epoch so workers stay deterministic); eval mode returns one
    fixed view when mask_prob is set, unmasked when mask_prob=0.
    """

    def __init__(self, rows, num_champs, mask_prob=0.15, role_keep_prob=0.1, role_only_prob=0.10,
                 seed=0, epoch=0):
        self.rows = rows
        self.num_champs = num_champs
        self.mask_prob = mask_prob
        self.role_keep_prob = role_keep_prob
        self.role_only_prob = role_only_prob
        self.seed = seed
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        """Call each epoch so train masks re-roll deterministically."""
        self.epoch = epoch

    def __getitem__(self, idx):
        champ, role = self.rows[idx]
        g = torch.Generator().manual_seed((self.seed + idx * 7919 +
                                           self.epoch * 104729) % (2 ** 63 - 1))
        cids, rids, rm, is_m, clab, rlab = apply_masking(
            champ, role, self.num_champs, self.mask_prob, self.role_keep_prob, self.role_only_prob, g)
        return {
            "champ_ids": cids, "role_ids": rids, "role_masked": rm,
            "is_masked": is_m, "champ_labels": clab, "role_labels": rlab,
            "pad_mask": torch.zeros(N_SLOTS, dtype=torch.bool),
        }


def soloq_collate(samples):
    """Stack per-draft dicts -> model/README.md Sec.3 batch [B,10]."""
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


def compute_popularity_baseline(rows, num_champs, names=None):
    """Global + per-role most-picked champions (eval baselines).

    Returns {"global_top": [...], "role_top": {role: [...]},
    "global_counts": Tensor[C], "role_counts": Tensor[5, C]} sorted by count
    desc; ties broken by smaller champ idx (deterministic). When `names`
    (idx -> display name list) is given, also adds "global_top_names" and
    "role_top_names" with the same ordering.
    """
    glob_counts = torch.zeros(num_champs, dtype=torch.long)
    role_counts = torch.zeros(5, num_champs, dtype=torch.long)
    for champ, role in rows:
        for c, r in zip(champ, role):
            glob_counts[c] += 1
            role_counts[r, c] += 1
    order = sorted(range(num_champs), key=lambda c: (-int(glob_counts[c]), c))
    role_top = {}
    for r in range(5):
        role_top[r] = sorted(range(num_champs),
                             key=lambda c: (-int(role_counts[r, c]), c))
    out = {"global_top": order, "role_top": role_top,
           "global_counts": glob_counts, "role_counts": role_counts}
    if names is not None:
        out["global_top_names"] = [name_of(c, names) for c in order]
        out["role_top_names"] = {r: [name_of(c, names) for c in role_top[r]]
                                 for r in range(5)}
    return out
