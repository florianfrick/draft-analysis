"""Shared plumbing for the LLM experiment (branch exp/finetune-llm).

The question this branch answers: **can the pro draft encoder be turned into a
"draft vision" encoder whose output an instruction-tuned LLM can read as soft
prompt tokens?** Three things are pinned here so the two trainers and the
evaluation cannot drift apart:

* **Corpus and split.** The same rows, the same `split_within_patch` call and
  the same seed as the checkpoint being adapted
  (`masking-proonly-randomvalsplit-through-wlds-ig-tes`), so its recorded val
  numbers (next-move top-1 0.183 / top-5 0.499 on 95,160 states) are the
  baseline this experiment is measured against rather than a nearby number.
* **The board rendering.** One function turns a prefix *tensor* sample into
  the text the LLM reads. It reads the tensors, not the row, so the prose and
  the soft tokens can never describe different drafts - and `swap_sides`
  works on the text for free.
* **Champion names.** The 173 display names in idx order, straight from
  `champ_names.json`, which is the same table the encoder's champion ids use.

Paths, constants and nothing that trains.
"""
import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent            # .../pro/model/llm
MODEL_DIR = HERE.parent                           # .../model
ROOT = MODEL_DIR.parents[1]                           # .../lol draft analysis
PRO_DIR = ROOT / "data" / "pro play"
SOLOQ_PROC_DIR = ROOT / "data" / "solo queue" / "process matchdata"
CKPT_DIR = MODEL_DIR / "checkpoints"

sys.path.insert(0, str(ROOT))

from solo_queue.model.pretraining.base_model import (  # noqa: E402
    SiameseBase, load_encoder_state)

# The encoder this experiment adapts, and the checkpoint whose val numbers are
# the baseline. Changing either invalidates every recorded comparison.
ENCODER_CKPT = "masking-proonly-randomvalsplit-through-wlds-ig-tes"
BASELINE_VAL = {  # nextpick-proonly-randomvalsplit-.../test_report.json
    "n": 95160, "top1": 0.1832177385456074, "top5": 0.4994955863808323,
    "top10": 0.6799390500210173,
}

# Split, identical to the encoder's own run (solo_queue/model/policy/train.py defaults for
# the -randomvalsplit- pair): 10% of every patch drawn at random per patch.
#
# `seed` is deliberately NOT passed. `split_within_patch` defaults it to
# `pipeline.VAL_SPLIT_SEED`, which is the seed the encoder and the head drew
# with; passing `seed=0` here drew a *second, independent* 10% per patch, and
# two independent 10% draws of the same corpus overlap on ~1% of it - i.e. 90%
# of the LLM's val sat inside the head's train. That coincidence is what
# `_split_overlap.py` used to report as a protocol difference and what shrank
# every cross-model comparison to a 478-game intersection. One seed, one split.
SPLIT = dict(val_frac=0.10, test_frac=0.0, val_random=True)

N_SLOTS = 10
TEAM_SIZE = 5
N_MOVES = 20
NUM_REAL_CHAMPS = 173
IGNORE_INDEX = -100
ROLE_NAMES = ["TOP", "JGL", "MID", "BOT", "SUP"]
# The prompt names the lane in words, so it uses the corpus' own vocabulary
# (`data/pro play/pipeline.py: ROLE_NAMES` = TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY)
# in a readable case rather than the terse forms above.
ROLE_DISPLAY = ["Top", "Jungle", "Mid", "Bot", "Support"]
SIDE_NAMES = ["Blue", "Red"]
PHASE_NAMES = ["bans1", "picks1", "bans2", "picks2"]

# Which board rendering the prompt uses. `v2` (default since 2026-11-08) splits
# the bans by side, labels the turn, and names the lane of the pick being asked
# for. `v1` is kept because every number recorded before that date was measured
# on it, and a silent format change would make those numbers unreproducible -
# a projector is trained on the text as much as on the soft tokens.
PROMPT_FORMAT = "v2"
# The corpus move order (data/pro play/pipeline.py), so the prompt can say
# which side is acting without the loader having to carry it separately.
MOVE_SIDE = ([0, 1, 0, 1, 0, 1] +          # bans1: B R B R B R
             [0, 1, 1, 0, 0, 1, 1, 0, 0, 1] +  # picks1: B R R B B R R B B R
             [1, 0, 1, 0] +                # bans2: R B R B
             [1, 0, 0, 1])                 # picks2: R B B R


def _load_module_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_module_by_path("pro_pipeline_llm", PRO_DIR / "pipeline.py")
DS = _load_module_by_path("pro_datasets_llm", PRO_DIR / "datasets.py")


# ------------------------------------------------------------------ names
def champ_names(num_champs=NUM_REAL_CHAMPS):
    """idx -> display name, in the encoder's champion-id order."""
    payload = json.loads((SOLOQ_PROC_DIR / "champ_names.json")
                         .read_text(encoding="utf-8"))
    riot_to_name = {int(k): v for k, v in payload["names"].items()}
    mapping = json.loads((SOLOQ_PROC_DIR / "champ_mapping.json")
                         .read_text(encoding="utf-8"))
    riot_to_idx = {int(k): int(v) for k, v in mapping["riot_to_idx"].items()}
    out = [None] * num_champs
    for riot, idx in riot_to_idx.items():
        if 0 <= idx < num_champs:
            out[idx] = riot_to_name[riot]
    if any(n is None for n in out):
        raise RuntimeError("champion name table has holes - the mapping and "
                           "names files disagree")
    return out


# ------------------------------------------------------------- corpus/split
# Which CSVs each corpus name scans. `pipeline.CSV_GLOB` is the 2022+ pattern,
# and it is what `load_rows` used to pass for both names - so the first
# `--corpus all` call found a signature mismatch, rescanned with the 2022+ glob
# and **overwrote `pro_drafts_all.pt` with the 2022 corpus**. The glob has to
# travel with the cache name or the cache silently changes meaning.
CORPUS_GLOB = {"2022": "202[2-9]_*OraclesElixir*.csv",
               "all": "*_OraclesElixir*.csv"}


def load_rows(corpus="2022"):
    """Pro draft rows (cached .pt), then recency weights attached.

    Falls back to reading the cache directly when the CSVs are absent, which is
    the Colab/Kaggle case: `pipeline.build_rows` validates the cache against a
    signature of the CSV files, so without them it would rescan and raise. The
    cache is the same object either way - it carries `version` and `csv_sig`,
    and this path only runs when there is nothing to revalidate against.
    """
    path = PRO_DIR / {"2022": "pro_drafts.pt",
                      "all": "pro_drafts_all.pt"}[corpus]
    try:
        rows, _stats = P.build_rows(cache_path=path,
                                    csv_glob=str(PRO_DIR / CORPUS_GLOB[corpus]))
    except FileNotFoundError:
        if not path.exists():
            raise
        print("no CSVs beside %s - reading the cache directly" % path.name)
        payload = torch.load(str(path), map_location="cpu",
                             weights_only=False)
        rows = payload["rows"]
    P.assign_weights(rows)
    return rows


def assert_patch_compat(cfg, patch_index):
    """Fail before the model loads if the corpus outruns the encoder's table.

    The frozen encoder's patch embedding has `n_patches` rows, trained on the
    vocabulary of the corpus it was built from. `make_patch_index` rebuilds that
    vocabulary from whatever rows are being used, so a wider corpus shifts every
    id and the encoder would index past its table - or, if the shift happened to
    stay in range, read the wrong patch row for the whole run. `--corpus all`
    (2017+, 219 patches) against the 2022+ encoder (107 rows) is exactly that
    case, and it cannot work without retraining the encoder.
    """
    n = int(cfg.get("n_patches") or 0)
    if not n or not patch_index:
        return
    top = max(patch_index.values())
    if top > n - 1:
        raise SystemExit(
            "patch vocabulary mismatch: corpus has ids up to %d but the frozen "
            "encoder's patch table has %d rows. The encoder was trained on the "
            "2022+ corpus; use --corpus 2022 or retrain the encoder on the "
            "wider one." % (top, n))


def split_rows(rows, verbose=False):
    """{'train': [...], 'val': [...], 'test': [...]} index lists."""
    return P.split_within_patch(rows, verbose=verbose, **SPLIT)


def patch_vocab(rows):
    return DS.build_patch_vocab(rows)


def encoder_patch_vocab(name=None):
    """The patch vocabulary the frozen encoder was trained with.

    Read from the checkpoint rather than rebuilt from the corpus, because the
    embedding rows are indexed by *this* list: a corpus that has gained a patch
    would otherwise shift every id by one and the encoder would read the wrong
    row for the entire run. Anything newer than the last entry here falls to
    `PATCH_UNK_ID`, which is the learned row reserved for it.
    """
    path = CKPT_DIR / (name or ENCODER_CKPT) / "best.pt"
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return list(ck.get("patch_vocab") or [])


def patch_coverage(rows, patch_index):
    """(n_rows_outside_the_table, [(patch, n_rows), ...] worst first).

    Printed by every entry point so a raised freeze is visible in the log
    instead of silently turning a patch into UNK.
    """
    unknown = {}
    for r in rows:
        k = DS.patch_key(r.get("patch"))
        if k and k not in patch_index:
            unknown[k] = unknown.get(k, 0) + 1
    return (sum(unknown.values()),
            sorted(unknown.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------- board rendering
def _team_line(sample, team, names):
    """v1's one-line board: 'Corki MID, Xin Zhao JGL'."""
    cids = sample["champ_ids"][team * TEAM_SIZE:(team + 1) * TEAM_SIZE]
    rids = sample["role_ids"][team * TEAM_SIZE:(team + 1) * TEAM_SIZE]
    pad = sample["pad_mask"][team * TEAM_SIZE:(team + 1) * TEAM_SIZE]
    picks = [("%s %s" % (names[int(c)], ROLE_NAMES[int(r)]))
             for c, r, p in zip(cids, rids, pad) if not bool(p)]
    return ", ".join(picks) if picks else "-"


def _picks(sample, team, names):
    """v2's board: [(champion, lane)] for one side, in pick order."""
    lo, hi = team * TEAM_SIZE, (team + 1) * TEAM_SIZE
    return [(names[int(c)], ROLE_DISPLAY[int(r)])
            for c, r, p in zip(sample["champ_ids"][lo:hi],
                               sample["role_ids"][lo:hi],
                               sample["pad_mask"][lo:hi]) if not bool(p)]


def _bans_by_side(sample):
    """[[blue bans], [red bans]] in slot order.

    The side of a ban is *derived*, never carried: `BAN_SIDE_BY_SLOT ^
    first_pick` (`data/pro play/datasets.py`). Deriving it here rather than
    reading a stored side is what keeps `swap_sides` working untouched - it
    flips `first_pick`, and the ban sides and the turn line flip with it.
    """
    n_bans = int(sample["n_bans"])
    sides = DS.ban_sides(int(sample["first_pick"]))
    out = [[], []]
    for i in range(n_bans):
        out[int(sides[i])].append(int(sample["ban_ids"][i]))
    return out


def _turn_line(sample):
    """'Turn: Red 2 (Pick 4 of 10)' - whose turn, and their n-th move.

    Both counts are of the *current* move (so the first move is "1"), and both
    are read off the board rather than passed in: the picks so far are the
    non-PAD slots, the bans so far are `n_bans`. The lane of a ban turn does
    not exist, which is why a ban is asked for without one.
    """
    to_move = int(sample["to_move"])
    is_ban = bool(int(sample["is_ban"]))
    n_b = int((~sample["pad_mask"][:TEAM_SIZE]).sum())
    n_r = int((~sample["pad_mask"][TEAM_SIZE:]).sum())
    bans = _bans_by_side(sample)
    if is_ban:
        made, total, kind = len(bans[to_move]), int(sample["n_bans"]), "Ban"
    else:
        made, total = (n_b if to_move == 0 else n_r), n_b + n_r
        kind = "Pick"
    return "Turn: %s %d (%s %d of 10)" % (SIDE_NAMES[to_move], made + 1,
                                          kind, total + 1)


def task_line(sample, prompt_format=None):
    fmt = prompt_format or PROMPT_FORMAT
    side = SIDE_NAMES[int(sample["to_move"])]
    if fmt == "v1":
        verb = "ban" if bool(int(sample["is_ban"])) else "pick"
        return ("Task: recommend the optimal champion %s for %s. "
                "Answer with the champion name only." % (verb, side))
    if fmt == "v4":
        # Closed set: the answer must be one of the listed candidates. Saying
        # so is not cosmetic - the decode is masked to those candidates' first
        # tokens, so a model that ignored the instruction would be scored
        # against a constraint the prompt never mentioned.
        if bool(int(sample["is_ban"])):
            return ("Task: Choose the best %s Side ban from the candidates. "
                    "Answer with champion name only." % side)
        role = int(sample["target_role"])
        if not 0 <= role < len(ROLE_DISPLAY):
            return ("Task: Choose the best %s Side pick from the candidates. "
                    "Answer with champion name only." % side)
        return ("Task: Choose the best %s Side pick for %s lane from the "
                "candidates. Answer with champion name only."
                % (side, ROLE_DISPLAY[role]))
    if bool(int(sample["is_ban"])):
        return ("Task: Recommend the optimal %s Side ban. Answer with "
                "champion name only." % side)
    role = int(sample["target_role"])
    if not 0 <= role < len(ROLE_DISPLAY):
        # A pick always carries a lane in the corpus (checked: 0-4, balanced
        # 20 % each). This branch exists so a malformed row degrades into the
        # lane-less question instead of rendering "lane -100".
        return ("Task: Recommend the optimal %s Side pick. Answer with "
                "champion name only." % side)
    return ("Task: Recommend the optimal %s Side pick for %s lane. Answer "
            "with champion name only." % (side, ROLE_DISPLAY[role]))


def _fmt_pct(p):
    """A probability as the prompt shows it: '31%', or '0.4%' when it is tiny.

    Integer percent keeps the line short - it is 5 candidates, so every
    character is paid 5 times over on a ~150-token prompt. The one-decimal
    fallback exists because a 5-way list of "0%" says nothing, and the head's
    tail probabilities really are under 1 % on ban moves with a wide pool.
    """
    pct = 100.0 * float(p)
    if pct >= 1.0:
        return "%d%%" % round(pct)
    return "%.1f%%" % pct


def candidates_line(names, cand_ids, cand_p):
    """'Candidates: Jax 31%, Renekton 22%, ...' - the head's shortlist.

    The order *is* the head's ranking, and the numbers are its raw softmax over
    legal champions (so they sum to ~1 over the whole legal set, not over the
    k listed). Both facts matter to a reranker: the point of the channel is to
    let the model disagree with a confident head and defer to a flat one.
    """
    return "Candidates: " + ", ".join(
        "%s %s" % (names[int(c)], _fmt_pct(p))
        for c, p in zip(cand_ids, cand_p))


def render_state(sample, names, include_fearless=True, prompt_format=None,
                 candidates=None):
    """Prefix tensors -> the factual part of the user turn.

    Reads only the sample, so it is exactly the state the encoder saw.

    `v2` (default): one line per fact, bans split by side, the turn labelled
    with the side's n-th move and the overall n-th pick/ban, and the lane of
    the pick being asked for. `v1`: the original two-team block plus one
    chronological ban list. Both are kept because a projector is trained on the
    text, so switching the format silently would invalidate every recorded
    number.

    `v4` is v2 plus a `Candidates:` line carrying the next-pick head's top-k
    with its probabilities, and a task line that asks for a choice *from* that
    list. It is the reranker's format: `candidates` is required, and the list
    is the head's ranking of the legal champions, so the answer is legal by
    construction.

    Kept deliberately terse. Every token here is a token the frozen 4B model
    has to forward and backward through, at ~380 tokens/s on this GPU
    (measured: 16 examples x ~150 tokens per 6.3 s step), so
    optional lines are omitted rather than written as "none".
    """
    fmt = prompt_format or PROMPT_FORMAT
    if fmt == "v1":
        return _render_state_v1(sample, names, include_fearless)
    if fmt == "v4" and not candidates:
        raise ValueError("prompt format v4 needs the candidate list; pass "
                         "candidates=[(champ_id, prob), ...]")

    lines = ["Patch: %s | %s" % (sample["meta"]["patch"], _turn_line(sample))]
    for team in (0, 1):
        picks = _picks(sample, team, names)
        if picks:
            lines.append("%s Picks: %s"
                         % (SIDE_NAMES[team],
                            ", ".join("%s (%s)" % p for p in picks)))
    bans = _bans_by_side(sample)
    for team in (0, 1):
        if bans[team]:
            lines.append("%s Bans: %s"
                         % (SIDE_NAMES[team],
                            ", ".join(names[c] for c in bans[team])))
    if include_fearless:
        unavail = sample["unavailable_mask"].nonzero().flatten().tolist()
        if unavail:
            # `v3` restores the gloss v1 had. It is 9 tokens, and the only
            # thing that says *why* the list matters: v2's "Fearless Bans:"
            # relies on the model already knowing the term.
            label = ("Fearless Bans (picked earlier this series, cannot pick "
                     "again)" if fmt == "v3" else "Fearless Bans")
            lines.append("%s: %s"
                         % (label,
                            ", ".join(names[int(c)] for c in unavail)))
    if fmt == "v4":
        ids, probs = zip(*candidates)
        lines.append(candidates_line(names, ids, probs))
    return "\n".join(lines)


def _render_state_v1(sample, names, include_fearless=True):
    """The original board rendering, frozen so old numbers stay reproducible."""
    move_idx = int(sample["move_idx"])
    to_move = int(sample["to_move"])
    is_ban = bool(int(sample["is_ban"]))
    n_bans = int(sample["n_bans"])
    banned = (", ".join(names[int(b)]
                        for b in sample["ban_ids"][:n_bans]) or "-")

    lines = ["Patch %s | move %d/%d %s | %s to %s | first pick %s"
             % (sample["meta"]["patch"], move_idx + 1, N_MOVES,
                PHASE_NAMES[int(sample["phase"])], SIDE_NAMES[to_move],
                "ban" if is_ban else "pick",
                SIDE_NAMES[int(sample["first_pick"])].lower()),
             "Blue: %s" % _team_line(sample, 0, names),
             "Red: %s" % _team_line(sample, 1, names),
             "Bans: %s" % banned]
    if include_fearless:
        unavail = sample["unavailable_mask"].nonzero().flatten().tolist()
        if unavail:
            lines.append("Fearless (picked earlier this series, cannot pick "
                         "again): %s"
                         % ", ".join(names[int(c)] for c in unavail))
    return "\n".join(lines)


# --------------------------------------------------------------- encoder
def build_frozen_encoder(ckpt_name=ENCODER_CKPT, device="cpu"):
    """Rebuild the pro-only masked encoder from its checkpoint and freeze it.

    Returns (encoder, config, patch_vocab_list). The encoder is returned in
    float32 and in eval mode: 2.5M parameters, so precision is free and the
    projector sees a deterministic target to fit.
    """
    path = CKPT_DIR / ckpt_name / "best.pt"
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    enc = SiameseBase(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        n_patches=int(cfg.get("n_patches", 0)))
    load_encoder_state(enc, payload["model"])
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad = False
    return enc, cfg, list(payload.get("patch_vocab") or [])


def encoder_batch(sample):
    """The subset of a prefix sample the encoder reads.

    `_encode` needs champ_ids / role_ids / pad_mask / patch_id and nothing
    else; keeping the projection explicit means an accidental read of
    `taken_mask` here cannot happen.
    """
    return {k: sample[k] for k in ("champ_ids", "role_ids", "pad_mask",
                                   "patch_id")}


def param_count(model):
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------ answer space
# The answer is emitted after "Answer:", so it always carries a leading space.
# Defined once because the tokenizer treats "Vi" and " Vi" as different
# tokens, and the candidate table has to match what the model produces.
ANSWER_PREFIX = " "


def candidate_tables(tok, names):
    """(first_tokens [M], owner [M, C]) for the 173 champion names.

    Tokenized **exactly as the model emits them**: `" Vi"`, with a leading
    space, not `"Vi"`. The two are different tokens in this vocabulary
    (`'Vi'` vs `'ĠVi'`), and mixing them makes every target look like it is
    not a candidate at all.

    The tokenizer gives far fewer distinct first tokens than there are
    champions - 24 collision groups covering 61 names, because "Vayne",
    "Viktor", "Viego", "Vex" and "Vladimir" all begin with the same `'ĠV'` and
    the largest group stands for 6 champions. A first-token readout is
    therefore an M-way problem with ties, not a 173-way classifier, and the
    owner matrix exists so the legality filter and the report can both say
    which champions a candidate token stands for.
    """
    first = [tok(ANSWER_PREFIX + n, add_special_tokens=False)["input_ids"][0]
             for n in names]
    uniq = sorted(set(first))
    idx = {t: i for i, t in enumerate(uniq)}
    owner = torch.zeros((len(uniq), len(names)), dtype=torch.bool)
    for c, t in enumerate(first):
        owner[idx[t], c] = True
    return torch.tensor(uniq, dtype=torch.long), owner


def name_token_stats(tok, names):
    """How the 173 names tokenize, for the report - never hard-coded."""
    lens = [len(tok(ANSWER_PREFIX + n, add_special_tokens=False)["input_ids"])
            for n in names]
    hist = {}
    for n in lens:
        hist[n] = hist.get(n, 0) + 1
    first, owner = candidate_tables(tok, names)
    return {"n_names": len(names),
            "length_histogram": dict(sorted(hist.items())),
            "single_token": sum(1 for n in lens if n == 1),
            "distinct_first_tokens": int(first.numel()),
            "first_token_collisions": int((owner.sum(dim=1) > 1).sum())}
