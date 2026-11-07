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
SPLIT = dict(val_frac=0.10, test_frac=0.0, val_random=True, seed=0)

N_SLOTS = 10
TEAM_SIZE = 5
N_MOVES = 20
NUM_REAL_CHAMPS = 173
IGNORE_INDEX = -100
ROLE_NAMES = ["TOP", "JGL", "MID", "BOT", "SUP"]
SIDE_NAMES = ["Blue", "Red"]
PHASE_NAMES = ["bans1", "picks1", "bans2", "picks2"]
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
        rows, _stats = P.build_rows(cache_path=path)
    except FileNotFoundError:
        if not path.exists():
            raise
        print("no CSVs beside %s - reading the cache directly" % path.name)
        payload = torch.load(str(path), map_location="cpu",
                             weights_only=False)
        rows = payload["rows"]
    P.assign_weights(rows)
    return rows


def split_rows(rows, verbose=False):
    """{'train': [...], 'val': [...], 'test': [...]} index lists."""
    return P.split_within_patch(rows, verbose=verbose, **SPLIT)


def patch_vocab(rows):
    return DS.build_patch_vocab(rows)


# --------------------------------------------------------- board rendering
def _team_line(sample, team, names):
    cids = sample["champ_ids"][team * TEAM_SIZE:(team + 1) * TEAM_SIZE]
    rids = sample["role_ids"][team * TEAM_SIZE:(team + 1) * TEAM_SIZE]
    pad = sample["pad_mask"][team * TEAM_SIZE:(team + 1) * TEAM_SIZE]
    picks = [("%s %s" % (names[int(c)], ROLE_NAMES[int(r)]))
             for c, r, p in zip(cids, rids, pad) if not bool(p)]
    return ", ".join(picks) if picks else "-"


def task_line(sample):
    side = SIDE_NAMES[int(sample["to_move"])]
    verb = "ban" if bool(int(sample["is_ban"])) else "pick"
    return ("Task: recommend the optimal champion %s for %s. "
            "Answer with the champion name only." % (verb, side))


def render_state(sample, names, include_fearless=True):
    """Prefix tensors -> the factual part of the user turn.

    Reads only the sample, so it is exactly the state the encoder saw. Bans are
    listed in chronological order (the corpus order is B R B R B R / R B R B,
    so a reader who knows the rule can also infer the banning side); fearless
    picks are listed as a name set because that is all the model is given.

    Kept deliberately terse. Every token here is a token the frozen 4B model
    has to forward and backward through, at ~70 tokens/s on this GPU, so
    "Blue picks: none yet" costs real money over 6,000 examples. Optional
    lines are omitted rather than written as "none".
    """
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
    champions - 32 collisions covering 77 names, because "Vayne", "Viktor",
    "Viego", "Vex" and "Vladimir" all begin with the same `'ĠV'`. A
    first-token readout is therefore an M-way problem with ties, not a 173-way
    classifier, and the owner matrix exists so the legality filter and the
    report can both say which champions a candidate token stands for.
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
