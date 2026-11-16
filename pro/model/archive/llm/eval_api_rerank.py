"""Score a hosted chat model as the reranker - no Qwen, no projector, no GPU.

## Why this exists

`eval_rerank5_zeroshot.json` records 32.17 % on the shared val's `common`
slice, against 19.14 % for the head's own top-1, and the README attributes
+13.0 pp to the shortlist + first-token mask and +3.1 pp to training. The
question this script answers is whether a hosted API reproduces that without
the open weights.

It cannot reproduce the configuration that produced 32.17 %, and the script
does not pretend otherwise. That checkpoint is `context: soft+text`: ten
2560-dim vectors spliced into `inputs_embeds` from the encoder's hidden
states. There is no `inputs_embeds` on a chat-completions endpoint, so the
soft half of that prompt is unavailable to any hosted model - not to DeepSeek
specifically. What *is* available is the text, and `_probe_soft.py` already
priced the difference: replacing the block with the projector's own PAD
embedding, text untouched, costs **-3.50 pp** (p<.001); swapping in a
*different board's* block costs -0.58 (p=.18); collapsing the ten vectors to
their mean is +0.00 exactly. The block is a "a real draft is in progress"
key, not information - the champions, bans, lane, patch and fearless pool all
reach the model through the text.

So this is the text-only arm, measured rather than assumed. Expect it to land
below 32.17 % and above the head's 19.14 %, and read the number off the run.

## What it deliberately does not measure

- **The masked decode.** The LoRA is decoded with the candidate first tokens
  at -inf everywhere else, so `top1_offlist` and `top5_illegal` are 0 by
  construction. An API can be asked to pick from the list and can still
  refuse, so `offlist` is a real outcome here and is reported. It is the price
  of moving the constraint from the decoder into the prompt.
- **`exact` in the LoRA report's sense.** That column is whole-name token
  accuracy (the answer is a 1-5 token sequence); a hosted API returns a
  string. `api_top1` here is a whole-champion-id match, so the LoRA column it
  belongs next to is `exact` (30.00 / 34.61), **not** `ft_top1` (32.17 /
  35.32) - `ft_top1` is first-token *group* accuracy and is up to ~2 pp more
  generous across the 24 collision groups. The `head_top1` in this report is
  a 173-way argmax on the same ids, so the API-vs-head delta is exact-match on
  both sides and is the one clean comparison.

## The comparison is paired, on purpose

Every number here is computed on the *same rows* as the head's, so the head's
top-1, the ceiling and the API's accuracy cannot drift apart through a
re-drawn val (`llm/data.eval_dataset` re-draws its subsample when the corpus
size changes - see `MEMORY.md`). The paired McNemar against the head's argmax
is the statistic that matters; it does not need the run to reproduce 95,260
states to be valid.

The published reference points are printed next to the result, with the caveat
that they are **not** on the same rows:

  heldout, 95,260 states   ceiling 50.51 %, reranker 33.85 %  (shipped)
  478-game `common` slice  ceiling 51.85 %, 32.17 / 35.31 %   (retired split)

`--split common` is refused by `rerank.split_games` today, which is why the
first pair is the one to quote and the second is only history.

## Usage

    # 200 games = 4,000 states, one API call each, ~20 min at 8 workers
    python -m pro.model.archive.llm.eval_api_rerank --k 5 --val-games 200 --workers 8

    # the k=20 arm - the 4B model converted only 29 % of an 86 % ceiling
    python -m pro.model.archive.llm.eval_api_rerank --k 20 --val-games 200

    # no key needed: print the prompt the API would see
    python -m pro.model.archive.llm.eval_api_rerank --k 5 --val-games 4 --dry-run

    # the anchoring ablation: names without the head's probabilities
    python -m pro.model.archive.llm.eval_api_rerank --k 5 --no-probs

Runs through the same env contract as the agent (`LLM_API_KEY`,
`LLM_BASE_URL`, `LLM_MODEL`), and reads `.env` the way `llm_client` does, so
a key set in the repo file works without exporting anything.
"""
import argparse
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from pro.model.agent import llm_client            # noqa: E402
from pro.model.agent import draft_state as DS     # noqa: E402
from pro.model.archive.llm import common as C             # noqa: E402
from pro.model.archive.llm import data as D               # noqa: E402
from pro.model.archive.llm import rerank as R             # noqa: E402
from pro.model.archive.llm.prompt import ANSWER_CUE, SYSTEM, build_text  # noqa: E402

SLICE_NAMES = {0: "bans1", 1: "picks1", 2: "bans2", 3: "picks2"}

# Printed next to every run. Not on the same rows as the run - they are the
# reference the run is being read against, and mixing them into one table is
# how a 478-game number got quoted as a shared-val number for months.
REFERENCE = {
    "heldout_95260": {"ceiling": 0.5051, "reranker_top1": 0.3385,
                      "reranker_exact": 0.3315,
                      "note": "shipped; the number to beat"},
    "common_478games": {"ceiling": 0.5185, "zeroshot_top1": 0.3217,
                        "zeroshot_exact": 0.3000, "trained_top1": 0.3532,
                        "trained_exact": 0.3461,
                        "note": "RETIRED split; history only"},
}


# ------------------------------------------------------------------ text/parse
def _norm(text):
    """Lowercase, alphanumerics only - `Kai'Sa` -> `kaisa`, `Dr. Mundo` -> `drmundo`."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _without_probs(state):
    """Drop the percentages from the `Candidates:` line only.

    The `--no-probs` arm exists to test whether the head's numbers anchor the
    answer: `candidates_line` prints the head's raw softmax so the model can
    "disagree with a confident head and defer to a flat one", and that is a
    hypothesis worth an arm rather than a belief. Done by rewriting the one
    line instead of adding a flag to `render_state`, because that function's
    output is pinned byte-identical by `test_board.py` and the training path
    shares it.
    """
    out, seen = [], False
    for line in state.split("\n"):
        if line.startswith("Candidates:"):
            seen = True
            line = re.sub(r" (\d+(?:\.\d+)?%)", "", line)
        out.append(line)
    if not seen:
        raise ValueError("no `Candidates:` line to strip - is this v4 text?")
    return "\n".join(out)


def _rank_all_task(body, k):
    """Rewrite the v4 answer cue so the arm asks for a permutation.

    Replaced rather than prepended: `task_line` already ends with "Answer with
    champion name only.", and an extra instruction above it would leave the
    prompt asking for two different answers. One changed sentence is also what
    makes `--rank-all` a single-variable arm against the default.
    """
    marker = "Answer with champion name only."
    if marker not in body:
        raise ValueError("v4 task line does not end the way this arm expects")
    return body.replace(
        marker, "Answer with all %d names in order from best to worst, "
                "comma-separated." % k)


def _state_text(sample, names, display):
    """Render the row through the agent's `DraftState`, not the LoRA's board.

    The two carry the same facts and differ in what they *do with them*. v4
    writes `Turn: Red 4 (Ban 7 of 10)` and leaves the reader to work out which
    side is on which slot of the pro schedule - which is where `deepseek-flash`
    spent 1,700-30,000 characters of reasoning before running out of budget.
    `DraftState.render` names the move ("to move: red ban (move 6 of 20)"),
    states the fearless pool's meaning, and gives the legal count, so nothing
    has to be derived.

    **The slots have to be compacted first.** The corpus stores a pick at
    `team * 5 + role` (lane-indexed, so blue's first pick can sit at index 3),
    while `log_from_board` reads slot `n` as that side's *n-th* pick and stops
    at the first gap - correct for the UI, whose strips are in pick order, and
    wrong for a corpus row. Compacting each side's block is the bridge. The
    order *within* a side is then lost, which does not matter here: legality,
    `legal_count()` and the fearless pool all depend on the set, and the
    rendered board lists each side's picks as a group.
    """
    picks = []
    for team in (0, 1):
        lo, hi = team * 5, (team + 1) * 5
        filled = [names[int(c)] for c, p in
                  zip(sample["champ_ids"][lo:hi], sample["pad_mask"][lo:hi])
                  if not bool(p)]
        picks.extend(filled + [None] * (5 - len(filled)))
    bans = [names[int(b)] for b in sample["ban_ids"][:int(sample["n_bans"])]]
    pool = [names[int(c)] for c in
            sample["unavailable_mask"].nonzero().flatten().tolist()]
    st = DS.log_from_board(picks, bans=bans,
                           red_first=bool(int(sample["first_pick"])),
                           patch=sample["meta"]["patch"], unavailable=pool)
    return st, st.render(display=display)


def _extract(text, allowed_ids, names):
    """-> candidate ids named in `text`, in the order they appear.

    Left-to-right, longest match wins, overlaps dropped: without that, `Vi`
    matches inside `Viktor` and a one-name answer reads as two. Matching is on
    the normalized string, so positions are only comparable to each other -
    which is all that is needed.
    """
    norm = _norm(text)
    if not norm:
        return []
    hits = []
    for cid in allowed_ids:
        n = _norm(names[int(cid)])
        start = norm.find(n)
        if start >= 0:
            hits.append((start, -len(n), int(cid)))
    hits.sort()
    out, taken = [], []
    for start, neg_len, cid in hits:
        end = start - neg_len
        if any(not (end <= a or start >= b) for a, b in taken):
            continue
        taken.append((start, end))
        out.append(cid)
    return out


def parse_answer(text, cand_ids, names):
    """-> (chosen_id or None, named_outside_list or None).

    `chosen_id` is the first candidate named. `named_outside_list` is set only
    when nothing in the list was named but a real champion was - that is the
    model refusing the constraint, which is a different failure from a
    non-answer and is worth counting separately.
    """
    inside = _extract(text, cand_ids, names)
    if inside:
        return inside[0], None
    outside = _extract(text, range(len(names)), names)
    return None, (outside[0] if outside else None)


# ------------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    """95 % interval for a binomial proportion. At n=200 the width is ±6 pp,
    which is the reason a 200-game run can only rule out large effects."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / d, (centre + margin) / d)


def mcnemar(b, c):
    """Two-sided exact McNemar p for discordant counts b and c.

    Exact rather than chi-square because the discordant pairs are the small
    number here: the API and the head agree on most rows, so the test runs on
    a few hundred pairs and the asymptotic version is not trustworthy there.
    """
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


# ---------------------------------------------------------------------- report
class Tally:
    """Per-slice counters. Rates are computed in `out()`, so a slice with no
    rows reports n=0 instead of a division by zero."""

    def __init__(self):
        self.n = 0
        self.hit = 0
        self.head_hit = 0
        self.ceiling = 0
        self.offlist = 0
        self.outside = 0
        self.error = 0
        self.pos_sum = 0
        self.top5 = 0
        self.ranked = 0

    def add(self, row, on=True):
        if not on:
            return
        self.n += 1
        self.hit += int(row["hit"])
        self.head_hit += int(row["head_hit"])
        self.ceiling += int(row["ceiling_hit"])
        self.offlist += int(row["chosen"] is None)
        self.outside += int(row["outside"] is not None)
        self.error += int(row["error"] is not None)
        if row["pos"] is not None:
            self.ranked += 1
            self.pos_sum += int(row["pos"])
            self.top5 += int(row["pos"] < 5)

    def out(self):
        if not self.n:
            return {"n": 0}
        out = {"n": self.n,
               "api_top1": self.hit / self.n,
               "head_top1": self.head_hit / self.n,
               "ceiling_recall": self.ceiling / self.n,
               "offlist": self.offlist / self.n,
               "named_outside_list": self.outside / self.n,
               "errors": self.error / self.n}
        if self.ranked:
            # 1-based, the same convention as `evaluate_llm`'s mean_rank: a
            # candidate the model never named is ranked last (k), not dropped.
            out["mean_rank"] = self.pos_sum / self.ranked + 1.0
            out["ft_top5"] = self.top5 / self.ranked
        return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=5,
                   help="shortlist size; 5 is the shipped reranker's k, 20 is "
                        "the arm the 4B model failed at (29 %% of an 86 %% "
                        "ceiling)")
    p.add_argument("--split", default="heldout", choices=["heldout", "rr"],
                   help="heldout = the whole shared val (ceiling only); rr = "
                        "the reranker's eval half. `common` is retired and "
                        "refused by rerank.split_games")
    p.add_argument("--corpus", default="2022", choices=["2022", "all"])
    p.add_argument("--val-games", type=int, default=200,
                   help="games to sample from the split, seed 0 - the same "
                        "games every run at a fixed corpus size, and the "
                        "gameids are written to --out so two runs can be "
                        "checked against each other")
    p.add_argument("--limit", type=int, default=0,
                   help="cap on states (rows), applied after the game sample")
    p.add_argument("--head", default=str(R.HEAD_CKPT))
    p.add_argument("--device", default="")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--board", dest="board", action="store_true", default=True,
                   help="render the board (default). --no-board keeps only the "
                        "candidate list - the `cand` context, which is the "
                        "arm that measures how much the board text is worth "
                        "(7.8 pp for the LoRA)")
    p.add_argument("--no-board", dest="board", action="store_false")
    p.add_argument("--render", default="v4", choices=["v4", "state"],
                   help="v4 = the LoRA's board text (default, and the only arm "
                        "comparable to the published numbers); state = the "
                        "agent's `DraftState.render()`, which names the move "
                        "instead of leaving the schedule to be inferred")
    p.add_argument("--probs", dest="probs", action="store_true", default=True,
                   help="print the head's probabilities in the candidate line "
                        "(default, and what the LoRA saw)")
    p.add_argument("--no-probs", dest="probs", action="store_false",
                   help="names only - tests whether the numbers anchor the "
                        "answer")
    p.add_argument("--rank-all", action="store_true",
                   help="ask for the whole shortlist in order instead of one "
                        "name. Buys mean_rank/ft_top5, costs comparability "
                        "with the single-answer task line, so it is off")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=0,
                   help="0 = 8192 for one name, 8*k+8192 for --rank-all. Not "
                        "a stylistic default: `deepseek-flash` is a reasoning "
                        "model and spends its budget on `reasoning_content` "
                        "before `content` - measured at 340-2,000 tokens per "
                        "row on this prompt (median 1,739 chars with the "
                        "head's probabilities, 6,553 without), so the budget "
                        "is a correctness setting, not a cost one")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--cache", default="",
                   help="jsonl of responses to resume from (default: "
                        "<out>.responses.jsonl when --out is given, else none)")
    p.add_argument("--dry-run", action="store_true",
                   help="print prompts and the call count, make no requests")
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    rows = C.load_rows(args.corpus)
    _, patch_index = D.make_patch_index(rows)
    names = C.champ_names()
    _, eval_idx = R.split_games(rows, args.split)
    ds = D.eval_dataset(rows, eval_idx, patch_index, seed=0,
                        val_games=args.val_games)
    if args.limit:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=D.collate, num_workers=0)
    print("split %s: %d games sampled of %d, %d states, k=%d"
          % (args.split, len(ds) // C.N_MOVES, len(eval_idx), len(ds), args.k),
          flush=True)

    head, cfg = R.load_head(args.head, device)
    C.assert_patch_compat(cfg, patch_index)
    print("head %s" % Path(args.head).parent.name, flush=True)

    added = llm_client.load_dotenv()
    info = llm_client.describe()
    if added:
        print("  .env supplied %s" % ", ".join(added))
    print("backend %s @ %s" % (info["model"], info["base_url"]), flush=True)
    max_tokens = args.max_tokens or (8 * args.k + 8192 if args.rank_all
                                     else 8192)
    call = None
    if not args.dry_run:
        call = llm_client.client_from_env(temperature=args.temperature,
                                          max_tokens=max_tokens)
        if call is None:
            raise SystemExit(
                "no LLM_API_KEY - set one in the environment or in .env, or "
                "run --dry-run to see the prompts without a key")

    cache_path = args.cache or (str(Path(args.out).with_suffix(""))
                               + ".responses.jsonl" if args.out else "")
    cache, lock = {}, threading.Lock()
    if cache_path and Path(cache_path).exists():
        for line in Path(cache_path).read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # `finish_reason` rides along so a resumed run still counts the
            # truncations it inherited instead of reporting zero.
            cache[rec["key"]] = (rec.get("text", ""), rec.get("finish_reason"))
        print("resuming: %d cached responses in %s"
              % (len(cache), cache_path), flush=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    def key_of(sample):
        # `max_tokens` and the model id are part of the key: a budget that was
        # too small caches an empty answer, and a re-run with a bigger budget -
        # or against another backend - must re-ask rather than resume into the
        # result it was trying to replace.
        return "%s|%d|k%d|b%d|p%d|r%d|m%d|%s|%s" % (
            sample["meta"]["gameid"], int(sample["move_idx"]), args.k,
            int(args.board), int(args.probs), int(args.rank_all), max_tokens,
            info["model"], args.render)

    def ask(item):
        key, messages = item
        if key in cache:
            text, finish = cache[key]
            return key, text, None, finish
        last = None
        for attempt in range(args.retries):
            try:
                reply = call(messages)
                text = (reply.get("content") or "").strip()
                finish = reply.get("finish_reason")
                with lock:
                    cache[key] = (text, finish)
                    if cache_path:
                        with open(cache_path, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(
                                {"key": key, "text": text,
                                 "model": reply.get("model"),
                                 "finish_reason": reply.get("finish_reason"),
                                 # Kept because it is the audit trail for the
                                 # `--probs` arm: a model that reasons "highest
                                 # percentage wins" is anchored on the head and
                                 # the accuracy will not show that.
                                 "reasoning": reply.get("reasoning") or ""})
                                + "\n")
                return key, text, None, finish
            except llm_client.LLMError as exc:
                last = str(exc)
                time.sleep(1.5 * (attempt + 1))
        return key, "", last, None

    overall = Tally()
    by_action = {0: Tally(), 1: Tally()}
    by_phase = {k: Tally() for k in SLICE_NAMES}
    by_pool = {(a, p): Tally() for a in (0, 1) for p in (0, 1)}
    # canonical key -> the corpus spelling, so `DraftState.render` prints
    # "Kai'Sa" rather than "kaisa". A callable, because `render_board` treats a
    # falsy `display` as "print the key" and a bare dict is not callable.
    _spelling = {DS.key(n): n for n in names}
    display = lambda k: _spelling.get(k, k)          # noqa: E731
    pairs = {"both": 0, "api_only": 0, "head_only": 0, "neither": 0}
    seen_games = set()
    shown = 0
    truncated = 0
    t0 = time.time()
    n_done = 0

    with torch.no_grad():
        for batch, samples in loader:
            batch = D.to_device(batch, device)
            hlog = head(batch)["policy_logits"]
            cid, cp = R.candidates(hlog, args.k)
            cands = [[(int(c), float(p)) for c, p in zip(a, b)]
                     for a, b in zip(cid.tolist(), cp.tolist())]
            rank = R.ranks(hlog, batch["target_champ"])

            work = []
            for i, s in enumerate(samples):
                seen_games.add(s["meta"]["gameid"])
                if args.render == "state":
                    st, state_text = _state_text(s, names, display)
                    due = st.to_move()
                    want = "ban" if int(s["is_ban"]) else "pick"
                    if (due is None or due["move_idx"] != int(s["move_idx"])
                            or due["kind"] != want):
                        raise SystemExit(
                            "the rebuilt state disagrees with the row: %s move "
                            "%d is %s, the state says %s - `log_from_board` and "
                            "the dataset have drifted"
                            % (s["meta"]["gameid"], int(s["move_idx"]), want,
                               due))
                    cand_line = C.candidates_line(
                        names, [c for c, _ in cands[i]], [p for _, p in cands[i]])
                    if not args.probs:
                        cand_line = _without_probs(cand_line)
                    body = "\n".join([state_text, cand_line,
                                      C.task_line(s, prompt_format="v4")]
                                     ) + "\n" + ANSWER_CUE
                else:
                    _, body, _ = build_text(
                        s, names, context="soft+text" if args.board else "cand",
                        prompt_format="v4", candidates=cands[i])
                if args.rank_all:
                    body = _rank_all_task(body, args.k)
                if not args.probs and args.render == "v4":
                    body = _without_probs(body)
                if not body.endswith(ANSWER_CUE):
                    raise SystemExit("prompt does not end in %r: %r"
                                     % (ANSWER_CUE, body[-80:]))
                messages = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": body}]
                if args.dry_run and shown < 3:
                    print("\n--- %s move %d (target %s) ---\n%s"
                          % (s["meta"]["gameid"], int(s["move_idx"]),
                             names[int(s["target_champ"])], body), flush=True)
                    shown += 1
                work.append((key_of(s), messages))

            if args.dry_run:
                n_done += len(work)
                continue

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                results = list(pool.map(ask, work))

            for i, (key, text, err, finish) in enumerate(results):
                allowed = [c for c, _ in cands[i]]
                chosen, outside = parse_answer(text, allowed, names)
                target = int(samples[i]["target_champ"])
                row = {"chosen": chosen, "outside": outside, "error": err,
                       "hit": chosen is not None and chosen == target,
                       "head_hit": int(rank[i]) == 0,
                       "ceiling_hit": int(rank[i]) < args.k,
                       "pos": None}
                if finish == "length":
                    truncated += 1
                if args.rank_all:
                    order = _extract(text, allowed, names)
                    row["pos"] = (order.index(target) if target in order
                                  else args.k)
                    row["hit"] = row["pos"] == 0
                overall.add(row, True)
                is_ban = int(samples[i]["is_ban"])
                by_action[is_ban].add(row, True)
                by_phase[int(samples[i]["phase"])].add(row, True)
                pool = bool(int(samples[i]["unavailable_mask"].any()))
                by_pool[(is_ban, int(pool))].add(row, True)
                if row["hit"] and row["head_hit"]:
                    pairs["both"] += 1
                elif row["hit"]:
                    pairs["api_only"] += 1
                elif row["head_hit"]:
                    pairs["head_only"] += 1
                else:
                    pairs["neither"] += 1
            n_done += len(results)
            rate = n_done / max(1e-9, time.time() - t0)
            print("  %d/%d states  %.1f/s  eta %.0f min"
                  % (n_done, len(ds), rate, (len(ds) - n_done) / max(1e-9, rate) / 60),
                  flush=True)

    if args.dry_run:
        print("\n%d prompts, %d calls, nothing sent. max_tokens=%d"
              % (n_done, n_done, max_tokens))
        return 0

    o = overall.out()
    if not o.get("n"):
        raise SystemExit("no rows scored")
    lo, hi = wilson(overall.hit, overall.n)
    b, c = pairs["api_only"], pairs["head_only"]
    p = mcnemar(b, c)
    delta = o["api_top1"] - o["head_top1"]
    report = {
        "checkpoint": {"backend": "api", "model": info["model"],
                       "base_url": info["base_url"], "context": "text",
                       "prompt_format": "v4", "board": args.board,
                       "probs": args.probs, "rank_all": args.rank_all,
                       "render": args.render},
        "split": args.split,
        "val_games": len(seen_games),
        "n_states": o["n"],
        "gameids": sorted(seen_games),
        "rerank": {"k": args.k, "ceiling_recall": o["ceiling_recall"],
                   "n_states": o["n"]},
        "overall": o,
        "paired_vs_head": {"both": pairs["both"], "api_only": b,
                           "head_only": c, "neither": pairs["neither"],
                           "delta_pp": 100.0 * delta, "mcnemar_p": p,
                           "wilson95": [lo, hi]},
        "by_action": {"pick": by_action[0].out(), "ban": by_action[1].out()},
        "by_phase": {SLICE_NAMES[k]: v.out() for k, v in by_phase.items()},
        "by_pool": {"%s_pool=%d" % ("ban" if a else "pick", q): v.out()
                    for (a, q), v in sorted(by_pool.items())},
        "truncated": truncated,
        "reference": REFERENCE,
    }

    print("\n== api reranker, k=%d, %s ==" % (args.k, info["model"]))
    print("  states %d   ceiling (head recall@%d) %.4f"
          % (o["n"], args.k, o["ceiling_recall"]))
    print("  head top-1            %.4f" % o["head_top1"])
    print("  api top-1             %.4f   [%.4f, %.4f] 95%%"
          % (o["api_top1"], lo, hi))
    print("  delta vs head         %+.4f (%+.1f pp)  McNemar p=%.4g "
          "(%d/%d discordant)" % (delta, 100 * delta, p, b, c))
    print("  api / ceiling         %.0f %% converted"
          % (100.0 * o["api_top1"] / max(1e-9, o["ceiling_recall"])))
    print("  off-list answers      %.4f   named a real champion outside the "
          "list %.4f   errors %.4f"
          % (o["offlist"], o["named_outside_list"], o["errors"]))
    if "mean_rank" in o:
        print("  mean rank %.2f   ft@5 %.4f  (rank-all arm; the LoRA's "
              "mean_rank was 1.62 at k=5)"
              % (o["mean_rank"], o["ft_top5"]))
    if truncated:
        print("  !! %d/%d answers hit max_tokens=%d and came back empty or "
              "cut - raise --max-tokens, do not read this as a refusal"
              % (truncated, o["n"], max_tokens))
    print("\n== by action ==")
    for name, d in report["by_action"].items():
        if d.get("n"):
            print("  %-5s n=%-5d api %.4f  head %.4f  ceiling %.4f  off %.4f"
                  % (name, d["n"], d["api_top1"], d["head_top1"],
                     d["ceiling_recall"], d["offlist"]))
    print("\n== reference (different rows - do not put these in one column) ==")
    print("  api_top1 is a whole-champion match: compare it to the LoRA's "
          "`exact`,\n  not `ft_top1`, which is first-token group accuracy and "
          "is ~2 pp kinder.")
    for name, d in REFERENCE.items():
        print("  %-16s %s" % (name, d))
    if args.k == 20:
        print("\n  k=20 reminder: the 4B adapter converted 29 % of an 86.35 % "
              "ceiling here\n  (25.27 %% top-1, +5.2 pp over the head's argmax) "
              "and was retired for it.")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1),
                                  encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
