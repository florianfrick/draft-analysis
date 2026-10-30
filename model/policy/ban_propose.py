"""Policy-rollout ban proposer: "ban what they are about to take".

The idea
--------
`ban_search.py` asks the value function to rank every legal ban on one
fixed board. That is structurally hopeless: V reads the *picked* slots, so
banning c does not change V at all until somebody picks differently, which
is why the score surface was 0.9 winrate points wide and flat.

This module inverts it. A ban is only worth something because of what
happens *after* it, so simulate the rest of the draft:

    my ban -> both teams keep drafting (my policy plays both sides)
           -> a complete draft
           -> V scores that draft

Now the arms being compared are different *drafts*, not different bans on
the same draft, and V is doing the job it was trained for and scored
0.5426 on. The flatness disappears by construction.

Who proposes, who verifies
--------------------------
Selection uses no value function at all: the candidate is the champion the
opponent is most likely to take at their first pick after my ban. That is
the user's rule verbatim, and it runs on the strongest component in the
repo (pick prediction at 1.98x/3.58x the frequency baseline) instead of
the weakest (V at 0.54).

V is therefore an *independent* evaluator of the arms, not part of the
selection. That matters: "pick the ban that maximises V, then report that
V likes it" would be circular. `--propose value` does exactly that and is
offered only as a diagnostic, never as the headline.

Phase 1 is deliberately not optimised
-------------------------------------
Pros ban the strongest champions, opponent specialties, or to serve a
unique game plan in phase 1 - there is no picks-on-the-board signal to
reason from, and `ban_intent.py` confirms it: observed phase-1 bans sit at
mean rank 36.9 of ~87 in the opponent's predicted pick distribution, with
no lift over a frequency table. The default is `--phase ban2`. Phase 1
runs, but the report labels it informational rather than a target.

Usage:
    python model/policy/ban_propose.py \
        --policy model/policy/ckpt/run_20261025_000606_ablF-stage1/best_policy.pt \
        --value model/finetune/ckpt/value_20261030_180102/best.pt \
        --verify model/finetune/ckpt/run_20261017_185016_stage1-full/best_win.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MODEL_DIR))

import train as T  # noqa: E402  (model/policy/train.py)
from model import PolicyValueModel  # noqa: E402
from soloq_model import (SoloQPretrainedModel, load_encoder_state,  # noqa: E402
                         CHAMP_PAD_ID)

PRO_DATA = T.PRO_DATA
EP = T.EP
PHASE_NAMES = {0: "bans1", 1: "picks1", 2: "bans2", 3: "picks2"}


def _load_by_path(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# `from model import X` resolves to model/policy/model.py here, so the
# finetune models have to be loaded under a private module name.
FIN_MODEL = _load_by_path("finetune_model", MODEL_DIR / "finetune" / "model.py")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Policy-rollout ban proposer")
    ap.add_argument("--policy", required=True,
                    help="policy checkpoint (the proposer, and the sim engine)")
    ap.add_argument("--value", required=True,
                    help="PrefixValueModel checkpoint (the verifier)")
    ap.add_argument("--verify", default="",
                    help="optional SECOND value model (finetune best_win.pt) "
                         "as an independent cross-check")
    ap.add_argument("--phase", default="bans2",
                    choices=("all", "bans1", "bans2"),
                    help="bans2 is the default: phase 1 is not a target "
                         "(see the module docstring)")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--topk", type=int, default=4,
                    help="shortlist size taken from the opponent's predicted "
                         "pick distribution")
    ap.add_argument("--propose", default="policy",
                    choices=("policy", "value"),
                    help="policy = ban their most likely pick (V-free, the "
                         "headline). value = ban the candidate with the best "
                         "V score (circular, diagnostic only)")
    ap.add_argument("--value-bans", action="store_true",
                    help="let the verifier V read the ban list too (its ban "
                         "path scored worse on AUC; off by default)")
    ap.add_argument("--limit-drafts", type=int, default=0)
    ap.add_argument("--half-life", type=float, default=75.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default="")
    return ap.parse_args(argv)


# ------------------------------------------------------------------- models
def load_policy(path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0))
    model = PolicyValueModel(
        enc, embed_dim=cfg["embed_dim"], state_dim=cfg["state_dim"],
        n_econ=cfg["n_econ"], dropout=cfg.get("head_dropout", 0.0),
        n_patches=cfg.get("n_patches", 0),
        with_value=cfg.get("with_value", True)).to(device)
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError("policy ckpt mismatch: %s / %s"
                           % (list(missing)[:4], list(unexpected)[:4]))
    model.eval()
    return model, cfg


def load_prefix_value(path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0),
        with_value=True, value_seeds=cfg.get("value_seeds", 4))
    model = FIN_MODEL.PrefixValueModel(
        enc, ban_heads=cfg["ban_heads"], ban_seeds=cfg["ban_seeds"],
        ban_dropout=cfg["ban_dropout"], use_bans=cfg["use_bans"]).to(device)
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError("value ckpt mismatch: %s / %s"
                           % (list(missing)[:4], list(unexpected)[:4]))
    model.eval()
    return model, cfg


def load_finetune_value(path, device):
    """The finetune win head (complete drafts only) as a second opinion.

    Loaded by path with a private module name: model/finetune/train.py and
    model/policy/train.py are both called `train`, so importing them in one
    process collides.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    enc = SoloQPretrainedModel(
        num_champs=cfg["num_champs"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_intra_layers=cfg["num_intra"],
        num_cross_layers=cfg["num_cross"], dropout=cfg.get("dropout", 0.0))
    model = FIN_MODEL.ProDraftModel(enc, n_econ=cfg.get("n_econ", 6)).to(device)
    # The checkpoint holds the whole ProDraftModel (encoder.* + heads.*),
    # so it is loaded as one piece rather than into the bare encoder.
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError("verifier ckpt mismatch: %s / %s"
                           % (list(missing)[:4], list(unexpected)[:4]))
    model.eval()
    return model


# -------------------------------------------------------------- simulation
class BatchedDraft:
    """The draft state for a batch of rows, advanced one move at a time.

    The sequence of (kind, team, role) is known in advance - only the
    champion is unknown - so the simulation walks the row's own move list
    and lets the policy choose the champion at every step.
    """

    def __init__(self, rows, row_ids, move_idx, num_champs, patch_index,
                 device):
        samples = [PRO_DATA.build_prefix(rows[i], move_idx, num_champs,
                                         patch_index) for i in row_ids]
        b = PRO_DATA.pro_collate(samples)
        self.device = device
        self.num_champs = num_champs
        self.champ_ids = b["champ_ids"].to(device)
        self.role_ids = b["role_ids"].to(device)
        self.pad_mask = b["pad_mask"].to(device)
        self.ban_ids = b["ban_ids"].to(device)
        self.n_bans = b["n_bans"].to(device)
        self.taken = b["taken_mask"].to(device)
        self.unavail = b["unavailable_mask"].to(device)
        self.first_pick = b["first_pick"].to(device)
        self.patch_id = b["patch_id"].to(device)
        self.B = len(row_ids)
        self.rows = row_ids
        self.rows_data = rows

    def batch(self, t, kind, team):
        """Policy-ready batch for move t (kind/team are [B] tensors)."""
        return {
            "champ_ids": self.champ_ids, "role_ids": self.role_ids,
            "pad_mask": self.pad_mask, "ban_ids": self.ban_ids,
            "taken_mask": self.taken, "unavailable_mask": self.unavail,
            "move_idx": torch.full((self.B,), t, dtype=torch.long,
                                   device=self.device),
            "phase": torch.full((self.B,), PRO_DATA.PHASE_OF[t],
                                dtype=torch.long, device=self.device),
            "to_move": team, "first_pick": self.first_pick,
            "is_ban": kind, "patch_id": self.patch_id,
        }

    def apply(self, champs, kind, team, role):
        """Write the chosen champion into the board and update legality."""
        idx = torch.arange(self.B, device=self.device)
        is_pick = kind == 0
        # picks land in the exact role slot: team*5 + role
        slot = (team * 5 + role).clamp(0, 9)
        pick_rows = is_pick.nonzero(as_tuple=True)[0]
        if pick_rows.numel():
            s = slot[pick_rows]
            self.champ_ids[pick_rows, s] = champs[pick_rows]
            self.role_ids[pick_rows, s] = role[pick_rows]
            self.pad_mask[pick_rows, s] = False
        ban_rows = (~is_pick).nonzero(as_tuple=True)[0]
        if ban_rows.numel():
            pos = self.n_bans[ban_rows].clamp(0, 9)
            self.ban_ids[ban_rows, pos] = champs[ban_rows]
            self.n_bans[ban_rows] = (self.n_bans[ban_rows] + 1).clamp(0, 10)
        self.taken[idx, champs] = True

    def complete_batch(self):
        """V-ready batch: the finished draft (all 10 slots filled)."""
        return {"champ_ids": self.champ_ids, "role_ids": self.role_ids,
                "pad_mask": self.pad_mask, "ban_ids": self.ban_ids}


def move_plan(rows, row_ids, start):
    """[(t, kind[B], team[B], role[B])] for t in start..N_MOVES-1."""
    plan = []
    for t in range(start, PRO_DATA.N_MOVES):
        kind = torch.tensor([0 if rows[i]["moves"][t][0] == PRO_DATA.MOVE_PICK
                             else 1 for i in row_ids], dtype=torch.long)
        team = torch.tensor([rows[i]["moves"][t][1] for i in row_ids],
                            dtype=torch.long)
        role = torch.tensor([rows[i]["moves"][t][3] for i in row_ids],
                            dtype=torch.long).clamp(0, 4)
        plan.append((t, kind, team, role))
    return plan


@torch.no_grad()
def rollout(policy, rows, row_ids, move_idx, num_champs, patch_index, device,
            forced_ban=None, watch_opp_pick=False, opp_team=None):
    """Simulate moves move_idx..19 with the policy playing every side.

    `forced_ban` overrides the champion chosen at the first move (the ban
    under evaluation); everything after it is the policy's own choice, so
    both teams "play like me" - the premise of the whole exercise.

    `watch_opp_pick` returns the opponent's pick distribution at their
    first pick after move_idx, which is the denial target. All rows must
    share the same banning team, or their opponent's first pick falls on
    different steps (first_pick decides who bans first).
    """
    sim = BatchedDraft(rows, row_ids, move_idx, num_champs, patch_index,
                       device)
    plan = move_plan(rows, row_ids, move_idx)
    watch = None
    for step, (t, kind, team, role) in enumerate(plan):
        kind = kind.to(device)
        team = team.to(device)
        role = role.to(device)
        logits = policy(sim.batch(t, kind, team))["policy_logits"]
        if step == 0 and forced_ban is not None:
            champs = forced_ban.to(device)
        else:
            champs = logits.argmax(dim=1)
        if watch_opp_pick and watch is None and opp_team is not None \
                and bool((kind == 0).all()) and bool((team == opp_team).all()):
            watch = {"t": t, "team": opp_team,
                     "probs": torch.softmax(logits, dim=1)}
        sim.apply(champs, kind, team, role)
    return sim, watch


@torch.no_grad()
def score_drafts(value_model, sims, device, use_bans=False):
    """P(blue wins) for each simulated finished draft."""
    out = []
    for sim in sims:
        b = sim.complete_batch()
        b = {k: v.to(device) for k, v in b.items()}
        logit = value_model(b, with_bans=use_bans)["win_logit"].squeeze(-1)
        out.append(torch.sigmoid(logit).cpu())
    return torch.stack(out)          # [n_arms, B]


@torch.no_grad()
def score_drafts_finetune(model, sims, device):
    """Second opinion from the finetune win head (ban-blind, full drafts)."""
    out = []
    for sim in sims:
        b = {k: v.to(device) for k, v in sim.complete_batch().items()}
        out.append(torch.sigmoid(model(b)["win_logits"][:, 1]).cpu())
    return torch.stack(out)


# ---------------------------------------------------------------- analysis
def paired(diff, boot, seed=0):
    """Mean paired difference + bootstrap CI + share of decisions > 0."""
    n = diff.numel()
    if n == 0:
        return {}
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (boot, n), generator=g)
    means = diff[idx].mean(dim=1)
    return {"mean": float(diff.mean()),
            "ci_lo": float(means.quantile(0.025)),
            "ci_hi": float(means.quantile(0.975)),
            "frac_positive": float((diff > 0).float().mean()),
            "n": int(n)}


def collect_decisions(rows, idxs, phase):
    want = {"all": (0, 2), "bans1": (0,), "bans2": (2,)}[phase]
    out = []
    for i in idxs:
        moves = rows[i]["moves"]
        for m, (kind, team, champ, _r) in enumerate(moves):
            if kind != PRO_DATA.MOVE_BAN:
                continue
            if PRO_DATA.PHASE_OF[m] not in want:
                continue
            out.append({"row": i, "move": m, "team": team, "champ": champ})
    return out


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    policy, pcfg = load_policy(args.policy, device)
    value, vcfg = load_prefix_value(args.value, device)
    fin = load_finetune_value(args.verify, device) if args.verify else None
    print("policy %s\nvalue  %s%s"
          % (args.policy, args.value,
             "\nverify %s" % args.verify if args.verify else ""))

    rows, splits = T.load_pro_rows(args)
    idxs = splits[args.split]
    num_champs = pcfg["num_champs"]
    patch_index = PRO_DATA.patch_index_from_vocab(pcfg.get("patch_vocab"))
    decisions = collect_decisions(rows, idxs, args.phase)
    print("%s: %d drafts, %d ban decisions (%s)"
          % (args.split, len(idxs), len(decisions), args.phase))

    # Group by (move, banning team). Rows at the same move index can have
    # different teams - first_pick decides who bans first - and then the
    # opponent's first pick lands at a different step. Batching them
    # together would read one row's "opponent pick" off another row's move.
    by_key = {}
    for d in decisions:
        by_key.setdefault((d["move"], d["team"]), []).append(d)

    arms = ["proposed", "observed", "random"]
    acc = {a: {"v": [], "fin": []} for a in arms}
    agree, shortlist_v = [], []
    changed, hamming_all = [], []
    gen = torch.Generator().manual_seed(args.seed)

    for (m, ban_team) in sorted(by_key):
        group = by_key[(m, ban_team)]
        row_ids = [d["row"] for d in group]
        n = len(group)
        opp = 1 - ban_team

        # 1. Self arm: my ban = the policy's own preference. Used ONLY to
        #    read the opponent's pick distribution, so the shortlist never
        #    sees the observed ban (that would leak the answer).
        _, watch = rollout(policy, rows, row_ids, m, num_champs, patch_index,
                           device, watch_opp_pick=True, opp_team=opp)
        probs = watch["probs"]
        k = min(args.topk, probs.shape[1])
        top = probs.topk(k, dim=1).indices          # [n, k]

        # 2. Candidate arms: ban each shortlisted champion in turn.
        sims = [rollout(policy, rows, row_ids, m, num_champs, patch_index,
                        device, forced_ban=top[:, j])[0]
                for j in range(k)]
        v_cand = score_drafts(value, sims, device, args.value_bans)

        # Proposal: the champion the opponent is most likely to take.
        # `--propose value` instead takes the best V score - circular, and
        # only useful as a diagnostic.
        pick = (torch.zeros(n, dtype=torch.long) if args.propose == "policy"
                else v_cand.argmax(dim=0))
        prop_champ = top[torch.arange(n, device=top.device),
                         pick.to(top.device)].cpu()

        # 3. Controls: the human ban, and a uniformly random legal ban.
        obs_champ = torch.tensor([d["champ"] for d in group])
        taken = BatchedDraft(rows, row_ids, m, num_champs, patch_index,
                             device).taken.cpu()
        legal = (~taken)
        rand_champ = torch.stack([
            legal[i].nonzero().flatten()[
                torch.randint(0, int(legal[i].sum()), (1,), generator=gen)]
            for i in range(n)]).squeeze(1)
        sim_obs = rollout(policy, rows, row_ids, m, num_champs, patch_index,
                          device, forced_ban=obs_champ)[0]
        sim_rnd = rollout(policy, rows, row_ids, m, num_champs, patch_index,
                          device, forced_ban=rand_champ)[0]

        v_prop = v_cand.gather(0, pick.view(1, -1)).squeeze(0)
        v_obs = score_drafts(value, [sim_obs], device, args.value_bans)[0]
        v_rnd = score_drafts(value, [sim_rnd], device, args.value_bans)[0]

        # Did the ban actually change the draft? If the simulated board is
        # identical, the arm is a no-op and V *must* return the same score -
        # so a null delta can mean "no difference" or "nothing happened".
        # This separates the two.
        boards = torch.stack([s.champ_ids.cpu() for s in sims])   # [k,n,10]
        prop_board = boards[pick, torch.arange(n)]                # [n,10]
        obs_board = sim_obs.champ_ids.cpu()
        hamming = (prop_board != obs_board).sum(1)
        changed.append((hamming > 0).float())
        hamming_all.append(hamming.float())
        acc["proposed"]["v"].append(v_prop)
        acc["observed"]["v"].append(v_obs)
        acc["random"]["v"].append(v_rnd)

        if fin is not None:
            f_cand = score_drafts_finetune(fin, sims, device)
            acc["proposed"]["fin"].append(
                f_cand.gather(0, pick.view(1, -1)).squeeze(0))
            acc["observed"]["fin"].append(
                score_drafts_finetune(fin, [sim_obs], device)[0])
            acc["random"]["fin"].append(
                score_drafts_finetune(fin, [sim_rnd], device)[0])

        agree.append((prop_champ == obs_champ).float())
        shortlist_v.append(v_cand)
        print("  move %2d team %d: %4d decisions" % (m, ban_team, n))

    res = {"policy": args.policy, "value": args.value, "verify": args.verify,
           "split": args.split, "phase": args.phase,
           "propose": args.propose, "topk": args.topk,
           "value_bans": args.value_bans, "arms": {}}
    v = {a: torch.cat(acc[a]["v"]) for a in arms}
    for a in arms:
        res["arms"][a] = {"mean_v": float(v[a].mean()),
                          "n": int(v[a].numel())}
    res["delta_proposed_minus_observed"] = paired(
        v["proposed"] - v["observed"], args.boot, args.seed)
    res["delta_proposed_minus_random"] = paired(
        v["proposed"] - v["random"], args.boot, args.seed)
    res["delta_observed_minus_random"] = paired(
        v["observed"] - v["random"], args.boot, args.seed)
    res["agreement_with_observed"] = float(torch.cat(agree).mean())
    res["shortlist_spread"] = float(torch.cat(shortlist_v, dim=1).std())
    res["proposal_changes_the_draft"] = float(torch.cat(changed).mean())
    res["proposal_hamming"] = float(torch.cat(hamming_all).mean())
    if fin is not None:
        f = {a: torch.cat(acc[a]["fin"]) for a in arms}
        for a in arms:
            res["arms"][a]["mean_verify"] = float(f[a].mean())
        res["verify_delta_proposed_minus_observed"] = paired(
            f["proposed"] - f["observed"], args.boot, args.seed)
        res["verify_delta_proposed_minus_random"] = paired(
            f["proposed"] - f["random"], args.boot, args.seed)

    print("\n=== %s / %s (%d decisions) ===" % (args.phase, args.split,
                                                res["arms"]["proposed"]["n"]))
    for a in arms:
        extra = ("" if "mean_verify" not in res["arms"][a]
                 else "  verify %.4f" % res["arms"][a]["mean_verify"])
        print("  %-9s V %.4f%s" % (a, res["arms"][a]["mean_v"], extra))
    for name in ("delta_proposed_minus_observed", "delta_proposed_minus_random",
                 "delta_observed_minus_random"):
        d = res[name]
        print("  %-30s %+0.5f  [%+0.5f, %+0.5f]  %.1f%% of decisions >0"
              % (name.replace("delta_", ""), d["mean"], d["ci_lo"],
                 d["ci_hi"], 100 * d["frac_positive"]))
    if "verify_delta_proposed_minus_observed" in res:
        for name in ("verify_delta_proposed_minus_observed",
                     "verify_delta_proposed_minus_random"):
            d = res[name]
            print("  %-30s %+0.5f  [%+0.5f, %+0.5f]  %.1f%% >0"
                  % (name.replace("verify_delta_", "VERIFIER "), d["mean"],
                     d["ci_lo"], d["ci_hi"], 100 * d["frac_positive"]))
    print("  agreement with the human ban: %.3f" %
          res["agreement_with_observed"])
    print("  the proposal changes the simulated draft: %.1f%% of decisions"
          " (mean %.2f champion slots)" %
          (100 * res["proposal_changes_the_draft"], res["proposal_hamming"]))

    out = Path(args.out) if args.out else HERE / (
        "ban_propose_%s.json" % args.phase)
    out.write_text(json.dumps(res, indent=1))
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
