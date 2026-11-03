# LoL Draft Analysis

> **Current state of the project: see [STATUS.md](STATUS.md)** — one section per
> stage (pretraining, finetune, policy, prefix value), the exact checkpoints
> kept, the evaluation numbers for each, what failed and why, and the plan.

Project Goals:

Create a draft analyzer/helper/predictor for league of legends pro games. It should be able to predict which champion will be picked/banned next.


First, generate an understanding of champions, their role, strength on the patch, and synergies/counters. Use high elo soloQ games for lots of data (just one patch for now).

## Repo layout & setup (keys + data stay local, never committed)

SoloQ masked-draft pretraining + pro-play finetune for League of Legends
draft prediction (patch 16.20).

### Layout

- `model/pretraining/soloq_model.py` + `model/README.md` — frozen I/O contract.
- `model/pretraining/train.py` — masked champ+role pretraining
  (`--role-only-prob 0.10` teaches P(role|champ)).
- `model/finetune/` — pro-play Stage 1 finetune (masked heads + win/econ).
- `model/proonly/` — the pro-only recipe: pro drafts only, no soloQ, patch
  embedding, within-patch split. Masked stage, next-pick stage, paired
  comparison. See `model/proonly/README.md`.
- `model/checkpoints/` — the five kept artefacts, one directory per inference
  class: `masking-*` answers the role board, `nextpick-*` the next move,
  `value-*` neither. Weights are git-ignored.
- `model/test_split/` — three pinned Worlds 2026 series replayed through every
  checkpoint, per class (what each model *said*, not just its accuracy). See
  `model/test_split/README.md`.
- `data/solo queue/` — Riot collection (puuids → match ids → match details)
  + `process matchdata/pipeline.py` (masking, batch collate).
- `data/pro play/` — Oracle's Elixir CSV → pro draft rows.
- `frontend/` — local draft-assistant UI (127.0.0.1 only).

## Setup (keys + data stay local, never committed)

### Environment: Docker

The Python environment is pinned in `Dockerfile` + `requirements.txt` (python
3.13, torch 2.14.1+cu126, numpy/pandas/pyarrow/scipy/sklearn/duckdb/wandb).
Use it instead of a machine-level install — the local interpreter's
`site-packages` got wiped once and took the whole toolchain with it.

```bash
docker compose build dev        # once
docker compose run --rm dev     # shell in the project, data bind-mounted
```

See `DOCKER.md` for tests, training, long runs and GPU checks. Everything below
also works verbatim on the host if you already have the pinned versions.

1. Copy `.env.example` to `.env` and fill in:
   - `RIOT_API_KEY` — Riot dev key (https://developer.riotgames.com,
     expires every 24h). Used by `data/solo queue/collect_*.py` via
     `data/solo queue/riot_api.py`.
   - `WANDB_API_KEY` — W&B key (https://wandb.ai/authorize), or pass
     `--no-wandb` to training. `docker-compose.yml` loads `.env` into the
     container, so every trainer reads it from the environment; there is no
     key file on disk.
2. Rebuild data locally (all raw/derived match data is git-ignored):
   - SoloQ: `collect_puuids.py` → `collect_matchids.py` →
     `collect_match_data.py`, then the process-matchdata pipeline caches
     `drafts_<patch>.pt`.
   - Pro: drop Oracle's Elixir CSVs into `data/pro play/` and run
     `pipeline.py --rescan` to build `pro_drafts.pt`.
3. Train: `python model/pretraining/train.py --epochs 30`
   (smoke: `--limit-drafts 4000 --epochs 2 --no-wandb`).
   Checkpoints are git-ignored (100MB+ each) — both the kept
   `model/checkpoints/` artefacts and any `ckpt/run_*/` a trainer writes.
   Share them out-of-band (releases, drive, etc.), never via git.

See each folder's README for full commands.

- Multiple embedding layers summed e.g. champion (num champions + \[MASK\] + \[PAD\]), role (num roles + unknown/flex), team and patch if necessary
    - e.g. ```class DraftSlotEmbedding(nn.Module):
        def __init__(self, num_champs=173, num_roles=6, embed_dim=128):
            super().__init__()
            # Static lookup tables
            self.champ_lookup = nn.Embedding(num_champs + 1, embed_dim) # +1 for [MASK]
            self.role_lookup  = nn.Embedding(num_roles, embed_dim)      # Top, Jg, Mid, Bot, Sup, [FLEX]
            self.layer_norm   = nn.LayerNorm(embed_dim)

        def forward(self, champ_ids, role_ids, team_ids):
            # Summing embeddings overlays role & team context onto the static champion ID
            x = (self.champ_lookup(champ_ids) + 
                self.role_lookup(role_ids))
            
            return self.layer_norm(x)```

- Masked modeling champion prediction in full draft (~15% chance to mask a champion and its role)
    - ```                      [Blue Champion Set]                     [Red Champion Set]
                                         │                                      │
                                         ▼                                      ▼
                             Intra-Team Self-Attention               Intra-Team Self-Attention
                                         │                                      │
                                         └───────────────┬──────────────────────┘
                                                         ▼
                                         Inter-Team Symmetric Cross-Attention
                                                         │
                                                         ▼ 
                                        ┌────────────────┴────--(potentially/finetune)──────────┐
                                        ▼                                                       ▼
                                [Slot-Level Vectors]                                     [PMA Set Pooling]
                                   Extract H_masks                                    v_Blue, v_Red 1D vectors
                                        │                                                       │
                                        │                                                       ▼
                                        │                                    Concat(v_Blue, v_Red, v_Blue - v_Red, v_Blue ⊙ v_Red)  
                                        ▼                                                       │
                    ┌───────────────────┴────────────────────┐                                  ▼
                    ▼                                        ▼                    Auxiliary heads (e.g. win prob / game metrics)
        Champion Projection Head                   Role Projection Head
        Linear Layer (173 Champions + mask?)               Linear Layer (5 Roles + flex?)
                    │                                        │
                    ▼                                        │
 Mask logits of already picked/banned champions (-inf)       │
                    │                                        │
                    ▼                                        ▼
        Soft Cross-Entropy Loss                      Cross-Entropy Loss
    (Cosine Similarity Targets / Softmax)           (Hard-Label Role Target)
                    │                                        │
                    └───────────────────┬────────────────────┘
                                        ▼
                    Total Loss = Loss_Champ + λ * Loss_Role ```
- Auxiliary prediction heads (game duration/outcome, team/lane gold lead at x minutes, kills, deaths, objectives). More comprehensive + win probability estimate.
- Possibly weigh loss differently based on game outcome (e.g. a team that loses might have made a draft mistake)



Then, finetune / use with another model architecture for pro game draft order and fearless bans
- Next pick/ban decision model
- Desired frontend UI that, given the current state of a draft, can show all of the top viable options for all remaining roles.
    This should allow a coach, for example, to see not just what counterpicks are available in the next pick, but what the opponent's ideas are.
    When evaluating the current draft and the remaining roles, keep in mind that a flex pick might be placed in several roles.
- Not sure on the architecture or training yet.





Why set attention blocks (and not ISAB):
- https://arxiv.org/pdf/1810.00825
- The same champion can not be picked twice and only the current state of the draft matters for the next pick. The order in which previous picks/bans were made doesn't matter.
- Thus, permutation invariance and no positional embedding is desired.
- Do NOT need induced set attention because the number of champions in a draft is very small.
- They should be Siamese (same weights for blue and red team)





Draft Format:

Fearless bans (previously played champions in the series can not be picked again)
First pick no longer always blue.
Ban Phase 1:	Blue / Red	Alternating 3 bans per side (3-3 split)
Pick Phase 1:   B1
                R1, R2
                B2, B3
                R3
Ban Phase 2:	Red / Blue	Alternating 2 bans per side (2-2 split)
Pick Phase 2:	R4
                B4, B5
                R5



Best Model So Far (the only checkpoints still on disk — see STATUS.md):
- masking-soloqueue   soloQ pretrain (kept, not retrained - it is the init)
  masking-finetune    soloQ -> pro finetune
  masking-proonly     pro drafts only, from scratch
  nextpick-policy     soloQ encoder + policy head
  nextpick-proonly    pro drafts only, encoder + policy from scratch