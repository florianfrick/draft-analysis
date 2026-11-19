# Project report — LoL Draft Analysis

## Summary

This project is a League of Legends draft assistant. It learns from pro drafts
and high-elo solo-queue games to answer two questions during a live draft:
**what should be picked or banned next**, and **which champion belongs in each
open lane slot**. The pro track is the maintained, shipping track; the soloQ
track exists as the reference point for whether pretraining helps.

The project now has two clients: the original local Python/browser app and a
native, fully offline Android app that runs the shipped models on-device with
ExecuTorch.

## Components

| Area | Role |
|---|---|
| `data/` | Raw Oracle's Elixir pro CSVs, soloQ match data, derived draft caches, and the pro data pipeline. |
| `solo_queue/` | Siamese base encoder, masked pretraining, soloQ→pro finetune, policy head, and soloQ analysis store. |
| `pro/` | Pro-only training, the canonical evaluation suite, baselines, checkpoints, and the board-holding agent. |
| `frontend/` | Local FastAPI/JS draft UI and model-serving endpoints. |
| `android/` | Native Kotlin/Compose offline client, ExecuTorch export tools, bundled assets, and demo videos. |
| `STATUS.md` / `FINDINGS.md` | Historical measurements, decisions, and dead ends. These are the source of truth for "what was measured". |

## Current shipped stack

- **Role board / masked completion:** `masking-proonly-nov13-through-lcp-df-sw`
- **Next move:** `nextpick-naivetransformer-role-d512-l8-h8-k5-through-lcp-df-sw`
- **Canonical evaluation split:** 39,720 / 4,771 / 2,448 drafts over 99 patches.
  Two suites are scored separately: `next_move` (prefix-only) and `masked`
  (one slot hidden, including later picks as context).
- **Frontend:** `bash frontend/serve.sh` serves the local UI on 127.0.0.1.
- **Android:** Kotlin/Compose with `policy.pte` (26.6 MB, quantized) and
  `masker.pte` (10.0 MB, FP32), bundled champion portraits, and static metadata.
  It needs no backend, network, auth, TLS, or API key. The optional Agent
  toggle is off by default and talks to `android/worker/` only on an explicit
  question.
- The Python/browser stack remains in the repository as the reproduction and
  evaluation history; it is not required to build or run the Android app.

## What works

- Draft state is modeled explicitly: ten slots, ten bans, first-pick side,
  patch, fearless pool, and series game index. Legality is applied as a logit
  mask before top-k, never as a post-softmax filter.
- Role assignment uses the champion-role table and a joint injective per-team
  assignment, not a per-champion argmax.
- The naive transformer serves next-move suggestions; the masking checkpoint
  covers role completion during the ban phase.
- Series/fearless is supported in both clients. `nextGame()` carries the
  current game's picks into the next pool; bans do not carry over.
- The Android engine reproduces the Python serving path on representative
  boards: empty, one-ban, fearless-pool, masker-cell, and head-role cases.
  Near-tie rank flips are explained by quantization, not contract drift.

## Recent work

The Android work began on branch `android` and was merged into `main` at
commit `34c3815`; the current release candidate is `98961ae` (`v0.1.0`). It
adds the offline ExecuTorch app, UI simplification, the opt-in Agent chat, the
2026 Worlds final walkthrough, and a rank-bar animation.

The walkthrough now replays the real four-game **BLG 3–1 Gen.G 2026 Worlds
final** using the app's `nextGame()` path, including the alternating first-pick
side and fearless pools of 0/10/20/30 champions. The rank animation measures
where each real pick appeared in the model's legal ranking: 12/40 picks were
top-five, distributed 2/3/4/3 across games 1–4.

## Important constraints

- Do not silently rebuild the pro corpus. The frozen `pro_drafts.pt` and the
  unfrozen `pro_drafts_full.pt` are different corpora; raising the freeze
  invalidates recorded numbers and the test split.
- The updated Oracle's Elixir CSV used for the 2026 final video lives in
  Downloads and is deliberately **not** committed. Only the exported demo JSON
  is checked in.
- Champion ids on the wire are Riot ids; tensors use 0–172 model indices.
  Patch ids are 1-based vocabulary rows, not list positions.
- Weights are git-ignored. Numbers and provenance live in `STATUS.md`,
  `FINDINGS.md`, and the dated work logs.

## Validation and next steps

Android `assembleDebug`, `testDebugUnitTest`, and `lintDebug` pass. The Python
training/evaluation suites are documented in `STATUS.md`; measured dead ends
should not be re-run.

Remaining candidates: a full 20-move end-to-end UI check, series persistence
across process death, and a quantized masker if ExecuTorch gains the missing
out-variants. Local `main` is ahead of `origin/main`; publishing was not part
of this change.
