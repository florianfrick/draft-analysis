# Findings index

`STATUS.md` is the single source of truth for measurements, decisions, and dead
ends. This file is only an index so older references to a top-level
`FINDINGS.md` still lead somewhere useful.

## Where the findings live

- `STATUS.md` §5 — ban analyses and why the ban-aware arm was kept separate
  from the policy comparison.
- `STATUS.md` §6 — measured dead ends: do not re-run them against the same
  corpus and split.
- `STATUS.md` §7 — the current numbers for the maintained pro-only stack.
- `android/README.md` — on-device parity checks and ExecuTorch constraints.

## Archived experiment snapshots

These are historical snapshots, not release branches. Their history is kept as
tags so the negative results remain citable without cluttering `main`:

| Tag | What it records |
|---|---|
| `archive/exp-bans` | Ban-aware policy experiment. |
| `archive/exp-langgraph` | Board-holding agent / canonical split work. |
| `archive/exp-pro-only-patch-emb` | Pro-only patch-embedding comparison. |
| `archive/exp-transformer` | Naive transformer policy. |
| `archive/exp-win-prediction` | Winner prediction dead end; final handoff is commit `e119572`. |
| `archive/android-mvp` | Android snapshot before the chat and header polish. |

The archived snapshots are local tags unless they have been pushed explicitly.
`exp/win-prediction` is the only former branch with commits not contained in
`main`; its nine commits are preserved by its tag.
