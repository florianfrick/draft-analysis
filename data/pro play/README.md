# Pro play data

Raw OE CSVs (2022-2026) + cached `pro_drafts.pt` (do NOT commit the cache
to git - rebuild anytime with `--rescan`).

## Run

```text
python pipeline.py --rescan          # full rescan (~1 min), refresh cache
python pipeline.py                   # loads cache, prints stats/splits
pytest test_pipeline.py -v           # ~20 s (uses the cache)
```

pipeline.py mirrors `../../solo queue/process matchdata/pipeline.py`
conventions (scan_and_index, kept/skipped stats, per-run provenance in
the cache) but parses per-game draft sequences instead of full drafts:

- 46,662 drafts (12-row OE blocks -> global 20-move sequence on the
  2026 First-Selection ban/pick order; pre-2026 defaults to blue-first).
- Shared champion space with soloQ: names resolve through the same
  Data Dragon `champ_mapping.json` (173/173 coverage in every year).
- Era handling: fearless `unavailable`/`prior_picks` are data-driven per
  series (2025+: mostly fearless; 2026: fully; pre-2025: factual prior
  picks only, `unavailable=[]`), First-Selection first-pick decoupling
  in 2026 only.
- Splits: series-grouped + time-ordered (train 40.5k / val 3.0k /
  test 3.1k games by default) - a series never straddles splits, so
  fearless context and team meta can't leak.
- Recency weighting: `assign_weights(rows, half_life=75d)`,
  w = 2^(-age/half_life), mean-normalized; used per-example by both
  datasets (`weight` key) and the train-fit popularity baseline.

datasets.py: `ProNextMoveDataset` (draft -> 20 prefix/target examples for
next-pick/ban finetuning, incl. turn info, taken/unavailable legality
masks, win_label) and `ProMaskedDraftDataset` (full pro drafts through
the verbatim soloQ masking, for encoder finetune + masked eval that
satisfies the soloQ contract).
