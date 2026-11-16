#!/usr/bin/env bash
# Four arms on the SAME split, SAME init, SAME seed: the only variable is
# ban-awareness. `control` reproduces the shipped baseline (measured: 0.4995
# top-5 against the shipped 0.4995), which is what makes the rest comparable.
# ~22 min per arm on a 4060.
set -euo pipefail
INIT=pro/model/archive/checkpoints/masking-proonly-randomvalsplit-through-wlds-ig-tes/best.pt
COMMON="--no-test --val-split random --seed 0 --no-wandb --init-ckpt $INIT"
run () {  # name, extra flags
  echo "=== $1 ==="
  python pro/model/proonly/train_policy.py $COMMON $2 \
    --out-dir "pro/model/archive/checkpoints/nextpick-bans-$1-through-wlds-ig-tes"
}
run control        ""
run counter-only   "--counter-rank 32"
run fearless-only  "--ban-fearless"
run aware          "--ban-side --ban-fearless --counter-rank 32"
