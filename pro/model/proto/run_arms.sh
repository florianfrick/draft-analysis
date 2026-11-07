#!/usr/bin/env bash
# Sequential arm runner. One GPU, so arms run one at a time; each writes its
# own log and its own best.pt, and the evaluator reads them all afterwards.
#
# Selection is on val exact top-5 (the trainer does that internally); the test
# split is only read once per arm by pro/model/proto/evaluate.py, after the fact.
set -u
cd "$(dirname "$0")/../.."

INIT=${INIT:-pro/model/proto/runs/masked-pro/best.pt}
EP=${EP:-36}
RUN="docker compose run --rm -T dev python pro/model/proto/train_policy.py --init $INIT --epochs $EP"

run () {  # name, extra flags...
  local name=$1; shift
  echo "=== $name  $*"
  $RUN --name "$name" "$@" 2>&1 | grep -E "^(arm=|ep |best )" || true
}

run base
run slotmask --slot-mask
run prior --prior-bias
run unfreeze --unfreeze 1 --lr-enc 1e-4
run combo --slot-mask --prior-bias --unfreeze 1 --lr-enc 1e-4
echo "=== arms done"
