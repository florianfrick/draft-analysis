#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
INIT=pro/model/proto/runs/masked-pro/best.pt
RUN="docker compose run --rm -T dev python pro/model/proto/train_policy.py --init $INIT --epochs 64"
run () { local name=$1; shift; echo "=== $name  $*"; $RUN --name "$name" "$@" 2>&1 | grep -E "^(arm=|ep |best )" || true; }
run base64
run combo64 --slot-mask --prior-bias --unfreeze 1 --lr-enc 1e-4
echo "=== round2 done"
