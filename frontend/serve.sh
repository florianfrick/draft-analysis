#!/usr/bin/env bash
# Serve the frontend with the LLM reranker in the loop.
#
# Three things have to be true at once, which is why this exists:
#
#   * the image must have transformers/peft/bitsandbytes  -> lol-draft:llm,
#     built from pro/model/llm/Dockerfile, not the base lol-draft:dev
#   * `python` must be that venv's interpreter           -> the image's ENV PATH
#     does that, but only for a NON-login shell (see the note at the bottom)
#   * the port must be published                         -> pro/model/llm/run.sh
#     publishes nothing, so a server started through it is unreachable from the
#     browser even though it is running fine
#
# The `agent` toggle in the chat panel needs one more thing, because langgraph is
# not in the image and pip installs into the image are lost with `--rm`:
#
#   pro/model/llm/.venv/bin/pip install langgraph
#
# The venv is inside the bind mount, so that survives. And `.env` is read when
# the container is created, so re-run this after adding a key.
#
#   bash frontend/serve.sh                       # http://127.0.0.1:8777
#   bash frontend/serve.sh --port 8780           # or PORT=8780
#   bash frontend/serve.sh --policy-ckpt pro/model/checkpoints/nextpick-bans-aware-through-wlds-ig-tes/best.pt
#
# Every argument is passed through to frontend/server.py; --port is consumed
# here as well, because the publish spec has to use the same number. The
# reranker is the next-move default, so the plain invocation is the fast one.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8777}"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --port)   PORT="$2"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    *)        ARGS+=("$1"); shift ;;
  esac
done

# The publish spec is bound to host loopback: `--host 0.0.0.0` inside the
# container is only reachable through this mapping, and never off the machine.
# `--device cuda` is not decoration either - the reranker is 4.2B in 4-bit and
# a CPU forward pass takes minutes; pass `--device cpu` after it to override.
#
# If you invoke the container by hand instead, use `bash -c`, never `bash -lc`:
# a login shell resets PATH, `python` becomes /usr/local/bin/python, and the
# frontend then serves the head while telling you bitsandbytes is not installed.
exec docker compose run --rm -T \
  -p "127.0.0.1:${PORT}:${PORT}" \
  llm python frontend/server.py --host 0.0.0.0 --port "$PORT" --device cuda \
  "${ARGS[@]+"${ARGS[@]}"}"
