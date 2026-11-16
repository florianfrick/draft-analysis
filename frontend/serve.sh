#!/usr/bin/env bash
# Serve the frontend: the board, the role board, the next move, the agent and
# its one-click suggestion.
#
# Two images can serve it, and the difference is one model:
#
#   dev  the role board, the head, and the remote-API agent. No GPU, no
#        `transformers`. This is the default, and it is the whole panel.
#   llm  the same, plus `llm-lora-rerank5` in the next-move slot - the 4.2 B
#        Qwen adapter. Needs a GPU and ~2 s per board.
#
# `--rerank` switches to the `llm` image and `--device cuda`. It is opt-in on
# purpose: the model's 34.27 % headline is measured with the *true lane* in the
# task line, and the panel's question has no lane, where it scores 20.85 %
# (frontend/README.md, "What the reranker is worth"). The role board does name
# a lane, which is the surface it is in-distribution for.
#
# Two things have to be true in either image:
#
#   * the port must be published  -> the base image publishes nothing, so a
#     server started with a bare `docker compose run` is unreachable from the
#     browser even though it is running fine
#   * `langgraph` must be installed for `/api/agent` -> it is in
#     requirements.txt, so a rebuilt `dev` image has it. Without it the server
#     starts anyway and `/api/agent` answers 503 with the reason.
#
# `LLM_API_KEY` (and `LLM_BASE_URL` / `LLM_MODEL` if the provider is not the
# default) comes from `.env`, which compose reads when the container is created
# - so re-run this after adding a key. Without a key the agent panel still
# opens and says there is none; the board and the role board are unaffected.
#
#   bash frontend/serve.sh                       # http://127.0.0.1:8777
#   bash frontend/serve.sh --rerank              # the 4B, needs a GPU
#   bash frontend/serve.sh --port 8780           # or PORT=8780
#   bash frontend/serve.sh --ckpt pro/model/archive/checkpoints/masking-proonly-randomvalsplit-through-wlds-ig-tes/best.pt
#   bash frontend/serve.sh --policy-ckpt pro/model/archive/checkpoints/nextpick-proto-ensemble/ensemble.json
#   bash frontend/serve.sh --no-policy           # role board only
#
# `pro/model/checkpoints/` holds exactly the shipped pair:
# `masking-proonly-nov13-through-lcp-df-sw` (role board) and
# `nextpick-proonly-nov13-k5-through-lcp-df-sw` (next move). Everything else is
# under `pro/model/archive/checkpoints/` and is reachable only by naming it, as
# the two examples above do. The reranker is served only when `--rerank` asks.
# There is no model picker in the browser - models are chosen at startup.
#
# Every argument is passed through to frontend/server.py; --port and --rerank
# are consumed here as well, because the publish spec and the image have to
# agree with them.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8777}"
SERVICE="dev"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --port)     PORT="$2"; shift 2 ;;
    --port=*)   PORT="${1#*=}"; shift ;;
    # The reranker needs the image that has transformers/peft/bitsandbytes and
    # a GPU reservation. Forwarding `--rerank` to a `dev` container would only
    # produce the head fallback the flag exists to avoid.
    --rerank)   SERVICE="llm"; ARGS+=(--rerank --device cuda); shift ;;
    --rerank=*) SERVICE="llm"; ARGS+=(--rerank "${1#*=}" --device cuda); shift ;;
    *)          ARGS+=("$1"); shift ;;
  esac
done

# The publish spec is bound to host loopback: `--host 0.0.0.0` inside the
# container is only reachable through this mapping, and never off the machine.
#
# If you invoke the container by hand instead, use `bash -c`, never `bash -lc`:
# a login shell resets PATH, `python` becomes a different interpreter, and the
# frontend then fails to import what this image installed.
exec docker compose run --rm -T \
  -p "127.0.0.1:${PORT}:${PORT}" \
  "$SERVICE" python frontend/server.py --host 0.0.0.0 --port "$PORT" \
  "${ARGS[@]+"${ARGS[@]}"}"
