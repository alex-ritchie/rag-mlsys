#!/usr/bin/env bash
# One-command local stack: DB + embedder + reranker + gateway + frontend, then open the browser.
#
#   make up                       # defaults: GPU services if nvidia-smi works, LLM_MODEL=fake
#   LLM_MODEL=fake make up
#   PROFILE=demo ANTHROPIC_API_KEY=sk-... make up      # real answers via Claude Haiku
#   LLM_BASE_URL=http://localhost:8003/v1 LLM_MODEL=qwen38-27b-w4a16 make up   # with vLLM already running
#   EMBEDDER_MODE=cpu RERANKER_MODE=cpu make up         # keep the GPU free
#   NO_BROWSER=1 make up
#
# Ctrl-C stops every service it started (the database is left running; `make db-down` stops it).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
LOGS=$ROOT/data/logs; mkdir -p "$LOGS"
export PATH="$HOME/.local/opt/node/bin:$PATH"
export HF_HOME="${HF_HOME:-$ROOT/data/hf-cache}"

# ---- defaults ---------------------------------------------------------------
if nvidia-smi >/dev/null 2>&1; then GPU_DEFAULT=gpu; else GPU_DEFAULT=cpu; fi
export EMBEDDER_MODE="${EMBEDDER_MODE:-$GPU_DEFAULT}"
export RERANKER_MODE="${RERANKER_MODE:-$GPU_DEFAULT}"
export PROFILE="${PROFILE:-local}"
export LLM_MODEL="${LLM_MODEL:-fake}"
export EMBEDDER_URL="${EMBEDDER_URL:-http://localhost:8001}"
export RERANKER_URL="${RERANKER_URL:-http://localhost:8002}"
if [[ "$PROFILE" == "demo" ]]; then
  # demo profile defaults to in-process ONNX; use the services unless the ONNX exports exist
  [[ -f data/onnx/bge-m3-int8/model.onnx ]] || export DEMO_EMBEDDER="${DEMO_EMBEDDER:-service}"
  [[ -f data/onnx/bge-reranker-v2-m3-int8/model.onnx ]] || export DEMO_RERANK="${DEMO_RERANK:-off}"
fi

# ---- database ---------------------------------------------------------------
if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "== database (pgserver under data/pg)"
  DATABASE_URL=$(uv run python scripts/local_db.py url | tail -1)
  export DATABASE_URL
fi
uv run python -m mlsys_common.db migrate >/dev/null
CHUNKS=$(uv run python - <<'PY'
import asyncio
from sqlalchemy import text
from mlsys_common.db import make_engine
async def go():
    e = make_engine()
    async with e.connect() as c:
        n, emb = (await c.execute(text("SELECT count(*), count(embedding) FROM chunks"))).first()
    await e.dispose(); print(f"{n} {emb}")
asyncio.run(go())
PY
)
read -r N_CHUNKS N_EMB <<<"$CHUNKS"
if [[ "$N_CHUNKS" == "0" ]]; then
  echo "!! no chunks in the database — run: make ingest && make index"; exit 1
elif [[ "$N_EMB" == "0" ]]; then
  echo "!! chunks are not embedded — run: make index"; exit 1
fi
echo "   $N_CHUNKS chunks, $N_EMB embedded"

# ---- services ---------------------------------------------------------------
PIDS=()
start() { # name port cmd...
  local name=$1 port=$2; shift 2
  echo "== $name (:$port, log: data/logs/$name.log)"
  "$@" >"$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}
wait_for() { # name url timeout_s
  local name=$1 url=$2 t=${3:-120}
  for ((i = 0; i < t; i++)); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 1
    if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then echo "!! $name exited — see data/logs/$name.log"; tail -20 "$LOGS/$name.log"; return 1; fi
  done
  echo "!! $name did not become healthy in ${t}s — see data/logs/$name.log"; return 1
}
cleanup() {
  echo; echo "== stopping"
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  pkill -P $$ 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start embedder 8001 uv run uvicorn mlsys_embedder.app:app --port 8001
wait_for embedder http://localhost:8001/health 300
start reranker 8002 uv run uvicorn mlsys_reranker.app:app --port 8002
wait_for reranker http://localhost:8002/health 300
start gateway 8000 uv run uvicorn mlsys_gateway.app:app --port 8000
wait_for gateway http://localhost:8000/api/health 60
( cd frontend && [[ -d node_modules ]] || pnpm install --silent )
start frontend 5173 bash -c 'cd frontend && exec pnpm dev --port 5173 --strictPort'
wait_for frontend http://localhost:5173/ 60

echo
echo "== ready"
echo "   app:      http://localhost:5173"
echo "   gateway:  http://localhost:8000/api/health   (profile=$PROFILE, model=$LLM_MODEL)"
echo "   services: embedder=$EMBEDDER_MODE reranker=$RERANKER_MODE"
[[ "$LLM_MODEL" == "fake" && "$PROFILE" == "local" ]] && echo "   note: LLM_MODEL=fake — real retrieval + citations, canned answer text (see README)"
if [[ -z "${NO_BROWSER:-}" ]]; then
  (xdg-open http://localhost:5173 >/dev/null 2>&1 || open http://localhost:5173 >/dev/null 2>&1 || true) &
fi
echo "   Ctrl-C to stop."
wait
