#!/usr/bin/env bash
# One-command local stack: DB + embedder + reranker + gateway + frontend, then open the browser.
#
#   make up                       # defaults: GPU services, Qwen3.8-27B W4A16 via vLLM on :8003 (started if not running)
#   LLM_MODEL=fake make up        # no LLM: real retrieval + citations, canned answer text
#   PROFILE=demo ANTHROPIC_API_KEY=sk-... make up      # real answers via Claude Haiku (no local LLM)
#   VLLM_CONFIG=config/serving/vllm-qwen35-9b.yaml LLM_MODEL=qwen35-9b-w4a16 make up   # another slate model
#   EMBEDDER_MODE=cpu RERANKER_MODE=cpu make up         # keep the GPU free
#   NO_BROWSER=1 make up
#
# Ctrl-C stops every service it started (the database is left running; `make db-down` stops it).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
LOGS=$ROOT/data/logs; mkdir -p "$LOGS"
export PATH="$HOME/.local/opt/node/bin:$PATH"
# All models live in the repo cache. Ignore HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE from the shell on purpose:
# an inherited value (e.g. ~/.cache/huggingface) made vLLM fail with "Cannot find an appropriate cached snapshot".
# Override only with MLSYS_HF_HOME.
export HF_HOME="${MLSYS_HF_HOME:-$ROOT/data/hf-cache}"
export HF_HUB_CACHE="$HF_HOME/hub"; unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_OFFLINE
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# models are cached after the first run; skip the hub round-trip (which can stall on a slow connection)
[[ -d "$HF_HOME/hub/models--BAAI--bge-m3" && -d "$HF_HOME/hub/models--BAAI--bge-reranker-v2-m3" ]] && export HF_HUB_OFFLINE=1
echo "   HF cache: $HF_HOME (offline=${HF_HUB_OFFLINE:-0})"

# ---- defaults ---------------------------------------------------------------
export PROFILE="${PROFILE:-local}"
export LLM_MODEL="${LLM_MODEL:-qwen38-27b-w4a16}"
# Placement (M3 attempt 5, docs/benchmarks/m3-baseline.md): the reranker shares the GPU with vLLM (util 0.90, 16K
# context, reranker max_length 512 -> 23.66 GB peak, no OOM); the embedder runs on CPU (71 ms, under target).
if nvidia-smi >/dev/null 2>&1; then HAVE_GPU=1; else HAVE_GPU=0; fi
export RERANKER_MODE="${RERANKER_MODE:-$([[ $HAVE_GPU == 1 ]] && echo gpu || echo cpu)}"
export RERANKER_MAX_LENGTH="${RERANKER_MAX_LENGTH:-512}"
export EMBEDDER_MODE="${EMBEDDER_MODE:-$([[ $HAVE_GPU == 1 && ( "$LLM_MODEL" == fake || "$PROFILE" == demo ) ]] && echo gpu || echo cpu)}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8003/v1}"
VLLM_CONFIG="${VLLM_CONFIG:-config/serving/vllm-qwen38-27b.yaml}"
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
for port in 8000 8001 8002 5173; do
  if ss -ltn 2>/dev/null | grep -qE ":$port "; then
    echo "!! port $port is already in use:"; ss -ltnp 2>/dev/null | grep -E ":$port " | grep -oP 'users:\(\(.*' | cut -c1-100
    echo "   stop it (e.g. kill the pid above) or a previous 'make up' is still running"; exit 1
  fi
done
PIDS=()
T_START=$(date +%s)
start() { # name port cmd...
  local name=$1 port=$2; shift 2
  echo "== $name (:$port, log: data/logs/$name.log)"
  "$@" >"$LOGS/$name.log" 2>&1 &
  PIDS+=($!)
}
wait_for() { # name url pid timeout_s
  local name=$1 url=$2 pid=$3 t=${4:-120}
  for ((i = 0; i < t * 2; i++)); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 0.5
    if ! kill -0 "$pid" 2>/dev/null; then echo "!! $name exited — see data/logs/$name.log"; tail -20 "$LOGS/$name.log"; return 1; fi
  done
  echo "!! $name did not become healthy in ${t}s — see data/logs/$name.log"; return 1
}
since() { echo "$(( $(date +%s) - T_START ))s"; }
killtree() { # kill a process and all of its descendants (uv run -> python -> vllm/uvicorn workers)
  local pid=$1 child
  for child in $(pgrep -P "$pid" 2>/dev/null); do killtree "$child"; done
  kill "$pid" 2>/dev/null || true
}
cleanup() {
  echo; echo "== stopping"
  for p in "${PIDS[@]}"; do killtree "$p"; done
  pkill -P $$ 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# embedder + reranker load in parallel (~5 s each on this box once cached; first run downloads ~2 GB each)
start embedder 8001 uv run uvicorn mlsys_embedder.app:app --port 8001; EMB_PID=${PIDS[-1]}
start reranker 8002 uv run uvicorn mlsys_reranker.app:app --port 8002; RR_PID=${PIDS[-1]}
wait_for embedder http://localhost:8001/health "$EMB_PID" 300; echo "   embedder ready at $(since)"
wait_for reranker http://localhost:8002/health "$RR_PID" 300;  echo "   reranker ready at $(since)"
# ---- LLM ----------------------------------------------------------------------
if [[ "$PROFILE" == "local" && "$LLM_MODEL" != "fake" ]]; then
  if curl -sf "${LLM_BASE_URL%/v1}/health" >/dev/null 2>&1; then
    echo "== llm already serving at $LLM_BASE_URL"
  elif [[ "$LLM_BASE_URL" == http://localhost:8003/v1 ]]; then
    if [[ ! -x data/vllm-venv/bin/vllm ]]; then
      echo "!! vLLM is not installed (data/vllm-venv). See docs/RUN_IT_YOURSELF.md §5, or use LLM_MODEL=fake / PROFILE=demo."; exit 1
    fi
    LLM_REPO=$(python3 -c "import yaml;print(yaml.safe_load(open('$VLLM_CONFIG'))['model'])")
    if ! SNAP=$(HF_HUB_OFFLINE=1 data/vllm-venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$LLM_REPO', local_files_only=True))" 2>/dev/null); then
      echo "!! $LLM_REPO is not in $HF_HOME/hub — downloading (HF_HUB_DISABLE_XET=1) before starting vLLM"
      HF_HUB_OFFLINE=0 uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$LLM_REPO', max_workers=8)" || { echo "!! download failed"; exit 1; }
    else
      echo "   model: $LLM_REPO -> ${SNAP##*/snapshots/}"
    fi
    echo "== vllm ($VLLM_CONFIG, :8003, log: data/logs/vllm.log) — model load takes a few minutes"
    uv run python scripts/serve_vllm.py "$VLLM_CONFIG" >"$LOGS/vllm.log" 2>&1 &
    PIDS+=($!)
    wait_for vllm http://localhost:8003/health "${PIDS[-1]}" 1200; echo "   vllm ready at $(since)"
  else
    echo "!! no LLM at $LLM_BASE_URL"; exit 1
  fi
fi
start gateway 8000 uv run uvicorn mlsys_gateway.app:app --port 8000
wait_for gateway http://localhost:8000/api/health "${PIDS[-1]}" 60; echo "   gateway ready at $(since)"
( cd frontend && [[ -d node_modules ]] || pnpm install --silent )
start frontend 5173 bash -c 'cd frontend && exec pnpm dev --port 5173 --strictPort'
wait_for frontend http://localhost:5173/ "${PIDS[-1]}" 60; echo "   frontend ready at $(since)"

echo
echo "== ready"
echo "   app:      http://localhost:5173"
echo "   gateway:  http://localhost:8000/api/health   (profile=$PROFILE, model=$LLM_MODEL)"
echo "   services: embedder=$EMBEDDER_MODE reranker=$RERANKER_MODE"
[[ "$LLM_MODEL" == "fake" && "$PROFILE" == "local" ]] && echo "   note: LLM_MODEL=fake — real retrieval + citations, canned answer text (see README)"
[[ "$PROFILE" == "local" && "$LLM_MODEL" != "fake" ]] && echo "   llm:      $LLM_BASE_URL ($LLM_MODEL)"
if [[ -z "${NO_BROWSER:-}" ]]; then
  (xdg-open http://localhost:5173 >/dev/null 2>&1 || open http://localhost:5173 >/dev/null 2>&1 || true) &
fi
echo "   Ctrl-C to stop."
wait
