#!/usr/bin/env bash
# One serving-ablation cell (spec §5.7): start vLLM from a serving config, record VRAM/KV, smoke, then
#   (a) engine-direct load at concurrency {1,4,8,16,32}  (b) end-to-end through the gateway with the
# embedder and reranker placed per the cell's config. Writes docs/benchmarks/cell-<tag>.json.
#
#   ./scripts/ablation_cell.sh config/serving/vllm-qwen35-9b.yaml qwen35-9b-w4a16 [EMBEDDER_MODE=gpu RERANKER_MODE=gpu RERANKER_MAX_LENGTH=1024]
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=$1; TAG=$2
export EMBEDDER_MODE=${EMBEDDER_MODE:-gpu} RERANKER_MODE=${RERANKER_MODE:-gpu} RERANKER_MAX_LENGTH=${RERANKER_MAX_LENGTH:-1024}
export HF_HOME=${HF_HOME:-$PWD/data/hf-cache} HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATABASE_URL=${DATABASE_URL:-postgresql://postgres:@/postgres?host=$PWD/data/pg}
export EMBEDDER_URL=http://localhost:8001 RERANKER_URL=http://localhost:8002 LLM_BASE_URL=http://localhost:8003/v1
MODEL=$(python3 -c "import yaml;print(yaml.safe_load(open('$CFG'))['served_model_name'])"); export LLM_MODEL=$MODEL
LOGS=data/logs; SP=data/logs/cell-$TAG; mkdir -p "$SP" docs/benchmarks
killtree() { local pid=$1 c; for c in $(pgrep -P "$pid" 2>/dev/null); do killtree "$c"; done; kill "$pid" 2>/dev/null || true; }
PIDS=(); cleanup() { for p in "${PIDS[@]}"; do killtree "$p"; done; wait 2>/dev/null || true; }; trap cleanup EXIT
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
for port in 8000 8001 8002 8003; do ss -ltn | grep -qE ":$port " && { echo "!! port $port busy"; exit 1; }; done

echo "== [$TAG] vLLM: $CFG"; T0=$(date +%s)
uv run python scripts/serve_vllm.py "$CFG" >"$SP/vllm.log" 2>&1 & PIDS+=($!); VP=$!
until curl -sf localhost:8003/health >/dev/null; do sleep 5; kill -0 $VP 2>/dev/null || { echo "!! vllm exited"; grep -E "ValueError|Error" "$SP/vllm.log" | tail -3; exit 1; }; done
LOAD_S=$(( $(date +%s) - T0 )); V_LLM=$(vram)
WEIGHTS=$(grep -oE "Model loading took [0-9.]+ GiB" "$SP/vllm.log" | head -1); KV=$(grep -oE "Available KV cache memory: [0-9.]+ GiB" "$SP/vllm.log" | head -1); KVTOK=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$SP/vllm.log" | head -1); CONC=$(grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$SP/vllm.log" | head -1)
echo "   up ${LOAD_S}s | $WEIGHTS | $KV | $KVTOK | $CONC | vram $V_LLM MiB"

echo "== smoke"; curl -s localhost:8003/v1/chat/completions -H 'content-type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what is a KV cache?\"}],\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("   ", d["choices"][0]["message"]["content"][:160].replace("\n"," "), "|", d["usage"]["completion_tokens"], "tok")'

echo "== engine-direct load (c 1/4/8/16/32)"
cat > "$SP/bench-direct.yaml" <<YAML
name: cell-$TAG-direct
target: openai
base_url: http://localhost:8003/v1
model: $MODEL
serving_config: $CFG
max_tokens: 512
extra_body:
  chat_template_kwargs: {enable_thinking: false}
concurrency: [1, 4, 8, 16, 32]
requests_per_level: 48
warmup: 2
YAML
uv run python -m mlsys_bench.run "$SP/bench-direct.yaml" | tee "$SP/bench-direct.log" | grep -E "^\s+rps|^-- "
DIRECT=$(ls -t bench/results/*cell-$TAG-direct*.json | head -1)

echo "== services: embedder=$EMBEDDER_MODE reranker=$RERANKER_MODE($RERANKER_MAX_LENGTH)"
uv run uvicorn mlsys_embedder.app:app --port 8001 >"$SP/embedder.log" 2>&1 & PIDS+=($!)
uv run uvicorn mlsys_reranker.app:app --port 8002 >"$SP/reranker.log" 2>&1 & PIDS+=($!)
until curl -sf localhost:8001/health >/dev/null && curl -sf localhost:8002/health >/dev/null; do sleep 2; done
uv run uvicorn mlsys_gateway.app:app --port 8000 >"$SP/gateway.log" 2>&1 & PIDS+=($!)
until curl -sf localhost:8000/api/health >/dev/null; do sleep 1; done
V_ALL=$(vram); echo "   vram with services: $V_ALL MiB"

echo "== gateway end-to-end load (c 1/4/8/16)"
cat > "$SP/bench-gateway.yaml" <<YAML
name: cell-$TAG-gateway
target: gateway
base_url: http://localhost:8000
serving_config: $CFG
concurrency: [1, 4, 8, 16]
requests_per_level: 32
warmup: 2
YAML
PEAK=$V_ALL; ( while true; do v=$(vram); [[ $v -gt $(cat "$SP/peak" 2>/dev/null || echo 0) ]] && echo $v > "$SP/peak"; sleep 2; done ) & MON=$!; PIDS+=($MON)  # cleanup must kill the monitor too, or `wait` hangs forever
uv run python -m mlsys_bench.run "$SP/bench-gateway.yaml" | tee "$SP/bench-gateway.log" | grep -E "^\s+rps|^-- "
kill $MON 2>/dev/null || true; PEAK=$(cat "$SP/peak" 2>/dev/null || echo $V_ALL)
GATEWAY=$(ls -t bench/results/*cell-$TAG-gateway*.json | head -1)
ERR=$(grep -ciE "out of memory|CUDA error" "$SP/vllm.log" "$SP/reranker.log" "$SP/embedder.log" | awk -F: '{s+=$2} END {print s+0}')
G500=$(grep -c "500 Internal" "$SP/gateway.log" || true)
curl -sf localhost:8003/health >/dev/null && ALIVE=true || ALIVE=false

python3 - "$TAG" "$CFG" "$DIRECT" "$GATEWAY" <<PY
import json, sys
tag, cfg, direct, gateway = sys.argv[1:]
d = json.load(open(direct)); g = json.load(open(gateway))
doc = {
  "tag": tag, "serving_config": cfg, "model": "$MODEL", "load_seconds": $LOAD_S,
  "weights": "$WEIGHTS", "kv": "$KV", "kv_tokens": "$KVTOK", "max_concurrency_at_full_context": "$CONC",
  "vram_mib": {"vllm_only": $V_LLM, "with_embedder_reranker": $V_ALL, "peak_gateway_load": $PEAK},
  "placement": {"embedder": "$EMBEDDER_MODE", "reranker": "$RERANKER_MODE", "reranker_max_length": $RERANKER_MAX_LENGTH},
  "oom_or_cuda_errors": $ERR, "gateway_500s": int("$G500" or 0), "vllm_alive_after": "$ALIVE" == "true",
  "engine_direct": d["runs"], "gateway_e2e": g["runs"],
  "bench_files": [direct, gateway], "vllm_version": d.get("server", {}).get("version"),
}
json.dump(doc, open(f"docs/benchmarks/cell-{tag}.json", "w"), indent=2)
print("wrote docs/benchmarks/cell-%s.json  (errors=%s, 500s=%s, alive=%s, peak %s MiB)" % (tag, $ERR, "$G500", "$ALIVE", $PEAK))
PY
