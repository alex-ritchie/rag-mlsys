#!/usr/bin/env bash
# M3 bring-up measurement (spec §4 M3): start vLLM from a serving config, record VRAM after load,
# run the 10-minute concurrency-8 soak, record peak VRAM and whether anything OOMed, write the
# numbers to docs/benchmarks/m3-<tag>.json (markdown is filled from it).
#
#   ./scripts/m3_measure.sh [config/serving/vllm-qwen38-27b.yaml] [--set key=val ...]
#   SOAK=bench/configs/m3-soak.yaml  TAG=graphs  DURATION_S=600
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-config/serving/vllm-qwen38-27b.yaml}; shift || true
TAG=${TAG:-$(basename "$CFG" .yaml)}
SOAK=${SOAK:-bench/configs/m3-soak.yaml}
LOGS=data/logs; mkdir -p "$LOGS" docs/benchmarks
export HF_HOME=${HF_HOME:-$PWD/data/hf-cache}
OUT=docs/benchmarks/m3-$TAG.json

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
if ss -ltn | grep -qE ":8003 "; then echo "!! :8003 already in use"; exit 1; fi
echo "== starting vLLM ($CFG $*) — log $LOGS/vllm-m3.log"
T0=$(date +%s)
uv run python scripts/serve_vllm.py "$CFG" "$@" >"$LOGS/vllm-m3.log" 2>&1 &
VPID=$!
killtree() { # kill a process and all of its descendants (uv run -> python -> vllm/uvicorn workers)
  local pid=$1 child
  for child in $(pgrep -P "$pid" 2>/dev/null); do killtree "$child"; done
  kill "$pid" 2>/dev/null || true
}
trap 'killtree $VPID; wait $VPID 2>/dev/null || true' EXIT
until curl -sf localhost:8003/health >/dev/null; do
  sleep 5
  if ! kill -0 $VPID 2>/dev/null; then echo "!! vLLM exited — tail of log:"; tail -40 "$LOGS/vllm-m3.log"; exit 1; fi
done
LOAD_S=$(( $(date +%s) - T0 ))
VRAM_LOADED=$(vram)
echo "   up in ${LOAD_S}s, VRAM after load: ${VRAM_LOADED} MiB"
VERSION=$(curl -s localhost:8003/version)
MODE=$(grep -qE -- "enforce-eager(=true)?( |$)" <<<"$*" && echo eager || (grep -qE "^\s*enforce-eager: true" "$CFG" && echo eager || echo cuda-graphs))
KV=$(grep -oE "GPU KV cache size: [0-9,]+ tokens|# GPU blocks: [0-9]+|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$LOGS/vllm-m3.log" | tr '\n' ';' || true)

echo "== smoke (single request, thinking off)"
curl -s localhost:8003/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"'"$(python3 -c "import yaml;print(yaml.safe_load(open('$CFG'))['served_model_name'])")"'","messages":[{"role":"user","content":"In one sentence, what is a KV cache?"}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print("   ", d["choices"][0]["message"]["content"][:200].replace("\n"," ")); print("   usage", d["usage"])'

echo "== soak: $SOAK"
PEAK=$VRAM_LOADED
( while kill -0 $VPID 2>/dev/null; do v=$(vram); [[ $v -gt $PEAK ]] && PEAK=$v && echo $PEAK > "$LOGS/m3-peak-vram"; sleep 5; done ) &
MON=$!
echo "$VRAM_LOADED" > "$LOGS/m3-peak-vram"
uv run python -m mlsys_bench.run "$SOAK" --tag "m3-$TAG" | tee "$LOGS/m3-soak-$TAG.log"
kill $MON 2>/dev/null || true
PEAK=$(cat "$LOGS/m3-peak-vram")
OOM=$(grep -ciE "out of memory|CUDA error|OOM" "$LOGS/vllm-m3.log" || true)
RESULT=$(ls -t bench/results/*m3-$TAG*.json | head -1)

python3 - "$OUT" "$RESULT" <<PY
import json, sys
out, result = sys.argv[1], sys.argv[2]
r = json.load(open(result))
run = r["runs"][0]
doc = {
  "tag": "$TAG", "config": "$CFG", "overrides": "$*", "mode": "$MODE",
  "vllm_version": $VERSION, "load_seconds": $LOAD_S,
  "vram_after_load_mib": $VRAM_LOADED, "vram_peak_soak_mib": $PEAK, "kv_cache_log": "$KV",
  "soak": {"concurrency": run["concurrency"], "requests": run["requests"], "errors": run["errors"], "wall_s": run["wall_s"],
           "requests_per_s": run["requests_per_s"], "output_tokens_per_s": run["output_tokens_per_s"],
           "ttft_ms": run["ttft_ms"], "total_ms": run["total_ms"]},
  "oom_or_cuda_errors_in_log": int("$OOM" or 0), "bench_result": result,
}
json.dump(doc, open(out, "w"), indent=2)
print(json.dumps(doc, indent=2))
PY
echo "wrote $OUT"
