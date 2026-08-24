#!/usr/bin/env bash
# Lever sweeps (spec §5.7): restart the engine per row with flag overrides, run the matching bench, collect JSON.
#   ./scripts/sweep.sh all | vllm | llamacpp
# Each row: <bench-config> <serving-config> [--set k=v ...]. Results land in bench/results/ (tagged by row) and
# `make bench-report` renders docs/benchmarks/results.md.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/data/hf-cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGS=data/logs/sweeps; mkdir -p "$LOGS"
killtree() { local pid=$1 c; for c in $(pgrep -P "$pid" 2>/dev/null); do killtree "$c"; done; kill "$pid" 2>/dev/null || true; }
row() { # name bench serving [overrides...]
  local name=$1 bench=$2 serving=$3; shift 3
  ss -ltn | grep -qE ":8003 " && { echo "!! :8003 busy, skipping $name"; return; }
  ls bench/results/*-sweep-*-"$name".json >/dev/null 2>&1 && { echo "-- $name already has a result, skipping (delete it to re-run)"; return; }
  echo "== $name  ($serving $*)  $(date -u +%T)"
  uv run python scripts/serve_vllm.py "$serving" "$@" >"$LOGS/$name.server.log" 2>&1 & local pid=$!
  local t0=$(date +%s)
  until curl -sf localhost:8003/health >/dev/null; do sleep 5; if ! kill -0 $pid 2>/dev/null; then echo "!! server exited for $name"; grep -E "ValueError|Error" "$LOGS/$name.server.log" | tail -2 | cut -c1-200; return; fi; [[ $(( $(date +%s) - t0 )) -gt 1500 ]] && { echo "!! timeout for $name"; killtree $pid; return; }; done
  echo "   up in $(( $(date +%s) - t0 ))s, vram $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"
  uv run python -m mlsys_bench.run "$bench" --tag "$name" 2>&1 | grep -E "^-- |rps=|wrote" | sed 's/^/   /'
  killtree $pid; wait $pid 2>/dev/null; sleep 5
}
B=config/serving/vllm-qwen38-27b.yaml           # adopted 27B default (variant B) = the baseline every vLLM lever is measured against
L=config/serving/llamacpp-qwen38-27b.yaml
if [[ "${1:-all}" != "llamacpp" ]]; then
  row vllm-baseline      bench/configs/sweeps/vllm-prefix-caching-off.yaml "$B"     # baseline at the same bench shape (the lever file only differs in flags)
  row vllm-prefix-off    bench/configs/sweeps/vllm-prefix-caching-off.yaml "$B" --set enable-prefix-caching=false
  row vllm-eager         bench/configs/sweeps/vllm-eager.yaml              "$B" --set enforce-eager=true
  row vllm-seqs-8        bench/configs/sweeps/vllm-max-num-seqs-8.yaml     "$B" --set max-num-seqs=8
  row vllm-seqs-64       bench/configs/sweeps/vllm-max-num-seqs-64.yaml    "$B" --set max-num-seqs=24   # 24 is the Mamba-block ceiling at this budget; 64 cannot start
  # MTP drafter costs ~3.3 GiB: at util 0.90/16K only 0.52 GiB KV is left (< 1.71 needed). llm-only row => no reranker co-resident, so util 0.95 + 8K.
  row vllm-mtp-n2        bench/configs/sweeps/vllm-mtp-n2.yaml             "$B" --set 'speculative-config={"method":"mtp","num_speculative_tokens":2}' --set gpu-memory-utilization=0.95 --set max-model-len=8192
  row vllm-reasoning-on  bench/configs/sweeps/vllm-reasoning-on.yaml       "$B"
fi
if [[ "${1:-all}" != "vllm" ]]; then
  for n in 0 2 3 4; do row llamacpp-mtp-n$n bench/configs/sweeps/llamacpp-mtp-n$n.yaml "$L" --set spec-draft-n-max=$n $( [[ $n == 0 ]] && echo --set spec-type=none ); done
  row llamacpp-kv-q4_0   bench/configs/sweeps/llamacpp-kv-q4_0.yaml        "$L" --set cache-type-k=q4_0 --set cache-type-v=q4_0 --set flash-attn=on
fi
uv run python -m mlsys_bench.report
