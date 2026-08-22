#!/usr/bin/env bash
# HPA demo (spec §5.8): drive the gateway 1 -> 4 -> 1 replicas with a load test; capture `kubectl get hpa -w`.
# Output: docs/benchmarks/hpa-demo-<timestamp>.log (paste/screenshot into the README) + the bench JSON.
set -euo pipefail
cd "$(dirname "$0")/.."
NS=mlsysbook
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=docs/benchmarks/hpa-demo-$TS.log
GW_URL=${GW_URL:-http://$(kubectl -n $NS get svc gateway -o jsonpath='{.spec.clusterIP}'):8000}
echo "gateway: $GW_URL" | tee "$LOG"
kubectl -n $NS get hpa gateway -w >> "$LOG" 2>&1 &
WATCH=$!
trap 'kill $WATCH 2>/dev/null || true' EXIT
echo "== load: 32 concurrent streams for 4 minutes" | tee -a "$LOG"
cat > /tmp/hpa-load.yaml <<YAML
name: hpa-demo
target: gateway
base_url: $GW_URL
concurrency: [32]
duration_s: 240
warmup: 1
YAML
uv run python -m mlsys_bench.run /tmp/hpa-load.yaml --tag hpa | tee -a "$LOG"
echo "== load finished; waiting for scale-down (stabilization 60 s + cooldown)" | tee -a "$LOG"
for _ in $(seq 1 30); do
  R=$(kubectl -n $NS get deploy gateway -o jsonpath='{.status.replicas}')
  echo "$(date -u +%H:%M:%S) replicas=$R" | tee -a "$LOG"
  [[ "$R" == "1" ]] && break
  sleep 20
done
echo "log: $LOG"
