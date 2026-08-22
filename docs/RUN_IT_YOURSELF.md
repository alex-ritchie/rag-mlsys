# Run it yourself

Two paths: the **developer path** (no Docker needed, used for all the tests in this repo) and the
**compose path** (the full stack as containers). Both start from a clean clone and end with a cited,
streaming answer. GPU steps assume an NVIDIA card with ≥ 24 GB and a 12.x-capable driver.

> Status on the author's workstation (2026-08-21): developer path verified end to end through
> ingestion, indexing, retrieval and the gateway with fake/real services; compose path written but not
> yet executed on this host (see `docs/DEVIATIONS.md` #4). This table is updated as each step is re-run.

## 0. Prerequisites

- Python 3.12 + [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Node ≥ 20 + pnpm 9 (`npm i -g pnpm@9`)
- `git`, `make`
- GPU path: NVIDIA driver ≥ 550 (CUDA 12.x). Check with `nvidia-smi`.
- Compose path: Docker 24+ with the NVIDIA container toolkit.

```bash
git clone https://github.com/alexritchie/mlsysbook-rag && cd mlsysbook-rag
cp .env.example .env           # every variable is documented there
make setup                     # uv workspace (CPU deps) + frontend deps + pre-commit
make setup-models              # adds torch (cu128), sentence-transformers, onnxruntime  (~4 GB)
make help                      # every target
```

## 1. Database

No Docker? `make db-up` starts an unprivileged Postgres 16 + pgvector under `data/pg` and prints the
`DATABASE_URL` to export. With Docker: `COMPOSE_PROFILES=core docker compose -f docker/docker-compose.yaml up -d db`
and keep the default `DATABASE_URL` from `.env`.

```bash
make db-up                     # prints: export DATABASE_URL='postgresql://...'
export DATABASE_URL='...'
make db-migrate
```

## 2. Ingest (CPU, ~1 min)

Fetches `harvard-edge/cs249r_book` at the pinned commit (shallow, ~1.7 GB with images) into the
gitignored `data/book`, parses both volumes, chunks, loads.

```bash
make ingest
# parsed 47 chapters -> 2815 chunks (BAAI/bge-m3 tokens)
# ingest @ 2bd97c5099: +2815 added, -0 removed, 0 unchanged
make ingest                    # second run is a no-op: +0 added, -0 removed, 2815 unchanged
```

`make ingest-dry` prints chunk statistics without a database.

## 3. Index (GPU batch job, ~1 min on a 3090 Ti; CPU works but is slow)

```bash
export HF_HOME=$PWD/data/hf-cache
make index                     # EMBEDDER_MODE=cpu make index  for the CPU fallback
```

## 4. Retrieval services + smoke test

```bash
EMBEDDER_MODE=cpu make embedder &      # :8001  (gpu if VRAM allows — see the M3 decision)
RERANKER_MODE=cpu make reranker &      # :8002
make retrieval-smoke                   # 10 spot-check queries with scores per stage
```

## 5. LLM

**vLLM (primary):**
```bash
uv pip install "vllm>=0.17" --extra-index-url https://download.pytorch.org/whl/cu128   # or use the vllm/vllm-openai image
make vllm                              # serves config/serving/vllm-qwen38-27b.yaml on :8003
```
Model download is ~17 GB. If CUDA-graph capture OOMs, restart with
`uv run python scripts/serve_vllm.py config/serving/vllm-qwen38-27b.yaml --set enforce-eager=true`.

**No GPU?** `LLM_MODEL=fake make gateway` runs the whole pipeline with a deterministic stand-in
generator — useful for frontend work; or point `LLM_BASE_URL`/`LLM_MODEL` at any OpenAI-compatible server.

## 6. Gateway + frontend

```bash
make gateway &                         # :8000
curl -N localhost:8000/api/ask -H 'content-type: application/json' \
     -d '{"question":"What is the difference between Software 1.0 and Software 2.0?"}'
# event: citations  ...  event: token ...  event: done {"usage":...,"latency_breakdown":{...}}
cd frontend && pnpm dev                # http://localhost:5173 (proxies /api to :8000)
```

OpenAI-compatible shim:
```python
from openai import OpenAI

c = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
print(
    c.chat.completions.create(
        model="qwen38-27b-w4a16", messages=[{"role": "user", "content": "What is quantization?"}]
    )
    .choices[0]
    .message.content
)
```

## 7. Compose path (all of the above as containers)

```bash
cp .env.example .env
./docker/build.sh                                   # images tagged with the git SHA
COMPOSE_PROFILES=core,llm,obs,frontend docker compose -f docker/docker-compose.yaml up -d
docker compose -f docker/docker-compose.yaml exec gateway python -m mlsys_ingest.cli run
docker compose -f docker/docker-compose.yaml exec embedder python -m mlsys_embedder.index --mode cpu
open http://localhost:8080      # frontend   (gateway :8000, Grafana :3000, Prometheus :9090)
```
Profiles: `core` (db, embedder, reranker, gateway) · `llm` (vLLM) · `llm-gguf` (llama.cpp) · `obs` · `frontend`.

## 8. Evaluate and benchmark

```bash
make golden-generate            # needs ANTHROPIC_API_KEY; writes gitignored candidates
make golden-verify              # MANDATORY human pass -> eval/golden/golden.jsonl + stamp
make eval                       # -> eval/results/<run-id>/{report.json,summary.md}
make bench                      # -> bench/results/*.json ; make bench-report -> docs/benchmarks/results.md
```

## 9. Kubernetes (k3s, single node)

```bash
curl -sfL https://get.k3s.io | sh -                   # Traefik + metrics-server bundled
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
sudo mkdir -p /var/lib/mlsysbook/{hf-cache,book}     # hostPath caches used by the manifests
docker save mlsysbook-rag/gateway:latest | sudo k3s ctr images import -   # (repeat per image, or push to a registry)
cp k8s/01-secrets.example.yaml k8s/secrets.yaml && $EDITOR k8s/secrets.yaml && kubectl apply -f k8s/secrets.yaml
make k8s-apply                                       # kubectl apply -f k8s/<numbered manifests>
kubectl -n mlsysbook get pods -w
make hpa-demo                                        # drives gateway replicas 1 -> 4 -> 1, logs to docs/benchmarks/
```
Grafana: `http://grafana.mlsysbook.local` (add both hosts to `/etc/hosts` → the node IP). Dashboards are
provisioned from `k8s/grafana/dashboards/*.json`.

## Troubleshooting

- `RuntimeError: The NVIDIA driver on your system is too old` → you have a CUDA-13 torch wheel; `make setup-models` uses the cu128 index (DEVIATIONS #1).
- `relation "chunks" does not exist` → `make db-migrate`.
- `golden set has not been human-verified` → by design; run `make golden-verify`.
- Demo returns 429 → per-IP limit or daily budget hit; both are `.env` settings.
