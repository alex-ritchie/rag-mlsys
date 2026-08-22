# MLSysBook RAG — workflow targets. `make help` lists them.
SHELL := /bin/bash
.DEFAULT_GOAL := help
UV ?= uv
PY := $(UV) run
export PATH := $(HOME)/.local/opt/node/bin:$(PATH)

## ---- setup ---------------------------------------------------------------
setup: ## Install Python workspace (CPU) + frontend deps + pre-commit hooks
	$(UV) sync --all-packages --dev
	cd frontend && pnpm install
	$(PY) pre-commit install

setup-models: ## Install the GPU/model extras (torch, sentence-transformers) — owner workstation only
	$(UV) sync --all-packages --dev --extra models --extra onnx

setup-vllm: ## Install vLLM (cu129 wheel, pinned torch) into its own venv at data/vllm-venv
	./scripts/setup_vllm.sh

## ---- quality --------------------------------------------------------------
lint: ## Ruff lint + format check, mypy, frontend lint
	$(PY) ruff check .
	$(PY) ruff format --check .
	$(PY) mypy
	cd frontend && pnpm lint && pnpm typecheck

fmt: ## Auto-format Python + frontend
	$(PY) ruff check --fix .
	$(PY) ruff format .
	cd frontend && pnpm format

test: ## Unit tests (no models, no book content)
	$(PY) pytest -m "not models and not integration" -q

test-integration: ## Unit + integration tests (spins up an unprivileged Postgres+pgvector via pgserver)
	$(PY) pytest -m "not models" -q

guard: ## Run the licensing/large-file guard over tracked files
	$(PY) python scripts/guard_content.py --tracked

ci: lint test guard ## Everything CI runs

## ---- data pipeline (local; needs DATABASE_URL) ------------------------------
db-up: ## Start a local unprivileged Postgres+pgvector (pgserver) under data/pg and print DATABASE_URL
	$(PY) python scripts/local_db.py up

db-down: ## Stop the local pgserver Postgres
	$(PY) python scripts/local_db.py down

db-migrate: ## Apply schema (idempotent)
	$(PY) python -m mlsys_common.db migrate

fetch: ## Shallow-fetch the book at the pinned commit into data/book (licensing check recorded in LICENSING.md)
	$(PY) python -m mlsys_ingest.cli fetch

ingest: db-migrate ## Fetch + parse + chunk + load chunks into Postgres (idempotent on the pinned SHA; INGEST_CONFIG=config/sweeps/... for chunk-size sweeps)
	$(PY) python -m mlsys_ingest.cli run $(if $(INGEST_CONFIG),--config $(INGEST_CONFIG),)

ingest-dry: ## Parse + chunk without a database; prints chunk statistics
	$(PY) python -m mlsys_ingest.cli dry-run

index: ## Embed all chunks with bge-m3 (GPU batch job) and build the HNSW index
	$(PY) python -m mlsys_embedder.index

retrieval-smoke: ## 10 spot-check queries through hybrid retrieval (+ rerank if RERANKER_URL set)
	$(PY) python -m mlsys_eval.smoke

## ---- services (dev) -----------------------------------------------------------
embedder: ## Run the embedder service (EMBEDDER_MODE=cpu|gpu|onnx)
	$(PY) uvicorn mlsys_embedder.app:app --host 0.0.0.0 --port 8001

reranker: ## Run the reranker service (RERANKER_MODE=cpu|gpu|onnx)
	$(PY) uvicorn mlsys_reranker.app:app --host 0.0.0.0 --port 8002

gateway: ## Run the gateway (PROFILE=local|demo)
	$(PY) uvicorn mlsys_gateway.app:app --host 0.0.0.0 --port 8000 --reload

up: ## One command: DB + embedder + reranker + gateway + frontend, opens http://localhost:5173 (LLM_MODEL=fake default)
	./scripts/up.sh

dev: ## Gateway + frontend dev server (expects embedder/reranker/vLLM already up)
	( $(PY) uvicorn mlsys_gateway.app:app --port 8000 & cd frontend && pnpm dev ); wait

vllm: ## Serve the primary model with vLLM using config/serving/vllm-qwen38-27b.yaml
	$(PY) python scripts/serve_vllm.py config/serving/vllm-qwen38-27b.yaml

## ---- evaluation & benchmarks -----------------------------------------------------
golden-generate: ## Generate candidate Q/A pairs with Claude Haiku into eval/golden/candidates.jsonl
	$(PY) python -m mlsys_eval.golden generate

golden-verify: ## Mandatory human verification CLI -> eval/golden/golden.jsonl (+ .verified stamp)
	$(PY) python -m mlsys_eval.golden verify

eval: ## Full eval (retrieval + LLM-judge + abstention) -> eval/results/<run-id>/
	$(PY) python -m mlsys_eval.run

eval-retrieval: ## Retrieval-only eval (dense / hybrid / hybrid+rerank)
	$(PY) python -m mlsys_eval.run --retrieval-only

judge-validate: ## Judge vs. 30 hand labels -> agreement + kappa
	$(PY) python -m mlsys_eval.judge_validate

bench: ## Latency/throughput benchmark against the gateway (BENCH_CONFIG=bench/configs/gateway-baseline.yaml)
	$(PY) python -m mlsys_bench.run $(or $(BENCH_CONFIG),bench/configs/gateway-baseline.yaml)

bench-sweeps: ## Run every sweep config in bench/configs/sweeps/
	for f in bench/configs/sweeps/*.yaml; do $(PY) python -m mlsys_bench.run $$f || exit 1; done

bench-report: ## Render bench/results/* into docs/benchmarks/*.md tables
	$(PY) python -m mlsys_bench.report

## ---- deployment -------------------------------------------------------------
compose-up: ## docker compose up (profiles via COMPOSE_PROFILES=core,llm,obs,frontend)
	docker compose -f docker/docker-compose.yaml up -d

compose-down: ## docker compose down
	docker compose -f docker/docker-compose.yaml down

images: ## Build all images tagged with the git SHA
	./docker/build.sh

k8s-apply: ## kubectl apply -f k8s/<numbered manifests> (plain manifests; k8s/_index-job.yaml is a one-shot run separately)
	kubectl apply $(foreach f,$(wildcard k8s/[0-9]*.yaml),-f $(f))

k8s-delete: ## Tear down the k8s stack
	kubectl delete $(foreach f,$(wildcard k8s/[0-9]*.yaml),-f $(f)) --ignore-not-found

hpa-demo: ## Drive the gateway HPA 1->4->1 with a load test and capture kubectl get hpa -w
	./scripts/hpa_demo.sh

demo-load-supabase: ## Push the local chunks table (with embeddings) to Supabase for the hosted demo
	$(PY) python scripts/load_supabase.py

demo-deploy: ## Deploy the demo backend (fly.io) + frontend (Cloudflare Pages)
	./scripts/demo_deploy.sh

## ---- misc ----------------------------------------------------------------------
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | awk -F'\t' '{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: up setup setup-models setup-vllm lint fmt test test-integration guard ci db-up db-down db-migrate fetch ingest ingest-dry index retrieval-smoke embedder reranker gateway dev vllm golden-generate golden-verify eval eval-retrieval judge-validate bench bench-sweeps bench-report compose-up compose-down images k8s-apply k8s-delete hpa-demo demo-load-supabase demo-deploy help
