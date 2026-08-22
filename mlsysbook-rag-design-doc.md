# MLSysBook RAG Learning Companion — Design Document

**Owner:** Alex Ritchie
**Last updated:** 2026-08-21

---

## 0. How to use this document

This document is the authoritative input for building the project. Work through the milestones in §4 **in order**; each has acceptance criteria that must pass before moving on. Component specifications in §5 are the source of truth for interfaces and behavior. The licensing requirements in §7 are **hard constraints** — they override convenience at every decision point.

When something in this doc conflicts with reality (a library API changed, a model repo moved, a flag was renamed), prefer reality, note the deviation in `docs/DEVIATIONS.md`, and keep the *intent* of the spec.

GPU-dependent steps run on the owner's workstation (§2). Ask before long downloads (>5 GB) or destructive operations.

---

## 1. Project summary

A Retrieval-Augmented Generation (RAG) learning companion for the two-volume *Machine Learning Systems* textbook by Vijay Janapa Reddi ([mlsysbook.ai](https://mlsysbook.ai/)). Users ask questions about the book and receive grounded answers with inline citations to specific chapters and sections.

The deliberate meta-angle: **an ML systems engineering project about an ML systems textbook.** The system demonstrates the same lifecycle the book teaches — ingestion, retrieval, serving, evaluation, monitoring, inference optimization — with every claim backed by measurements.

**Design principles (from the approved spec):**

1. **Cost-conscious** — develop and benchmark on owned hardware; the public demo uses only free tiers plus pennies-per-query API generation.
2. **Measured, not just built** — every performance and quality claim ships with numbers: eval metrics, p50/p99 latency tables, monitoring dashboards.
3. **Scoped to ship** — retrieval + serving + evaluation + monitoring, cleanly. Fine-tuning and multi-agent designs are out of scope for v1.
4. **Provenance-aware** — licensing handled explicitly and documented (§7).

**Headline differentiators for reviewers:**

- A four-model ablation slate where each pairwise comparison varies exactly one factor (size, architecture, engine) with model family held constant — experimental design, not a model zoo.
- Serving a two-week-old hybrid-attention 27B (Qwen3.8-27B) on consumer Ampere in vLLM, with the batching/latency story that architecture enables.
- A real evaluation harness with a hand-verified golden set, abstention measurement, and a validated LLM-as-judge — most RAG portfolios have none.
- Kubernetes on bare metal with an honest scaling story: HPA on the stateless tier, and a written analysis of *why* horizontal scaling is the wrong axis for a single-GPU LLM tier.

---

## 2. Target environment

| Item | Value |
|---|---|
| OS | Ubuntu 22.04 |
| CPU | AMD Ryzen 9 7900X3D (12c/24t) |
| GPU | NVIDIA RTX 3090 Ti, 24 GB VRAM, **Ampere (sm_86)** |
| RAM | 96 GB |
| NVIDIA driver | ≥ 550 (CUDA 12.x runtime) — verify with `nvidia-smi` before M3 |
| Python | 3.12, managed with `uv` |
| Node | ≥ 20, `pnpm` |

Ampere is a load-bearing constraint: no FP8/NVFP4 compute kernels. All vLLM-served models use **W4A16 (AWQ/GPTQ/AutoRound in compressed-tensors format)**, which runs through the Marlin kernel on sm_86. GGUF quants are llama.cpp-only in this project.

---

## 3. Architecture overview

Two serving paths share one codebase:

**Local stack (the project):**

```
                                  ┌─────────────────────────────────────┐
                                  │              k3s cluster            │
 ┌──────────┐   HTTPS/SSE   ┌─────┴────┐  ┌──────────┐  ┌────────────┐  │
 │ Frontend │ ───────────▶  │ Gateway  │─▶│ Embedder │  │  Reranker  │  │
 │ (React)  │               │ (FastAPI)│  │ (bge-m3) │  │ (bge-r-v2) │  │
 └──────────┘               │ HPA 1-5  │  └──────────┘  └────────────┘  │
                            └─────┬────┘        │             ▲         │
                                  │             ▼             │         │
                                  │      ┌────────────┐       │         │
                                  ├─────▶│  Postgres  │───────┘         │
                                  │      │ + pgvector │ hybrid top-30   │
                                  │      └────────────┘  → rerank top-5 │
                                  ▼                                     │
                            ┌───────────┐   ┌────────────┐              │
                            │   vLLM    │   │ Prometheus │              │
                            │ Qwen3.8-  │   │ + Grafana  │              │
                            │ 27B W4A16 │   │ + DCGM     │              │
                            └───────────┘   └────────────┘              │
                                  └─────────────────────────────────────┘
```

**Hosted demo (accessibility layer, not the system):** static frontend (Cloudflare Pages) → thin demo backend (same gateway code, demo profile) → Supabase Postgres+pgvector for retrieval → Claude Haiku for generation. Framed everywhere as "the local stack is the project; the hosted demo lets you try it without a GPU."

**Request lifecycle (local):** query → gateway → embed query (bge-m3) → hybrid retrieval (dense HNSW + Postgres FTS, RRF fusion, top-30) → rerank (bge-reranker-v2-m3, top-5) → prompt assembly with numbered context blocks → vLLM streaming generation → SSE to client: first a `citations` event (chunk metadata + scores), then `token` events, then a `done` event with usage stats. Every stage emits Prometheus metrics and a per-query log row.

---

## 4. Milestones and acceptance criteria

Work strictly in order. M3 is the designated risk milestone — it validates the newest moving part before anything is built on top of it.

### M0 — Repo scaffold and tooling
- Monorepo layout per §6; `uv`-managed Python 3.12 workspace; `pnpm` frontend workspace.
- Ruff (lint + format), pytest, mypy (permissive to start), pre-commit hooks.
- GitHub Actions CI: lint + unit tests + frontend typecheck/build. **CI never touches book content or GPU.**
- `.gitignore` and a CI guard step that fails if `data/`, `*.gguf`, `*.safetensors`, or any file > 20 MB is staged.
- **Done when:** CI is green on an empty-but-wired repo; `make help` lists all workflow targets.

### M1 — Ingestion pipeline
- Fetch `harvard-edge/cs249r_book` at a **pinned commit SHA** (config value); parse `.qmd` sources; structure-aware chunking per §5.1; load chunks into Postgres.
- **Done when:** `make ingest` produces a populated `chunks` table from a clean checkout; unit tests cover the chunker on fixture `.qmd` files (fixtures are synthetic, not book content); zero book text is committed.

### M2 — Indexing and retrieval
- bge-m3 embedding of all chunks (GPU batch job); HNSW index; FTS column + GIN index; hybrid RRF retrieval function; reranker service.
- **Done when:** `make index` completes; a retrieval smoke test returns sensible chunks for 10 spot-check queries; retrieval-only eval harness runs end to end (metrics may be poor — that's fine, they're the baseline).

### M3 — vLLM serving bring-up (risk milestone)
- Qwen3.8-27B W4A16 serving on the 3090 Ti per §5.4, text-only, OpenAI-compatible endpoint up.
- Measure the actual VRAM budget; decide reranker/embedder placement (GPU vs CPU) per the contingency table in §5.4.
- **Done when:** sustained generation at concurrency 8 for 10 minutes without OOM; TTFT and tokens/s recorded in `docs/benchmarks/m3-baseline.md`. If the model cannot be made to serve reliably within ~2 days of effort, execute the fallback in §9 and record the decision.

### M4 — Gateway
- FastAPI service per §5.5: `/api/ask` SSE pipeline, native JSON endpoints, OpenAI-compatible shim, structured per-query logging, Prometheus metrics.
- **Done when:** end-to-end question → cited streaming answer works via `curl`; the shim works with the official `openai` Python client pointed at the gateway; unit tests mock vLLM/DB; p50 gateway overhead (excluding model time) < 150 ms measured.

### M5 — Evaluation harness and golden set
- Golden set built per §5.6 (generated → **hand-verified by the owner**; the harness must block on a human-review step, not silently accept generated QA).
- Retrieval metrics, LLM-judge generation metrics, abstention scoring; judge validated against 30 hand-labeled examples with agreement reported.
- **Done when:** `make eval` produces `eval/results/<run-id>/report.json` + a markdown summary; baseline numbers for the primary model are recorded.

### M6 — docker-compose reproduction path
- Full local stack via `docker compose up` with profiles: `core` (db, embedder, reranker, gateway), `llm` (vLLM), `llm-gguf` (llama.cpp), `obs` (Prometheus, Grafana), `frontend`.
- **Done when:** a fresh-machine runbook in `docs/RUN_IT_YOURSELF.md` goes from clone → ingested → answering questions, tested by actually following it top to bottom.

### M7 — Kubernetes deployment and monitoring
- k3s single-node install; NVIDIA device plugin; plain manifests per §5.8; HPA on gateway; Prometheus + Grafana + DCGM exporter; nightly groundedness CronJob.
- **Done when:** `kubectl apply -k` is NOT used — `kubectl apply -f k8s/` brings up the full stack; HPA demo works (load test drives gateway 1→4 replicas and back, screenshotted); all three Grafana dashboards render with live data; dashboard JSON exported to repo.

### M8 — Benchmarks and ablations
- Latency/throughput benchmark harness; optimization-lever sweeps; four-model ablation matrix per §5.7 (quality × serving).
- **Done when:** `docs/benchmarks/` contains the full results (machine-readable JSON + markdown tables) for every cell of the ablation matrix, plus the lever sweeps, each with the exact commands to reproduce.

### M9 — Frontend
- React + TypeScript + Vite app per §5.10: streaming chat with citations, retrieval inspector, book coverage browser, eval dashboard, dark mode.
- **Done when:** all five features work against the local gateway; `pnpm build` output is static-hostable; Lighthouse performance ≥ 90 on the built app.

### M10 — Hosted demo
- Supabase project, index load script, demo backend profile, rate limiting + spend cap, Cloudflare Pages deploy.
- **Done when:** a stranger with a browser gets a cited answer in < 10 s; the daily budget hard-stop is tested by lowering it to a trivial value and hitting it.

### M11 — Writeups
- README (with dashboard screenshots and the HPA demo), `LICENSING.md`, and the blog-style benchmark report per §5.12.
- **Done when:** the reader-testing pass (fresh reviewer or fresh Claude session with only the repo) can answer: what is this, how do I run it, what was measured, what were the findings.

---

## 5. Component specifications

### 5.1 Ingestion pipeline (`ingest/`)

**Source:** `harvard-edge/cs249r_book`, cloned shallow at the pinned commit SHA from `config/ingest.yaml`. The SHA is part of index provenance: it is stored on every chunk row and displayed in the frontend's About page ("index built from commit `abc1234`").

**Parsing:** walk the Quarto `.qmd` sources for both volumes. Strip Quarto-specific directives (callout fences, `{{< >}}` shortcodes, div attributes) while preserving: headings, prose, code blocks, tables, figure captions (with figure IDs), and equations (kept as LaTeX text). Cross-references (`@sec-...`, `@fig-...`) resolve to human-readable labels where the target is in the parsed corpus, else pass through verbatim.

**Chunking (structure-aware):**
- Split at heading boundaries first; pack sibling content greedily into chunks of **400–800 tokens** (bge-m3 tokenizer count), never splitting a code block, table, or equation across chunks. Oversized atomic blocks become their own chunk with a `oversize=true` flag.
- Prepend the **heading path** to each chunk's embedded text: `"Vol 1 > Ch 8: Model Optimization > 8.3 Quantization\n\n<chunk text>"`. The stored `text` column keeps the raw text; the prepended form exists only at embedding time.
- Metadata per chunk: `id`, `volume`, `chapter_num`, `chapter_title`, `section_path` (array), `heading_path` (display string), `source_file`, `char_start`, `char_end`, `token_count`, `commit_sha`, `content_hash`.

**Idempotency:** re-running `make ingest` against the same SHA is a no-op (content-hash comparison); against a new SHA it rebuilds and reports a diff summary (chunks added/removed/changed).

**Hard licensing constraints (see §7):** fetched book content lives only under `data/` (gitignored). No fixture, test, snapshot, or eval artifact committed to the repo may contain book text. Golden-set questions are committed; the *reference answers* are committed only in paraphrased form written during hand-verification, never as extracted passages.

### 5.2 Retrieval stack

**Embeddings:** `BAAI/bge-m3`, dense vectors (1024-d, normalized). Served by a small internal FastAPI service (`embedder/`) with two modes: GPU (bulk indexing, batched) and CPU (query-time; also the demo path). Target query-embed latency on CPU: ≤ 100 ms (verify in M2; if exceeded, use the ONNX int8 export).

**Storage:** Postgres 16 + pgvector.

```sql
CREATE TABLE chunks (
  id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  volume        SMALLINT NOT NULL,
  chapter_num   SMALLINT NOT NULL,
  chapter_title TEXT NOT NULL,
  section_path  TEXT[] NOT NULL,
  heading_path  TEXT NOT NULL,
  source_file   TEXT NOT NULL,
  char_start    INT NOT NULL,
  char_end      INT NOT NULL,
  token_count   INT NOT NULL,
  commit_sha    TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  text          TEXT NOT NULL,
  embedding     vector(1024),
  fts           tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING gin (fts);
```

**Hybrid retrieval:** run dense top-50 (cosine) and FTS top-50 (`ts_rank_cd` over `websearch_to_tsquery`) in one SQL round trip; fuse with **Reciprocal Rank Fusion**, k = 60; take fused **top-30**. Both a `hybrid` and pure `dense` mode are config flags so the ablation table gets a dense-only row for free.

**Reranker:** `BAAI/bge-reranker-v2-m3` served by `reranker/` (internal FastAPI). Scores the fused top-30 against the query; returns **top-5** with scores. Placement (GPU vs CPU) is decided by the M3 VRAM measurement (§5.4). All of top-30-with-fusion-scores and top-5-with-rerank-scores are logged per query and returned to the frontend for the retrieval inspector.

### 5.3 Model slate (the ablation backbone)

| Role | Model | Quant / format | Engine | Varies (vs primary) |
|---|---|---|---|---|
| **Primary serving** | Qwen3.8-27B | W4A16 compressed-tensors (candidate repos, in order: `dbirks/Qwen3.8-27B-W4A16-AutoRound`; a base-model W4A16 AWQ such as the `philbert440` build referenced in community 3090 recipes — verify on HF at build time) | vLLM | — |
| Size ablation | Qwen3.5-9B | W4A16 (community AWQ/GPTQ; if none exists, quantize with `llm-compressor` and document the recipe) | vLLM | parameter count, same family |
| Architecture ablation | Qwen3.6-35B-A3B | W4A16/AWQ (community build with vLLM 3090 precedent) | vLLM | dense vs MoE, same family |
| Engine ablation | Qwen3.8-27B | `unsloth/Qwen3.8-27B-GGUF` **UD-Q4_K_M** (16.5 GB, MTP tensors included) | llama.cpp | serving engine, same model |

All Apache 2.0. Model repo IDs are config values, not hardcoded — they will drift.

### 5.4 LLM serving

**vLLM (primary path):**
- Version: **≥ 0.17** (the qwen3_5 hybrid-attention architecture requires it); `transformers ≥ 5.8`. Pin exact working versions in `uv.lock` / the serving image the moment M3 passes.
- Serve text-only: pass the language-model-only flag (name as of Qwen3.6 recipes: `--language-model-only`) to skip loading the vision tower.
- Baseline flags: `--max-model-len 32768 --enable-prefix-caching --reasoning-parser qwen3 --gpu-memory-utilization 0.90` (tune in M3).
- Known failure mode: CUDA graph capture can OOM on tight VRAM; `--enforce-eager` is the documented escape hatch. Try graphs first (they matter for decode throughput), fall back if needed, and record which mode every benchmark ran in.
- Reasoning mode: Qwen3.8 reasons by default and it dominates TTFT. The RAG system prompt disables thinking for the serving path; reasoning on/off is a measured lever in M8, not a default.
- MTP speculative decoding in vLLM (`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`) is an M8 lever, benchmarked at concurrency 1 and 8 (expectation to verify: helps single-stream, washes out under batching).

**VRAM budget (verify empirically in M3 — these are planning numbers):**

| Component | Est. VRAM |
|---|---|
| Qwen3.8-27B W4A16 weights | ~17–19.5 GB |
| KV cache @ 32K, hybrid attention | ~2 GB (grows with concurrency, but 48/64 layers hold constant-size recurrent state) |
| CUDA context + activations | ~1–1.5 GB |
| bge-reranker-v2-m3 (fp16) | ~1.3 GB |
| bge-m3 (fp16) | ~1.3 GB |

Contingency ladder if over budget, in order: (1) reranker to CPU; (2) embedder to CPU (query-time only — bulk indexing is a batch job that can borrow the GPU when vLLM is down); (3) `--gpu-memory-utilization` down to 0.85; (4) `--max-model-len` 16384. The chosen configuration is documented with the measured numbers that forced it — that paragraph goes in the writeup.

**llama.cpp (engine-ablation path):** current build (MTP support landed July 2026 — anything older rejects the qwen35 architecture). Server invocation for the comparison:

```
llama-server -m Qwen3.8-27B-UD-Q4_K_M.gguf \
  --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -c 32768 -ngl 99 --parallel 1
```

Benchmark protocol notes that are findings in themselves: measure MTP at `--parallel 1` (the speculative advantage decays by ~4 concurrent streams); sweep `--spec-draft-n-max` ∈ {0, 2, 3, 4}; sweep KV cache type f16 vs q4_0; reasoning on vs off (affects both TTFT and draft acceptance rate).

### 5.5 API gateway (`gateway/`)

FastAPI + Pydantic v2, fully async. Prescribed libraries (to avoid sync-in-async event-loop stalls): `asyncpg` via SQLAlchemy 2 async for Postgres, `httpx.AsyncClient` for vLLM/embedder/reranker calls, `sse-starlette` for streaming.

**Native API:**
- `POST /api/ask` — body `{question, top_k?, mode?}` → SSE stream: event `citations` (array of `{chunk_id, heading_path, rerank_score, fusion_score, text_preview}`), then `token` events, then `done` `{usage, latency_breakdown}`. `latency_breakdown` = per-stage ms: embed, retrieve, rerank, ttft, generate.
- `GET /api/chunks/{id}` — full chunk with metadata (retrieval inspector).
- `GET /api/coverage` — chapter tree with per-chapter chunk counts and query-hit counts (coverage browser).
- `GET /api/eval/summary` — latest eval report JSON (eval dashboard).
- `GET /api/health`, `GET /metrics` (Prometheus).

**OpenAI-compatible shim:** `POST /v1/chat/completions` (+ `GET /v1/models`). Accepts standard chat payloads, streams `chat.completion.chunk` deltas; the last user message is the question; retrieved citations are appended to the final assistant message as a markdown "Sources" block (the shim can't use custom SSE events, so citations degrade gracefully into content). Acceptance test: the official `openai` Python client works against it unmodified.

**Prompting:** system prompt instructs answer-only-from-context with numbered inline citations `[1]..[5]` mapping to the context blocks, and an explicit abstention instruction: if the context does not contain the answer, say the book does not appear to cover it — never answer from parametric knowledge. Prompt templates live in `gateway/prompts/` as versioned files; the prompt version is logged per query.

**Per-query log row** (Postgres `query_log`): question hash, timestamps, per-stage latencies, retrieved ids + scores, model id, prompt version, token usage, answer text. This table feeds the drift CronJob and the coverage browser. No user identifiers are stored.

**Statelessness:** the gateway holds nothing between requests — this is what makes it the HPA target.

### 5.6 Evaluation (`eval/`)

**Golden set:** 120–150 questions, stratified across chapters proportional to chunk counts. Three types:
- ~60% single-chunk factual (answer lives in one chunk),
- ~30% multi-chunk synthesis (requires ≥ 2 labeled chunks),
- 10–15 **unanswerable/out-of-scope** questions (plausible topic, not covered by the book) to measure abstention.

Generation procedure: sample chunks → generate candidate Q/A pairs with an API model (Claude Haiku) → **mandatory human-verification pass** (CLI tool presents each pair with its source chunk; owner accepts/edits/rejects; the harness refuses to run against an unverified set). Committed format: JSONL with `{id, question, answer_key_points (paraphrased), source_chunk_content_hashes, type, chapter}`. Chunk *hashes* rather than ids make the labels survive re-ingestion; a resolver maps hashes → current ids.

**Retrieval metrics:** recall@{5,10,30} and MRR against labeled chunks, reported overall and per question type; computed for dense-only, hybrid, and hybrid+rerank (the three-row ablation).

**Generation metrics (own thin LLM-as-judge, not RAGAS-as-black-box):** three judge prompts — faithfulness (is every claim supported by the provided chunks), answer relevance, groundedness/abstention correctness — each returning a structured verdict via the judge model (Claude Haiku). **Judge validation:** 30 examples hand-labeled by the owner; report judge–human agreement (percent + Cohen's kappa) in the eval report. If agreement < 80%, iterate on judge prompts before trusting the numbers.

**Abstention scoring:** on unanswerable questions, correct behavior = explicit "not covered" response; any substantive answer counts as hallucination. Report abstention precision/recall.

**Execution:** `make eval` (local, needs the serving stack up) → `eval/results/<run-id>/` with `report.json` + markdown summary. CI runs eval *unit tests* only (metric math on synthetic fixtures) — never book content, never GPU. A copy of the latest `report.json` is committed to feed the frontend eval dashboard (it contains only questions, scores, and paraphrased key points — no book text).

### 5.7 Benchmarks and ablation matrix (`bench/`)

**Harness:** async load generator (self-built on `httpx` + `asyncio`, or `vllm bench serve` where applicable) driving the *gateway* (end-to-end) and vLLM directly (engine-isolated). Report per run: TTFT p50/p99, total latency p50/p99, output tokens/s, requests/s, at concurrency ∈ {1, 4, 8, 16, 32}. Flame graphs of the gateway under load via `py-spy`. Every result JSON embeds the full config (model, flags, concurrency, git SHA).

**Ablation matrix (all cells get golden-set quality + the serving profile above):**

| | Quality (faithfulness / relevance / abstention) | Serving (TTFT, p99, tok/s vs concurrency) |
|---|---|---|
| Qwen3.8-27B W4A16 / vLLM | ✓ | ✓ |
| Qwen3.5-9B / vLLM | ✓ | ✓ |
| Qwen3.6-35B-A3B / vLLM | ✓ | ✓ |
| Qwen3.8-27B GGUF / llama.cpp (+MTP) | ✓ | ✓ (single-stream focus) |

**Lever sweeps (primary model unless noted):** prefix caching on/off (RAG shares a large prompt prefix — expected to matter and worth quantifying); CUDA graphs vs eager; max-num-seqs; vLLM MTP at concurrency 1 vs 8; llama.cpp MTP draft-depth sweep; llama.cpp KV q4_0 vs f16; reasoning on/off. Each sweep is one table row: config, delta vs baseline, and one sentence of interpretation.

**Framing questions the report must answer with numbers:** Was the 27B worth it over the 9B for grounded textbook QA? Does the MoE's throughput advantage survive reranked-RAG prompt shapes? Where does MTP stop paying? What did hybrid attention actually buy in KV memory and achievable concurrency on 24 GB?

### 5.8 Kubernetes deployment (`k8s/`)

**Distribution:** k3s, single node, on the workstation. Traefik (bundled) for ingress; metrics-server (bundled) backs the HPA. NVIDIA device plugin installed so pods can request `nvidia.com/gpu: 1`.

**Style: plain manifests only.** No Helm, no Kustomize in v1. One resource per file, readable top to bottom — the manifests are a portfolio artifact in their own right. `kubectl apply -f k8s/` (directory apply) must bring up the whole stack; files are numbered for apply order (`00-namespace.yaml`, `10-postgres-*.yaml`, ...).

**Workloads:**

| Resource | Kind | Replicas | Notes |
|---|---|---|---|
| `postgres` | StatefulSet + PVC (local-path) + Service | 1 | pgvector image; stateful, never scaled |
| `vllm` | Deployment + Service | 1 | requests `nvidia.com/gpu: 1`; generous startup probe (model load is minutes); `Recreate` strategy (two copies can't share the GPU) |
| `embedder` | Deployment + Service | 1 | CPU mode in-cluster |
| `reranker` | Deployment + Service | 1 | placement per M3 decision |
| `gateway` | Deployment + Service + **HPA** | 1–5 | CPU target 70%; resource requests set realistically from load-test data, or the HPA math is meaningless |
| `frontend` | Deployment + Service + Ingress | 1 | nginx serving the built static app |
| `prometheus` | Deployment + ConfigMap + PVC | 1 | minimal hand-written scrape config: gateway, vLLM `/metrics`, DCGM, kubelet/cAdvisor |
| `grafana` | Deployment + ConfigMaps | 1 | dashboards provisioned from JSON in-repo |
| `dcgm-exporter` | DaemonSet | — | GPU util/VRAM/power metrics |
| `drift-judge` | **CronJob** (nightly) | — | samples N answers from `query_log`, judges groundedness via the judge harness, writes scores to Postgres |

Quality metrics reach Prometheus without a pushgateway: the gateway's `/metrics` exposes gauges (`rag_nightly_groundedness`, `rag_retrieval_score_p50`, etc.) read from the tables the CronJob and query logging populate.

**The scaling story (writeup requirement, not just config):** HPA lives on the stateless CPU tier only. The GPU tier is a single replica by design; its interesting signals are vLLM queue depth, batch occupancy, and request-level SLOs. `docs/writeups/scaling.md` must explain why horizontal scaling is the wrong axis for a single-GPU LLM tier and what changes with N GPUs (replica-per-GPU vs tensor parallel, and when each wins). The HPA demo (load test drives gateway replicas 1→4→1, captured from `kubectl get hpa -w` and the Grafana panel) is a README centerpiece.

### 5.9 Monitoring (three layers, one dashboard JSON per layer, exported to `k8s/grafana/dashboards/`)

1. **Infra:** GPU utilization, VRAM, power (DCGM); pod CPU/memory; gateway replica count (the HPA panel).
2. **Serving:** request rate, TTFT and end-to-end latency p50/p99 (gateway histograms), vLLM queue depth / running & waiting sequences / KV-cache utilization (vLLM's own `/metrics`), tokens/s, error rate.
3. **Quality (the LLM-specific layer):** nightly groundedness gauge with trend, per-query mean rerank score p50 (a cheap online retrieval-quality proxy — a sustained drop means retrieval drift), abstention rate, judge-flagged answer count. Alert rules (Prometheus, even if they only ever fire in a demo): groundedness below threshold, p99 above SLO, GPU memory near ceiling.

### 5.10 Frontend (`frontend/`)

React + TypeScript + Vite + Tailwind. Dark mode default. Static-hostable build; API base URL from env (`VITE_API_BASE`) so the same build serves local and demo.

1. **Chat** — streaming answers; inline `[n]` citations rendered as chips; hovering a chip highlights the matching chunk in the inspector; per-message latency breakdown (embed/retrieve/rerank/TTFT/total) in a subtle footer — the "measured, not just built" principle made visible in the UI.
2. **Retrieval inspector** — side panel per answer: the reranked top-5 with scores, expandable to the fused top-30 with fusion scores; each chunk shows heading path + text. This is the anti-black-box feature; spend polish here.
3. **Book coverage browser** — chapter tree (volume → chapter → section) from `/api/coverage`; shows chunk counts and which chapters recent answers drew from.
4. **Eval dashboard** — renders the committed `report.json`: metric cards, per-question-type breakdown, the retrieval three-row ablation, model-comparison table, judge-agreement stat. Static data; no backend dependency.
5. **About / licensing** — attribution block (book, author, license, link), index commit SHA, and the demo-vs-local framing.

SSE via `fetch` + `ReadableStream` (not `EventSource` — need POST). Graceful degradation when the demo backend rate-limits: show the limit message and point to the run-it-yourself path.

### 5.11 Hosted demo

- **Retrieval:** Supabase free tier (Postgres + pgvector). `scripts/load_supabase.py` pushes the locally built `chunks` table (schema-compatible; HNSW index created remotely). This is *private hosting for the demo's own use*, not redistribution — one sentence in `LICENSING.md` covers it.
- **Demo backend:** the same gateway code with `PROFILE=demo`: query embedding via CPU ONNX bge-m3 in-process; retrieval against Supabase; rerank via CPU int8 ONNX bge-reranker if it fits the latency budget (< 1 s for the rerank stage), else hybrid-only (config flag, decided in M10); generation via **Claude Haiku** (`claude-haiku-4-5`) with the same prompt templates and citation contract. Hosted on a Fly.io free-allowance micro-VM (or equivalent; smallest thing that runs a 2 GB-RAM Python process).
- **Cost controls:** per-IP sliding-window rate limit (default 10 questions/day, stored in Supabase); **global daily spend cap** with a hard stop (returns a friendly "demo budget exhausted for today" + run-it-yourself link). Both limits are config. API key only ever server-side.
- **Frontend:** Cloudflare Pages, same build, `VITE_API_BASE` pointed at the demo backend.
- **Framing (enforced in copy):** the demo page footer and README both state that the hosted demo is an accessibility layer over a cheap API model, and the k8s/vLLM local stack is the actual project — with the demo video and `docker compose up` path linked.

### 5.12 Writeups (`docs/`)

- `README.md` — what/why, architecture diagram, dashboard + HPA screenshots, quickstart, links to everything.
- `docs/RUN_IT_YOURSELF.md` — the tested fresh-machine runbook.
- `docs/writeups/benchmark-report.md` — the blog-post-style report: methodology, the four-model matrix, lever sweeps, flame graphs, and the framing questions from §5.7 answered with numbers.
- `docs/writeups/scaling.md` — the HPA/GPU scaling analysis.
- `LICENSING.md` — per §7.
- `docs/DEVIATIONS.md` — running log of spec deviations and why.

---

## 6. Repository layout

```
mlsysbook-rag/
├── README.md, LICENSE (code: MIT), LICENSING.md
├── Makefile                  # ingest, index, eval, bench, dev, demo-deploy, help
├── pyproject.toml, uv.lock   # uv workspace: ingest, embedder, reranker, gateway, eval, bench
├── config/                   # ingest.yaml (pinned SHA), models.yaml, serving profiles
├── ingest/                   # fetch + parse + chunk + load
├── embedder/                 # bge-m3 service (gpu|cpu|onnx modes)
├── reranker/                 # bge-reranker-v2-m3 service
├── gateway/                  # FastAPI app, prompts/, OpenAI shim, demo profile
├── eval/                     # golden set (JSONL), verification CLI, judge harness, results/
├── bench/                    # load generator, sweep configs, results/
├── frontend/                 # React + TS + Vite app
├── k8s/                      # numbered plain manifests + grafana/dashboards/*.json
├── docker/                   # Dockerfiles (multi-stage) + docker-compose.yaml (profiles)
├── scripts/                  # load_supabase.py, dev helpers
├── docs/                     # runbook, writeups, benchmarks, DEVIATIONS.md
└── data/                     # GITIGNORED: book checkout, model files, local artifacts
```

---

## 7. Licensing and compliance (hard constraints)

The book is **CC BY-NC-SA 4.0** (per `LICENSE.md` in `harvard-edge/cs249r_book`). The project's own code is MIT. These requirements bind every milestone:

1. **Attribution in three places:** README, app UI (About page + footer), and per-answer source citations. Credit Vijay Janapa Reddi and *Machine Learning Systems*, link mlsysbook.ai, state the license.
2. **Never commit book content or derived indexes.** No book text in fixtures, tests, snapshots, eval reference answers (paraphrase only), or the repo history. Enforced by `.gitignore` + the M0 CI guard. Ship the *pipeline*; every user builds their own index.
3. **Non-commercial:** the deployed demo stays free and non-monetized. No paywalls, no paid tiers, ever, without revisiting the license.
4. **ShareAlike:** if any derived artifact is ever published, it is licensed CC BY-NC-SA 4.0. Current design publishes none (the Supabase index is private hosting for the demo's own use — documented, not distributed).
5. **`LICENSING.md`** documents this reasoning explicitly, including the demo-hosting analysis and the pre-ingestion check of mlsysbook.ai site terms for AI/data-mining clauses (do this check during M1 and record the result).

---

## 8. Tooling and conventions

- Python 3.12, `uv` workspace; Ruff for lint + format; pytest (+ `pytest-asyncio`); mypy on gateway and eval first. Type-annotated public interfaces; Pydantic models are the contract between services.
- TypeScript strict; ESLint + Prettier; pnpm.
- Docker multi-stage builds; images tagged with git SHA; CPU and GPU images separated (GPU images only for vllm/embedder/reranker).
- GitHub Actions: lint, unit tests, frontend build, the large-file/content guard. No GPU jobs, no book content, no eval-with-models in CI.
- Secrets (Anthropic key, Supabase creds) via env only; `.env.example` documents every variable; k8s Secrets for in-cluster.
- Conventional commits; PR-sized changes per milestone chunk.

---

## 9. Risks and fallbacks

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| vLLM qwen3_5 support unstable on Ampere (nightly churn, CUDA graph OOM, kernel gaps) | Medium | M3 is first GPU milestone; `--enforce-eager`; pin the first working version set. **Fallback:** promote Qwen3.6-27B AWQ (identical architecture, months of community 3090 recipes) to primary and move 3.8 to the llama.cpp column — the ablation design survives intact. |
| No usable W4A16 quant for Qwen3.5-9B or 3.6-35B-A3B | Low–Med | Quantize with `llm-compressor` (documented recipe = bonus artifact), or substitute the nearest same-family model and note it in DEVIATIONS.md. |
| VRAM contention (weights + reranker + embedder) | Medium | Contingency ladder in §5.4; measured decision recorded in the writeup. |
| Supabase free-tier limits (500 MB DB) | Low | ~2–3K chunks × 1024-d vectors fits comfortably; if not, halfvec or drop stored `fts` on the demo copy. |
| Demo abuse / cost overrun | Medium | Per-IP limit + global hard cap, tested in M10. |
| Scope creep | High | §10 is binding. New ideas go to a `FUTURE.md`, not the codebase. |

---

## 10. Out of scope for v1

Fine-tuning; multi-agent designs; conversation memory / multi-turn retrieval; user accounts; ingesting sources beyond the book; a jailbreak-eval harness (noted as a possible future AI-safety extension); Helm/Kustomize; multi-node k8s.

---

## 11. Glossary (for readers, not the implementer)

**RRF** — Reciprocal Rank Fusion, a rank-based method for merging result lists. **HPA** — Horizontal Pod Autoscaler. **TTFT** — time to first token. **MTP** — multi-token prediction; trained-in draft layers enabling self-speculative decoding. **W4A16** — 4-bit weights, 16-bit activations. **Hybrid attention** — Qwen3.5+ architecture mixing full-attention layers with linear-attention (constant-state) layers, shrinking KV cache ~4×.
