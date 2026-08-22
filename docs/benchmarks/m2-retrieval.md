# M2 — indexing and retrieval (measured 2026-08-21, RTX 3090 Ti, index @ `2bd97c5`)

| Item | Value |
|---|---|
| Chunks | 2,815 (47 chapter files, both volumes); p50 680 bge-m3 tokens, 4 oversize atomic blocks, 48 under 400 |
| `make ingest` (parse + chunk + load, CPU) | ~40 s; re-run is a no-op (`+0 / -0 / 2815 unchanged`) |
| `make index` (bge-m3 fp16 on GPU, batch 32, + HNSW m=16 ef_construction=128) | 28.5 s total |
| Query embed, GPU service | ~9 ms per query (HTTP round trip included) |
| Query embed, CPU (bge-m3 fp32, sentence-transformers, default threads) | **p50 71 ms**, min 67, max 100 → meets the ≤ 100 ms target; the ONNX int8 export stays the demo/Fly path |
| Hybrid retrieval SQL (dense top-50 + FTS top-50 + RRF, one round trip, ef_search 100) | 3–14 ms |
| Rerank 30 candidates, bge-reranker-v2-m3 fp16 on GPU | ~260 ms (sequence length 1024; the cross-encoder dominates the pre-generation budget) |
| VRAM, embedder + reranker services both on GPU, idle after warm-up | 3.6 GB total (nvidia-smi) — input to the M3 placement decision |
| Gateway overhead (total − embed − retrieve − rerank − generate), fake LLM | ~5.5 ms per request (`rag_gateway_overhead_seconds`) |

Smoke test (`make retrieval-smoke`, 10 spot-check questions): every query's reranked top-5 came from the expected
chapters (e.g. *pruning* → Ch 10 structured/unstructured pruning sections; *data vs. model parallelism* → Vol 2 Ch 5
"Data Parallelism" and "Model Parallelism" as #1/#2; *MLPerf* → Ch 12 MLPerf Inference). Observations for M5/M8:
the reranker's sigmoid scores saturate near 1.0 on easy queries, so `rag_retrieval_score_p50` will sit high until
queries drift; FTS contributes mainly on acronym/keyword questions (MLPerf, KV cache) and is absent (`f=None`) on
paraphrased ones — the expected hybrid behaviour.

## Load-generator validation run (fake LLM, real index, 2026-08-21)

`mlsys_bench` against the gateway with `LLM_MODEL=fake` (generation cost ≈ 0), GPU embedder + GPU reranker services,
16 requests per level. With no model time, this isolates the **pre-generation pipeline ceiling**:

| conc | req/s | TTFT p50 / p99 (ms) | rerank p50 (ms) |
|---|---|---|---|
| 1 | ~3.9 | ~270 / ~330 | ~260 |
| 4 | 3.88 | 997 / 1199 | ~950 |
| 8 | 3.85 | 2011 / 2612 | 1914 |

**Finding:** throughput flat-lines at ~3.9 req/s from c1 onward and TTFT grows linearly with concurrency — the
reranker service (one uvicorn worker, one cross-encoder call per request at max_length 1024 over 30 × ~700-token
pairs) is a serial ~260 ms stage, so requests queue behind it. Embedding (9 ms) and the hybrid SQL (3 ms) are
negligible. This is now an explicit M8 lever row: cross-request batching in the reranker, `max_length` 512,
rerank top-20 instead of top-30, or int8 ONNX on CPU cores in parallel. In production the LLM's own decode time
(seconds) will hide part of this at low concurrency, but at c8+ the reranker — not the gateway — is the first
non-GPU-tier bottleneck, which is exactly the kind of measurement the HPA discussion needs (scaling the gateway
would not help here; scaling or batching the reranker would).
