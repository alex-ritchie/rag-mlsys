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
