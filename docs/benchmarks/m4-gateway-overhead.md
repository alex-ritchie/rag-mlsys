# M4 — gateway overhead

Acceptance: p50 gateway overhead (excluding model time) < 150 ms.

**In-process measurement (2026-08-21, this repo's CI-style test):** `gateway/tests/test_api.py::test_gateway_overhead_under_150ms`
issues 15 `/api/ask` requests through the ASGI app with a real Postgres (pgserver, 6 synthetic chunks),
a hash-based embedder, a lexical reranker, and a zero-delay fake LLM, so the measured time is almost
entirely gateway work (request parsing, retrieval SQL round trip, prompt assembly, SSE framing, query-log
insert). p50 on the workstation: ~15–25 ms. The assertion threshold is 150 ms.

**Network-level measurement:** the `rag_gateway_overhead_seconds` histogram (total − embed − retrieve −
rerank − generate, per request) is exported on `/metrics` and graphed on the Serving dashboard; the M8
gateway run records its p50/p99 in `bench/results/*gateway-baseline*.json` (`stage_ms_p50`).
