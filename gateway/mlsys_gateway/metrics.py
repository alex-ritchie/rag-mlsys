"""Prometheus metrics (spec §5.5, §5.9)."""

from __future__ import annotations

import time

from prometheus_client import Counter, Gauge, Histogram

STAGE_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)

REQUESTS = Counter("rag_requests_total", "RAG requests", ["profile", "route", "status"])
STAGE_SECONDS = Histogram(
    "rag_stage_seconds", "Per-stage latency", ["stage"], buckets=STAGE_BUCKETS
)
TTFT_SECONDS = Histogram(
    "rag_ttft_seconds", "Time to first token (from request start)", buckets=STAGE_BUCKETS
)
E2E_SECONDS = Histogram("rag_e2e_seconds", "End-to-end request latency", buckets=STAGE_BUCKETS)
GATEWAY_OVERHEAD_SECONDS = Histogram(
    "rag_gateway_overhead_seconds",
    "Total minus model + retrieval service time",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5),
)
TOKENS = Counter("rag_tokens_total", "Tokens", ["kind"])
ABSTENTIONS = Counter("rag_abstentions_total", "Answers that abstained")
TRUNCATED = Counter(
    "rag_truncated_answers_total", "Answers cut off by MAX_OUTPUT_TOKENS (finish_reason=length)"
)
ERRORS = Counter("rag_errors_total", "Errors by stage", ["stage"])
INFLIGHT = Gauge("rag_inflight_requests", "In-flight /api/ask requests")
RERANK_TOP1 = Histogram(
    "rag_rerank_top1_score",
    "Top-1 rerank score per query",
    buckets=(-10, -5, -2, -1, 0, 1, 2, 3, 5, 8, 12),
)
DEMO_LIMITED = Counter("rag_demo_limited_total", "Demo requests refused", ["reason"])
DEMO_SPEND = Gauge("rag_demo_spend_usd_24h", "Demo spend in the trailing 24h")

# quality gauges populated from Postgres on scrape (no pushgateway, spec §5.8)
QUALITY = {
    name: Gauge(name, desc)
    for name, desc in {
        "rag_queries_24h": "Queries in the trailing 24h",
        "rag_abstention_rate_24h": "Abstention rate in the trailing 24h",
        "rag_retrieval_score_p50": "p50 of per-query mean rerank score (24h) - retrieval drift proxy",
        "rag_retrieval_top1_score_p50": "p50 of per-query top-1 rerank score (24h)",
        "rag_nightly_groundedness": "Mean groundedness from the nightly judge (7d)",
        "rag_nightly_faithfulness": "Mean faithfulness from the nightly judge (7d)",
        "rag_judge_flagged_7d": "Judge-flagged answers (7d)",
        "rag_judged_answers_7d": "Answers judged (7d)",
        "rag_last_judge_run_timestamp": "Unix time of the last judge run",
    }.items()
}

_last_refresh = 0.0
REFRESH_INTERVAL_S = 30.0


async def refresh_quality_gauges(engine) -> None:
    global _last_refresh
    if time.monotonic() - _last_refresh < REFRESH_INTERVAL_S:
        return
    from mlsys_gateway.query_log import quality_gauges

    try:
        vals = await quality_gauges(engine)
    except Exception:
        ERRORS.labels(stage="metrics").inc()
        return
    for k, v in vals.items():
        QUALITY[k].set(v)
    _last_refresh = time.monotonic()
