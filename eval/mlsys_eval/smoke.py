"""`make retrieval-smoke`: 10 spot-check queries through hybrid retrieval (+rerank when RERANKER_URL is set)."""

from __future__ import annotations

import asyncio

from mlsys_common.models import LatencyBreakdown
from mlsys_common.settings import get_settings
from mlsys_gateway.app import build_deps
from mlsys_gateway.pipeline import retrieve_and_rerank

QUERIES = [
    "What is the difference between Software 1.0 and Software 2.0?",
    "How does quantization reduce model memory and what accuracy cost does it have?",
    "What is the roofline model and how does arithmetic intensity relate to it?",
    "Explain structured versus unstructured pruning.",
    "What is knowledge distillation?",
    "How does data parallelism differ from model parallelism in distributed training?",
    "What does the iron law of performance say?",
    "Why is the KV cache a bottleneck for LLM inference?",
    "What is MLPerf and what does it benchmark?",
    "What are the main causes of silent data corruption in ML fleets?",
]


async def main() -> None:
    s = get_settings()
    deps = build_deps(s)
    for q in QUERIES:
        lat = LatencyBreakdown()
        fused, reranked = await retrieve_and_rerank(
            deps, q, mode=s.retrieval_mode, top_n=s.retrieval_top_n, top_k=s.rerank_top_k, lat=lat
        )
        print(
            f"\n=== {q}\n    embed {lat.embed_ms:.0f} ms · retrieve {lat.retrieve_ms:.0f} ms · rerank {lat.rerank_ms:.0f} ms · fused {len(fused)}"
        )
        for c in reranked:
            rs = f"rr={c.rerank_score:+.2f} " if c.rerank_score is not None else ""
            print(
                f"    {rs}rrf={c.fusion_score:.4f} d={c.dense_rank} f={c.fts_rank}  {c.heading_path}"
            )
    await deps.engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
