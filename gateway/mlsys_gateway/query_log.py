"""Per-query log rows (spec §5.5) and the quality gauges read by /metrics (spec §5.8)."""

from __future__ import annotations

from dataclasses import dataclass, field

from mlsys_common.hashing import question_hash
from mlsys_common.models import LatencyBreakdown, RetrievedChunk, Usage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass
class QueryLogRow:
    question: str
    profile: str
    model: str
    prompt_version: str
    retrieval_mode: str
    latency: LatencyBreakdown
    fused: list[RetrievedChunk] = field(default_factory=list)
    reranked: list[RetrievedChunk] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    answer: str = ""
    abstained: bool = False
    error: str | None = None


INSERT = text(
    """
    INSERT INTO query_log (question_hash, question, finished_at, profile, model, prompt_version, retrieval_mode,
        embed_ms, retrieve_ms, rerank_ms, ttft_ms, generate_ms, total_ms,
        fused_ids, fused_scores, reranked_ids, rerank_scores, prompt_tokens, completion_tokens, answer, abstained, error)
    VALUES (:qh, :q, now(), :profile, :model, :pv, :mode,
        :embed, :retrieve, :rerank, :ttft, :generate, :total,
        :fused_ids, :fused_scores, :reranked_ids, :rerank_scores, :pt, :ct, :answer, :abstained, :error)
    RETURNING id
    """
)


async def insert_query_log(engine: AsyncEngine, row: QueryLogRow) -> int:
    params = {
        "qh": question_hash(row.question),
        "q": row.question,
        "profile": row.profile,
        "model": row.model,
        "pv": row.prompt_version,
        "mode": row.retrieval_mode,
        "embed": row.latency.embed_ms,
        "retrieve": row.latency.retrieve_ms,
        "rerank": row.latency.rerank_ms,
        "ttft": row.latency.ttft_ms,
        "generate": row.latency.generate_ms,
        "total": row.latency.total_ms,
        "fused_ids": [c.chunk_id for c in row.fused],
        "fused_scores": [float(c.fusion_score) for c in row.fused],
        "reranked_ids": [c.chunk_id for c in row.reranked],
        "rerank_scores": [float(c.rerank_score or 0.0) for c in row.reranked],
        "pt": row.usage.prompt_tokens,
        "ct": row.usage.completion_tokens,
        "answer": row.answer,
        "abstained": row.abstained,
        "error": row.error,
    }
    async with engine.begin() as conn:
        return int((await conn.execute(INSERT, params)).scalar_one())


async def quality_gauges(engine: AsyncEngine) -> dict[str, float]:
    """Cheap online quality signals (spec §5.9 layer 3)."""
    async with engine.connect() as conn:
        q = (
            await conn.execute(
                text(
                    """
                    SELECT count(*) AS n,
                           COALESCE(avg(CASE WHEN abstained THEN 1 ELSE 0 END), 0) AS abstain_rate,
                           COALESCE(percentile_cont(0.5) WITHIN GROUP (ORDER BY rerank_scores[1]), 0) AS top1_p50,
                           COALESCE(percentile_cont(0.5) WITHIN GROUP (ORDER BY (SELECT avg(s) FROM unnest(rerank_scores) s)), 0) AS mean_p50
                    FROM query_log WHERE started_at > now() - interval '24 hours' AND error IS NULL
                    """
                )
            )
        ).first()
        j = (
            await conn.execute(
                text(
                    """
                    SELECT COALESCE(avg(groundedness), 0), COALESCE(avg(faithfulness), 0), COALESCE(sum(CASE WHEN flagged THEN 1 ELSE 0 END), 0), count(*),
                           COALESCE(EXTRACT(EPOCH FROM max(judged_at)), 0)
                    FROM judge_scores WHERE judged_at > now() - interval '7 days'
                    """
                )
            )
        ).first()
    assert q is not None and j is not None
    return {
        "rag_queries_24h": float(q[0]),
        "rag_abstention_rate_24h": float(q[1]),
        "rag_retrieval_score_p50": float(q[3]),
        "rag_retrieval_top1_score_p50": float(q[2]),
        "rag_nightly_groundedness": float(j[0]),
        "rag_nightly_faithfulness": float(j[1]),
        "rag_judge_flagged_7d": float(j[2]),
        "rag_judged_answers_7d": float(j[3]),
        "rag_last_judge_run_timestamp": float(j[4]),
    }
