"""The /api/ask pipeline (spec §3 request lifecycle): embed -> retrieve -> rerank -> prompt -> stream -> log."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from mlsys_common.models import (
    AskRequest,
    Citation,
    DoneEvent,
    LatencyBreakdown,
    RetrievedChunk,
    Usage,
)
from mlsys_common.settings import Settings
from mlsys_embedder.client import Embedder
from mlsys_reranker.client import Reranker
from sqlalchemy.ext.asyncio import AsyncEngine

from mlsys_gateway import metrics as M
from mlsys_gateway.llm import LLM, now_ms
from mlsys_gateway.prompts import build_messages, is_abstention
from mlsys_gateway.query_log import QueryLogRow, insert_query_log
from mlsys_gateway.retrieval import RetrievalConfig, retrieve


@dataclass
class Deps:
    settings: Settings
    engine: AsyncEngine
    embedder: Embedder
    reranker: Reranker | None
    llm: LLM


@dataclass
class PipelineEvent:
    event: str  # citations | token | done | error
    data: Any


def preview(text: str, n: int = 240) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


def to_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            n=i + 1,
            chunk_id=c.chunk_id,
            heading_path=c.heading_path,
            rerank_score=c.rerank_score,
            fusion_score=c.fusion_score,
            text_preview=preview(c.text),
        )
        for i, c in enumerate(chunks)
    ]


async def retrieve_and_rerank(
    deps: Deps,
    question: str,
    *,
    mode: str,
    top_n: int,
    top_k: int,
    lat: LatencyBreakdown,
    rerank: bool = True,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Shared by /api/ask, the shim, and the eval harness. Returns (fused top-N, reranked top-K)."""
    t0 = now_ms()
    vec = await deps.embedder.embed_query(question)
    lat.embed_ms = now_ms() - t0
    M.STAGE_SECONDS.labels("embed").observe(lat.embed_ms / 1000)

    t1 = now_ms()
    fused = await retrieve(deps.engine, vec, question, RetrievalConfig(mode=mode, top_n=top_n))
    lat.retrieve_ms = now_ms() - t1
    M.STAGE_SECONDS.labels("retrieve").observe(lat.retrieve_ms / 1000)

    t2 = now_ms()
    rerank_error: str | None = None
    if rerank and deps.reranker is not None and fused:
        try:
            results = await deps.reranker.rerank(question, [c.text for c in fused], top_k)
        except (
            Exception
        ) as e:  # reranker overloaded/OOM: degrade to the fused order rather than fail the answer
            M.ERRORS.labels(stage="rerank").inc()
            results = None
            rerank_error = f"{type(e).__name__}: {str(e)[:120]}"
        if results is not None:
            reranked = []
            for r in results:
                c = fused[r.index].model_copy()
                c.rerank_score = r.score
                reranked.append(c)
        else:
            reranked = [c.model_copy() for c in fused[:top_k]]
    else:
        reranked = [c.model_copy() for c in fused[:top_k]]
    lat.rerank_ms = now_ms() - t2
    M.STAGE_SECONDS.labels("rerank").observe(lat.rerank_ms / 1000)
    if reranked and reranked[0].rerank_score is not None:
        M.RERANK_TOP1.observe(reranked[0].rerank_score)
    if rerank_error:
        lat.rerank_error = rerank_error
    return fused, reranked


async def ask(deps: Deps, req: AskRequest) -> AsyncIterator[PipelineEvent]:
    s = deps.settings
    mode = req.mode or s.retrieval_mode
    top_k = req.top_k or s.rerank_top_k
    lat = LatencyBreakdown()
    t_start = now_ms()
    fused: list[RetrievedChunk] = []
    reranked: list[RetrievedChunk] = []
    answer_parts: list[str] = []
    usage = Usage()
    finish_reason: str | None = None
    error: str | None = None
    M.INFLIGHT.inc()
    try:
        fused, reranked = await retrieve_and_rerank(
            deps, req.question, mode=mode, top_n=s.retrieval_top_n, top_k=top_k, lat=lat
        )
        yield PipelineEvent("citations", [c.model_dump() for c in to_citations(reranked)])

        messages = build_messages(req.question, reranked, s.prompt_version)
        t3 = now_ms()
        first = True
        async for ev in deps.llm.stream(messages, s.max_output_tokens):
            if ev.kind == "token":
                if first:
                    lat.ttft_ms = now_ms() - t_start
                    M.TTFT_SECONDS.observe(lat.ttft_ms / 1000)
                    first = False
                answer_parts.append(ev.text)
                yield PipelineEvent("token", ev.text)
            elif ev.kind == "usage" and ev.usage:
                usage = ev.usage
            elif ev.kind == "done":
                finish_reason = ev.finish_reason
        lat.generate_ms = now_ms() - t3
        M.STAGE_SECONDS.labels("generate").observe(lat.generate_ms / 1000)
    except Exception as e:  # surface as an SSE error event, log the row, re-raise nothing
        error = f"{type(e).__name__}: {e}"
        M.ERRORS.labels(stage="pipeline").inc()
        yield PipelineEvent("error", {"message": error})
    finally:
        M.INFLIGHT.dec()

    lat.total_ms = now_ms() - t_start
    M.E2E_SECONDS.observe(lat.total_ms / 1000)
    overhead = lat.total_ms - (lat.embed_ms + lat.retrieve_ms + lat.rerank_ms + lat.generate_ms)
    M.GATEWAY_OVERHEAD_SECONDS.observe(max(overhead, 0) / 1000)
    M.TOKENS.labels("prompt").inc(usage.prompt_tokens)
    M.TOKENS.labels("completion").inc(usage.completion_tokens)
    answer = "".join(answer_parts)
    abstained = is_abstention(answer)
    if abstained:
        M.ABSTENTIONS.inc()
    if finish_reason == "length":
        M.TRUNCATED.inc()
    M.REQUESTS.labels(s.profile, "ask", "error" if error else "ok").inc()

    log_id: int | None = None
    try:
        log_id = await insert_query_log(
            deps.engine,
            QueryLogRow(
                question=req.question,
                profile=s.profile,
                model=deps.llm.model,
                prompt_version=s.prompt_version,
                retrieval_mode=mode,
                latency=lat,
                fused=fused,
                reranked=reranked,
                usage=usage,
                answer=answer,
                abstained=abstained,
                error=error,
            ),
        )
    except Exception:
        M.ERRORS.labels(stage="query_log").inc()
    if error is None:
        yield PipelineEvent(
            "done",
            DoneEvent(
                usage=usage,
                latency_breakdown=lat,
                model=deps.llm.model,
                prompt_version=s.prompt_version,
                abstained=abstained,
                query_log_id=log_id,
                finish_reason=finish_reason,
                truncated=finish_reason == "length",
            ).model_dump(),
        )
