"""Internal reranker service. POST /rerank, GET /health, GET /metrics."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import anyio
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from mlsys_common.models import RerankItem, RerankRequest, RerankResponse
from mlsys_common.settings import get_settings
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from mlsys_reranker.backends import RerankBackend, load_backend

RERANK_LATENCY = Histogram(
    "reranker_request_seconds",
    "Rerank request latency",
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
RERANK_DOCS = Counter("reranker_docs_total", "Documents scored")
RERANK_OOM = Counter("reranker_oom_total", "CUDA OOMs recovered (request retried once, then 503)")
RERANK_QUEUE = Histogram(
    "reranker_queue_seconds",
    "Time waiting for the inference slot",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
_backend: RerankBackend | None = None
# The cross-encoder runs one request at a time on the GPU: concurrent forward passes multiply activation memory and
# OOM when co-resident with vLLM (M8 27B cell: 7 OOMs at gateway concurrency 4). Throughput = batching, not parallelism.
_slots: anyio.Semaphore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _backend
    s = get_settings()
    mode = os.environ.get("RERANKER_MODE", s.reranker_mode)
    _backend = await anyio.to_thread.run_sync(load_backend, mode, s.reranker_model)
    app.state.mode = mode
    global _slots
    _slots = anyio.Semaphore(int(os.environ.get("RERANKER_CONCURRENCY", "1")))
    yield


app = FastAPI(title="mlsys-reranker", lifespan=lifespan)


def _score_with_oom_recovery(query: str, docs: list[str]) -> np.ndarray:
    """Score; on CUDA OOM free the cache and retry once in halves; then fail with 503 (not a crash, not a 500)."""
    assert _backend is not None
    try:
        return _backend.score(query, docs)
    except Exception as e:  # torch.OutOfMemoryError is not importable without torch
        if "out of memory" not in str(e).lower():
            raise
        RERANK_OOM.inc()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            mid = max(1, len(docs) // 2)
            return (
                np.concatenate(
                    [_backend.score(query, docs[:mid]), _backend.score(query, docs[mid:])]
                )
                if len(docs) > 1
                else _backend.score(query, docs)
            )
        except Exception as e2:
            if "out of memory" in str(e2).lower():
                raise HTTPException(503, "reranker out of GPU memory; retry later") from e2
            raise


@app.get("/health")
async def health() -> dict:
    return {
        "ok": _backend is not None,
        "model": getattr(_backend, "name", None),
        "mode": getattr(app.state, "mode", None),
    }


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest) -> RerankResponse:
    assert _backend is not None
    t0 = time.perf_counter()
    scores: np.ndarray = await anyio.to_thread.run_sync(
        lambda: _backend.score(req.query, req.documents)
    )
    RERANK_LATENCY.observe(time.perf_counter() - t0)
    RERANK_DOCS.inc(len(req.documents))
    order = np.argsort(-scores)
    if req.top_k:
        order = order[: req.top_k]
    return RerankResponse(
        results=[RerankItem(index=int(i), score=float(scores[i])) for i in order],
        model=_backend.name,
    )


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
