"""Internal reranker service. POST /rerank, GET /health, GET /metrics."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import anyio
import numpy as np
from fastapi import FastAPI, Response
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
_backend: RerankBackend | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _backend
    s = get_settings()
    mode = os.environ.get("RERANKER_MODE", s.reranker_mode)
    _backend = await anyio.to_thread.run_sync(load_backend, mode, s.reranker_model)
    app.state.mode = mode
    yield


app = FastAPI(title="mlsys-reranker", lifespan=lifespan)


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
