"""Internal embedding service. POST /embed, GET /health, GET /metrics."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Response
from mlsys_common.models import EmbedRequest, EmbedResponse
from mlsys_common.settings import get_settings
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from mlsys_embedder.backends import EmbeddingBackend, load_backend

EMBED_LATENCY = Histogram(
    "embedder_request_seconds",
    "Embed request latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
EMBED_TEXTS = Counter("embedder_texts_total", "Texts embedded")

_backend: EmbeddingBackend | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _backend
    s = get_settings()
    mode = os.environ.get("EMBEDDER_MODE", s.embedder_mode)
    _backend = await anyio.to_thread.run_sync(load_backend, mode, s.embedder_model)
    app.state.mode = mode
    yield


app = FastAPI(title="mlsys-embedder", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": _backend is not None,
        "model": getattr(_backend, "name", None),
        "mode": getattr(app.state, "mode", None),
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    assert _backend is not None
    t0 = time.perf_counter()
    # run the model off the event loop (sync torch/onnx call)
    vecs = await anyio.to_thread.run_sync(lambda: _backend.embed(req.texts, req.normalize))
    EMBED_LATENCY.observe(time.perf_counter() - t0)
    EMBED_TEXTS.inc(len(req.texts))
    return EmbedResponse(embeddings=vecs.tolist(), model=_backend.name, dim=_backend.dim)


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
