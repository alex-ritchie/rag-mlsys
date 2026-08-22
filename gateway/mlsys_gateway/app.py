"""FastAPI gateway (spec §5.5). Stateless: nothing is held between requests (the HPA target)."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from mlsys_common.db import make_engine
from mlsys_common.models import AskRequest, Usage
from mlsys_common.settings import REPO_ROOT, Settings, get_settings
from mlsys_embedder.client import HttpEmbedder, InProcessEmbedder
from mlsys_reranker.client import HttpReranker, InProcessReranker
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sse_starlette.sse import EventSourceResponse

from mlsys_gateway import metrics as M
from mlsys_gateway.demo import DemoGuard, DemoLimitError
from mlsys_gateway.llm import AnthropicLLM, FakeLLM, OpenAICompatLLM
from mlsys_gateway.openai_shim import ChatCompletionRequest, complete_chat, stream_chat
from mlsys_gateway.pipeline import Deps, ask
from mlsys_gateway.retrieval import coverage, get_chunk

EVAL_REPORT_PATH = Path(
    os.environ.get("EVAL_REPORT_PATH", REPO_ROOT / "eval" / "results" / "latest" / "report.json")
)


def build_deps(s: Settings) -> Deps:
    engine = make_engine(s.effective_database_url, pool_size=10, max_overflow=20)
    http = httpx.AsyncClient(timeout=60.0)
    if s.profile == "demo":
        from mlsys_embedder.backends import OnnxBackend
        from mlsys_reranker.backends import OnnxRerankBackend

        embedder = (
            InProcessEmbedder(OnnxBackend())
            if s.demo_embedder == "inprocess-onnx"
            else HttpEmbedder(s.embedder_url, http)
        )
        reranker = InProcessReranker(OnnxRerankBackend()) if s.demo_rerank == "onnx" else None
        llm = (
            AnthropicLLM(s.demo_generation_model, s.anthropic_api_key)
            if s.anthropic_api_key
            else FakeLLM()
        )
    else:
        embedder = HttpEmbedder(s.embedder_url, http)
        reranker = HttpReranker(s.reranker_url, http) if s.reranker_url else None
        llm = (
            FakeLLM()
            if s.llm_model == "fake"
            else OpenAICompatLLM(s.llm_base_url, s.llm_model, s.llm_api_key, s.disable_thinking)
        )
    return Deps(settings=s, engine=engine, embedder=embedder, reranker=reranker, llm=llm)


def create_app(deps: Deps | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.deps = deps or build_deps(get_settings())
        d: Deps = app.state.deps
        app.state.demo = DemoGuard(d.engine, d.settings) if d.settings.profile == "demo" else None
        yield
        await d.engine.dispose()

    app = FastAPI(title="mlsysbook-rag gateway", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def D(request: Request) -> Deps:  # noqa: N802
        return request.app.state.deps

    def client_ip(request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for") or request.headers.get("fly-client-ip")
        return (fwd.split(",")[0].strip() if fwd else None) or (
            request.client.host if request.client else "unknown"
        )

    # ---- native API -------------------------------------------------------------------
    @app.post("/api/ask")
    async def api_ask(req: AskRequest, request: Request):
        deps = D(request)
        guard: DemoGuard | None = request.app.state.demo
        ip = client_ip(request)
        if guard is not None:
            try:
                await guard.check(ip)
            except DemoLimitError as e:
                M.DEMO_LIMITED.labels(e.reason).inc()
                return JSONResponse(
                    status_code=429, content={"error": e.reason, "message": e.message}
                )
            await guard.reserve(ip)

        async def gen():
            async for ev in ask(deps, req):
                if guard is not None and ev.event == "done":
                    cost = await guard.settle(ip, Usage(**ev.data["usage"]))
                    M.DEMO_SPEND.set(await guard.spend_24h())
                    ev.data["demo_cost_usd"] = round(cost, 6)
                yield {
                    "event": ev.event,
                    "data": json.dumps(
                        ev.data if not isinstance(ev.data, str) else {"text": ev.data}
                    ),
                }

        return EventSourceResponse(gen(), ping=15)

    @app.get("/api/chunks/{chunk_id}")
    async def api_chunk(chunk_id: int, request: Request):
        c = await get_chunk(D(request).engine, chunk_id)
        if c is None:
            raise HTTPException(404, "chunk not found")
        return c

    @app.get("/api/coverage")
    async def api_coverage(request: Request, days: int = 30):
        return await coverage(D(request).engine, days=days)

    @app.get("/api/eval/summary")
    async def api_eval_summary():
        if not EVAL_REPORT_PATH.exists():
            return JSONResponse(
                status_code=404,
                content={"error": "no eval report committed yet", "path": str(EVAL_REPORT_PATH)},
            )
        return Response(EVAL_REPORT_PATH.read_bytes(), media_type="application/json")

    @app.get("/api/health")
    async def api_health(request: Request):
        d = D(request)
        return {
            "ok": True,
            "profile": d.settings.profile,
            "model": d.llm.model,
            "prompt_version": d.settings.prompt_version,
            "retrieval_mode": d.settings.retrieval_mode,
            "reranker": d.reranker is not None,
        }

    @app.get("/api/config")
    async def api_config(request: Request):
        """Non-secret runtime config for the frontend About page."""
        d = D(request)
        from sqlalchemy import text

        async with d.engine.connect() as conn:
            row = (await conn.execute(text("SELECT min(commit_sha), count(*) FROM chunks"))).first()
        return {
            "profile": d.settings.profile,
            "model": d.llm.model,
            "index_commit_sha": row[0] if row else None,
            "chunks": row[1] if row else 0,
            "prompt_version": d.settings.prompt_version,
        }

    @app.get("/metrics")
    async def metrics(request: Request):
        await M.refresh_quality_gauges(D(request).engine)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ---- OpenAI-compatible shim -----------------------------------------------------------
    @app.get("/v1/models")
    async def v1_models(request: Request):
        d = D(request)
        return {
            "object": "list",
            "data": [
                {"id": d.llm.model, "object": "model", "created": 0, "owned_by": "mlsysbook-rag"}
            ],
        }

    @app.post("/v1/chat/completions")
    async def v1_chat(req: ChatCompletionRequest, request: Request):
        deps = D(request)
        guard: DemoGuard | None = request.app.state.demo
        if guard is not None:
            try:
                await guard.check(client_ip(request))
            except DemoLimitError as e:
                M.DEMO_LIMITED.labels(e.reason).inc()
                return JSONResponse(
                    status_code=429, content={"error": {"message": e.message, "type": e.reason}}
                )
            await guard.reserve(client_ip(request))
        if req.stream:
            return StreamingResponse(stream_chat(deps, req), media_type="text/event-stream")
        return await complete_chat(deps, req)

    return app


app = create_app()
