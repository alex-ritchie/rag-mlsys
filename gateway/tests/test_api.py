"""End-to-end through the ASGI app with a real Postgres (pgserver) and fake model services."""

import json
import time

import pytest

pytestmark = pytest.mark.integration


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        ev, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if ev and data is not None:
            events.append((ev, json.loads(data)))
    return events


async def test_ask_streams_citations_tokens_done(client):
    r = await client.post("/api/ask", json={"question": "How does widget sizing use tooth pitch?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "citations" and kinds[-1] == "done" and "token" in kinds
    cits = events[0][1]
    assert 1 <= len(cits) <= 3
    assert (
        cits[0]["n"] == 1
        and "heading_path" in cits[0]
        and "rerank_score" in cits[0]
        and "fusion_score" in cits[0]
    )
    assert "sizing" in cits[0]["text_preview"].lower() or "pitch" in cits[0]["text_preview"].lower()
    done = events[-1][1]
    lb = done["latency_breakdown"]
    assert {"embed_ms", "retrieve_ms", "rerank_ms", "ttft_ms", "generate_ms", "total_ms"} <= set(lb)
    assert (
        done["usage"]["completion_tokens"] > 0
        and done["query_log_id"] is not None
        and done["abstained"] is False
    )
    assert done["finish_reason"] == "stop" and done["truncated"] is False
    answer = "".join(d["text"] for e, d in events if e == "token")
    assert "[1]" in answer


async def test_dense_mode_and_top_k(client):
    r = await client.post(
        "/api/ask", json={"question": "gear train ratio backlash", "mode": "dense", "top_k": 2}
    )
    events = parse_sse(r.text)
    assert len(events[0][1]) == 2


async def test_chunk_coverage_health_metrics(client):
    r = await client.post("/api/ask", json={"question": "conveyor throughput sprocket"})
    cid = parse_sse(r.text)[0][1][0]["chunk_id"]
    c = await client.get(f"/api/chunks/{cid}")
    assert c.status_code == 200 and c.json()["id"] == cid and c.json()["commit_sha"] == "fixturesha"
    assert (await client.get("/api/chunks/999999")).status_code == 404
    cov = (await client.get("/api/coverage")).json()
    assert cov["total_chunks"] == 6 and cov["volumes"][0]["volume"] == 9
    chs = {ch["chapter_num"]: ch for ch in cov["volumes"][0]["chapters"]}
    assert chs[3]["chunks"] == 2 and chs[3]["answers"] >= 1  # the conveyor question drew from ch 3
    h = (await client.get("/api/health")).json()
    assert h["ok"] and h["profile"] == "local" and h["reranker"] is True
    m = await client.get("/metrics")
    txt = m.text
    assert (
        "rag_requests_total" in txt
        and "rag_stage_seconds_bucket" in txt
        and "rag_nightly_groundedness" in txt
        and "rag_retrieval_score_p50" in txt
    )
    cfg = (await client.get("/api/config")).json()
    assert cfg["index_commit_sha"] == "fixturesha" and cfg["chunks"] == 6


async def test_query_log_row_written(client, deps):
    from sqlalchemy import text

    await client.post("/api/ask", json={"question": "lubricant viscosity bearing"})
    async with deps.engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT question, model, prompt_version, retrieval_mode, reranked_ids, answer, abstained FROM query_log ORDER BY id DESC LIMIT 1"
                )
            )
        ).first()
    assert (
        row[0] == "lubricant viscosity bearing"
        and row[1] == "fake-llm"
        and row[2] == "v1"
        and row[3] == "hybrid"
    )
    assert len(row[4]) >= 1 and "[1]" in row[5] and row[6] is False


async def test_openai_shim_non_stream_and_stream(client):
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "widget deployment duty cycle"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert (
        body["object"] == "chat.completion"
        and "**Sources**" in body["choices"][0]["message"]["content"]
    )
    assert body["usage"]["completion_tokens"] > 0
    r = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "widget deployment"}], "stream": True},
    )
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data:")]
    assert lines[-1].strip() == "data: [DONE]"
    chunks = [json.loads(ln[5:]) for ln in lines[:-1]]
    assert (
        chunks[0]["object"] == "chat.completion.chunk"
        and chunks[-1]["choices"][0]["finish_reason"] == "stop"
    )
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert "**Sources**" in content
    models = (await client.get("/v1/models")).json()
    assert models["data"][0]["id"] == "fake-llm"


async def test_official_openai_client_works_against_shim(client, deps):
    """Acceptance test (spec §4 M4): the official `openai` Python client, unmodified."""
    import httpx
    from openai import AsyncOpenAI

    transport = httpx.ASGITransport(app=client._transport.app)  # type: ignore[attr-defined]
    oai = AsyncOpenAI(
        base_url="http://t/v1", api_key="none", http_client=httpx.AsyncClient(transport=transport)
    )
    res = await oai.chat.completions.create(
        model="fake-llm", messages=[{"role": "user", "content": "gear ratio"}]
    )
    assert res.choices[0].message.content and "Sources" in res.choices[0].message.content
    stream = await oai.chat.completions.create(
        model="fake-llm", messages=[{"role": "user", "content": "gear ratio"}], stream=True
    )
    got = ""
    async for ch in stream:
        got += ch.choices[0].delta.content or ""
    assert "[1]" in got and "Sources" in got
    ms = await oai.models.list()
    assert ms.data[0].id == "fake-llm"


async def test_gateway_overhead_under_150ms(client):
    """p50 gateway overhead excluding model time (spec §4 M4). Fake services => nearly all time is gateway."""
    totals = []
    for _ in range(15):
        t0 = time.perf_counter()
        r = await client.post("/api/ask", json={"question": "conveyor throughput"})
        totals.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
    totals.sort()
    p50 = totals[len(totals) // 2]
    assert p50 < 150, f"p50 gateway overhead {p50:.1f} ms"


async def test_demo_profile_rate_limit_and_budget(seeded_engine):
    from httpx import ASGITransport, AsyncClient
    from mlsys_common.settings import Settings
    from mlsys_gateway.app import create_app
    from mlsys_gateway.llm import FakeLLM
    from mlsys_gateway.pipeline import Deps
    from mlsys_gateway.tests_support import FakeEmbedder, FakeReranker

    s = Settings(
        profile="demo",
        demo_rate_limit_per_day=2,
        demo_daily_budget_usd=100.0,
        rerank_top_k=2,
        retrieval_top_n=4,
    )
    deps = Deps(
        settings=s,
        engine=seeded_engine,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )
    app = create_app(deps)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c,
    ):
        ok1 = await c.post(
            "/api/ask", json={"question": "widgets"}, headers={"x-forwarded-for": "203.0.113.7"}
        )
        ok2 = await c.post(
            "/api/ask", json={"question": "widgets"}, headers={"x-forwarded-for": "203.0.113.7"}
        )
        assert ok1.status_code == 200 and ok2.status_code == 200
        assert "demo_cost_usd" in parse_sse(ok2.text)[-1][1]
        limited = await c.post(
            "/api/ask", json={"question": "widgets"}, headers={"x-forwarded-for": "203.0.113.7"}
        )
        assert limited.status_code == 429 and limited.json()["error"] == "rate_limit"
        other = await c.post(
            "/api/ask", json={"question": "widgets"}, headers={"x-forwarded-for": "203.0.113.8"}
        )
        assert other.status_code == 200
        # lower the budget to a trivial value and hit it (spec §4 M10 test)
        s.demo_daily_budget_usd = 0.0
        budget = await c.post(
            "/api/ask", json={"question": "widgets"}, headers={"x-forwarded-for": "203.0.113.9"}
        )
        assert budget.status_code == 429 and budget.json()["error"] == "budget"
        shim = await c.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x"}]},
            headers={"x-forwarded-for": "203.0.113.9"},
        )
        assert shim.status_code == 429
