"""Gateway test fixtures: a real (unprivileged) Postgres with synthetic chunks + fake embedder/reranker/LLM."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from mlsys_common.db import make_engine, migrate
from mlsys_common.settings import Settings
from mlsys_embedder.backends import HashBackend
from mlsys_gateway.app import create_app
from mlsys_gateway.llm import FakeLLM
from mlsys_gateway.pipeline import Deps
from mlsys_gateway.tests_support import FakeEmbedder, FakeReranker
from sqlalchemy import text

SYNTHETIC = [
    (
        "Vol 9 > Ch 1: Widget Systems > Sizing",
        "Widget sizing multiplies gear count by tooth pitch. Pitch is the cheaper lever.",
    ),
    (
        "Vol 9 > Ch 1: Widget Systems > Deployment",
        "Widgets deploy to carts, shelves, or warehouses with different duty cycles.",
    ),
    (
        "Vol 9 > Ch 2: Gear Trains > Ratios",
        "A gear train ratio is the product of stage ratios; backlash accumulates per stage.",
    ),
    (
        "Vol 9 > Ch 2: Gear Trains > Lubrication",
        "Lubricant viscosity sets the bearing load limit in a gear housing.",
    ),
    (
        "Vol 9 > Ch 3: Conveyors > Throughput",
        "Conveyor throughput is bounded by belt speed and sprocket duty cycle.",
    ),
    (
        "Vol 9 > Ch 3: Conveyors > Quantization",
        "Quantization of sensor readings to int8 reduces memory four-fold at some accuracy cost.",
    ),
]


@pytest.fixture
async def seeded_engine(pg_url):
    engine = make_engine(pg_url)
    await migrate(engine)
    hb = HashBackend()
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM chunks"))
        for i, (hp, body) in enumerate(SYNTHETIC):
            _vol, rest = hp.split(" > ", 1)
            ch_title = rest.split(" > ")[0].split(": ", 1)[1]
            ch_num = int(rest.split(" > ")[0].split(":")[0].replace("Ch ", ""))
            vec = hb.embed([f"{hp}\n\n{body}"])[0]
            await conn.execute(
                text(
                    """INSERT INTO chunks (volume, chapter_num, chapter_title, section_path, heading_path, source_file, char_start, char_end,
                       token_count, commit_sha, content_hash, text, embedding)
                       VALUES (9, :cn, :ct, :sp, :hp, :sf, 0, 10, 20, 'fixturesha', :h, :t, CAST(:v AS vector))"""
                ),
                {
                    "cn": ch_num,
                    "ct": ch_title,
                    "sp": [rest.split(" > ")[1]],
                    "hp": hp,
                    "sf": f"fixture{i}.qmd",
                    "h": f"hash{i}",
                    "t": body,
                    "v": "[" + ",".join(f"{x:.6f}" for x in vec) + "]",
                },
            )
    yield engine
    await engine.dispose()


@pytest.fixture
def settings() -> Settings:
    return Settings(profile="local", rerank_top_k=3, retrieval_top_n=6, llm_model="fake")


@pytest.fixture
def deps(seeded_engine, settings) -> Deps:
    return Deps(
        settings=settings,
        engine=seeded_engine,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )


@pytest.fixture
async def client(deps):
    app = create_app(deps)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c,
    ):
        yield c
