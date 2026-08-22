import os

import pytest
from httpx import ASGITransport, AsyncClient

os.environ["EMBEDDER_MODE"] = "test"


@pytest.fixture
async def client():
    from mlsys_embedder.app import app

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c,
    ):
        yield c


async def test_embed_roundtrip(client):
    r = await client.post("/embed", json={"texts": ["hello world", "hello"]})
    assert r.status_code == 200
    body = r.json()
    assert (
        body["dim"] == 1024 and len(body["embeddings"]) == 2 and len(body["embeddings"][0]) == 1024
    )
    h = await client.get("/health")
    assert h.json()["ok"] is True
    m = await client.get("/metrics")
    assert b"embedder_texts_total" in m.content
