import os

import pytest
from httpx import ASGITransport, AsyncClient

os.environ["RERANKER_MODE"] = "test"


@pytest.fixture
async def client():
    from mlsys_reranker.app import app

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c,
    ):
        yield c


async def test_rerank_orders_by_score(client):
    r = await client.post(
        "/rerank",
        json={
            "query": "quantization int8 weights",
            "documents": ["pruning removes weights", "int8 quantization of weights", "unrelated"],
            "top_k": 2,
        },
    )
    assert r.status_code == 200
    res = r.json()["results"]
    assert len(res) == 2 and res[0]["index"] == 1 and res[0]["score"] >= res[1]["score"]
