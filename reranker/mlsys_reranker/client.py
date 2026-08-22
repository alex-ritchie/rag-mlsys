from __future__ import annotations

from typing import Protocol

import httpx
from mlsys_common.models import RerankItem, RerankRequest, RerankResponse


class Reranker(Protocol):
    async def rerank(self, query: str, docs: list[str], top_k: int) -> list[RerankItem]: ...


class HttpReranker:
    def __init__(
        self, base_url: str, client: httpx.AsyncClient | None = None, timeout: float = 30.0
    ) -> None:
        self._url = base_url.rstrip("/") + "/rerank"
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def rerank(self, query: str, docs: list[str], top_k: int) -> list[RerankItem]:
        r = await self._client.post(
            self._url, json=RerankRequest(query=query, documents=docs, top_k=top_k).model_dump()
        )
        r.raise_for_status()
        return RerankResponse.model_validate(r.json()).results


class InProcessReranker:
    def __init__(self, backend) -> None:
        self._b = backend

    async def rerank(self, query: str, docs: list[str], top_k: int) -> list[RerankItem]:
        import anyio
        import numpy as np

        scores: np.ndarray = await anyio.to_thread.run_sync(lambda: self._b.score(query, docs))
        order = np.argsort(-scores)[:top_k]
        return [RerankItem(index=int(i), score=float(scores[i])) for i in order]
