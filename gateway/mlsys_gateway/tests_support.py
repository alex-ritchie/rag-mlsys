"""Fakes for tests and the no-GPU dev path (kept importable so `make dev` can run without models)."""

from __future__ import annotations

from mlsys_common.models import RerankItem
from mlsys_embedder.backends import HashBackend


class FakeEmbedder:
    def __init__(self) -> None:
        self.b = HashBackend()

    async def embed_query(self, text: str) -> list[float]:
        return self.b.embed([text])[0].tolist()


class FakeReranker:
    async def rerank(self, query: str, docs: list[str], top_k: int) -> list[RerankItem]:
        q = set(query.lower().split())
        scored = sorted(
            ((len(q & set(d.lower().split())), i) for i, d in enumerate(docs)), reverse=True
        )
        return [RerankItem(index=i, score=float(s)) for s, i in scored[:top_k]]
