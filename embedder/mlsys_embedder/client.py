"""Async HTTP client for the embedder service; also an in-process adapter for the demo profile."""

from __future__ import annotations

from typing import Protocol

import httpx
import numpy as np
from mlsys_common.models import EmbedRequest, EmbedResponse


class Embedder(Protocol):
    async def embed_query(self, text: str) -> list[float]: ...


class HttpEmbedder:
    def __init__(
        self, base_url: str, client: httpx.AsyncClient | None = None, timeout: float = 30.0
    ) -> None:
        self._url = base_url.rstrip("/") + "/embed"
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def embed_query(self, text: str) -> list[float]:
        r = await self._client.post(self._url, json=EmbedRequest(texts=[text]).model_dump())
        r.raise_for_status()
        return EmbedResponse.model_validate(r.json()).embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        r = await self._client.post(self._url, json=EmbedRequest(texts=texts).model_dump())
        r.raise_for_status()
        return EmbedResponse.model_validate(r.json()).embeddings


class InProcessEmbedder:
    """Wraps a backend (ONNX in the demo) and runs it in a worker thread."""

    def __init__(self, backend) -> None:
        self._b = backend

    async def embed_query(self, text: str) -> list[float]:
        import anyio

        vec: np.ndarray = await anyio.to_thread.run_sync(lambda: self._b.embed([text]))
        return vec[0].tolist()
