"""Pydantic contracts shared between services (spec §8: the contract between services)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    """One row of the `chunks` table (sans embedding)."""

    id: int
    volume: int
    chapter_num: int
    chapter_title: str
    section_path: list[str]
    heading_path: str
    source_file: str
    char_start: int
    char_end: int
    token_count: int
    commit_sha: str
    content_hash: str
    text: str
    oversize: bool = False


class ChunkPreview(BaseModel):
    chunk_id: int
    heading_path: str
    text_preview: str


class RetrievedChunk(BaseModel):
    """A chunk with retrieval scores. fusion_score is RRF (hybrid) or cosine similarity (dense)."""

    chunk_id: int
    volume: int
    chapter_num: int
    chapter_title: str
    heading_path: str
    text: str
    content_hash: str
    fusion_score: float
    dense_rank: int | None = None
    fts_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    """The `citations` SSE event payload item (spec §5.5)."""

    n: int = Field(description="1-based index matching [n] in the answer")
    chunk_id: int
    heading_path: str
    rerank_score: float | None
    fusion_score: float
    text_preview: str


class LatencyBreakdown(BaseModel):
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    ttft_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    mode: Literal["hybrid", "dense"] | None = None


class DoneEvent(BaseModel):
    usage: Usage
    latency_breakdown: LatencyBreakdown
    model: str
    prompt_version: str
    abstained: bool
    query_log_id: int | None = None


# ---- embedder / reranker service contracts --------------------------------------


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=512)
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dim: int


class RerankRequest(BaseModel):
    query: str
    documents: list[str] = Field(min_length=1, max_length=200)
    top_k: int | None = None


class RerankItem(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    results: list[RerankItem]
    model: str
