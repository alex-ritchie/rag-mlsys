"""Hybrid retrieval (spec §5.2): dense top-50 + FTS top-50 in one round trip, RRF k=60, fused top-30.

`mode="dense"` is the ablation row: dense-only, score = cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlsys_common.models import ChunkRecord, RetrievedChunk
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "hybrid"  # hybrid | dense
    top_n: int = 30  # fused list size handed to the reranker
    per_list: int = 50  # candidates per modality
    rrf_k: int = 60
    ef_search: int = 100  # hnsw.ef_search (>= per_list)


HYBRID_SQL = text(
    """
    WITH dense AS (
        SELECT id, row_number() OVER (ORDER BY embedding <=> CAST(:v AS vector)) AS rnk
        FROM chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:v AS vector)
        LIMIT :per_list
    ),
    fts AS (
        SELECT id, row_number() OVER (ORDER BY ts_rank_cd(fts, q) DESC) AS rnk
        FROM chunks, websearch_to_tsquery('english', :q) AS q
        WHERE fts @@ q
        ORDER BY ts_rank_cd(fts, q) DESC
        LIMIT :per_list
    ),
    fused AS (
        SELECT COALESCE(d.id, f.id) AS id, d.rnk AS dense_rank, f.rnk AS fts_rank,
               COALESCE(1.0 / (:k + d.rnk), 0) + COALESCE(1.0 / (:k + f.rnk), 0) AS score
        FROM dense d FULL OUTER JOIN fts f ON d.id = f.id
    )
    SELECT c.id, c.volume, c.chapter_num, c.chapter_title, c.heading_path, c.text, c.content_hash,
           fused.score, fused.dense_rank, fused.fts_rank
    FROM fused JOIN chunks c ON c.id = fused.id
    ORDER BY fused.score DESC, c.id
    LIMIT :top_n
    """
)

DENSE_SQL = text(
    """
    SELECT c.id, c.volume, c.chapter_num, c.chapter_title, c.heading_path, c.text, c.content_hash,
           1 - (c.embedding <=> CAST(:v AS vector)) AS score,
           row_number() OVER (ORDER BY c.embedding <=> CAST(:v AS vector)) AS dense_rank,
           NULL::bigint AS fts_rank
    FROM chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> CAST(:v AS vector)
    LIMIT :top_n
    """
)


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


async def retrieve(
    engine: AsyncEngine, query_vec: list[float], question: str, cfg: RetrievalConfig
) -> list[RetrievedChunk]:
    params = {
        "v": _vec(query_vec),
        "q": question,
        "per_list": cfg.per_list,
        "k": cfg.rrf_k,
        "top_n": cfg.top_n,
    }
    sql = HYBRID_SQL if cfg.mode == "hybrid" else DENSE_SQL
    async with engine.connect() as conn:
        await conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(cfg.ef_search)}"))
        rows = (await conn.execute(sql, params)).all()
    return [
        RetrievedChunk(
            chunk_id=r[0],
            volume=r[1],
            chapter_num=r[2],
            chapter_title=r[3],
            heading_path=r[4],
            text=r[5],
            content_hash=r[6],
            fusion_score=float(r[7]),
            dense_rank=r[8],
            fts_rank=r[9],
        )
        for r in rows
    ]


async def get_chunk(engine: AsyncEngine, chunk_id: int) -> ChunkRecord | None:
    async with engine.connect() as conn:
        r = (
            await conn.execute(
                text(
                    "SELECT id, volume, chapter_num, chapter_title, section_path, heading_path, source_file, char_start, char_end, token_count, commit_sha, content_hash, text, oversize FROM chunks WHERE id = :id"
                ),
                {"id": chunk_id},
            )
        ).first()
    if not r:
        return None
    return ChunkRecord(
        id=r[0],
        volume=r[1],
        chapter_num=r[2],
        chapter_title=r[3],
        section_path=list(r[4]),
        heading_path=r[5],
        source_file=r[6],
        char_start=r[7],
        char_end=r[8],
        token_count=r[9],
        commit_sha=r[10],
        content_hash=r[11],
        text=r[12],
        oversize=r[13],
    )


async def coverage(engine: AsyncEngine, days: int = 30) -> dict:
    """Chapter tree with chunk counts and how many recent answers drew from each chapter (spec §5.5)."""
    async with engine.connect() as conn:
        chapters = (
            await conn.execute(
                text(
                    """
                    SELECT volume, chapter_num, chapter_title, count(*) AS chunks, min(commit_sha),
                           sum(token_count) AS tokens
                    FROM chunks GROUP BY volume, chapter_num, chapter_title ORDER BY volume, chapter_num
                    """
                )
            )
        ).all()
        sections = (
            await conn.execute(
                text(
                    """
                    SELECT volume, chapter_num, COALESCE(section_path[1], '') AS section, count(*)
                    FROM chunks GROUP BY volume, chapter_num, section ORDER BY volume, chapter_num, min(id)
                    """
                )
            )
        ).all()
        hits = (
            await conn.execute(
                text(
                    """
                    SELECT c.volume, c.chapter_num, count(DISTINCT q.id) AS answers, count(*) AS citations
                    FROM query_log q
                    JOIN LATERAL unnest(q.reranked_ids) AS rid(id) ON true
                    JOIN chunks c ON c.id = rid.id
                    WHERE q.started_at > now() - make_interval(days => :days)
                    GROUP BY c.volume, c.chapter_num
                    """
                ),
                {"days": days},
            )
        ).all()
    hit_map = {(h[0], h[1]): {"answers": h[2], "citations": h[3]} for h in hits}
    sec_map: dict[tuple[int, int], list[dict]] = {}
    for s in sections:
        sec_map.setdefault((s[0], s[1]), []).append({"title": s[2], "chunks": s[3]})
    volumes: dict[int, dict] = {}
    commit = None
    for ch in chapters:
        commit = commit or ch[4]
        vol = volumes.setdefault(ch[0], {"volume": ch[0], "chapters": [], "chunks": 0})
        entry = {
            "chapter_num": ch[1],
            "title": ch[2],
            "chunks": ch[3],
            "tokens": int(ch[5] or 0),
            "sections": sec_map.get((ch[0], ch[1]), []),
            **hit_map.get((ch[0], ch[1]), {"answers": 0, "citations": 0}),
        }
        vol["chapters"].append(entry)
        vol["chunks"] += ch[3]
    return {
        "commit_sha": commit,
        "window_days": days,
        "total_chunks": sum(v["chunks"] for v in volumes.values()),
        "volumes": list(volumes.values()),
    }
