"""`make index`: embed every chunk lacking an embedding (GPU batch job) and build the HNSW index (spec §4 M2).

Embeds `heading_path + "\n\n" + text` (spec §5.1), stores the 1024-d normalized vector.
Idempotent: only rows with NULL embedding are processed, so re-running after an incremental
ingest embeds only the new chunks.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

from mlsys_common.db import make_engine, migrate
from sqlalchemy import text

from mlsys_embedder.backends import load_backend

HNSW_SQL = "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128)"


async def run(mode: str, model: str, batch: int, rebuild_index: bool) -> None:
    engine = make_engine()
    await migrate(engine)
    backend = load_backend(mode, model)
    print(f"backend={backend.name} mode={mode} dim={backend.dim}")
    async with engine.connect() as conn:
        total = (
            await conn.execute(text("SELECT count(*) FROM chunks WHERE embedding IS NULL"))
        ).scalar_one()
    print(f"{total} chunks to embed")
    done = 0
    t0 = time.perf_counter()
    while True:
        async with engine.begin() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT id, heading_path, text FROM chunks WHERE embedding IS NULL ORDER BY id LIMIT :n"
                    ),
                    {"n": batch},
                )
            ).all()
            if not rows:
                break
            texts = [f"{r[1]}\n\n{r[2]}" for r in rows]
            vecs = await asyncio.to_thread(backend.embed, texts, True, batch)
            for r, v in zip(rows, vecs, strict=True):
                await conn.execute(
                    text("UPDATE chunks SET embedding = CAST(:v AS vector) WHERE id = :id"),
                    {"v": "[" + ",".join(f"{x:.6f}" for x in v) + "]", "id": r[0]},
                )
        done += len(rows)
        el = time.perf_counter() - t0
        print(f"  {done}/{total}  {done / el:.1f} chunks/s", flush=True)
    async with engine.begin() as conn:
        if rebuild_index:
            await conn.execute(text("DROP INDEX IF EXISTS chunks_embedding_hnsw"))
        await conn.execute(text(HNSW_SQL))
        n = (
            await conn.execute(text("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"))
        ).scalar_one()
    print(f"index ready: {n} embedded chunks, HNSW built in {time.perf_counter() - t0:.1f}s total")
    await engine.dispose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        default=os.environ.get("EMBEDDER_MODE", "gpu"),
        choices=["gpu", "cpu", "onnx", "test"],
    )
    ap.add_argument("--model", default=os.environ.get("EMBEDDER_MODEL", "BAAI/bge-m3"))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("INDEX_BATCH", "32")))
    ap.add_argument("--rebuild-index", action="store_true", help="drop and rebuild the HNSW index")
    a = ap.parse_args()
    asyncio.run(run(a.mode, a.model, a.batch, a.rebuild_index))


if __name__ == "__main__":
    main()
