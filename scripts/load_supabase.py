#!/usr/bin/env python3
"""Push the locally built `chunks` table (with embeddings) to the demo's Supabase Postgres (spec §5.11).

  SUPABASE_DB_URL=postgresql://postgres:...@db.<ref>.supabase.co:5432/postgres \
  DATABASE_URL=<local> python scripts/load_supabase.py [--batch 200] [--drop]

Private hosting for the demo's own use — not redistribution (LICENSING.md §4). Schema-compatible:
the same schema.sql is applied remotely, rows are copied by content_hash (idempotent), and the
HNSW index is created remotely at the end. ~2.8K chunks x 1024-d fits the free tier (~25 MB of vectors).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from mlsys_common.db import make_engine, migrate
from sqlalchemy import text

COLS = "volume, chapter_num, chapter_title, section_path, heading_path, source_file, char_start, char_end, token_count, commit_sha, content_hash, oversize, text, embedding"


async def run(batch: int, drop: bool) -> None:
    remote_url = os.environ.get("SUPABASE_DB_URL")
    if not remote_url:
        sys.exit("SUPABASE_DB_URL is required")
    local = make_engine(os.environ["DATABASE_URL"])
    remote = make_engine(remote_url, pool_size=2)
    await migrate(remote)
    async with remote.begin() as conn:
        if drop:
            await conn.execute(text("TRUNCATE chunks"))
        have = {r[0] for r in (await conn.execute(text("SELECT content_hash FROM chunks"))).all()}
    async with local.connect() as conn:
        total = (
            await conn.execute(text("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"))
        ).scalar_one()
    print(f"remote has {len(have)} chunks; local has {total} embedded chunks")
    sent = 0
    last_id = 0
    while True:
        async with local.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"SELECT id, {COLS.replace('embedding', 'embedding::text')} FROM chunks WHERE embedding IS NOT NULL AND id > :last ORDER BY id LIMIT :n"
                    ),
                    {"last": last_id, "n": batch},
                )
            ).all()
        if not rows:
            break
        last_id = rows[-1][0]
        todo = [r for r in rows if r[11] not in have]
        if todo:
            async with remote.begin() as conn:
                for r in todo:
                    await conn.execute(
                        text(
                            f"INSERT INTO chunks ({COLS}) VALUES (:v, :cn, :ct, :sp, :hp, :sf, :cs, :ce, :tc, :sha, :h, :os, :t, CAST(:e AS vector))"
                        ),
                        dict(
                            v=r[1],
                            cn=r[2],
                            ct=r[3],
                            sp=list(r[4]),
                            hp=r[5],
                            sf=r[6],
                            cs=r[7],
                            ce=r[8],
                            tc=r[9],
                            sha=r[10],
                            h=r[11],
                            os=r[12],
                            t=r[13],
                            e=r[14],
                        ),
                    )
            sent += len(todo)
            print(f"  +{sent}", flush=True)
    async with remote.begin() as conn:
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128)"
            )
        )
        n = (await conn.execute(text("SELECT count(*) FROM chunks"))).scalar_one()
    print(f"remote index ready: {n} chunks")
    await local.dispose()
    await remote.dispose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--drop", action="store_true", help="truncate the remote table first")
    a = ap.parse_args()
    asyncio.run(run(a.batch, a.drop))


if __name__ == "__main__":
    main()
