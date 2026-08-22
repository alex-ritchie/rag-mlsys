"""Idempotent chunk loading (spec §5.1): content-hash diff against the current table."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from mlsys_ingest.chunk import Chunk


@dataclass
class LoadSummary:
    commit_sha: str
    added: int
    removed: int
    unchanged: int
    previous_sha: str | None

    def __str__(self) -> str:
        prev = f" (previous index: {self.previous_sha[:10]})" if self.previous_sha else ""
        return f"ingest @ {self.commit_sha[:10]}{prev}: +{self.added} added, -{self.removed} removed, {self.unchanged} unchanged"


INSERT_SQL = text(
    """
    INSERT INTO chunks (volume, chapter_num, chapter_title, section_path, heading_path, source_file,
                        char_start, char_end, token_count, commit_sha, content_hash, oversize, text)
    VALUES (:volume, :chapter_num, :chapter_title, :section_path, :heading_path, :source_file,
            :char_start, :char_end, :token_count, :commit_sha, :content_hash, :oversize, :text)
    """
)


async def load_chunks(engine: AsyncEngine, chunks: list[Chunk], commit_sha: str) -> LoadSummary:
    by_hash = {c.content_hash: c for c in chunks}
    async with engine.begin() as conn:
        rows = (await conn.execute(text("SELECT content_hash, commit_sha FROM chunks"))).all()
        existing = {r[0]: r[1] for r in rows}
        prev_sha = next(iter(set(existing.values())), None) if existing else None

        to_add = [c for h, c in by_hash.items() if h not in existing]
        to_remove = [h for h in existing if h not in by_hash]
        unchanged = len(by_hash) - len(to_add)

        run_id = (
            await conn.execute(
                text("INSERT INTO ingest_runs (commit_sha) VALUES (:sha) RETURNING id"),
                {"sha": commit_sha},
            )
        ).scalar_one()

        if to_remove:
            await conn.execute(
                text("DELETE FROM chunks WHERE content_hash = ANY(:hs)"), {"hs": to_remove}
            )
        for c in to_add:
            await conn.execute(
                INSERT_SQL,
                {
                    "volume": c.volume,
                    "chapter_num": c.chapter_num,
                    "chapter_title": c.chapter_title,
                    "section_path": c.section_path,
                    "heading_path": c.heading_path,
                    "source_file": c.source_file,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "token_count": c.token_count,
                    "commit_sha": c.commit_sha,
                    "content_hash": c.content_hash,
                    "oversize": c.oversize,
                    "text": c.text,
                },
            )
        # unchanged rows keep their embedding; refresh provenance sha on them
        if unchanged and prev_sha != commit_sha:
            await conn.execute(
                text("UPDATE chunks SET commit_sha = :sha WHERE commit_sha <> :sha"),
                {"sha": commit_sha},
            )
        await conn.execute(
            text(
                "UPDATE ingest_runs SET finished_at = now(), chunks_added=:a, chunks_removed=:r, chunks_unchanged=:u WHERE id=:id"
            ),
            {"a": len(to_add), "r": len(to_remove), "u": unchanged, "id": run_id},
        )
    return LoadSummary(commit_sha, len(to_add), len(to_remove), unchanged, prev_sha)
