"""Async DB access (SQLAlchemy 2 async + asyncpg, spec §5.5)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mlsys_common.settings import get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def normalize_url(url: str) -> str:
    """Accept plain postgresql:// URLs (pgserver, Supabase) and make them asyncpg URLs."""
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    return url


def make_engine(url: str | None = None, **kw: object) -> AsyncEngine:
    url = normalize_url(url or get_settings().effective_database_url)
    return create_async_engine(url, pool_pre_ping=True, **kw)  # type: ignore[arg-type]


def _statements(sql: str) -> list[str]:
    return [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]


async def migrate(engine: AsyncEngine) -> None:
    sql = SCHEMA_PATH.read_text()
    async with engine.begin() as conn:
        for stmt in _statements(sql):
            await conn.execute(text(stmt))


async def _main(argv: list[str]) -> int:
    if argv[:1] != ["migrate"]:
        print("usage: python -m mlsys_common.db migrate", file=sys.stderr)
        return 2
    engine = make_engine()
    await migrate(engine)
    await engine.dispose()
    print("schema applied")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
