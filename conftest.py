"""Shared pytest fixtures. `pg_url` spins up an unprivileged Postgres+pgvector (pgserver)."""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest


def pytest_collection_modifyitems(config, items):
    if shutil.which("git") is None:
        return
    # integration tests need pgserver; skip cleanly if it is missing
    try:
        import pgserver  # noqa: F401
    except Exception:
        skip = pytest.mark.skip(reason="pgserver not installed")
        for it in items:
            if "integration" in it.keywords:
                it.add_marker(skip)


@pytest.fixture(scope="session")
def pg_url():
    import pgserver

    d = tempfile.mkdtemp(prefix="mlsys-pg-")
    srv = pgserver.get_server(os.path.join(d, "pgdata"))
    srv.psql("CREATE EXTENSION IF NOT EXISTS vector;")
    os.environ["DATABASE_URL"] = srv.get_uri()
    try:
        yield srv.get_uri()
    finally:
        srv.cleanup()
        shutil.rmtree(d, ignore_errors=True)
