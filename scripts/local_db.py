#!/usr/bin/env python3
"""Unprivileged local Postgres+pgvector via `pgserver` (no Docker/sudo needed).

python scripts/local_db.py up    -> starts (or reuses) data/pg, prints DATABASE_URL
python scripts/local_db.py down  -> stops it
python scripts/local_db.py url   -> prints the URL only
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PGDATA = ROOT / "data" / "pg"


def main() -> int:
    import pgserver

    cmd = sys.argv[1] if len(sys.argv) > 1 else "up"
    PGDATA.parent.mkdir(parents=True, exist_ok=True)
    if cmd in ("up", "url"):
        srv = pgserver.get_server(str(PGDATA), cleanup_mode=None)
        uri = srv.get_uri()
        srv.psql("CREATE EXTENSION IF NOT EXISTS vector;")
        if cmd == "up":
            print(f"postgres ready at {PGDATA}")
            print(f"export DATABASE_URL='{uri}'")
        else:
            print(uri)
        return 0
    if cmd == "down":
        srv = pgserver.get_server(str(PGDATA), cleanup_mode=None)
        srv.cleanup()
        print("stopped")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
