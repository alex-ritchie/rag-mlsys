#!/usr/bin/env python3
"""Keep golden-set chunk labels valid across a re-ingest that changes chunk text (e.g. materialised values).

  python scripts/migrate_golden_hashes.py snapshot   # BEFORE re-ingest: save (hash, source_file, char_start, char_end)
  python scripts/migrate_golden_hashes.py apply      # AFTER  re-ingest: remap hashes by source-span overlap, re-stamp

Labels are content hashes (spec §5.6) so they survive *identical* re-ingests; when text changes, the only stable
identity is the source span (file + character offsets), so old -> new is resolved by maximum overlap.
"""

from __future__ import annotations

import asyncio
import json
import sys

from mlsys_common.db import make_engine
from mlsys_common.settings import REPO_ROOT
from mlsys_eval.schema import (
    CANDIDATES_PATH,
    GOLDEN_PATH,
    STAMP_PATH,
    Candidate,
    GoldenItem,
    read_jsonl,
    write_jsonl,
    write_stamp,
)
from sqlalchemy import text

SNAP = REPO_ROOT / "data" / "chunk-snapshot.json"


async def _rows():
    e = make_engine()
    async with e.connect() as c:
        rows = (
            await c.execute(
                text("SELECT content_hash, source_file, char_start, char_end FROM chunks")
            )
        ).all()
    await e.dispose()
    return [{"hash": r[0], "file": r[1], "start": r[2], "end": r[3]} for r in rows]


def snapshot() -> None:
    rows = asyncio.run(_rows())
    SNAP.write_text(json.dumps(rows))
    print(f"snapshot: {len(rows)} chunks -> {SNAP}")


def _best_match(old: dict, new_by_file: dict[str, list[dict]]) -> str | None:
    best, best_ov = None, 0
    for n in new_by_file.get(old["file"], []):
        ov = min(old["end"], n["end"]) - max(old["start"], n["start"])
        if ov > best_ov:
            best, best_ov = n["hash"], ov
    return best


def apply() -> None:
    old = json.loads(SNAP.read_text())
    new = asyncio.run(_rows())
    new_hashes = {n["hash"] for n in new}
    new_by_file: dict[str, list[dict]] = {}
    for n in new:
        new_by_file.setdefault(n["file"], []).append(n)
    old_by_hash = {o["hash"]: o for o in old}
    mapping: dict[str, str | None] = {}

    def remap(h: str) -> str | None:
        if h in new_hashes:
            return h
        if h not in mapping:
            mapping[h] = _best_match(old_by_hash[h], new_by_file) if h in old_by_hash else None
        return mapping[h]

    for path, model in ((CANDIDATES_PATH, Candidate), (GOLDEN_PATH, GoldenItem)):
        items = read_jsonl(path, model)
        if not items:
            continue
        lost = 0
        for it in items:
            out = []
            for h in it.source_chunk_content_hashes:
                m = remap(h)
                if m is None:
                    lost += 1
                else:
                    out.append(m)
            it.source_chunk_content_hashes = out
        write_jsonl(path, items)
        print(
            f"{path.name}: {len(items)} items, {sum(1 for v in mapping.values() if v)} hashes remapped, {lost} unresolvable"
        )
    if GOLDEN_PATH.exists() and STAMP_PATH.exists():
        st = write_stamp(GOLDEN_PATH, STAMP_PATH)
        print(f"re-stamped golden set ({st['count']} items)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"snapshot": snapshot, "apply": apply}.get(cmd, lambda: print(__doc__))()
