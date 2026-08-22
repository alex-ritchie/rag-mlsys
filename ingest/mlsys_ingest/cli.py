"""`python -m mlsys_ingest.cli {fetch|dry-run|run}`"""

from __future__ import annotations

import asyncio
import json
from collections import Counter

import typer
from mlsys_common.db import make_engine, migrate

from mlsys_ingest.chunk import chunk_stats
from mlsys_ingest.config import load_config
from mlsys_ingest.fetch import fetch as _fetch
from mlsys_ingest.load import load_chunks
from mlsys_ingest.pipeline import chunk_corpus, parse_corpus
from mlsys_ingest.tokens import get_counter

app = typer.Typer(add_completion=False, help="MLSysBook ingestion pipeline")


@app.command()
def fetch(config: str = typer.Option(None, help="config/ingest.yaml")) -> None:
    """Shallow-fetch the book at the pinned SHA into data/."""
    cfg = load_config(config)
    _fetch(cfg)


@app.command(name="dry-run")
def dry_run(
    config: str = typer.Option(None),
    approx: bool = typer.Option(False, help="use the offline approximate tokenizer"),
    show: int = typer.Option(0, help="print the heading path + first line of N chunks"),
) -> None:
    """Parse + chunk without a database; print statistics."""
    cfg = load_config(config)
    checkout = _fetch(cfg, quiet=True)
    parsed = parse_corpus(cfg, checkout)
    counter = get_counter(None if approx else cfg.chunking.tokenizer)
    chunks = chunk_corpus(cfg, parsed, counter)
    stats = chunk_stats(chunks)
    stats["under_min"] = sum(c.token_count < cfg.chunking.min_tokens for c in chunks)
    stats["tokenizer"] = counter.name
    stats["chapters"] = len(parsed)
    per_vol = Counter(c.volume for c in chunks)
    stats["per_volume"] = {f"vol{v}": n for v, n in sorted(per_vol.items())}
    print(json.dumps(stats, indent=2))
    for c in chunks[:show]:
        print(f"- [{c.token_count:4d} tok] {c.heading_path} :: {c.text[:80]!r}")


@app.command()
def run(config: str = typer.Option(None), approx: bool = typer.Option(False)) -> None:
    """Fetch + parse + chunk + load into Postgres (idempotent on the pinned SHA)."""
    cfg = load_config(config)
    checkout = _fetch(cfg)
    parsed = parse_corpus(cfg, checkout)
    counter = get_counter(None if approx else cfg.chunking.tokenizer)
    chunks = chunk_corpus(cfg, parsed, counter)
    print(f"parsed {len(parsed)} chapters -> {len(chunks)} chunks ({counter.name} tokens)")

    async def _go() -> None:
        engine = make_engine()
        await migrate(engine)
        summary = await load_chunks(engine, chunks, cfg.source.commit_sha)
        await engine.dispose()
        print(summary)

    asyncio.run(_go())


if __name__ == "__main__":
    app()
