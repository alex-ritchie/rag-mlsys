"""Golden-set generation and the MANDATORY human verification CLI (spec §5.6).

python -m mlsys_eval.golden generate   -> eval/golden/candidates.jsonl (gitignored)
python -m mlsys_eval.golden verify     -> eval/golden/golden.jsonl + golden.verified.json
python -m mlsys_eval.golden stats
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections import Counter

import typer
from mlsys_common.db import make_engine
from mlsys_common.settings import get_settings
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from sqlalchemy import text

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

app = typer.Typer(add_completion=False)
console = Console()

GEN_SINGLE = """You write exam-style questions for a textbook chapter. Below is one passage.

<passage heading="{heading}">
{text}
</passage>

Write ONE question a student could answer from this passage alone, plus 2-4 short key points a correct answer must contain, paraphrased in your own words (do not copy sentences from the passage).
The question must be self-contained, phrased the way a student would ask a study assistant about the textbook — never refer to "the passage", "the text", or "the author".
Respond with JSON only: {{"question": "...", "key_points": ["...", "..."]}}"""

GEN_MULTI = """You write exam-style synthesis questions for a textbook. Below are two passages from the same chapter.

<passage_1 heading="{heading1}">
{text1}
</passage_1>
<passage_2 heading="{heading2}">
{text2}
</passage_2>

Write ONE question whose answer REQUIRES combining both passages, plus 2-4 short key points a correct answer must contain, paraphrased (no copied sentences).
The question must be self-contained, phrased the way a student would ask a study assistant about the textbook — never refer to "the passage(s)", "the text", or "the author".
Respond with JSON only: {{"question": "...", "key_points": ["...", "..."]}}"""

GEN_UNANSWERABLE = """You write trap questions for a retrieval system over a textbook about machine learning systems. Below is a passage showing the textbook's style and scope.

<passage heading="{heading}">
{text}
</passage>

Write ONE plausible-sounding question on a closely related topic that this passage (and a textbook like it) would NOT answer — e.g. a specific vendor pricing detail, a named product's internal implementation, a very recent event, or a numeric fact the passage does not state. It must sound like a reasonable, self-contained student question (no reference to "the passage").
Respond with JSON only: {{"question": "...", "why_unanswerable": "..."}}"""


async def _sample_chunks(engine, n: int, seed: int) -> list[dict]:
    """Stratified by chapter proportional to chunk counts (spec §5.6) — i.e. uniform over chunks."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT id, volume, chapter_num, chapter_title, heading_path, text, content_hash FROM chunks WHERE token_count >= 200"
                )
            )
        ).all()
    rng = random.Random(seed)
    picks = rng.sample(rows, min(n, len(rows)))
    return [
        dict(
            id=r[0],
            volume=r[1],
            chapter_num=r[2],
            chapter_title=r[3],
            heading_path=r[4],
            text=r[5],
            content_hash=r[6],
        )
        for r in picks
    ]


async def _sibling(engine, chunk: dict, rng: random.Random) -> dict | None:
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT id, heading_path, text, content_hash FROM chunks WHERE volume=:v AND chapter_num=:c AND id<>:id AND token_count >= 200"
                ),
                {"v": chunk["volume"], "c": chunk["chapter_num"], "id": chunk["id"]},
            )
        ).all()
    if not rows:
        return None
    r = rng.choice(rows)
    return dict(id=r[0], heading_path=r[1], text=r[2], content_hash=r[3])


def _parse_json(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.S)
    return json.loads(m.group(0)) if m else {}


@app.command()
def generate(
    n: int = typer.Option(140, help="target number of candidates"),
    seed: int = 42,
    model: str = typer.Option(None),
) -> None:
    """Generate candidate Q/A pairs with Claude Haiku (60% single, 30% multi, 10% unanswerable)."""
    s = get_settings()
    if not s.anthropic_api_key:
        raise typer.BadParameter("ANTHROPIC_API_KEY is required for generation")
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=s.anthropic_api_key)
    model = model or s.golden_gen_model
    n_single, n_multi = round(n * 0.6), round(n * 0.3)
    rng = random.Random(seed)

    async def ask_model(prompt: str) -> dict:
        msg = await client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        obj = _parse_json("".join(getattr(b, "text", "") for b in msg.content))
        if "question" not in obj:
            raise ValueError(
                f"no JSON question in response (stop_reason={msg.stop_reason}, output_tokens={msg.usage.output_tokens})"
            )
        return obj

    async def go() -> list[Candidate]:
        engine = make_engine()
        chunks = await _sample_chunks(engine, n, seed)
        out: list[Candidate] = []
        sem = asyncio.Semaphore(8)

        async def one(i: int, ch: dict) -> None:
            kind = (
                "single" if i < n_single else "multi" if i < n_single + n_multi else "unanswerable"
            )
            chapter = f"vol{ch['volume']}/ch{ch['chapter_num']}"
            async with sem:
                try:
                    if kind == "single":
                        r = await ask_model(
                            GEN_SINGLE.format(heading=ch["heading_path"], text=ch["text"])
                        )
                        out.append(
                            Candidate(
                                id=f"g{i:03d}",
                                question=r["question"],
                                answer_key_points=r.get("key_points", []),
                                source_chunk_content_hashes=[ch["content_hash"]],
                                type="single",
                                chapter=chapter,
                                generated_by=model,
                            )
                        )
                    elif kind == "multi":
                        sib = await _sibling(engine, ch, rng)
                        if not sib:
                            return
                        r = await ask_model(
                            GEN_MULTI.format(
                                heading1=ch["heading_path"],
                                text1=ch["text"],
                                heading2=sib["heading_path"],
                                text2=sib["text"],
                            )
                        )
                        out.append(
                            Candidate(
                                id=f"g{i:03d}",
                                question=r["question"],
                                answer_key_points=r.get("key_points", []),
                                source_chunk_content_hashes=[
                                    ch["content_hash"],
                                    sib["content_hash"],
                                ],
                                type="multi",
                                chapter=chapter,
                                generated_by=model,
                            )
                        )
                    else:
                        r = await ask_model(
                            GEN_UNANSWERABLE.format(heading=ch["heading_path"], text=ch["text"])
                        )
                        out.append(
                            Candidate(
                                id=f"g{i:03d}",
                                question=r["question"],
                                answer_key_points=[],
                                source_chunk_content_hashes=[],
                                type="unanswerable",
                                chapter=chapter,
                                generated_by=model,
                                notes=r.get("why_unanswerable", ""),
                            )
                        )
                except Exception as e:  # keep going; the human pass filters
                    console.print(f"[yellow]skip {i}: {e}")

        await asyncio.gather(*(one(i, ch) for i, ch in enumerate(chunks)))
        await engine.dispose()
        return sorted(out, key=lambda c: c.id)

    cands = asyncio.run(go())
    write_jsonl(CANDIDATES_PATH, cands)
    console.print(
        f"wrote {len(cands)} candidates -> {CANDIDATES_PATH} (gitignored; run `make golden-verify`)"
    )
    console.print(Counter(c.type for c in cands))


@app.command()
def verify(
    resume: bool = typer.Option(True, help="skip candidates already in golden.jsonl"),
) -> None:
    """Interactive human verification: accept / edit / reject each candidate with its source chunk(s)."""
    cands = read_jsonl(CANDIDATES_PATH, Candidate)
    if not cands:
        raise typer.BadParameter(f"no candidates at {CANDIDATES_PATH}; run `make golden-generate`")
    golden = read_jsonl(GOLDEN_PATH, GoldenItem) if resume else []
    done_ids = {g.id for g in golden}
    todo = [c for c in cands if c.id not in done_ids]
    console.print(f"{len(todo)} candidates to review ({len(golden)} already verified)")

    async def run() -> None:
        # one event loop for the whole session: the async engine's pool is bound to it
        engine = make_engine()

        async def chunk_text(h: str) -> tuple[str, str]:
            async with engine.connect() as conn:
                r = (
                    await conn.execute(
                        text("SELECT heading_path, text FROM chunks WHERE content_hash = :h"),
                        {"h": h},
                    )
                ).first()
            return (r[0], r[1]) if r else ("<missing>", "<chunk not found - re-ingested?>")

        try:
            for i, c in enumerate(todo):
                console.rule(f"[{i + 1}/{len(todo)}] {c.id}  type={c.type}  {c.chapter}")
                for h in c.source_chunk_content_hashes:
                    hp, txt = await chunk_text(h)
                    console.print(
                        Panel(escape(txt[:2500]), title=escape(hp), subtitle=h[:12], expand=False)
                    )
                console.print(f"[bold cyan]Q:[/] {escape(c.question)}")
                for kp in c.answer_key_points:
                    console.print(f"   • {escape(kp)}")
                if c.notes:
                    console.print(f"   [dim]{escape(c.notes)}[/]")
                choice = Prompt.ask(
                    r"\[a]ccept / \[e]dit / \[r]eject / \[q]uit",
                    choices=["a", "e", "r", "q"],
                    default="a",
                )
                if choice == "q":
                    break
                if choice == "r":
                    continue
                item = GoldenItem(
                    **{k: v for k, v in c.model_dump().items() if k in GoldenItem.model_fields}
                )
                if choice == "e":
                    item.question = Prompt.ask("question", default=item.question)
                    kps = Prompt.ask(
                        "key points (paraphrase; separate with ' | ')",
                        default=" | ".join(item.answer_key_points),
                    )
                    item.answer_key_points = [k.strip() for k in kps.split("|") if k.strip()]
                    item.type = Prompt.ask(
                        "type", choices=["single", "multi", "unanswerable"], default=item.type
                    )  # type: ignore[assignment]
                item.verified = True
                golden.append(item)
                write_jsonl(GOLDEN_PATH, golden)  # save progress after every item
        finally:
            await engine.dispose()

    asyncio.run(run())
    write_jsonl(GOLDEN_PATH, golden)
    stamp = write_stamp(GOLDEN_PATH, STAMP_PATH)
    console.print(
        f"[green]verified set: {stamp['count']} items {stamp['by_type']} -> {GOLDEN_PATH}; stamp {STAMP_PATH}"
    )


@app.command()
def stats() -> None:
    items = read_jsonl(GOLDEN_PATH, GoldenItem)
    console.print(
        {
            "count": len(items),
            "by_type": Counter(i.type for i in items),
            "by_chapter": Counter(i.chapter for i in items),
        }
    )


if __name__ == "__main__":
    app()
