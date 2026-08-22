"""Evaluation harness core (spec §5.6): retrieval three-row ablation + generation judging + abstention."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from mlsys_common.models import AskRequest, LatencyBreakdown
from mlsys_gateway.pipeline import Deps, ask, retrieve_and_rerank
from sqlalchemy import text

from mlsys_eval.judge import JudgeClient, format_context, judge
from mlsys_eval.metrics import abstention_scores, mean, mrr, percentile, recall_at_k
from mlsys_eval.report import GenerationSummary, PerQuestion, RetrievalRow
from mlsys_eval.schema import GoldenItem


async def resolve_hashes(engine, items: list[GoldenItem]) -> dict[str, int | None]:
    """content_hash -> current chunk id (labels survive re-ingestion)."""
    hashes = sorted({h for it in items for h in it.source_chunk_content_hashes})
    if not hashes:
        return {}
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT content_hash, id FROM chunks WHERE content_hash = ANY(:hs)"),
                {"hs": hashes},
            )
        ).all()
    found = {r[0]: r[1] for r in rows}
    return {h: found.get(h) for h in hashes}


@dataclass
class RetrievalRun:
    config: str
    per_item: dict[str, list[str]] = field(
        default_factory=dict
    )  # item id -> retrieved hashes (ordered)


async def _retrieve_all(
    deps: Deps, items: list[GoldenItem], *, mode: str, use_rerank: bool, sem: asyncio.Semaphore
) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}

    async def one(it: GoldenItem) -> None:
        async with sem:
            lat = LatencyBreakdown()
            fused, reranked = await retrieve_and_rerank(
                deps, it.question, mode=mode, top_n=30, top_k=30, lat=lat, rerank=use_rerank
            )
            results[it.id] = [c.content_hash for c in (reranked if use_rerank else fused)]

    await asyncio.gather(*(one(it) for it in items))
    return results


async def run_retrieval(
    deps: Deps,
    items: list[GoldenItem],
    configs: tuple[str, ...] = ("dense", "hybrid", "hybrid+rerank"),
    concurrency: int = 4,
) -> list[RetrievalRow]:
    answerable = [
        it for it in items if it.type != "unanswerable" and it.source_chunk_content_hashes
    ]
    rows: list[RetrievalRow] = []
    sem = asyncio.Semaphore(concurrency)
    for cfg in configs:
        mode = "dense" if cfg == "dense" else "hybrid"
        use_rerank = cfg.endswith("+rerank")
        results = await _retrieve_all(deps, answerable, mode=mode, use_rerank=use_rerank, sem=sem)
        by_type: dict[str, dict[str, list[float]]] = {}
        r5, r10, r30, mr = [], [], [], []
        for it in answerable:
            rel = set(it.source_chunk_content_hashes)
            got = results.get(it.id, [])
            vals = {
                "recall_at_5": recall_at_k(got, rel, 5),
                "recall_at_10": recall_at_k(got, rel, 10),
                "recall_at_30": recall_at_k(got, rel, 30),
                "mrr": mrr(got, rel),
            }
            r5.append(vals["recall_at_5"])
            r10.append(vals["recall_at_10"])
            r30.append(vals["recall_at_30"])
            mr.append(vals["mrr"])
            bt = by_type.setdefault(it.type, {k: [] for k in vals})
            for k, v in vals.items():
                bt[k].append(v)
        rows.append(
            RetrievalRow(
                config=cfg,
                recall_at_5=mean(r5),
                recall_at_10=mean(r10),
                recall_at_30=mean(r30),
                mrr=mean(mr),
                n=len(answerable),
                by_type={t: {k: mean(v) for k, v in m.items()} for t, m in by_type.items()},
            )
        )
    return rows


@dataclass
class Answered:
    item: GoldenItem
    answer: str
    abstained: bool
    context: list[tuple[str, str]]
    latency_ms: float
    cited_hashes: list[str]


async def run_generation(
    deps: Deps, items: list[GoldenItem], concurrency: int = 4
) -> list[Answered]:
    sem = asyncio.Semaphore(concurrency)
    out: list[Answered] = []

    async def one(it: GoldenItem) -> None:
        async with sem:
            parts: list[str] = []
            cits: list[dict] = []
            done: dict = {}
            async for ev in ask(deps, AskRequest(question=it.question)):
                if ev.event == "citations":
                    cits = ev.data
                elif ev.event == "token":
                    parts.append(ev.data)
                elif ev.event == "done":
                    done = ev.data
            # fetch the full context texts for the judges
            ids = [c["chunk_id"] for c in cits]
            ctx: list[tuple[str, str]] = []
            hashes: list[str] = []
            if ids:
                async with deps.engine.connect() as conn:
                    rows = (
                        await conn.execute(
                            text(
                                "SELECT id, heading_path, text, content_hash FROM chunks WHERE id = ANY(:ids)"
                            ),
                            {"ids": ids},
                        )
                    ).all()
                by_id = {r[0]: r for r in rows}
                for i in ids:
                    if i in by_id:
                        ctx.append((by_id[i][1], by_id[i][2]))
                        hashes.append(by_id[i][3])
            out.append(
                Answered(
                    it,
                    "".join(parts),
                    bool(done.get("abstained")),
                    ctx,
                    float(done.get("latency_breakdown", {}).get("total_ms", 0.0)),
                    hashes,
                )
            )

    await asyncio.gather(*(one(it) for it in items))
    return sorted(out, key=lambda a: a.item.id)


async def judge_answers(
    client: JudgeClient, answered: list[Answered], concurrency: int = 8
) -> tuple[GenerationSummary, list[PerQuestion]]:
    sem = asyncio.Semaphore(concurrency)
    per: dict[str, PerQuestion] = {}

    async def one(a: Answered) -> None:
        async with sem:
            ctx = format_context(a.context)
            expected = (
                "abstain: the book does not cover this"
                if a.item.type == "unanswerable"
                else "answer from the context"
            )
            g = await judge(
                client,
                "groundedness",
                question=a.item.question,
                answer=a.answer,
                context=ctx,
                expected=expected,
            )
            pq = PerQuestion(
                id=a.item.id,
                type=a.item.type,
                chapter=a.item.chapter,
                question=a.item.question,
                abstained=a.abstained,
                groundedness=g.score,
                latency_ms=a.latency_ms,
                recall_at_5_rerank=recall_at_k(
                    a.cited_hashes, set(a.item.source_chunk_content_hashes), 5
                )
                if a.item.source_chunk_content_hashes
                else None,
            )
            if a.item.type != "unanswerable":
                f = await judge(
                    client, "faithfulness", question=a.item.question, answer=a.answer, context=ctx
                )
                r = await judge(
                    client,
                    "relevance",
                    question=a.item.question,
                    answer=a.answer,
                    key_points=a.item.answer_key_points,
                )
                pq.faithfulness, pq.relevance = f.score, r.score
            per[a.item.id] = pq

    await asyncio.gather(*(one(a) for a in answered))
    pqs = [per[a.item.id] for a in answered]
    answerable = [p for p in pqs if p.type != "unanswerable"]
    f_scores = [p.faithfulness for p in answerable if p.faithfulness is not None]
    r_scores = [p.relevance for p in answerable if p.relevance is not None]
    g_scores = [p.groundedness for p in pqs if p.groundedness is not None]
    abst = abstention_scores(
        [p.type == "unanswerable" for p in pqs], [bool(p.abstained) for p in pqs]
    )
    by_type: dict[str, dict[str, float]] = {}
    for t in sorted({p.type for p in pqs}):
        ps = [p for p in pqs if p.type == t]
        by_type[t] = {
            "n": float(len(ps)),
            "groundedness_mean": mean([p.groundedness for p in ps if p.groundedness is not None]),
            "faithfulness_mean": mean([p.faithfulness for p in ps if p.faithfulness is not None]),
            "relevance_mean": mean([p.relevance for p in ps if p.relevance is not None]),
            "abstention_rate": mean([1.0 if p.abstained else 0.0 for p in ps]),
        }
    lat = [p.latency_ms for p in pqs if p.latency_ms]
    summary = GenerationSummary(
        model=client.model,
        n=len(pqs),
        faithfulness_mean=mean(f_scores),
        faithfulness_pass_rate=mean([1.0 if s >= 4 else 0.0 for s in f_scores]),
        relevance_mean=mean(r_scores),
        relevance_pass_rate=mean([1.0 if s >= 4 else 0.0 for s in r_scores]),
        groundedness_mean=mean(g_scores),
        groundedness_pass_rate=mean([1.0 if s >= 4 else 0.0 for s in g_scores]),
        abstention=abst,
        by_type=by_type,
        latency_ms={
            "p50": percentile(lat, 50),
            "p90": percentile(lat, 90),
            "p99": percentile(lat, 99),
        },
    )
    return summary, pqs
