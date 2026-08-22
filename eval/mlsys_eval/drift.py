"""Nightly groundedness CronJob (spec §5.8): sample N recent answers from query_log, judge, write judge_scores."""

from __future__ import annotations

import argparse
import asyncio

from mlsys_common.db import make_engine
from mlsys_common.settings import get_settings
from sqlalchemy import text

from mlsys_eval.judge import AnthropicJudge, FakeJudge, format_context, judge


async def run(n: int, fake: bool) -> None:
    s = get_settings()
    client = (
        FakeJudge()
        if fake or not s.anthropic_api_key
        else AnthropicJudge(s.judge_model, s.anthropic_api_key)
    )
    engine = make_engine()
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    """SELECT q.id, q.question, q.answer, q.reranked_ids FROM query_log q
                       WHERE q.finished_at > now() - interval '24 hours' AND q.error IS NULL AND q.answer IS NOT NULL
                         AND NOT EXISTS (SELECT 1 FROM judge_scores j WHERE j.query_log_id = q.id)
                       ORDER BY random() LIMIT :n"""
                ),
                {"n": n},
            )
        ).all()
    print(f"judging {len(rows)} answers with {client.model}")
    for qid, question, answer, ids in rows:
        async with engine.connect() as conn:
            ch = (
                await conn.execute(
                    text("SELECT heading_path, text FROM chunks WHERE id = ANY(:ids)"),
                    {"ids": list(ids or [])},
                )
            ).all()
        ctx = format_context([(c[0], c[1]) for c in ch])
        g = await judge(
            client,
            "groundedness",
            question=question,
            answer=answer,
            context=ctx,
            expected="answer from the context, or abstain if it is not covered",
        )
        f = await judge(client, "faithfulness", question=question, answer=answer, context=ctx)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO judge_scores (query_log_id, judge_model, faithfulness, groundedness, flagged, rationale) VALUES (:q, :m, :f, :g, :fl, :r)"
                ),
                {
                    "q": qid,
                    "m": client.model,
                    "f": f.score / 5,
                    "g": g.score / 5,
                    "fl": not (g.passed and f.passed),
                    "r": f"g: {g.rationale} | f: {f.rationale}",
                },
            )
    await engine.dispose()
    print("done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--fake-judge", action="store_true")
    a = ap.parse_args()
    asyncio.run(run(a.n, a.fake_judge))


if __name__ == "__main__":
    main()
