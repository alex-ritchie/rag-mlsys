"""`make eval`: full run -> eval/results/<run-id>/{report.json,summary.md}; copy to eval/results/latest/."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from mlsys_common.settings import REPO_ROOT, get_settings
from mlsys_gateway.app import build_deps
from sqlalchemy import text

from mlsys_eval.harness import judge_answers, run_generation, run_retrieval
from mlsys_eval.judge import AnthropicJudge, FakeJudge
from mlsys_eval.judge_validate import validate as validate_judge
from mlsys_eval.report import EvalReport, PerQuestion, render_markdown
from mlsys_eval.schema import JUDGE_LABELS_PATH, load_verified_golden

RESULTS = REPO_ROOT / "eval" / "results"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


async def main_async(
    retrieval_only: bool,
    fake_judge: bool,
    limit: int | None,
    run_id: str | None,
    golden: str | None = None,
    stamp: str | None = None,
    update_latest: bool = True,
) -> EvalReport:
    s = get_settings()
    items = (
        load_verified_golden(Path(golden), Path(stamp))
        if golden and stamp
        else load_verified_golden()
    )
    if limit:
        items = items[:limit]
    deps = build_deps(s)
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{s.llm_model}"
    async with deps.engine.connect() as conn:
        idx_sha = (await conn.execute(text("SELECT min(commit_sha) FROM chunks"))).scalar()
    print(f"run {run_id}: {len(items)} golden items, model={deps.llm.model}, index={idx_sha}")

    retrieval = await run_retrieval(deps, items)
    for row in retrieval:
        print(
            f"  {row.config:14s} R@5={row.recall_at_5:.3f} R@10={row.recall_at_10:.3f} R@30={row.recall_at_30:.3f} MRR={row.mrr:.3f}"
        )

    gen = None
    per_q: list[PerQuestion] = []
    agreement = None
    judge_model = "none"
    if not retrieval_only:
        judge_client = (
            FakeJudge()
            if fake_judge or not s.anthropic_api_key
            else AnthropicJudge(s.judge_model, s.anthropic_api_key)
        )
        judge_model = judge_client.model
        answered = await run_generation(deps, items)
        gen, per_q = await judge_answers(judge_client, answered)
        print(
            f"  faithfulness={gen.faithfulness_mean:.2f} relevance={gen.relevance_mean:.2f} groundedness={gen.groundedness_mean:.2f} abstention={gen.abstention}"
        )
        if JUDGE_LABELS_PATH.exists():
            agreement = await validate_judge(judge_client)
    await deps.engine.dispose()

    report = EvalReport(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        git_sha=git_sha(),
        index_commit_sha=idx_sha,
        model=deps.llm.model,
        prompt_version=s.prompt_version,
        judge_model=judge_model,
        golden_count=len(items),
        golden_by_type={
            t: sum(i.type == t for i in items) for t in ("single", "multi", "unanswerable")
        },
        retrieval=retrieval,
        generation=gen,
        judge_agreement=agreement,
        per_question=per_q,
        config={
            "retrieval_mode": s.retrieval_mode,
            "retrieval_top_n": s.retrieval_top_n,
            "rerank_top_k": s.rerank_top_k,
            "reranker": deps.reranker is not None,
            "max_output_tokens": s.max_output_tokens,
            "disable_thinking": s.disable_thinking,
        },
    )
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(report.model_dump_json(indent=2))
    (out / "summary.md").write_text(render_markdown(report))
    if update_latest:
        latest = RESULTS / "latest"
        latest.mkdir(exist_ok=True)
        shutil.copy(out / "report.json", latest / "report.json")
        shutil.copy(out / "summary.md", latest / "summary.md")
    print(
        f"wrote {out}/report.json and summary.md"
        + (" (copied to eval/results/latest/)" if update_latest else "")
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument(
        "--fake-judge",
        action="store_true",
        help="deterministic lexical judge (plumbing check, not a real score)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--golden", default=None, help="alternate golden.jsonl (with --stamp)")
    ap.add_argument("--stamp", default=None)
    ap.add_argument(
        "--no-latest",
        action="store_true",
        help="do not copy into eval/results/latest/ (plumbing runs)",
    )
    a = ap.parse_args()
    asyncio.run(
        main_async(
            a.retrieval_only, a.fake_judge, a.limit, a.run_id, a.golden, a.stamp, not a.no_latest
        )
    )


if __name__ == "__main__":
    main()
