"""Report model + markdown rendering. report.json contains questions, scores, paraphrased key points — no book text."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalRow(BaseModel):
    config: str  # dense | hybrid | hybrid+rerank
    recall_at_5: float
    recall_at_10: float
    recall_at_30: float
    mrr: float
    by_type: dict[str, dict[str, float]] = Field(default_factory=dict)
    n: int = 0


class GenerationSummary(BaseModel):
    model: str
    n: int
    faithfulness_mean: float
    faithfulness_pass_rate: float
    relevance_mean: float
    relevance_pass_rate: float
    groundedness_mean: float
    groundedness_pass_rate: float
    abstention: dict[str, float]
    by_type: dict[str, dict[str, float]] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)


class JudgeAgreement(BaseModel):
    n: int
    percent_agreement: float
    cohen_kappa: float
    per_judge: dict[str, dict[str, float]] = Field(default_factory=dict)
    trusted: bool


class PerQuestion(BaseModel):
    id: str
    type: str
    chapter: str
    question: str
    abstained: bool | None = None
    faithfulness: int | None = None
    relevance: int | None = None
    groundedness: int | None = None
    recall_at_5_rerank: float | None = None
    latency_ms: float | None = None


class EvalReport(BaseModel):
    run_id: str
    created_at: str
    git_sha: str
    index_commit_sha: str | None
    model: str
    prompt_version: str
    judge_model: str
    golden_count: int
    golden_by_type: dict[str, int]
    retrieval: list[RetrievalRow]
    generation: GenerationSummary | None
    judge_agreement: JudgeAgreement | None
    per_question: list[PerQuestion] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_markdown(r: EvalReport) -> str:
    out = [f"# Eval report `{r.run_id}`", ""]
    out += [
        f"- model: `{r.model}` · prompt `{r.prompt_version}` · judge `{r.judge_model}`",
        f"- index commit `{(r.index_commit_sha or '?')[:10]}` · code `{r.git_sha[:10]}`",
        f"- golden set: {r.golden_count} questions {r.golden_by_type}",
        "",
    ]
    out += [
        "## Retrieval",
        "",
        "| config | R@5 | R@10 | R@30 | MRR | n |",
        "|---|---|---|---|---|---|",
    ]
    for row in r.retrieval:
        out.append(
            f"| {row.config} | {_pct(row.recall_at_5)} | {_pct(row.recall_at_10)} | {_pct(row.recall_at_30)} | {row.mrr:.3f} | {row.n} |"
        )
    for row in r.retrieval:
        if row.by_type:
            out += [
                "",
                f"### {row.config} by question type",
                "",
                "| type | R@5 | R@10 | R@30 | MRR |",
                "|---|---|---|---|---|",
            ]
            for t, m in row.by_type.items():
                out.append(
                    f"| {t} | {_pct(m['recall_at_5'])} | {_pct(m['recall_at_10'])} | {_pct(m['recall_at_30'])} | {m['mrr']:.3f} |"
                )
    if r.generation:
        g = r.generation
        out += [
            "",
            "## Generation (LLM-as-judge)",
            "",
            "| metric | mean (1-5) | pass rate |",
            "|---|---|---|",
        ]
        out.append(
            f"| faithfulness | {g.faithfulness_mean:.2f} | {_pct(g.faithfulness_pass_rate)} |"
        )
        out.append(f"| relevance | {g.relevance_mean:.2f} | {_pct(g.relevance_pass_rate)} |")
        out.append(
            f"| groundedness | {g.groundedness_mean:.2f} | {_pct(g.groundedness_pass_rate)} |"
        )
        a = g.abstention
        out += [
            "",
            f"**Abstention** (positive = abstained): precision {_pct(a['precision'])}, recall {_pct(a['recall'])}, F1 {_pct(a['f1'])}; hallucination rate on unanswerable {_pct(a['hallucination_rate_on_unanswerable'])}; false abstention rate {_pct(a['false_abstention_rate'])}",
        ]
        if g.latency_ms:
            out += [
                "",
                "Latency (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in g.latency_ms.items()),
            ]
    if r.judge_agreement:
        j = r.judge_agreement
        out += [
            "",
            "## Judge validation",
            "",
            f"{j.n} hand-labeled examples: agreement {_pct(j.percent_agreement)}, Cohen's κ {j.cohen_kappa:.3f} → {'trusted' if j.trusted else 'NOT trusted (< 80%) — iterate on judge prompts'}",
        ]
    return "\n".join(out) + "\n"
