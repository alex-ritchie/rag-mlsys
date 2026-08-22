"""Judge validation (spec §5.6): 30 hand-labeled examples -> percent agreement + Cohen's kappa.

eval/golden/judge_labels.jsonl rows: {"id", "judge": "faithfulness|relevance|groundedness", "question", "answer",
"context": [["heading", "text"], ...] OR "context_hashes": [...], "key_points": [...], "expected": "...", "human_pass": true/false}
Context given inline must be paraphrased/synthetic or resolved from hashes at run time (no book text committed).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mlsys_common.db import make_engine
from mlsys_common.settings import get_settings
from sqlalchemy import text

from mlsys_eval.judge import AnthropicJudge, FakeJudge, JudgeClient, format_context, judge
from mlsys_eval.metrics import cohen_kappa, percent_agreement
from mlsys_eval.report import JudgeAgreement
from mlsys_eval.schema import JUDGE_LABELS_PATH


async def _resolve(engine, hashes: list[str]) -> list[tuple[str, str]]:
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT heading_path, text FROM chunks WHERE content_hash = ANY(:hs)"),
                {"hs": hashes},
            )
        ).all()
    return [(r[0], r[1]) for r in rows]


async def validate(
    client: JudgeClient, labels_path: Path = JUDGE_LABELS_PATH, threshold: float = 0.8
) -> JudgeAgreement:
    rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    engine = make_engine()
    human: list[int] = []
    machine: list[int] = []
    per: dict[str, dict[str, list[int]]] = {}
    for r in rows:
        ctx = r.get("context") or []
        if r.get("context_hashes"):
            ctx = await _resolve(engine, r["context_hashes"])
        v = await judge(
            client,
            r["judge"],
            question=r["question"],
            answer=r["answer"],
            context=format_context([tuple(c) for c in ctx]),
            key_points=r.get("key_points"),
            expected=r.get("expected", ""),
        )
        h, m = int(bool(r["human_pass"])), int(v.passed)
        human.append(h)
        machine.append(m)
        d = per.setdefault(r["judge"], {"h": [], "m": []})
        d["h"].append(h)
        d["m"].append(m)
    await engine.dispose()
    agreement = percent_agreement(human, machine)
    return JudgeAgreement(
        n=len(rows),
        percent_agreement=agreement,
        cohen_kappa=cohen_kappa(human, machine),
        per_judge={
            k: {
                "n": float(len(v["h"])),
                "agreement": percent_agreement(v["h"], v["m"]),
                "kappa": cohen_kappa(v["h"], v["m"]),
            }
            for k, v in per.items()
        },
        trusted=agreement >= threshold,
    )


def main() -> None:
    s = get_settings()
    client = (
        AnthropicJudge(s.judge_model, s.anthropic_api_key) if s.anthropic_api_key else FakeJudge()
    )
    res = asyncio.run(validate(client))
    print(res.model_dump_json(indent=2))
    if not res.trusted:
        print(
            "agreement < 80%: iterate on judge prompts before trusting generation metrics (spec §5.6)"
        )


if __name__ == "__main__":
    main()
