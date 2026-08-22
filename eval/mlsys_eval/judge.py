"""Thin LLM-as-judge (spec §5.6): three prompts, structured verdicts, Claude Haiku by default.

Each judge returns {"score": int 1-5, "pass": bool, "rationale": str}. `faithfulness` and
`relevance` apply to answered questions; `groundedness` judges abstention correctness: did the
answer stay inside the provided context (or correctly abstain)?
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

JUDGE_PROMPTS: dict[str, str] = {
    "faithfulness": """You are grading a retrieval-augmented answer about a textbook.
Context blocks (numbered) are the ONLY admissible evidence.

<context>
{context}
</context>

<question>{question}</question>
<answer>{answer}</answer>

Task: Is every factual claim in the answer supported by the context blocks? Ignore citation formatting; judge content.
Score 5 = fully supported; 3 = mostly supported with a minor unsupported detail; 1 = contains claims not in the context or contradicting it.
Respond with JSON only: {{"score": <1-5>, "pass": <true if score >= 4>, "rationale": "<one sentence>"}}""",
    "relevance": """You are grading whether an answer addresses the question asked.

<question>{question}</question>
<answer>{answer}</answer>
<key_points>{key_points}</key_points>

Task: Does the answer address the question and cover the key points? Do not reward padding or unrelated material.
Score 5 = directly answers and covers the key points; 3 = partially; 1 = off-topic or empty.
Respond with JSON only: {{"score": <1-5>, "pass": <true if score >= 4>, "rationale": "<one sentence>"}}""",
    "groundedness": """You are auditing whether a RAG system stayed inside its evidence.

<context>
{context}
</context>

<question>{question}</question>
<answer>{answer}</answer>
<expected_behavior>{expected}</expected_behavior>

Task: If the context does not cover the question, the correct behavior is to say the book does not cover it (an abstention). If it does, the answer must use only the context.
Score 5 = behavior exactly right (correct abstention, or grounded answer); 3 = right behavior with small leakage; 1 = wrong behavior (hallucinated an answer to an uncovered question, or abstained although the context clearly answers it).
Respond with JSON only: {{"score": <1-5>, "pass": <true if score >= 4>, "rationale": "<one sentence>"}}""",
}


@dataclass
class Verdict:
    score: int
    passed: bool
    rationale: str
    raw: str = ""


class JudgeClient(Protocol):
    model: str

    async def complete(self, prompt: str) -> str: ...


class AnthropicJudge:
    def __init__(self, model: str, api_key: str) -> None:
        from anthropic import AsyncAnthropic

        self.model = model
        self._c = AsyncAnthropic(api_key=api_key)

    async def complete(self, prompt: str) -> str:
        msg = await self._c.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(b, "text", "") for b in msg.content)


class FakeJudge:
    """Scores by lexical overlap: enough to unit-test the harness plumbing deterministically."""

    model = "fake-judge"

    async def complete(self, prompt: str) -> str:
        ans = re.search(r"<answer>(.*?)</answer>", prompt, re.S)
        ctx = re.search(r"<context>(.*?)</context>", prompt, re.S)
        a = set((ans.group(1) if ans else "").lower().split())
        c = set((ctx.group(1) if ctx else "").lower().split())
        overlap = len(a & c) / (len(a) or 1)
        score = 5 if overlap > 0.6 else 3 if overlap > 0.3 else 1
        if "does not appear to cover" in (ans.group(1) if ans else "").lower():
            score = 5 if "abstain" in prompt.lower() else 2
        return json.dumps(
            {"score": score, "pass": score >= 4, "rationale": f"overlap={overlap:.2f}"}
        )


def parse_verdict(text: str) -> Verdict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return Verdict(score=1, passed=False, rationale="unparseable judge output", raw=text)
    try:
        obj = json.loads(m.group(0))
        score = int(obj.get("score", 1))
        return Verdict(
            score=score,
            passed=bool(obj.get("pass", score >= 4)),
            rationale=str(obj.get("rationale", "")),
            raw=text,
        )
    except (ValueError, TypeError):
        return Verdict(score=1, passed=False, rationale="invalid judge JSON", raw=text)


def format_context(chunks: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[{i + 1}] ({hp})\n{txt}" for i, (hp, txt) in enumerate(chunks))


async def judge(
    client: JudgeClient,
    kind: str,
    *,
    question: str,
    answer: str,
    context: str = "",
    key_points: list[str] | None = None,
    expected: str = "",
) -> Verdict:
    prompt = JUDGE_PROMPTS[kind].format(
        context=context,
        question=question,
        answer=answer,
        key_points="; ".join(key_points or []),
        expected=expected,
    )
    return parse_verdict(await client.complete(prompt))
