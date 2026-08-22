"""Versioned prompt templates (spec §5.5). The version string is logged per query."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from mlsys_common.models import RetrievedChunk

PROMPTS_DIR = Path(__file__).parent
ABSTAIN_PHRASE = "The book does not appear to cover this"


@lru_cache
def load_prompt(version: str) -> dict[str, str]:
    d = PROMPTS_DIR / version
    if not d.is_dir():
        raise FileNotFoundError(f"prompt version {version!r} not found under {PROMPTS_DIR}")
    return {p.stem: p.read_text() for p in d.glob("*.txt")}


def build_messages(
    question: str, chunks: list[RetrievedChunk], version: str = "v1"
) -> list[dict[str, str]]:
    tpl = load_prompt(version)
    blocks = "\n\n".join(
        tpl["block"].format(n=i + 1, heading_path=c.heading_path, text=c.text).rstrip()
        for i, c in enumerate(chunks)
    )
    return [
        {"role": "system", "content": tpl["system"].strip()},
        {
            "role": "user",
            "content": tpl["user"].format(context=blocks, question=question.strip()).strip(),
        },
    ]


def is_abstention(answer: str) -> bool:
    return ABSTAIN_PHRASE.lower() in answer.strip().lower()[:200]
