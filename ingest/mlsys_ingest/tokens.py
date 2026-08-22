"""Token counting in the bge-m3 tokenizer (spec §5.1), with an offline approximation for tests."""

from __future__ import annotations

import re
from typing import Protocol


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class ApproxTokenCounter:
    """Deterministic offline estimate (~XLM-R on English prose: ~1.35 tokens/word)."""

    name = "approx"
    _word = re.compile(r"\w+|[^\w\s]")

    def count(self, text: str) -> int:
        return int(len(self._word.findall(text)) * 1.35) + 1


class HFTokenCounter:
    """Exact counts via the `tokenizers` fast tokenizer (downloads tokenizer.json only, ~17 MB)."""

    def __init__(self, repo: str = "BAAI/bge-m3") -> None:
        from tokenizers import Tokenizer

        self.name = repo
        self._tok = Tokenizer.from_pretrained(repo)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)


def get_counter(repo: str | None = "BAAI/bge-m3", allow_approx: bool = True) -> TokenCounter:
    if repo:
        try:
            return HFTokenCounter(repo)
        except Exception as e:  # offline, or extra not installed
            if not allow_approx:
                raise
            print(f"warning: using approximate token counter ({type(e).__name__}: {e})")
    return ApproxTokenCounter()
