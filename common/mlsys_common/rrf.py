"""Reciprocal Rank Fusion (spec §5.2): score(d) = sum_i 1 / (k + rank_i(d))."""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def rrf[T: Hashable](rankings: Sequence[Sequence[T]], k: int = 60) -> list[tuple[T, float]]:
    """Fuse ranked lists. Each inner sequence is ordered best-first. Returns (item, score) best-first."""
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
