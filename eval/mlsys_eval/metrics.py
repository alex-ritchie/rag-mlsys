"""Metric math (spec §5.6). Pure functions; unit-tested on synthetic fixtures in CI."""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def abstention_scores(
    should_abstain: Sequence[bool], did_abstain: Sequence[bool]
) -> dict[str, float]:
    """Positive class = abstained. FP = abstained on an answerable question; FN = answered an unanswerable one (hallucination)."""
    tp = sum(s and d for s, d in zip(should_abstain, did_abstain, strict=True))
    fp = sum((not s) and d for s, d in zip(should_abstain, did_abstain, strict=True))
    fn = sum(s and (not d) for s, d in zip(should_abstain, did_abstain, strict=True))
    out = precision_recall_f1(tp, fp, fn)
    n_unans = sum(should_abstain)
    out["hallucination_rate_on_unanswerable"] = fn / n_unans if n_unans else 0.0
    out["false_abstention_rate"] = (
        fp / (len(should_abstain) - n_unans) if len(should_abstain) - n_unans else 0.0
    )
    return out


def percent_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else 0.0


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for two raters over nominal labels."""
    n = len(a)
    if n == 0:
        return 0.0
    labels = sorted(set(a) | set(b))
    po = percent_agreement(a, b)
    pe = sum((sum(x == lab for x in a) / n) * (sum(y == lab for y in b) / n) for lab in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, round((p / 100) * (len(s) - 1))))
    return s[idx]
