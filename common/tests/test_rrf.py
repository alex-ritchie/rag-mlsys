from mlsys_common.hashing import content_hash
from mlsys_common.rrf import rrf


def test_rrf_prefers_items_in_both_lists():
    fused = rrf([["a", "b", "c"], ["c", "d", "a"]], k=60)
    order = [x for x, _ in fused]
    assert order[0] in {"a", "c"}
    assert set(order[:2]) == {"a", "c"}
    assert order[-1] in {"b", "d"}


def test_rrf_scores_match_formula():
    fused = dict(rrf([["a"], ["a"]], k=60))
    assert abs(fused["a"] - 2 / 61) < 1e-12


def test_rrf_deterministic_tiebreak():
    assert rrf([["b"], ["a"]]) == rrf([["b"], ["a"]])


def test_content_hash_stable():
    assert content_hash("x") == content_hash("x")
    assert len(content_hash("x")) == 32
