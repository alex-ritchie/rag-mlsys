import json

import pytest
from mlsys_eval.judge import FakeJudge, parse_verdict
from mlsys_eval.metrics import (
    abstention_scores,
    cohen_kappa,
    mrr,
    percent_agreement,
    percentile,
    recall_at_k,
)
from mlsys_eval.report import (
    EvalReport,
    GenerationSummary,
    JudgeAgreement,
    RetrievalRow,
    render_markdown,
)
from mlsys_eval.schema import (
    GoldenItem,
    UnverifiedGoldenSetError,
    load_verified_golden,
    write_jsonl,
    write_stamp,
)


def test_recall_and_mrr():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 1) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert recall_at_k([], {"a"}, 5) == 0.0
    assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert mrr(["x"], {"a"}) == 0.0


def test_abstention_scores():
    should = [True, True, False, False, True]
    did = [True, False, False, True, True]
    s = abstention_scores(should, did)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["hallucination_rate_on_unanswerable"] == pytest.approx(1 / 3)
    assert s["false_abstention_rate"] == pytest.approx(1 / 2)


def test_kappa_and_agreement():
    assert percent_agreement([1, 0, 1], [1, 0, 1]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    # textbook example: po=0.8, pe=0.5 -> kappa 0.6
    a = [1] * 5 + [0] * 5
    b = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    assert percent_agreement(a, b) == pytest.approx(0.8)
    assert cohen_kappa(a, b) == pytest.approx(0.6)
    assert percentile([5, 1, 3], 50) == 3


async def test_fake_judge_and_parse():
    j = FakeJudge()
    out = await j.complete(
        "<context>alpha beta gamma delta</context><answer>alpha beta gamma</answer>"
    )
    v = parse_verdict(out)
    assert v.score == 5 and v.passed
    assert parse_verdict("garbage").passed is False
    assert parse_verdict('prefix {"score": 2, "pass": false, "rationale": "x"} suffix').score == 2


def test_verified_golden_gate(tmp_path):
    g = tmp_path / "golden.jsonl"
    st = tmp_path / "stamp.json"
    items = [
        GoldenItem(
            id="g1",
            question="q?",
            answer_key_points=["kp"],
            source_chunk_content_hashes=["h1"],
            type="single",
            chapter="vol1/ch1",
            verified=True,
        )
    ]
    write_jsonl(g, items)
    with pytest.raises(UnverifiedGoldenSetError):
        load_verified_golden(g, st)  # no stamp
    write_stamp(g, st)
    assert len(load_verified_golden(g, st)) == 1
    items[0].question = "changed?"
    write_jsonl(g, items)
    with pytest.raises(UnverifiedGoldenSetError):
        load_verified_golden(g, st)  # hash mismatch
    write_stamp(g, st)
    items[0].verified = False
    write_jsonl(g, items)
    write_stamp(g, st)
    with pytest.raises(UnverifiedGoldenSetError):
        load_verified_golden(g, st)  # unverified flag


def test_report_renders():
    r = EvalReport(
        run_id="t",
        created_at="now",
        git_sha="abc",
        index_commit_sha="def",
        model="m",
        prompt_version="v1",
        judge_model="j",
        golden_count=2,
        golden_by_type={"single": 1, "multi": 0, "unanswerable": 1},
        retrieval=[
            RetrievalRow(
                config="hybrid",
                recall_at_5=0.5,
                recall_at_10=0.6,
                recall_at_30=0.9,
                mrr=0.4,
                n=1,
                by_type={
                    "single": {
                        "recall_at_5": 0.5,
                        "recall_at_10": 0.6,
                        "recall_at_30": 0.9,
                        "mrr": 0.4,
                    }
                },
            )
        ],
        generation=GenerationSummary(
            model="m",
            n=2,
            faithfulness_mean=4.0,
            faithfulness_pass_rate=1.0,
            relevance_mean=4,
            relevance_pass_rate=1,
            groundedness_mean=4.5,
            groundedness_pass_rate=1,
            abstention={
                "precision": 1,
                "recall": 1,
                "f1": 1,
                "hallucination_rate_on_unanswerable": 0,
                "false_abstention_rate": 0,
            },
        ),
        judge_agreement=JudgeAgreement(n=30, percent_agreement=0.87, cohen_kappa=0.7, trusted=True),
    )
    md = render_markdown(r)
    assert (
        "| hybrid | 50.0% | 60.0% | 90.0% | 0.400 | 1 |" in md
        and "trusted" in md
        and "Abstention" in md
    )
    json.loads(r.model_dump_json())
