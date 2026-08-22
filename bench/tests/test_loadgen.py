import json

from mlsys_bench.loadgen import Sample, summarize
from mlsys_bench.report import render


def test_summarize():
    s = [
        Sample(True, 100, 1000, 50, stage_ms={"embed_ms": 10}),
        Sample(True, 200, 2000, 50),
        Sample(False, 0, 10, 0, "boom"),
    ]
    out = summarize(s, wall_s=2.0, concurrency=2)
    assert out["requests"] == 3 and out["errors"] == 1
    assert out["output_tokens_per_s"] == 50.0 and out["requests_per_s"] == 1.0
    assert out["ttft_ms"]["p50"] in (100, 200) and out["total_ms"]["p99"] == 2000
    assert out["stage_ms_p50"]["embed_ms"] == 10
    json.dumps(out)


def test_render_tables():
    md = render(
        [
            {
                "name": "x",
                "created_at": "t",
                "git_sha": "abcdef1234",
                "config": {"base_url": "u", "model": "m"},
                "reproduce": "cmd",
                "runs": [
                    {
                        "concurrency": 1,
                        "requests_per_s": 1.0,
                        "output_tokens_per_s": 2.0,
                        "ttft_ms": {"p50": 1, "p99": 2},
                        "total_ms": {"p50": 3, "p99": 4},
                        "errors": 0,
                    }
                ],
            }
        ]
    )
    assert "| 1 | 1.0 | 2.0 | 1 | 2 | 3 | 4 | 0 |" in md
