"""Async load generator (spec §5.7). Two targets:
  rag-e2e  — the full RAG pipeline through the gateway (/api/ask SSE)
  llm-only — vLLM's OpenAI endpoint directly, plain prompts, no retrieval
(`gateway` / `openai` are accepted as legacy aliases.)

Per run: TTFT p50/p99, total latency p50/p99, output tokens/s, requests/s at a fixed concurrency.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class Sample:
    ok: bool
    ttft_ms: float
    total_ms: float
    output_tokens: int
    error: str = ""
    stage_ms: dict[str, float] = field(default_factory=dict)
    rerank_fallback: bool = False  # gateway degraded to the fused order because the reranker failed


async def _one_gateway(
    client: httpx.AsyncClient, base: str, question: str, top_k: int | None
) -> Sample:
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    stages: dict[str, float] = {}
    fallback = False
    try:
        async with client.stream(
            "POST",
            f"{base}/api/ask",
            json={"question": question, **({"top_k": top_k} if top_k else {})},
        ) as r:
            if r.status_code != 200:
                return Sample(
                    False, 0, (time.perf_counter() - t0) * 1000, 0, f"http {r.status_code}"
                )
            ev = None
            async for line in r.aiter_lines():
                if line.startswith("event:"):
                    ev = line[6:].strip()
                elif line.startswith("data:") and ev == "token":
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    tokens += (
                        1  # token events ~ streamed chunks; usage from `done` is authoritative
                    )
                elif line.startswith("data:") and ev == "done":
                    d = json.loads(line[5:])
                    tokens = d["usage"]["completion_tokens"] or tokens
                    # keep only numeric stage timings (the breakdown also carries e.g. rerank_error: str|None)
                    stages = {
                        k: float(v)
                        for k, v in d["latency_breakdown"].items()
                        if isinstance(v, int | float)
                    }
                elif line.startswith("data:") and ev == "error":
                    return Sample(
                        False,
                        ttft or 0,
                        (time.perf_counter() - t0) * 1000,
                        tokens,
                        json.loads(line[5:]).get("message", "error"),
                    )
    except Exception as e:
        return Sample(
            False, ttft or 0, (time.perf_counter() - t0) * 1000, tokens, f"{type(e).__name__}: {e}"
        )
    return Sample(
        True,
        ttft or 0,
        (time.perf_counter() - t0) * 1000,
        tokens,
        stage_ms=stages,
        rerank_fallback=fallback,
    )


async def _one_openai(
    client: httpx.AsyncClient, base: str, model: str, prompt: str, max_tokens: int, extra: dict
) -> Sample:
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **extra,
    }
    try:
        async with client.stream("POST", f"{base}/chat/completions", json=body) as r:
            if r.status_code != 200:
                return Sample(
                    False, 0, (time.perf_counter() - t0) * 1000, 0, f"http {r.status_code}"
                )
            async for line in r.aiter_lines():
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                obj = json.loads(line[5:])
                for ch in obj.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        tokens += 1
                if obj.get("usage"):
                    tokens = obj["usage"].get("completion_tokens", tokens)
    except Exception as e:
        return Sample(
            False, ttft or 0, (time.perf_counter() - t0) * 1000, tokens, f"{type(e).__name__}: {e}"
        )
    return Sample(True, ttft or 0, (time.perf_counter() - t0) * 1000, tokens)


async def run_load(
    *,
    target: str,
    base_url: str,
    prompts: list[str],
    concurrency: int,
    requests: int | None = None,
    duration_s: float | None = None,
    model: str = "",
    max_tokens: int = 512,
    top_k: int | None = None,
    extra_body: dict | None = None,
    warmup: int = 2,
) -> tuple[list[Sample], float]:
    """Closed-loop: `concurrency` workers each issue requests back-to-back until the budget is spent."""
    samples: list[Sample] = []
    lock = asyncio.Lock()
    counter = {"issued": 0}
    limits = httpx.Limits(
        max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4
    )
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=10.0), limits=limits
    ) as client:

        async def fire(i: int) -> Sample:
            p = prompts[i % len(prompts)]
            if target in (
                "rag-e2e",
                "gateway",
            ):  # full pipeline through the gateway (old name: gateway)
                return await _one_gateway(client, base_url, p, top_k)
            return await _one_openai(client, base_url, model, p, max_tokens, extra_body or {})

        for i in range(warmup):
            await fire(i)
        t_start = time.perf_counter()
        deadline = t_start + duration_s if duration_s else None

        async def worker() -> None:
            while True:
                async with lock:
                    if requests is not None and counter["issued"] >= requests:
                        return
                    if deadline and time.perf_counter() >= deadline:
                        return
                    i = counter["issued"]
                    counter["issued"] += 1
                s = await fire(i)
                async with lock:
                    samples.append(s)

        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t_start
    return samples, wall


def summarize(samples: list[Sample], wall_s: float, concurrency: int) -> dict:
    from mlsys_eval.metrics import percentile  # pure function, no heavy imports

    ok = [s for s in samples if s.ok]
    ttft = [s.ttft_ms for s in ok]
    tot = [s.total_ms for s in ok]
    toks = sum(s.output_tokens for s in ok)
    stages: dict[str, list[float]] = {}
    for s in ok:
        for k, v in s.stage_ms.items():
            stages.setdefault(k, []).append(v)
    return {
        "concurrency": concurrency,
        "requests": len(samples),
        "errors": len(samples) - len(ok),
        "wall_s": round(wall_s, 2),
        "requests_per_s": round(len(ok) / wall_s, 3) if wall_s else 0,
        "output_tokens_per_s": round(toks / wall_s, 1) if wall_s else 0,
        "ttft_ms": {
            "p50": round(percentile(ttft, 50), 1),
            "p90": round(percentile(ttft, 90), 1),
            "p99": round(percentile(ttft, 99), 1),
        },
        "total_ms": {
            "p50": round(percentile(tot, 50), 1),
            "p90": round(percentile(tot, 90), 1),
            "p99": round(percentile(tot, 99), 1),
        },
        "stage_ms_p50": {k: round(percentile(v, 50), 1) for k, v in stages.items()},
        "rerank_fallbacks": sum(s.rerank_fallback for s in ok),
        "error_samples": [s.error for s in samples if not s.ok][:5],
    }
