# M3 baseline — vLLM bring-up (risk milestone)

**Status: PASSED (2026-08-21) on attempt 2.** Acceptance criterion: sustained generation at concurrency 8 for
10 minutes without OOM (`scripts/m3_measure.sh`, which wraps `bench/configs/m3-soak.yaml`).

## Attempt log

| # | Config | Outcome | Measured |
|---|---|---|---|
| 1 | spec baseline: CUDA graphs, `--max-model-len 32768`, `--gpu-memory-utilization 0.90` | **engine refused to start** — not an OOM crash, a pre-flight check: "2.3 GiB KV cache is needed … available 1.52 GiB … estimated maximum model length is 19600" | weights **16.84 GiB** (Marlin W4A16 kernel selected, text-only mode, 8 shards loaded in 3.4 s); torch.compile 32 s + warm-up 40 s; CUDA-graph memory estimate 0.25 GiB; **KV available 1.52 GiB at 0.90** → the context + activation + compile overhead on this 24 GB card is ~3.6 GiB, larger than the 1–1.5 GiB the planning table assumed. Log: `data/logs/vllm-m3-graphs-util090-FAILED.log`. |
| 2 | CUDA graphs, 32768, **`--gpu-memory-utilization 0.95`** | **PASS** — 10-min soak at c8: 417 requests, **0 errors**, no OOM/CUDA errors in the log | weights 16.84 GiB; **KV 4.78 GiB = 67,584 tokens** (2.06× concurrency at the full 32K); CUDA graphs captured (11 piecewise + 7 full-decode); warm start 40 s (compile cache); VRAM **22.93 GB** after load and the same at soak peak (vLLM pre-allocates); smoke answer correct with thinking off (22 prompt → 43 completion tokens). Soak: **351 output tok/s aggregate**, 0.69 req/s, **TTFT p50 172 ms / p99 177 ms**, total p50 11.6 s per 512-token answer (≈44 tok/s per stream). `docs/benchmarks/m3-graphs-util095.json`, `bench/results/*m3-graphs-util095.json`. |

| 3 | CUDA graphs, **`--max-model-len 16384`, util 0.88** (free VRAM for a GPU reranker) | refused: needs 1.29 GiB KV, had **1.05 GiB** | overhead is not linear in utilization: 0.88 → 1.05 GiB KV, 0.90 → 1.52, 0.92 → 4.07, 0.95 → 4.78. |
| 4 | CUDA graphs, 16384, **util 0.92 + bge-reranker-v2-m3 on the same GPU** | **works alone, fails under load**: KV 4.07 GiB (51,579 tokens, 3.15× at 16K); vLLM 22.26 GB, +reranker 23.62 GB; GPU rerank **261 ms** (30 cands) / 182 ms (20); an 8-way generation burst pushed VRAM to 24,087 of 24,564 MiB and the reranker then OOMed (`Tried to allocate 26 MiB … 22 MiB free`); with the reranker at `max_length` 512 one RAG query reranked in 465 ms, then **vLLM itself died** under a 6-way burst. | Co-residency at 0.92 leaves ~0.5 GB headroom — not enough for both processes' transient activations. Next: util 0.90 + 16K (KV 1.52 GiB ≈ 19K tokens ≈ 4 concurrent RAG requests) + reranker 512, and measure the concurrency ceiling honestly instead of the context ceiling. |

| 5 | CUDA graphs, 16384, **util 0.90 + GPU reranker at `max_length` 512**, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` | **PASS — adopted** | Available KV cache memory: 3.8 GiB (GPU KV cache size: 47,938 tokens, 2.93× at 16K); vLLM 21.77 GB, +reranker 23.14 GB, **peak 23.66 GB** during an 8-way 400-token generation burst running concurrently with three RAG queries; 0 OOM, 0 gateway errors, vLLM alive afterwards. End to end: rerank **169–471 ms**, TTFT 2.2–2.6 s (≈2.9K-token prompt prefill while the burst occupied the batch), totals 3.6–11.5 s; the unanswerable question abstained correctly. |

**`make up` with the adopted defaults (cold start, 2026-08-22):** embedder (CPU) and reranker (GPU) ready in 4 s,
vLLM in 33 s (compile cache warm), gateway and frontend at 34 s. Question "How does data parallelism differ from
model parallelism?": 3,511-token prompt → 233-token answer citing Vol 2 Ch 5 *Data Parallelism* / *Model Parallelism*;
embed 104 ms · retrieve 14 ms · **rerank 282 ms** · TTFT 2.77 s · generate 6.96 s · total 7.36 s; 23.66 GB VRAM;
Ctrl-C returned the GPU to 18 MiB.

## Open issue: reranker `max_length` 512 truncates what the reranker *reads*

The adopted placement sets the cross-encoder's input window to 512 tokens per (query, chunk) pair. Chunks are 400–800
tokens (p50 680), so for most candidates the reranker scores only the first ~450 tokens of the chunk; the tail is
invisible to ranking. Answers themselves are **not** truncated: in the attempt-5 run the three answers were 368, 377
and 73 completion tokens, all ending naturally (the console excerpts in the log were clipped to 230 characters for
readability). Answer length is governed by `MAX_OUTPUT_TOKENS` (1024), and the `done` event's `truncated` flag —
plus the `rag_truncated_answers_total` counter and the red *truncated* tag in the UI — now report if that cap is
ever hit. Whether the
blind tail hurts is an eval question (recall@5 / MRR with 512 vs 1024, on the verified golden set). Candidate fixes,
cheapest first:

1. **Chunk size 400** (`config/sweeps/ingest-chunk-400.yaml`): every chunk fits the 512 window; more, smaller chunks
   also cut prompt size. Already on the M8 list.
2. **Head + tail truncation**: score the first 256 and last 256 tokens of each chunk instead of the first 512 — a
   few lines in the reranker backend; keeps section-ending summaries visible.
3. **Windowed scoring**: split each chunk into 512-token windows, score all, take the max — 1.5–2× reranker cost
   (still ~0.5 s on the GPU), no blind spots.
4. **Small-to-big**: rerank on 256-token sub-chunks, send the parent chunk/section to the LLM (needs a parent
   lookup at prompt time). Best ranking precision per token; already on the M8 list.
5. **Buy the VRAM back**: `--max-model-len 8192` (RAG prompts are ~3.5K tokens) and/or `--max-num-seqs 8` to let
   the reranker run at `max_length` 1024 alongside vLLM. Costs concurrency headroom — reported honestly in the
   ablation as "c8 not sustainable at this setting" if that is what the soak shows.
6. **Smaller LLM cell** (Qwen3.5-9B, ~6 GB): the ablation's natural answer — 32K context, reranker at 1024, GPU
   embedder, more candidates, all at once.

| 6 | **8K context, util 0.88** (co-residency variant for the M8 gateway cell) | refused at first: `max_num_seqs (32) exceeds available Mamba cache blocks (26)` — Qwen3.8's 48 linear-attention layers hold a fixed-size recurrent state per sequence, allocated as "Mamba cache blocks"; the budget left 26 → `--max-num-seqs 24`. | A second, architecture-specific concurrency ceiling next to the KV cache: the hybrid design shrinks per-token KV ~4× but adds a per-*sequence* constant, so max concurrency is bounded by `min(KV tokens / tokens per request, Mamba blocks)`. |

| 7 | 8K, util 0.88, `--max-num-seqs 24` + GPU reranker@512 (M8 cell) | **stable**: 0 OOMs, peak 23.17 GB, gateway c1–c16 all error-free | KV 1.27 GiB (12,528 tokens, 1.53× at 8K) — the price: direct throughput saturates at 275 tok/s from c8 and gateway TTFT queues to 9.4 s at c4, 26 s at c8. Safe but KV-starved; variant B (16K + bounded prefill) is the next test. |

Environment: vLLM 0.27.1 (+cu129 wheel), torch 2.13.0+cu129, transformers 5.15.1, driver 575.57 (CUDA 12.9),
model `dbirks/Qwen3.8-27B-W4A16-AutoRound` (compressed-tensors pack-quantized, group 128, symmetric int4),
architecture `Qwen3_5ForConditionalGeneration` (64 layers: full + linear attention), attention block size auto-set to
784 tokens so the attention page size matches the mamba (linear-attention) state page.

## Procedure

1. `uv pip install "vllm>=0.17" --extra-index-url https://download.pytorch.org/whl/cu128` (cu12x build — DEVIATIONS #1), `transformers ≥ 5.8`.
2. `make vllm` (config `config/serving/vllm-qwen38-27b.yaml`; first candidate repo `dbirks/Qwen3.8-27B-W4A16-AutoRound`). Record the exact vLLM/transformers versions that worked in `uv.lock` and here.
3. If CUDA-graph capture OOMs: `--set enforce-eager=true`; record the mode.
4. `nvidia-smi` at idle-loaded and under the c8 soak → fill the VRAM table.
5. Decide reranker/embedder placement per the contingency ladder (design doc §5.4) and record it below.

## Results

| Item | Value |
|---|---|
| vLLM / torch / transformers | 0.27.1 (+cu129 GitHub wheel) / 2.13.0+cu129 / 5.15.1 — pinned in `scripts/setup_vllm.sh` |
| Model repo | `dbirks/Qwen3.8-27B-W4A16-AutoRound` (19.5 GB on disk, 18.12 GiB checkpoint) |
| CUDA graphs or eager | **CUDA graphs** (capture succeeded; eager never needed) |
| Weights VRAM | 16.84 GiB (vLLM) |
| VRAM after load / under c8 soak (nvidia-smi) | 22.93 GB / 22.93 GB |
| KV cache | 4.78 GiB, 67,584 tokens, block size 784 (attention page matched to the linear-attention state page) |
| TTFT p50 / p99 @ c8 | 172 ms / 177 ms |
| Output tok/s @ c8 | 351 aggregate (≈44 per stream) |
| 10-min soak at c8 | 417 requests, 0 errors, 0 OOM |

**What the planning table got wrong:** "CUDA context + activations ~1–1.5 GB" was ~3.6 GiB in practice at 0.90
(weights 16.84 + KV 1.52 = 18.4 of the 21.6 GiB budget); the remainder is the torch.compile / CUDA-graph workspace and
the profiling run's activations. Raising utilization to 0.95 recovered 3.3 GiB of KV, not 1.2 — vLLM's profiling
reserve is not linear in the utilization fraction.

## Placement decision (contingency ladder, spec §5.4) — **settled, attempt 5**

| Component | Placement | Status |
|---|---|---|
| bge-m3 (query-time) | **CPU** — settled | query-embed p50 71–115 ms, within the ≤100 ms target; not worth VRAM |
| bge-reranker-v2-m3 | **GPU, co-resident with vLLM** (vLLM util 0.90, 16K context; reranker `max_length` 512) | rung 1 (CPU) measured and rejected: 22.6 s/query (`m2-retrieval.md`); GPU costs 170–470 ms. Util 0.92 left too little headroom (attempt 4); 0.90 survives bursts with ~0.9 GB spare (attempt 5). Cost: context 32K → 16K, which RAG never needs, and ~2.9× instead of ~2× concurrency at the full context. |

Bulk indexing borrows the GPU as a batch job while vLLM is down (28 s for the whole corpus). The ladder's ordering
assumed a CPU reranker is acceptable; measurement says it is 20–80× over budget, so the VRAM for the reranker comes
from context length (rung 4) rather than from reranking itself.
