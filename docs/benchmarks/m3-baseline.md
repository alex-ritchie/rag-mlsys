# M3 baseline — vLLM bring-up (risk milestone)

**Status: PASSED (2026-08-21) on attempt 2.** Acceptance criterion: sustained generation at concurrency 8 for
10 minutes without OOM (`scripts/m3_measure.sh`, which wraps `bench/configs/m3-soak.yaml`).

## Attempt log

| # | Config | Outcome | Measured |
|---|---|---|---|
| 1 | spec baseline: CUDA graphs, `--max-model-len 32768`, `--gpu-memory-utilization 0.90` | **engine refused to start** — not an OOM crash, a pre-flight check: "2.3 GiB KV cache is needed … available 1.52 GiB … estimated maximum model length is 19600" | weights **16.84 GiB** (Marlin W4A16 kernel selected, text-only mode, 8 shards loaded in 3.4 s); torch.compile 32 s + warm-up 40 s; CUDA-graph memory estimate 0.25 GiB; **KV available 1.52 GiB at 0.90** → the context + activation + compile overhead on this 24 GB card is ~3.6 GiB, larger than the 1–1.5 GiB the planning table assumed. Log: `data/logs/vllm-m3-graphs-util090-FAILED.log`. |
| 2 | CUDA graphs, 32768, **`--gpu-memory-utilization 0.95`** | **PASS** — 10-min soak at c8: 417 requests, **0 errors**, no OOM/CUDA errors in the log | weights 16.84 GiB; **KV 4.78 GiB = 67,584 tokens** (2.06× concurrency at the full 32K); CUDA graphs captured (11 piecewise + 7 full-decode); warm start 40 s (compile cache); VRAM **22.93 GB** after load and the same at soak peak (vLLM pre-allocates); smoke answer correct with thinking off (22 prompt → 43 completion tokens). Soak: **351 output tok/s aggregate**, 0.69 req/s, **TTFT p50 172 ms / p99 177 ms**, total p50 11.6 s per 512-token answer (≈44 tok/s per stream). `docs/benchmarks/m3-graphs-util095.json`, `bench/results/*m3-graphs-util095.json`. |

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

## Placement decision (contingency ladder, spec §5.4)

| Component | Placement | Reason (measured) |
|---|---|---|
| bge-reranker-v2-m3 | **CPU** | 22.93 GB of 24 GB is taken by vLLM at the configuration that serves 32K; the two services need ~3.6 GB together (M2 measurement) |
| bge-m3 (query-time) | **CPU** | same; CPU query-embed p50 is 71 ms, within the ≤100 ms target (M2) |

Bulk indexing still borrows the GPU as a batch job while vLLM is down (28 s for the whole corpus). Ladder steps 3–4
(utilization 0.85, `--max-model-len 16384`) were not needed; the opposite direction (0.95) was. Rungs 1–2 (both
services to CPU) are the configuration `make up`, compose, and the k8s manifests now default to.
