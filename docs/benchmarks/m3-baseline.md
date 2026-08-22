# M3 baseline — vLLM bring-up (risk milestone)

**Status: in progress (2026-08-21).** Acceptance criterion: sustained generation at concurrency 8 for
10 minutes without OOM (`scripts/m3_measure.sh`, which wraps `bench/configs/m3-soak.yaml`).

## Attempt log

| # | Config | Outcome | Measured |
|---|---|---|---|
| 1 | spec baseline: CUDA graphs, `--max-model-len 32768`, `--gpu-memory-utilization 0.90` | **engine refused to start** — not an OOM crash, a pre-flight check: "2.3 GiB KV cache is needed … available 1.52 GiB … estimated maximum model length is 19600" | weights **16.84 GiB** (Marlin W4A16 kernel selected, text-only mode, 8 shards loaded in 3.4 s); torch.compile 32 s + warm-up 40 s; CUDA-graph memory estimate 0.25 GiB; **KV available 1.52 GiB at 0.90** → the context + activation + compile overhead on this 24 GB card is ~3.6 GiB, larger than the 1–1.5 GiB the planning table assumed. Log: `data/logs/vllm-m3-graphs-util090-FAILED.log`. |
| 2 | CUDA graphs, 32768, **`--gpu-memory-utilization 0.95`** | _running_ | |

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

## Results (to fill)

| Item | Value |
|---|---|
| vLLM version / transformers version | |
| Model repo + revision | |
| CUDA graphs or eager | |
| Weights VRAM (nvidia-smi after load, before traffic) | |
| VRAM under c8 soak (peak) | |
| KV cache blocks reported by vLLM (`# GPU blocks`) | |
| TTFT p50 / p99 @ c8 | |
| Output tok/s @ c8 | |
| 10-min soak at c8: errors / OOM | |

## Placement decision

| Component | Placement | Reason (measured) |
|---|---|---|
| bge-reranker-v2-m3 | | |
| bge-m3 (query-time) | | |
