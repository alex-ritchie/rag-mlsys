# M3 baseline — vLLM bring-up (risk milestone)

**Status: not yet run.** This file is filled in on the workstation when M3 executes. The acceptance
criterion is sustained generation at concurrency 8 for 10 minutes without OOM
(`make bench BENCH_CONFIG=bench/configs/m3-soak.yaml`).

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
