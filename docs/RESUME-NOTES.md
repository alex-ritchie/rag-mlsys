# Resume notes — 2026-08-22 (written before a workstation reboot)

## What was running / scheduled when the machine went down

| Job | State | How to resume |
|---|---|---|
| Lever sweeps (`scripts/sweep.sh all`) | vLLM rows done (baseline, prefix-off, eager, seqs-8, seqs-24, reasoning-on); **vllm-mtp-n2 failed to start** (see below); llama.cpp rows (MTP n=0/2/3/4, KV q4_0) **not done** — n0 was mid-run when stopped | `./scripts/sweep.sh llamacpp` then `make bench-report`; commit `docs/benchmarks/results.md` + copy new JSONs into `docs/benchmarks/sweeps/` |
| Completed sweep JSONs | copied to `docs/benchmarks/sweeps/` and committed (bench/results is gitignored) | — |
| Serving ablation cells | **all complete and pushed** (13 cells: 27B ×5 configs, 9B W4A16/W8A8/BF16 at 32K and 8K, MoE, llama.cpp) | `docs/benchmarks/ablation-serving.md`, `docs/writeups/benchmark-report.md` |
| Golden-set verification | owner task in progress (`eval/golden/golden.jsonl`, saved after every item) | `make golden-verify` resumes where it left off |

## After the reboot

1. `make db-up` — the unprivileged Postgres under `data/pg` does not start itself; `make` targets find it automatically once it runs.
2. `make up` — the full local stack (vLLM default = 27B variant B, reranker on GPU, embedder on CPU).
3. `./scripts/sweep.sh llamacpp` — finish the llama.cpp lever rows (~1 h), then `make bench-report`.
4. `make golden-verify` → `make eval` once stamped.

## Open items noted during the sweeps

- `vllm-mtp-n2`: vLLM refused the `speculative-config` MTP row — error recorded in `data/logs/sweeps/vllm-mtp-n2.server.log`; to be diagnosed (likely a flag-shape or model-support issue for Qwen3.8's MTP head in vLLM 0.27.1).
- `max-num-seqs 64` cannot start at the adopted 27B budget (Mamba state blocks cap at 26), so that row was run at 24.
