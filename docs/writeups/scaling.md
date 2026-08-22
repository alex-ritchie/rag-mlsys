# Scaling story: why the HPA lives on the gateway and not on vLLM

*(Writeup requirement, design doc §5.8. Numbers marked **[M8]** are filled from the benchmark runs.)*

## The two tiers have different bottlenecks

The stack has a **stateless CPU tier** (gateway: embed call, one SQL round trip, rerank call, prompt
assembly, SSE fan-out) and a **stateful GPU tier** (vLLM holding ~18 GB of W4A16 weights plus a KV
cache in 24 GB of VRAM). They fail differently under load:

| | Gateway | vLLM |
|---|---|---|
| Holds state between requests | no | yes: weights, KV cache, prefix cache |
| Cost of a second replica | ~250 m CPU, 400 MB RAM, seconds to start | a second 24 GB GPU; minutes to load |
| What saturates first | CPU (JSON, SSE, asyncpg) at ~N concurrent streams/replica **[M8]** | KV-cache memory, then compute: queue depth grows, TTFT climbs |
| Useful scaling signal | CPU utilization (HPA) | `vllm:num_requests_waiting`, KV-cache utilization, TTFT p99 |

The HPA is therefore configured on the gateway only (`k8s/42-gateway-hpa.yaml`, CPU target 70 %,
1–5 replicas), with resource requests derived from load-test data so the 70 % threshold corresponds
to a real concurrency number rather than a guess. The demo (`make hpa-demo`) drives replicas
1 → 4 → 1 with a 32-stream load test; the `kubectl get hpa -w` capture and the Grafana replica
panel are in the README.

## Why horizontal scaling is the wrong axis for a single-GPU LLM tier

1. **There is nothing to scale onto.** `replicas: 2` with `nvidia.com/gpu: 1` simply leaves the
   second pod `Pending`. The manifest uses `strategy: Recreate` for the same reason: a rolling update
   would need two GPUs for the overlap.
2. **Throughput on one GPU comes from batching, not replicas.** vLLM's continuous batching already
   turns N concurrent requests into one forward pass per decode step; the marginal cost of the 8th
   concurrent stream is far below the first **[M8: tok/s at concurrency 1/4/8/16/32]**. A second
   vLLM process on the same GPU would split the KV cache, halve the batch, and fight over SMs — strictly
   worse than one process with a higher `--max-num-seqs`.
3. **The memory wall is the real limit.** On 24 GB with an 18 GB model, the KV cache is what bounds
   achievable concurrency at a given context length. Qwen3.8's hybrid attention (48 of 64 layers with
   constant-size recurrent state) shrinks per-token KV by roughly 4× vs. full attention, which is
   exactly why a 27B at 32K context fits at all **[M8: measured KV budget and max concurrency]**. No
   replica count changes that; quantizing the KV cache or shortening `--max-model-len` does.
4. **The right signals are queue depth and SLOs, not CPU.** The Serving dashboard shows running vs.
   waiting sequences, KV-cache utilization and prefix-cache hit rate, TTFT/e2e p99. A rising *waiting*
   count with flat GPU utilization means the GPU is memory-bound, not compute-bound — the actionable
   levers are context length, `max-num-seqs`, and KV dtype, all of which are M8 sweep rows.

## What changes with N GPUs

| Option | When it wins | Cost |
|---|---|---|
| **Replica-per-GPU** (N vLLM pods, each `gpu: 1`, a Service in front) | Model fits on one GPU (our case). Linear throughput, independent failure domains, rolling updates become possible, and the HPA *can* target the LLM tier using a custom metric (queue depth via the Prometheus adapter). | No latency improvement per request; N copies of the weights; prefix cache is per-replica so RAG's shared-prefix benefit fragments unless requests are routed by prefix. |
| **Tensor parallel** (`--tensor-parallel-size N`) | Model does *not* fit on one GPU, or single-stream latency must drop (each layer's matmuls are split, so decode steps get faster when the interconnect is fast enough). | All-reduce on every layer: needs NVLink/PCIe bandwidth; one failure domain; one (bigger) KV cache. On consumer cards over PCIe the communication overhead often eats the gain for a 27B that already fits. |
| **Pipeline parallel** | Very large models across slow interconnects. | Bubbles; latency gets worse, only throughput scales. |

Rule of thumb the numbers support **[M8]**: if the model fits on one GPU, run replica-per-GPU and
scale on queue depth; reach for tensor parallelism only when it does not fit or when p50 latency is
the SLO you are failing.
