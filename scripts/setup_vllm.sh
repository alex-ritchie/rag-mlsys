#!/usr/bin/env bash
# Install vLLM into its own venv (data/vllm-venv). Separate from the uv workspace because vLLM pins torch exactly.
# Pinned here the moment M3 passed (spec §5.4). Driver 575 => CUDA 12.9 => the +cu129 wheel from the GitHub release
# (the PyPI wheel targets CUDA 13, which this driver rejects) — see docs/DEVIATIONS.md #1.
set -euo pipefail
cd "$(dirname "$0")/.."
VLLM_VERSION=${VLLM_VERSION:-0.27.1}
CUDA=${CUDA_VARIANT:-cu129}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-900}
uv venv -q -p 3.12 data/vllm-venv
export VIRTUAL_ENV=$PWD/data/vllm-venv
uv pip install "torch==2.13.0" torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA"
uv pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA}-cp38-abi3-manylinux_2_28_x86_64.whl" \
  --extra-index-url "https://download.pytorch.org/whl/$CUDA"
data/vllm-venv/bin/python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
