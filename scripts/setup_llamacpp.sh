#!/usr/bin/env bash
# Build llama.cpp with CUDA for the engine-ablation cell (spec §5.4: a build from July 2026 or later for Qwen3.8 MTP).
# Uses the system CUDA toolkit at /usr/local/cuda (nvcc is not on PATH by default on this workstation).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc
[[ -d data/llama.cpp ]] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git data/llama.cpp
cd data/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(nproc)" --target llama-server llama-cli
./build/bin/llama-server --version
