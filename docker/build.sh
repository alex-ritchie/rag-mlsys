#!/usr/bin/env bash
# Build all images tagged with the git SHA (spec §8). CPU and GPU images are separate.
set -euo pipefail
cd "$(dirname "$0")/.."
SHA=$(git rev-parse --short HEAD)
export DOCKER_BUILDKIT=1
echo "== python base (cpu)";          docker build -f docker/python-base.Dockerfile -t mlsysbook-rag/python-base:cpu .
echo "== python base (cpu + models)"; docker build -f docker/python-base.Dockerfile --build-arg EXTRAS="--extra models --extra onnx" -t mlsysbook-rag/python-base:cpu-models .
echo "== gateway";  docker build -f docker/gateway.Dockerfile  -t mlsysbook-rag/gateway:$SHA  -t mlsysbook-rag/gateway:latest .
echo "== embedder"; docker build -f docker/embedder.Dockerfile -t mlsysbook-rag/embedder:$SHA -t mlsysbook-rag/embedder:latest .
echo "== reranker"; docker build -f docker/reranker.Dockerfile -t mlsysbook-rag/reranker:$SHA -t mlsysbook-rag/reranker:latest .
echo "== frontend"; docker build -f docker/frontend.Dockerfile -t mlsysbook-rag/frontend:$SHA -t mlsysbook-rag/frontend:latest .
if [[ "${GPU_IMAGES:-0}" == "1" ]]; then
  echo "== python base (gpu + models)"; docker build -f docker/python-gpu-base.Dockerfile -t mlsysbook-rag/python-base:gpu-models .
  docker build -f docker/embedder.Dockerfile --build-arg BASE=mlsysbook-rag/python-base:gpu-models -t mlsysbook-rag/embedder:$SHA-gpu .
  docker build -f docker/reranker.Dockerfile --build-arg BASE=mlsysbook-rag/python-base:gpu-models -t mlsysbook-rag/reranker:$SHA-gpu .
fi
echo "built tag $SHA"
