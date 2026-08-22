# GPU variant of the python base (embedder/reranker in GPU mode). CUDA 12.x runtime for Ampere.
# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv ca-certificates && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 app
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
COPY --chown=app:app . /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra models --extra onnx -p python3.12 && chown -R app:app /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 HF_HOME=/data/hf-cache
USER app
