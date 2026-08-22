# Shared CPU base for gateway / embedder / reranker / eval jobs. Multi-stage: build wheels with uv, run slim.
# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
COPY common/pyproject.toml common/
COPY ingest/pyproject.toml ingest/
COPY embedder/pyproject.toml embedder/
COPY reranker/pyproject.toml reranker/
COPY gateway/pyproject.toml gateway/
COPY eval/pyproject.toml eval/
COPY bench/pyproject.toml bench/
ARG EXTRAS=""
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace --no-dev ${EXTRAS}
COPY common common
COPY ingest ingest
COPY embedder embedder
COPY reranker reranker
COPY gateway gateway
COPY eval eval
COPY bench bench
COPY config config
COPY scripts scripts
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev ${EXTRAS}

FROM python:3.12-slim AS runtime
RUN useradd -m -u 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 HF_HOME=/data/hf-cache
USER app
