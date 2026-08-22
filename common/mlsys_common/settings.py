"""Environment-driven settings. Every variable is documented in .env.example."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    profile: Literal["local", "demo"] = "local"
    database_url: str = "postgresql+asyncpg://mlsys:mlsys@localhost:5432/mlsys"
    log_level: str = "info"

    # gateway
    gateway_port: int = 8000
    embedder_url: str = "http://localhost:8001"
    reranker_url: str = "http://localhost:8002"  # "" => skip reranking
    llm_base_url: str = "http://localhost:8003/v1"
    llm_model: str = "qwen38-27b-w4a16"
    llm_api_key: str = "none"
    retrieval_mode: Literal["hybrid", "dense"] = "hybrid"
    retrieval_top_n: int = 30
    rerank_top_k: int = 5
    prompt_version: str = "v1"
    max_output_tokens: int = 1024
    disable_thinking: bool = True

    # embedder / reranker services
    embedder_mode: Literal["cpu", "gpu", "onnx", "test"] = "cpu"
    embedder_model: str = "BAAI/bge-m3"
    reranker_mode: Literal["cpu", "gpu", "onnx", "test"] = "cpu"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # demo profile
    anthropic_api_key: str = ""
    demo_generation_model: str = "claude-haiku-4-5"
    demo_embedder: Literal["inprocess-onnx", "service"] = "inprocess-onnx"
    demo_rerank: Literal["onnx", "off"] = "onnx"
    demo_rate_limit_per_day: int = 10
    demo_daily_budget_usd: float = 2.0
    demo_haiku_input_usd_per_mtok: float = 1.0
    demo_haiku_output_usd_per_mtok: float = 5.0
    supabase_db_url: str = ""

    # eval
    judge_model: str = "claude-haiku-4-5"
    golden_gen_model: str = "claude-haiku-4-5"

    @property
    def effective_database_url(self) -> str:
        if self.profile == "demo" and self.supabase_db_url:
            return self.supabase_db_url
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
