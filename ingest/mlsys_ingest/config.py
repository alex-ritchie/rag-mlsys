from __future__ import annotations

from pathlib import Path

import yaml
from mlsys_common.settings import REPO_ROOT
from pydantic import BaseModel


class SourceConfig(BaseModel):
    repo: str
    commit_sha: str
    checkout_dir: str = "data/book"
    license: str = "CC-BY-NC-SA-4.0"


class VolumeConfig(BaseModel):
    number: int
    title: str
    quarto_config: str
    quarto_root: str = "book/quarto"


class SelectionConfig(BaseModel):
    skip_path_substrings: list[str] = []
    appendix_base: int = 100
    glossary_base: int = 200


class ChunkingConfig(BaseModel):
    tokenizer: str = "BAAI/bge-m3"
    min_tokens: int = 400
    max_tokens: int = 800
    drop_code_langs: list[str] = ["python", "r", "tikz", "=html", "=latex"]
    drop_div_classes: list[str] = []
    drop_div_attrs: list[str] = []
    resolve_inline_values: bool = (
        True  # execute the book's calculation cells (mlsysim) to materialise numbers
    )


class IngestConfig(BaseModel):
    source: SourceConfig
    volumes: list[VolumeConfig]
    selection: SelectionConfig = SelectionConfig()
    chunking: ChunkingConfig = ChunkingConfig()

    @property
    def checkout_path(self) -> Path:
        p = Path(self.source.checkout_dir)
        return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path | None = None) -> IngestConfig:
    path = Path(path) if path else REPO_ROOT / "config" / "ingest.yaml"
    with open(path) as f:
        return IngestConfig.model_validate(yaml.safe_load(f))
