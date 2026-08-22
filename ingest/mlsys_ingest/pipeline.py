"""Chapter selection + numbering, parse, chunk — everything before the database."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mlsys_ingest.chunk import Chunk, chunk_document
from mlsys_ingest.config import IngestConfig, VolumeConfig
from mlsys_ingest.quarto import (
    Document,
    QmdParser,
    chapter_files_from_quarto_config,
    collect_labels,
    parse_file,
    resolve_crossrefs,
)
from mlsys_ingest.tokens import TokenCounter
from mlsys_ingest.values import ValueStats, materialise, resolver_available


@dataclass
class ChapterRef:
    volume: int
    chapter_num: int
    rel_path: str  # relative to the book checkout
    kind: str  # chapter | appendix | glossary


def select_chapters(cfg: IngestConfig, checkout: Path, volume: VolumeConfig) -> list[ChapterRef]:
    files = chapter_files_from_quarto_config(checkout / volume.quarto_config)
    sel = cfg.selection
    refs: list[ChapterRef] = []
    n_ch = n_app = 0
    for rel in files:
        if any(s in ("/" + rel) for s in sel.skip_path_substrings):
            continue
        full_rel = f"{volume.quarto_root}/{rel}"
        if "/backmatter/glossary" in rel:
            refs.append(ChapterRef(volume.number, sel.glossary_base, full_rel, "glossary"))
        elif "/backmatter/" in rel:
            refs.append(ChapterRef(volume.number, sel.appendix_base + n_app, full_rel, "appendix"))
            n_app += 1
        else:
            n_ch += 1
            refs.append(ChapterRef(volume.number, n_ch, full_rel, "chapter"))
    return refs


def parse_corpus(cfg: IngestConfig, checkout: Path) -> list[tuple[ChapterRef, Document]]:
    parser = QmdParser(
        drop_code_langs=cfg.chunking.drop_code_langs,
        drop_div_classes=cfg.chunking.drop_div_classes,
        drop_div_attrs=cfg.chunking.drop_div_attrs,
    )
    out: list[tuple[ChapterRef, Document]] = []
    resolve = cfg.chunking.resolve_inline_values and resolver_available()
    if cfg.chunking.resolve_inline_values and not resolver_available():
        print(
            "warning: data/mlsysim-venv missing (run `make setup-mlsysim`); inline values will be `[value]`"
        )
    vstats = ValueStats()
    for vol in cfg.volumes:
        for ref in select_chapters(cfg, checkout, vol):
            p = checkout / ref.rel_path
            if not p.exists():
                print(f"warning: missing {ref.rel_path}")
                continue
            if resolve:
                src = materialise(p, vstats)
                out.append((ref, parser.parse(src, ref.rel_path)))
            else:
                out.append((ref, parse_file(p, parser, source_file=ref.rel_path)))
    if resolve:
        print(
            f"inline values: {vstats.resolved}/{vstats.inline} resolved ({vstats.cells_ok}/{vstats.cells} cells)"
        )
    docs = [d for _, d in out]
    resolve_crossrefs(docs, collect_labels(docs))
    return out


def chunk_corpus(
    cfg: IngestConfig, parsed: list[tuple[ChapterRef, Document]], counter: TokenCounter
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for ref, doc in parsed:
        chunks.extend(
            chunk_document(
                doc,
                volume=ref.volume,
                chapter_num=ref.chapter_num,
                chapter_title=doc.title or Path(ref.rel_path).stem,
                commit_sha=cfg.source.commit_sha,
                counter=counter,
                min_tokens=cfg.chunking.min_tokens,
                max_tokens=cfg.chunking.max_tokens,
            )
        )
    return chunks
