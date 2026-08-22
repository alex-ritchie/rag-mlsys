"""Structure-aware chunking (spec §5.1).

Rules:
  * split at heading boundaries first; pack sibling content greedily into 400-800 tokens;
  * never split code / table / equation / figure blocks; oversized atomic blocks become a
    chunk of their own with oversize=True;
  * long paragraphs are split at sentence boundaries;
  * heading path is prepended only at embedding time (`embed_text`), `text` stays raw.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mlsys_common.hashing import content_hash

from mlsys_ingest.quarto import ATOMIC_KINDS, Document
from mlsys_ingest.tokens import TokenCounter

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    volume: int
    chapter_num: int
    chapter_title: str
    section_path: list[str]
    source_file: str
    char_start: int
    char_end: int
    token_count: int
    commit_sha: str
    text: str
    oversize: bool = False
    content_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.text)

    @property
    def heading_path(self) -> str:
        return heading_path_string(
            self.volume, self.chapter_num, self.chapter_title, self.section_path
        )

    def embed_text(self) -> str:
        """The form that is embedded: heading path + raw text (spec §5.1)."""
        return f"{self.heading_path}\n\n{self.text}"


def _chapter_label(chapter_num: int) -> str:
    if chapter_num >= 200:
        return "Glossary"
    if chapter_num >= 100:
        return f"Appendix {chr(ord('A') + chapter_num - 100)}"
    return f"Ch {chapter_num}"


def heading_path_string(
    volume: int, chapter_num: int, chapter_title: str, section_path: list[str]
) -> str:
    parts = [f"Vol {volume}", f"{_chapter_label(chapter_num)}: {chapter_title}", *section_path]
    return " > ".join(parts)


@dataclass
class _Item:
    """A packable unit: a block (or piece of a paragraph) under a heading context."""

    text: str
    tokens: int
    section_path: list[str]
    char_start: int
    char_end: int
    atomic: bool
    section_key: tuple[str, ...]


def _split_paragraph(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    sents = _SENT.split(text)
    out: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for s in sents:
        t = counter.count(s)
        if cur and cur_tokens + t > max_tokens:
            out.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(s)
        cur_tokens += t
    if cur:
        out.append(" ".join(cur))
    return out


def _items(doc: Document, counter: TokenCounter, max_tokens: int) -> list[_Item]:
    stack: list[tuple[int, str]] = []  # (level, title) for levels >= 2
    items: list[_Item] = []
    for b in doc.blocks:
        if b.kind == "heading":
            if b.level == 1:
                continue
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.text))
            continue
        path = [t for _, t in stack]
        key = tuple(path)
        if b.kind in ATOMIC_KINDS:
            items.append(
                _Item(b.text, counter.count(b.text), path, b.char_start, b.char_end, True, key)
            )
            continue
        t = counter.count(b.text)
        if t <= max_tokens:
            items.append(_Item(b.text, t, path, b.char_start, b.char_end, False, key))
        else:
            for piece in _split_paragraph(b.text, counter, max_tokens):
                items.append(
                    _Item(piece, counter.count(piece), path, b.char_start, b.char_end, False, key)
                )
    return items


def chunk_document(
    doc: Document,
    *,
    volume: int,
    chapter_num: int,
    chapter_title: str,
    commit_sha: str,
    counter: TokenCounter,
    min_tokens: int = 400,
    max_tokens: int = 800,
) -> list[Chunk]:
    items = _items(doc, counter, max_tokens)
    chunks: list[Chunk] = []
    cur: list[_Item] = []
    cur_tokens = 0

    def flush(oversize: bool = False) -> None:
        nonlocal cur, cur_tokens
        if not cur:
            return
        text = "\n\n".join(it.text for it in cur)
        chunks.append(
            Chunk(
                volume=volume,
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                section_path=list(cur[0].section_path),
                source_file=doc.source_file,
                char_start=cur[0].char_start,
                char_end=cur[-1].char_end,
                token_count=cur_tokens,
                commit_sha=commit_sha,
                text=text,
                oversize=oversize,
            )
        )
        cur, cur_tokens = [], 0

    for it in items:
        # oversized atomic block: its own chunk
        if it.atomic and it.tokens > max_tokens:
            flush()
            cur, cur_tokens = [it], it.tokens
            flush(oversize=True)
            continue
        # heading boundary: flush if we already have a full-enough chunk; else keep packing siblings
        at_boundary = bool(cur) and it.section_key != cur[-1].section_key
        if at_boundary and cur_tokens >= min_tokens:
            flush()
        if cur and cur_tokens + it.tokens > max_tokens:
            flush()
        cur.append(it)
        cur_tokens += it.tokens
    flush()
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict[str, float]:
    if not chunks:
        return {"count": 0}
    toks = sorted(c.token_count for c in chunks)
    return {
        "count": len(chunks),
        "tokens_min": toks[0],
        "tokens_p50": toks[len(toks) // 2],
        "tokens_max": toks[-1],
        "tokens_mean": round(sum(toks) / len(toks), 1),
        "oversize": sum(c.oversize for c in chunks),
        "under_min": 0,  # filled by caller with the configured min
    }
