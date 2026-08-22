from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    """Stable identity of a chunk's text; survives re-ingestion (spec §5.6 uses hashes as labels)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def question_hash(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()[:32]
