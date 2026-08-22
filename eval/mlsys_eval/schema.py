"""Golden-set record format (spec §5.6). Chunk *hashes* label the sources so labels survive re-ingestion."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from mlsys_common.settings import REPO_ROOT
from pydantic import BaseModel, Field

GOLDEN_DIR = REPO_ROOT / "eval" / "golden"
CANDIDATES_PATH = GOLDEN_DIR / "candidates.jsonl"  # gitignored: may contain near-verbatim extracts
GOLDEN_PATH = GOLDEN_DIR / "golden.jsonl"  # committed: questions + paraphrased key points only
STAMP_PATH = GOLDEN_DIR / "golden.verified.json"  # committed: proves the human pass happened
JUDGE_LABELS_PATH = (
    GOLDEN_DIR / "judge_labels.jsonl"
)  # committed: 30 hand labels for judge validation

QType = Literal["single", "multi", "unanswerable"]


class GoldenItem(BaseModel):
    id: str
    question: str
    answer_key_points: list[str] = Field(
        default_factory=list, description="paraphrased by the verifier, never extracted text"
    )
    source_chunk_content_hashes: list[str] = Field(default_factory=list)
    type: QType
    chapter: str  # "vol1/ch8" style label for stratification reporting
    verified: bool = False


class Candidate(GoldenItem):
    """Generated Q/A before human review. Kept out of git."""

    generated_by: str = ""
    notes: str = ""


def read_jsonl(path: Path, model):
    if not path.exists():
        return []
    with open(path) as f:
        return [model.model_validate_json(line) for line in f if line.strip()]


def write_jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_stamp(
    golden_path: Path = GOLDEN_PATH, stamp_path: Path = STAMP_PATH, verifier: str = "owner"
) -> dict:
    items = read_jsonl(golden_path, GoldenItem)
    stamp = {
        "golden_sha256": file_sha256(golden_path),
        "count": len(items),
        "by_type": {
            t: sum(i.type == t for i in items) for t in ("single", "multi", "unanswerable")
        },
        "verified_at": datetime.now(UTC).isoformat(),
        "verifier": verifier,
    }
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")
    return stamp


class UnverifiedGoldenSetError(RuntimeError):
    pass


def load_verified_golden(
    golden_path: Path = GOLDEN_PATH, stamp_path: Path = STAMP_PATH
) -> list[GoldenItem]:
    """The harness refuses to run against an unverified set (spec §4 M5)."""
    if not golden_path.exists():
        raise UnverifiedGoldenSetError(
            f"{golden_path} missing — run `make golden-generate` then `make golden-verify`"
        )
    if not stamp_path.exists():
        raise UnverifiedGoldenSetError(
            f"{stamp_path} missing — the golden set has not been human-verified (`make golden-verify`)"
        )
    stamp = json.loads(stamp_path.read_text())
    if stamp.get("golden_sha256") != file_sha256(golden_path):
        raise UnverifiedGoldenSetError(
            "golden.jsonl changed since verification — re-run `make golden-verify`"
        )
    items = read_jsonl(golden_path, GoldenItem)
    unverified = [i.id for i in items if not i.verified]
    if unverified:
        raise UnverifiedGoldenSetError(
            f"{len(unverified)} items not marked verified: {unverified[:5]}"
        )
    return items
