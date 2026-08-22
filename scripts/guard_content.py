#!/usr/bin/env python3
"""Licensing guard (spec §4 M0 / §7.2).

Fails if any staged (or tracked, in CI) file is:
  * under data/ (book checkout, indexes, model files)
  * a model artifact (*.gguf, *.safetensors, *.bin, *.pt, *.onnx)
  * a Quarto source (*.qmd) outside the synthetic fixture directory
  * larger than 20 MB
  * or contains the book's distinctive boilerplate (a cheap tripwire for
    accidentally committed chapter text; fixtures must be synthetic).

Usage:
  guard_content.py --staged      # pre-commit: files in the index
  guard_content.py --tracked     # CI: every tracked file in HEAD
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys

MAX_BYTES = 20 * 1024 * 1024
BLOCKED_GLOBS = ["data/*", "data/**", "*.gguf", "*.safetensors", "*.bin", "*.pt", "*.onnx"]
QMD_ALLOWED_PREFIX = "ingest/tests/fixtures/"
# Tripwires: strings that occur in every real chapter source and never in synthetic fixtures.
TRIPWIRES = [
    b"\\mlsysstack{",
    b"quiz: ",
    b"concepts: ",
    b"from mlsysim",
    b"harvard-edge/cs249r_book/blob",
]
TEXT_EXT = {".qmd", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".py", ".ts", ".tsx"}


def git_files(mode: str) -> list[str]:
    if mode == "staged":
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    else:
        cmd = ["git", "ls-files"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def check(path: str) -> list[str]:
    problems: list[str] = []
    for g in BLOCKED_GLOBS:
        if fnmatch.fnmatch(path, g):
            problems.append(f"{path}: matches blocked pattern {g!r}")
    if path.endswith(".qmd") and not path.startswith(QMD_ALLOWED_PREFIX):
        problems.append(f"{path}: .qmd outside {QMD_ALLOWED_PREFIX}")
    if os.path.isfile(path):
        size = os.path.getsize(path)
        if size > MAX_BYTES:
            problems.append(f"{path}: {size / 1e6:.1f} MB > 20 MB")
        if os.path.splitext(path)[1] in TEXT_EXT and path != "scripts/guard_content.py":
            with open(path, "rb") as f:
                blob = f.read()
            for tw in TRIPWIRES:
                if tw in blob:
                    problems.append(f"{path}: contains tripwire {tw!r} (book content?)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--staged", action="store_true")
    g.add_argument("--tracked", action="store_true")
    args = ap.parse_args()
    files = git_files("staged" if args.staged else "tracked")
    problems = [p for f in files for p in check(f)]
    if problems:
        print("CONTENT GUARD FAILED:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
        print("\nSee LICENSING.md §2 — book content must never enter the repo.", file=sys.stderr)
        return 1
    print(f"content guard: {len(files)} files OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
