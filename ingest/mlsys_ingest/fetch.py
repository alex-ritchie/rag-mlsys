"""Shallow-fetch the book repo at the pinned commit SHA (spec §5.1). Content lives only under data/."""

from __future__ import annotations

import subprocess
from pathlib import Path

from mlsys_ingest.config import IngestConfig


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def current_sha(checkout: Path) -> str | None:
    if not (checkout / ".git").exists():
        return None
    try:
        return _git("rev-parse", "HEAD", cwd=checkout).strip()
    except subprocess.CalledProcessError:
        return None


def fetch(cfg: IngestConfig, quiet: bool = False) -> Path:
    """Ensure `checkout_dir` holds exactly `commit_sha`. Shallow (depth 1) by SHA; idempotent."""
    checkout = cfg.checkout_path
    sha = cfg.source.commit_sha
    if current_sha(checkout) == sha:
        if not quiet:
            print(f"book already at {sha[:10]} in {checkout}")
        return checkout
    checkout.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").exists():
        _git("init", "-q", cwd=checkout)
        _git("remote", "add", "origin", cfg.source.repo, cwd=checkout)
    else:
        _git("remote", "set-url", "origin", cfg.source.repo, cwd=checkout)
    if not quiet:
        print(f"fetching {cfg.source.repo} @ {sha[:10]} (depth 1) ...")
    _git("fetch", "-q", "--depth", "1", "origin", sha, cwd=checkout)
    _git("checkout", "-q", "--force", "FETCH_HEAD", cwd=checkout)
    got = current_sha(checkout)
    if got != sha:
        raise RuntimeError(f"checkout SHA mismatch: wanted {sha}, got {got}")
    # Belt and braces: make sure the checkout can never be mistaken for part of this repo.
    (checkout / ".mlsysbook-content").write_text(
        "CC BY-NC-SA 4.0 book content. Never commit. See LICENSING.md.\n"
    )
    return checkout
