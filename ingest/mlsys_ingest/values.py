"""Materialise the book's computed numbers (spec §5.1 intent: preserve prose content).

Quarto renders `` `{python} expr` `` inline expressions from hidden calculation cells backed by the repo's `mlsysim`
package. A third of the chunks (927/2815 at commit 2bd97c5) contain such values — the training-memory multipliers,
parameter counts, bandwidth figures — so leaving `[value]` placeholders strips the book's quantitative content.
This module runs `_run_cells.py` inside data/mlsysim-venv for each chapter and substitutes the rendered strings
into the source *before* parsing; anything that fails to evaluate keeps the `[value]` placeholder.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mlsys_common.settings import REPO_ROOT

VENV_PY = REPO_ROOT / "data" / "mlsysim-venv" / "bin" / "python"
RUNNER = Path(__file__).with_name("_run_cells.py")
INLINE_RE = re.compile(r"`\{python\}\s*([^`]+)`")
CELL_RE = re.compile(r"^```\{python\}[^\n]*\n(.*?)^```", re.S | re.M)


@dataclass
class ValueStats:
    inline: int = 0
    resolved: int = 0
    cells: int = 0
    cells_ok: int = 0


def resolver_available() -> bool:
    return VENV_PY.exists()


def evaluate(qmd_path: Path, python: Path = VENV_PY, timeout: int = 600) -> dict | None:
    """Run the chapter's cells; returns {"cells", "cells_ok", "values": [one per inline expression, None = unresolved]}."""
    if not python.exists():
        return None
    try:
        proc = subprocess.run(
            [str(python), str(RUNNER), str(qmd_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, ValueError, KeyError, IndexError):
        return None


def substitute(src: str, values: list[str | None] | None, stats: ValueStats | None = None) -> str:
    """Replace inline expressions (outside cell bodies) with their rendered values; keep `[value]` when unknown."""
    cell_spans = [(m.start(), m.end()) for m in CELL_RE.finditer(src)]
    it = iter(values or [])
    out: list[str] = []
    last = 0
    for m in INLINE_RE.finditer(src):
        if any(a <= m.start() < b for a, b in cell_spans):
            continue
        v = next(it, None)
        if stats is not None:
            stats.inline += 1
            stats.resolved += v is not None
        out.append(src[last : m.start()])
        out.append(v if v is not None else "[value]")
        last = m.end()
    out.append(src[last:])
    return "".join(out)


def materialise(qmd_path: Path, stats: ValueStats | None = None) -> str:
    src = qmd_path.read_text(encoding="utf-8")
    res = evaluate(qmd_path) if resolver_available() else None
    if res and stats is not None:
        stats.cells += res.get("cells", 0)
        stats.cells_ok += res.get("cells_ok", 0)
    return substitute(src, res["values"] if res else None, stats)
