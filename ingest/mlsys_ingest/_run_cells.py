"""Subprocess helper: execute a chapter's `{python}` cells in order and evaluate its inline `{python}` expressions.

Runs inside data/mlsysim-venv (see scripts/setup_mlsysim.sh). Reads a .qmd path from argv, prints JSON:
  {"cells": N, "cells_ok": M, "values": [str|null, ...]}   (one entry per inline expression, in document order)
Figure-styling imports from the book's build tooling are stubbed; cells that fail are skipped (their values resolve
to null), which keeps a broken figure cell from blanking a chapter's numbers.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import types
from pathlib import Path

CELL_RE = re.compile(r"^```\{python\}[^\n]*\n(.*?)^```", re.S | re.M)
INLINE_RE = re.compile(r"`\{python\}\s*([^`]+)`")


def _stub_book_tooling() -> None:
    class _Anything(types.ModuleType):
        def __getattr__(self, name: str):  # any attribute is a no-op callable
            return lambda *a, **k: None

    for name in ("book", "book.tools", "book.tools.figures", "book.tools.figures.style"):
        sys.modules.setdefault(name, _Anything(name))


def main(path: str) -> None:
    _stub_book_tooling()
    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:
        pass
    src = Path(path).read_text(encoding="utf-8")
    # inline expressions must be evaluated against the cells *before* them, so walk the document in order
    ns: dict = {"__name__": "__chapter__"}
    out: list[str | None] = []
    cells = 0
    cells_ok = 0
    events = sorted(
        [(m.start(), "cell", m) for m in CELL_RE.finditer(src)]
        + [(m.start(), "inline", m) for m in INLINE_RE.finditer(src)]
    )
    cell_spans = [(m.start(), m.end()) for _, k, m in events if k == "cell"]
    for start, kind, m in events:
        if kind == "cell":
            cells += 1
            body = "\n".join(
                line for line in m.group(1).splitlines() if not line.lstrip().startswith("#|")
            )
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    exec(compile(body, f"{Path(path).name}:cell{cells}", "exec"), ns)
                cells_ok += 1
            except BaseException:  # a failing figure cell must not abort the chapter
                pass
        else:
            if any(a <= start < b for a, b in cell_spans):
                continue  # an inline-looking match inside a cell body (a string literal), not document text
            try:
                v = eval(m.group(1).strip(), ns)  # the book's own source, executed on purpose
                out.append(str(v))
            except BaseException:
                out.append(None)
    print(json.dumps({"cells": cells, "cells_ok": cells_ok, "values": out}))


if __name__ == "__main__":
    main(sys.argv[1])
