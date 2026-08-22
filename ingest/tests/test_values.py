import json
import subprocess
import sys
from pathlib import Path

from mlsys_ingest.values import RUNNER, ValueStats, substitute

SYNTH = """---
title: synthetic
---

# Widgets

```{python}
#| echo: false
class Stats:
    gears = 8
    gears_str = f"{gears} gears"
    ratio_str = "4$\\\\times$"
```

A standard widget has `{python} Stats.gears_str` and costs `{python} Stats.ratio_str` more.

```{python}
#| echo: false
raise RuntimeError("a broken figure cell must not abort the chapter")
```

Still `{python} Stats.gears * 2` teeth, and `{python} undefined_name` stays a placeholder.
"""


def test_runner_executes_cells_in_order_and_tolerates_failures(tmp_path: Path):
    q = tmp_path / "w.qmd"
    q.write_text(SYNTH)
    out = subprocess.run(
        [sys.executable, str(RUNNER), str(q)], capture_output=True, text=True, check=True
    ).stdout
    res = json.loads(out.strip().splitlines()[-1])
    assert res["cells"] == 2 and res["cells_ok"] == 1
    assert res["values"] == ["8 gears", "4$\\times$", "16", None]


def test_substitute_replaces_outside_cells_only():
    stats = ValueStats()
    text = substitute(SYNTH, ["8 gears", "4$\\times$", "16", None], stats)
    assert "has 8 gears and costs 4$\\times$ more" in text
    assert "Still 16 teeth, and [value] stays" in text
    assert "`{python} Stats.gears_str`" not in text
    assert (
        'gears_str = f"{gears} gears"' in text
    )  # cell body untouched (dropped later by the parser)
    assert stats.inline == 4 and stats.resolved == 3


def test_substitute_without_values_keeps_placeholders():
    text = substitute(SYNTH, None)
    assert text.count("[value]") == 4
