#!/usr/bin/env bash
# Install the book's own calculation package (mlsysim, from the fetched checkout) into data/mlsysim-venv so the
# ingestion pipeline can execute the chapters' hidden `{python}` cells and materialise inline `{python}` values
# (the numbers in the text). Isolated from the workspace because it pins pint/pydantic/numpy independently.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -d data/book/mlsysim ]] || { echo "book checkout missing: run make fetch"; exit 1; }
uv venv -q -p 3.12 data/mlsysim-venv
VIRTUAL_ENV=$PWD/data/mlsysim-venv uv pip install -q -e data/book/mlsysim numpy matplotlib pandas
data/mlsysim-venv/bin/python -c "import mlsysim, numpy; print('mlsysim ok')"
