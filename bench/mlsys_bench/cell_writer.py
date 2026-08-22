"""Write docs/benchmarks/cell-<tag>.json from a cell's log directory + its two bench result files.

Replaces the bash heredoc writer in scripts/ablation_cell.sh (which failed silently twice). Everything is derived
from artifacts on disk, so it can be re-run after the fact:  python -m mlsys_bench.cell_writer <tag> <serving.yaml>
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import yaml

from mlsys_common.settings import REPO_ROOT


def _first(pat: str, text: str) -> str:
    m = re.search(pat, text)
    return m.group(0) if m else ""


def write_cell(
    tag: str, serving_cfg: str, placement: dict | None = None, extra: dict | None = None
) -> Path:
    sp = REPO_ROOT / "data" / "logs" / f"cell-{tag}"
    vllm_log = (sp / "vllm.log").read_text(errors="replace") if (sp / "vllm.log").exists() else ""
    direct = sorted(glob.glob(str(REPO_ROOT / "bench" / "results" / f"*cell-{tag}-direct*.json")))
    gateway = sorted(glob.glob(str(REPO_ROOT / "bench" / "results" / f"*cell-{tag}-gateway*.json")))
    d = json.load(open(direct[-1])) if direct else {"runs": []}
    g = json.load(open(gateway[-1])) if gateway else {"runs": []}
    errs = 0
    for name in ("vllm.log", "reranker.log", "embedder.log"):
        f = sp / name
        if f.exists():
            errs += len(re.findall(r"(?i)out of memory|CUDA error", f.read_text(errors="replace")))
    g500 = (
        len(re.findall("500 Internal", (sp / "gateway.log").read_text(errors="replace")))
        if (sp / "gateway.log").exists()
        else 0
    )
    peak = int((sp / "peak").read_text().strip() or 0) if (sp / "peak").exists() else None
    vram = json.loads((sp / "vram.json").read_text()) if (sp / "vram.json").exists() else {}
    cfg = yaml.safe_load(open(REPO_ROOT / serving_cfg))
    doc = {
        "tag": tag,
        "serving_config": serving_cfg,
        "model": cfg.get("served_model_name"),
        "serving_args": cfg.get("args", {}),
        "load_seconds": vram.get("load_seconds"),
        "weights": _first(r"Model loading took [0-9.]+ GiB", vllm_log),
        "kv": _first(r"Available KV cache memory: [0-9.]+ GiB", vllm_log),
        "kv_tokens": _first(r"GPU KV cache size: [0-9,]+ tokens", vllm_log),
        "max_concurrency_at_full_context": _first(
            r"Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x", vllm_log
        ),
        "vram_mib": {
            "vllm_only": vram.get("vllm_only"),
            "with_embedder_reranker": vram.get("with_services"),
            "peak_gateway_load": peak,
        },
        "placement": placement or vram.get("placement") or {},
        "oom_or_cuda_errors": errs,
        "gateway_500s": g500,
        "vllm_alive_after": vram.get("alive"),
        "engine_direct": d["runs"],
        "gateway_e2e": g["runs"],
        "bench_files": [
            Path(p).relative_to(REPO_ROOT).as_posix() for p in (direct[-1:] + gateway[-1:])
        ],
        "vllm_version": (d.get("server") or {}).get("version"),
        **(extra or {}),
    }
    out = REPO_ROOT / "docs" / "benchmarks" / f"cell-{tag}.json"
    out.write_text(json.dumps(doc, indent=2))
    return out


def main() -> None:
    tag, cfg = sys.argv[1], sys.argv[2]
    placement = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    print("wrote", write_cell(tag, cfg, placement))


if __name__ == "__main__":
    main()
