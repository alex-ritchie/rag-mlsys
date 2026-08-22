"""`python -m mlsys_bench.run <config.yaml>` -> bench/results/<run-id>.json (full config embedded)."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from mlsys_common.settings import REPO_ROOT

from mlsys_bench.loadgen import run_load, summarize

RESULTS = REPO_ROOT / "bench" / "results"
DEFAULT_PROMPTS = REPO_ROOT / "bench" / "configs" / "prompts.txt"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def gpu_info() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        name, drv, mem = [x.strip() for x in out.split(",")]
        return {"name": name, "driver": drv, "memory": mem}
    except Exception:
        return {}


async def probe(base_url: str, target: str) -> dict:
    """Record what the server says about itself (model id, vLLM version) for provenance."""
    info: dict = {}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            if target in ("rag-e2e", "gateway"):
                info["health"] = (await c.get(f"{base_url}/api/health")).json()
            else:
                info["models"] = (await c.get(f"{base_url}/models")).json()
                v = await c.get(f"{base_url.rsplit('/v1', 1)[0]}/version")
                if v.status_code == 200:
                    info["version"] = v.json()
    except Exception as e:
        info["probe_error"] = str(e)
    return info


async def main_async(cfg_path: Path, out_dir: Path, tag: str | None) -> Path:
    cfg = yaml.safe_load(cfg_path.read_text())
    prompts_file = Path(cfg.get("prompts_file", DEFAULT_PROMPTS))
    prompts = [
        ln.strip()
        for ln in prompts_file.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    target = cfg.get("target", "rag-e2e")
    base_url = cfg["base_url"]
    server = await probe(base_url, target)
    runs = []
    for conc in cfg.get("concurrency", [1, 4, 8, 16, 32]):
        print(f"-- concurrency {conc}")
        samples, wall = await run_load(
            target=target,
            base_url=base_url,
            prompts=prompts,
            concurrency=conc,
            requests=cfg.get("requests_per_level"),
            duration_s=cfg.get("duration_s"),
            model=cfg.get("model", ""),
            max_tokens=cfg.get("max_tokens", 512),
            top_k=cfg.get("top_k"),
            extra_body=cfg.get("extra_body"),
            warmup=cfg.get("warmup", 2),
        )
        s = summarize(samples, wall, conc)
        runs.append(s)
        print(
            f"   rps={s['requests_per_s']} tok/s={s['output_tokens_per_s']} ttft p50/p99={s['ttft_ms']['p50']}/{s['ttft_ms']['p99']} total p50/p99={s['total_ms']['p50']}/{s['total_ms']['p99']} errors={s['errors']}"
        )
    name = cfg.get("name", cfg_path.stem) + (f"-{tag}" if tag else "")
    result = {
        "name": name,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "gpu": gpu_info(),
        "server": server,
        "config": cfg,
        "config_file": str(cfg_path.relative_to(REPO_ROOT))
        if cfg_path.is_relative_to(REPO_ROOT)
        else str(cfg_path),
        "serving_config": yaml.safe_load((REPO_ROOT / cfg["serving_config"]).read_text())
        if cfg.get("serving_config")
        else None,
        "reproduce": f"python -m mlsys_bench.run {cfg_path}",
        "runs": runs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{name}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    asyncio.run(main_async(Path(a.config), Path(a.out_dir), a.tag))


if __name__ == "__main__":
    main()
