#!/usr/bin/env python3
"""Launch `vllm serve` from a config/serving/*.yaml so every benchmark records the exact flags.

  python scripts/serve_vllm.py config/serving/vllm-qwen38-27b.yaml [--dry-run] [--set key=value ...]

`--set` overrides (e.g. --set enforce-eager=true --set gpu-memory-utilization=0.85) are the
M8 lever mechanism; the effective config is printed as JSON before launch and can be captured
by the bench harness.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# vLLM lives in its own venv (data/vllm-venv) because it pins torch exactly; see docs/RUN_IT_YOURSELF.md §5.
VLLM_BIN = ROOT / "data" / "vllm-venv" / "bin" / "vllm"


def build_cmd(cfg: dict) -> list[str]:
    if cfg.get("engine") == "llamacpp":
        cmd = [
            "llama-server",
            "-m",
            cfg["model_file"],
            "--port",
            str(cfg.get("port", 8003)),
            "--alias",
            cfg.get("served_model_name", "model"),
        ]
        for k, v in cfg.get("args", {}).items():
            flag = ("-" if len(k) == 1 else "--") + k
            if v is True:
                cmd.append(flag)
            elif v is False or v is None:
                continue
            else:
                cmd += [flag, str(v)]
        return cmd
    vllm = str(VLLM_BIN) if VLLM_BIN.exists() else "vllm"
    cmd = [
        vllm,
        "serve",
        cfg["model"],
        "--port",
        str(cfg.get("port", 8003)),
        "--served-model-name",
        cfg.get("served_model_name", cfg["model"]),
    ]
    for k, v in cfg.get("args", {}).items():
        flag = "--" + k
        if v is True:
            cmd.append(flag)
        elif v is False or v is None:
            continue
        else:
            cmd += [flag, str(v)]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="key=value override of args.*")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    for kv in a.set:
        k, _, v = kv.partition("=")
        vv: object = v
        if v.lower() in ("true", "false"):
            vv = v.lower() == "true"
        elif v.lower() in ("none", "null", ""):
            vv = None
        else:
            try:
                vv = int(v) if v.isdigit() else float(v)
            except ValueError:
                vv = v
        cfg.setdefault("args", {})[k] = vv
    cmd = build_cmd(cfg)
    print(
        json.dumps(
            {"effective_config": cfg, "cmd": " ".join(shlex.quote(c) for c in cmd)}, indent=2
        ),
        file=sys.stderr,
    )
    if a.dry_run:
        return 0
    env = {**os.environ, **{k: str(v) for k, v in cfg.get("env", {}).items()}}
    env["HF_HOME"] = env.get("MLSYS_HF_HOME") or str(
        ROOT / "data" / "hf-cache"
    )  # repo cache, regardless of the shell
    env["HF_HUB_CACHE"] = str(Path(env["HF_HOME"]) / "hub")
    env.pop("TRANSFORMERS_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    env.setdefault(
        "HF_HUB_DISABLE_XET", "1"
    )  # the Xet transport stalled repeatedly on this link; plain HTTPS is fine
    # exec so the server *is* this process: whoever started us (make up, m3_measure.sh) can stop it by PID.
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main())
