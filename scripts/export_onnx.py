#!/usr/bin/env python3
"""Export bge-m3 and bge-reranker-v2-m3 to ONNX + dynamic int8 quantization for the CPU/demo path (spec §5.2, §5.11).

Writes data/onnx/bge-m3-int8/{model.onnx,tokenizer.json} and data/onnx/bge-reranker-v2-m3-int8/...
Requires the `onnx` extra (optimum[onnxruntime]). Run on the workstation, not in CI.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export(repo: str, task: str, out: Path) -> None:
    from optimum.onnxruntime import (
        ORTModelForFeatureExtraction,
        ORTModelForSequenceClassification,
        ORTQuantizer,
    )
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    cls = (
        ORTModelForFeatureExtraction
        if task == "feature-extraction"
        else ORTModelForSequenceClassification
    )
    tmp = out.parent / (out.name + "-fp32")
    print(f"exporting {repo} -> {tmp}")
    model = cls.from_pretrained(repo, export=True)
    model.save_pretrained(tmp)
    AutoTokenizer.from_pretrained(repo).save_pretrained(tmp)
    print("quantizing (dynamic int8, avx512_vnni if available)")
    q = ORTQuantizer.from_pretrained(tmp)
    qcfg = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    q.quantize(save_dir=out, quantization_config=qcfg)
    # the quantizer names the file model_quantized.onnx; normalise
    qf = out / "model_quantized.onnx"
    if qf.exists():
        qf.replace(out / "model.onnx")
    shutil.copy(tmp / "tokenizer.json", out / "tokenizer.json")
    print(f"done: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["embedder", "reranker", "both"], default="both")
    a = ap.parse_args()
    base = ROOT / "data" / "onnx"
    base.mkdir(parents=True, exist_ok=True)
    if a.which in ("embedder", "both"):
        export("BAAI/bge-m3", "feature-extraction", base / "bge-m3-int8")
    if a.which in ("reranker", "both"):
        export(
            "BAAI/bge-reranker-v2-m3", "sequence-classification", base / "bge-reranker-v2-m3-int8"
        )


if __name__ == "__main__":
    main()
