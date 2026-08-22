"""bge-reranker-v2-m3 cross-encoder backends: gpu | cpu | onnx (int8) | test."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np
from mlsys_common.settings import REPO_ROOT


class RerankBackend(Protocol):
    name: str

    def score(self, query: str, docs: list[str]) -> np.ndarray: ...


class CrossEncoderBackend:
    def __init__(
        self, model: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu", max_length: int = 1024
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        kw = {}
        if device.startswith("cuda"):
            kw["model_kwargs"] = {"torch_dtype": torch.float16}
        self.name = model
        self._m = CrossEncoder(model, device=device, max_length=max_length, **kw)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        s = self._m.predict(
            [(query, d) for d in docs],
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(s, dtype=np.float32)


class OnnxRerankBackend:
    def __init__(self, model_dir: str | Path | None = None, max_length: int = 512) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(
            model_dir
            or os.environ.get(
                "RERANKER_ONNX_DIR", REPO_ROOT / "data" / "onnx" / "bge-reranker-v2-m3-int8"
            )
        )
        self.name = f"onnx:{model_dir.name}"
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "4"))
        self._sess = ort.InferenceSession(
            str(model_dir / "model.onnx"), so, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._sess.get_inputs()}

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(docs), 8):
            enc = self._tok.encode_batch([(query, d) for d in docs[i : i + 8]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            logits = self._sess.run(None, feed)[0]
            out.append(logits.reshape(-1))
        return np.concatenate(out).astype(np.float32)


class LexicalOverlapBackend:
    """Deterministic test stand-in: Jaccard overlap between query and doc tokens."""

    name = "lexical-test"

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        q = set(query.lower().split())
        return np.array(
            [
                len(q & set(d.lower().split())) / (len(q | set(d.lower().split())) or 1)
                for d in docs
            ],
            dtype=np.float32,
        )


def load_backend(mode: str, model: str = "BAAI/bge-reranker-v2-m3") -> RerankBackend:
    max_length = int(os.environ.get("RERANKER_MAX_LENGTH", "1024"))  # 512 halves activation memory when co-resident with vLLM
    if mode == "gpu":
        return CrossEncoderBackend(model, device="cuda", max_length=max_length)
    if mode == "cpu":
        return CrossEncoderBackend(model, device="cpu", max_length=max_length)
    if mode == "onnx":
        return OnnxRerankBackend()
    if mode == "test":
        return LexicalOverlapBackend()
    raise ValueError(f"unknown RERANKER_MODE {mode!r}")
