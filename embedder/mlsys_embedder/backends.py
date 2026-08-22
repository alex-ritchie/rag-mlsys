"""bge-m3 dense embedding backends (spec §5.2): gpu | cpu (sentence-transformers) | onnx (int8, demo path)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np
from mlsys_common.settings import REPO_ROOT


class EmbeddingBackend(Protocol):
    name: str
    dim: int

    def embed(
        self, texts: list[str], normalize: bool = True, batch_size: int = 32
    ) -> np.ndarray: ...


class SentenceTransformerBackend:
    """GPU or CPU via sentence-transformers (fp16 on GPU)."""

    def __init__(
        self, model: str = "BAAI/bge-m3", device: str = "cpu", max_seq_length: int = 1024
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.name = model
        kwargs = {}
        if device.startswith("cuda"):
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        self._m = SentenceTransformer(model, device=device, **kwargs)
        self._m.max_seq_length = max_seq_length
        self.dim = int(self._m.get_sentence_embedding_dimension() or 1024)
        self.device = device

    def embed(self, texts: list[str], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        out = self._m.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(out, dtype=np.float32)


class OnnxBackend:
    """CPU ONNX Runtime (int8 export produced by scripts/export_onnx.py). Used in-process by the demo profile."""

    def __init__(self, model_dir: str | Path | None = None, max_seq_length: int = 512) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(
            model_dir
            or os.environ.get("EMBEDDER_ONNX_DIR", REPO_ROOT / "data" / "onnx" / "bge-m3-int8")
        )
        self.name = f"onnx:{model_dir.name}"
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_seq_length)
        self._tok.enable_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "4"))
        self._sess = ort.InferenceSession(
            str(model_dir / "model.onnx"), so, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self.dim = 1024

    def embed(self, texts: list[str], normalize: bool = True, batch_size: int = 16) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), batch_size):
            enc = self._tok.encode_batch(texts[i : i + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._sess.run(None, feed)[0]  # (B, T, H)
            cls = hidden[:, 0, :]  # bge-m3 dense = CLS pooling
            outs.append(cls)
        emb = np.concatenate(outs, axis=0).astype(np.float32)
        if normalize:
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return emb


class HashBackend:
    """Deterministic fake embeddings for unit/integration tests (no model download)."""

    name = "hash-test"
    dim = 1024

    def embed(self, texts: list[str], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, hash(tok) % self.dim] += 1.0
            if normalize:
                n = np.linalg.norm(out[i]) or 1.0
                out[i] /= n
        return out


def load_backend(mode: str, model: str = "BAAI/bge-m3") -> EmbeddingBackend:
    if mode == "gpu":
        return SentenceTransformerBackend(model, device="cuda")
    if mode == "cpu":
        return SentenceTransformerBackend(model, device="cpu")
    if mode == "onnx":
        return OnnxBackend()
    if mode == "test":
        return HashBackend()
    raise ValueError(f"unknown EMBEDDER_MODE {mode!r}")
