"""Shared embedding provider implementations."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a)
    vb = np.array(b)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(np.dot(va, vb) / norm)


class SentenceTransformerProvider:
    """EmbeddingProvider backed by sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install 'mnemebrain-benchmark[embeddings]'"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Embed text into a vector."""
        return list(self._model.encode(text).tolist())

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        return cosine_similarity(a, b)
