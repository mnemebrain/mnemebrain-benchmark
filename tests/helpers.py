"""Shared test helpers for mnemebrain_benchmark tests."""
from __future__ import annotations

import math


class FakeEmbedder:
    """Deterministic embedder for testing adapters that use EmbeddingProvider protocol."""

    def embed(self, text: str) -> list[float]:
        h = hash(text) % 1000
        return [h / 1000.0, (h * 7 % 1000) / 1000.0, (h * 13 % 1000) / 1000.0]

    def similarity(self, a: list[float], b: list[float]) -> float:
        if a == b:
            return 1.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class HighSimEmbedder:
    """Embedder that always returns identical embeddings (similarity=1.0)."""

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def similarity(self, a: list[float], b: list[float]) -> float:
        return 1.0


class LowSimEmbedder:
    """Embedder where all texts have low mutual similarity."""

    _counter = 0

    def embed(self, text: str) -> list[float]:
        LowSimEmbedder._counter += 1
        c = LowSimEmbedder._counter
        return [float(c), 0.0, 0.0]

    def similarity(self, a: list[float], b: list[float]) -> float:
        return 0.1
