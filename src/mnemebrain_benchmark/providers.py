"""Shared embedding provider implementations."""

from __future__ import annotations

import os

import numpy as np


EMBEDDER_CHOICES = [
    "sentence_transformers",
    "openai",
    "ollama",
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a)
    vb = np.array(b)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(np.dot(va, vb) / norm)


def build_embedder(name: str | None = None, model: str | None = None):
    """Factory: build an EmbeddingProvider by name.

    Args:
        name: One of "sentence_transformers", "openai", "ollama".
              If None, auto-detects: OPENAI_API_KEY -> openai, else sentence_transformers.
        model: Optional model override (provider-specific).
    """
    if name is None:
        if os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "sentence_transformers"

    if name == "sentence_transformers":
        return SentenceTransformerProvider(model_name=model or "all-MiniLM-L6-v2")
    elif name == "openai":
        return OpenAIEmbeddingProvider(model=model or "text-embedding-3-small")
    elif name == "ollama":
        return OllamaEmbeddingProvider(model=model or "nomic-embed-text")
    else:
        raise ValueError(
            f"Unknown embedder: {name!r}. Choose from: {', '.join(EMBEDDER_CHOICES)}"
        )


class SentenceTransformerProvider:
    """EmbeddingProvider backed by sentence-transformers (local)."""

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


class OpenAIEmbeddingProvider:
    """EmbeddingProvider backed by OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai is required: pip install 'mnemebrain-benchmark[openai]'"
            ) from exc
        self._client = openai.OpenAI()
        self._model = model

    def embed(self, text: str) -> list[float]:
        """Embed text into a vector."""
        resp = self._client.embeddings.create(input=text, model=self._model)
        return resp.data[0].embedding

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        return cosine_similarity(a, b)


class OllamaEmbeddingProvider:
    """EmbeddingProvider backed by a local Ollama instance."""

    def __init__(
        self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx is required for Ollama embeddings") from exc
        self._client = httpx.Client(base_url=base_url, timeout=60.0)
        self._model = model

    def embed(self, text: str) -> list[float]:
        """Embed text into a vector."""
        resp = self._client.post("/api/embed", json={"model": self._model, "input": text})
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        return cosine_similarity(a, b)
