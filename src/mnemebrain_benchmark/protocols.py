"""Local protocol definitions replacing mnemebrain_core imports."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers used by benchmark adapters."""

    def embed(self, text: str) -> list[float]: ...

    def similarity(self, a: list[float], b: list[float]) -> float: ...
