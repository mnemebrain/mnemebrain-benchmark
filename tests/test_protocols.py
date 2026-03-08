"""Tests for mnemebrain_benchmark.protocols."""

from __future__ import annotations

from mnemebrain_benchmark.protocols import EmbeddingProvider


class TestEmbeddingProviderProtocol:
    def test_runtime_checkable(self):
        class ValidProvider:
            def embed(self, text: str) -> list[float]:
                return [0.0]

            def similarity(self, a: list[float], b: list[float]) -> float:
                return 0.0

        assert isinstance(ValidProvider(), EmbeddingProvider)

    def test_not_implementing(self):
        class Invalid:
            pass

        assert not isinstance(Invalid(), EmbeddingProvider)

    def test_partial_not_implementing(self):
        class PartialProvider:
            def embed(self, text: str) -> list[float]:
                return [0.0]

        assert not isinstance(PartialProvider(), EmbeddingProvider)
