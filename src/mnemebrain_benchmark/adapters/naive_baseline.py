"""Naive baseline memory system -- flat vector store, no belief logic."""
from __future__ import annotations

from uuid import uuid4

from mnemebrain_benchmark.interface import (
    Capability,
    MemorySystem,
    QueryResult,
    StoreResult,
)
from mnemebrain_benchmark.protocols import EmbeddingProvider


class NaiveBaseline(MemorySystem):
    """Flat in-memory vector store with cosine-similarity deduplication."""

    def __init__(self, embedder: EmbeddingProvider, threshold: float = 0.92) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._beliefs: list[dict] = []

    def name(self) -> str:
        return "naive_baseline"

    def capabilities(self) -> set[Capability]:
        return {Capability.STORE, Capability.QUERY}

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        embedding = self._embedder.embed(claim)
        for b in self._beliefs:
            sim = self._embedder.similarity(embedding, b["embedding"])
            if sim >= self._threshold:
                return StoreResult(
                    belief_id=b["id"],
                    merged=True,
                    contradiction_detected=False,
                    truth_state=None,
                    confidence=None,
                )
        belief_id = str(uuid4())
        self._beliefs.append({"id": belief_id, "claim": claim, "embedding": embedding})
        return StoreResult(
            belief_id=belief_id,
            merged=False,
            contradiction_detected=False,
            truth_state=None,
            confidence=None,
        )

    def query(self, claim: str) -> list[QueryResult]:
        if not self._beliefs:
            return []
        embedding = self._embedder.embed(claim)
        results = []
        for b in self._beliefs:
            sim = self._embedder.similarity(embedding, b["embedding"])
            if sim >= 0.5:
                results.append(QueryResult(
                    belief_id=b["id"],
                    claim=b["claim"],
                    confidence=None,
                    truth_state=None,
                ))
        return results

    def reset(self) -> None:
        self._beliefs.clear()
