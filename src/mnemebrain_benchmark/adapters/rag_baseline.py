"""RAG (Retrieval-Augmented Generation) baseline adapter."""

from __future__ import annotations

from uuid import uuid4

from mnemebrain_benchmark.interface import (
    Capability,
    MemorySystem,
    QueryResult,
    StoreResult,
)
from mnemebrain_benchmark.protocols import EmbeddingProvider


class RAGBaseline(MemorySystem):
    """Vector-store RAG memory with overwrite-on-conflict semantics."""

    def __init__(self, embedder: EmbeddingProvider, threshold: float = 0.75) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._entries: list[dict] = []

    def name(self) -> str:
        return "rag_baseline"

    def capabilities(self) -> set[Capability]:
        return {Capability.STORE, Capability.QUERY}

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        embedding = self._embedder.embed(claim)
        for i, entry in enumerate(self._entries):
            sim = self._embedder.similarity(embedding, entry["embedding"])
            if sim >= self._threshold:
                self._entries[i] = {
                    "id": entry["id"],
                    "claim": claim,
                    "embedding": embedding,
                }
                return StoreResult(
                    belief_id=entry["id"],
                    merged=True,
                    contradiction_detected=False,
                    truth_state=None,
                    confidence=None,
                )
        entry_id = str(uuid4())
        self._entries.append(
            {
                "id": entry_id,
                "claim": claim,
                "embedding": embedding,
            }
        )
        return StoreResult(
            belief_id=entry_id,
            merged=False,
            contradiction_detected=False,
            truth_state=None,
            confidence=None,
        )

    def query(self, claim: str) -> list[QueryResult]:
        if not self._entries:
            return []
        embedding = self._embedder.embed(claim)
        results = []
        for entry in self._entries:
            sim = self._embedder.similarity(embedding, entry["embedding"])
            if sim >= 0.5:
                results.append(
                    QueryResult(
                        belief_id=entry["id"],
                        claim=entry["claim"],
                        confidence=None,
                        truth_state=None,
                    )
                )
        return results

    def reset(self) -> None:
        self._entries.clear()
