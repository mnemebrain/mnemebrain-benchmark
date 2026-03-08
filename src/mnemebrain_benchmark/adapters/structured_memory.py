"""Structured memory baseline (Mem0-style) adapter."""
from __future__ import annotations

from uuid import uuid4

from mnemebrain_benchmark.interface import (
    Capability,
    ExplainResult,
    MemorySystem,
    QueryResult,
    RetractResult,
    ReviseResult,
    StoreResult,
)
from mnemebrain_benchmark.protocols import EmbeddingProvider


class StructuredMemoryBaseline(MemorySystem):
    """Structured key-value memory with slot updates (Mem0-style)."""

    def __init__(self, embedder: EmbeddingProvider, threshold: float = 0.75) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._slots: list[dict] = []

    def name(self) -> str:
        return "structured_memory"

    def capabilities(self) -> set[Capability]:
        return {
            Capability.STORE,
            Capability.QUERY,
            Capability.RETRACT,
            Capability.EXPLAIN,
            Capability.REVISE,
        }

    def _find_similar(self, embedding: list[float]) -> int | None:
        best_idx = None
        best_sim = 0.0
        for i, slot in enumerate(self._slots):
            if slot.get("deleted"):
                continue
            sim = self._embedder.similarity(embedding, slot["embedding"])
            if sim >= self._threshold and sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        embedding = self._embedder.embed(claim)
        idx = self._find_similar(embedding)

        if idx is not None:
            slot = self._slots[idx]
            slot["claim"] = claim
            slot["embedding"] = embedding
            slot["evidence_count"] = slot.get("evidence_count", 0) + len(evidence)
            slot["confidence"] = min(0.5 + 0.1 * slot["evidence_count"], 0.95)
            return StoreResult(
                belief_id=slot["id"],
                merged=True,
                contradiction_detected=False,
                truth_state="true",
                confidence=slot["confidence"],
            )

        slot_id = str(uuid4())
        confidence = min(0.5 + 0.1 * len(evidence), 0.95)
        self._slots.append({
            "id": slot_id,
            "claim": claim,
            "embedding": embedding,
            "evidence_count": len(evidence),
            "confidence": confidence,
            "deleted": False,
        })
        return StoreResult(
            belief_id=slot_id,
            merged=False,
            contradiction_detected=False,
            truth_state="true",
            confidence=confidence,
        )

    def query(self, claim: str) -> list[QueryResult]:
        if not self._slots:
            return []
        embedding = self._embedder.embed(claim)
        results = []
        for slot in self._slots:
            if slot.get("deleted"):
                continue
            sim = self._embedder.similarity(embedding, slot["embedding"])
            if sim >= 0.5:
                results.append(QueryResult(
                    belief_id=slot["id"],
                    claim=slot["claim"],
                    confidence=slot["confidence"],
                    truth_state="true",
                ))
        return results

    def retract(self, evidence_id: str) -> RetractResult:
        for slot in self._slots:
            if slot["id"] == evidence_id:
                slot["deleted"] = True
                return RetractResult(affected_beliefs=1, truth_states_changed=1)
        return RetractResult(affected_beliefs=0, truth_states_changed=0)

    def explain(self, claim: str) -> ExplainResult:
        embedding = self._embedder.embed(claim)
        idx = self._find_similar(embedding)
        if idx is None:
            return ExplainResult(
                claim=claim,
                has_evidence=False,
                supporting_count=0,
                attacking_count=0,
                truth_state=None,
                confidence=None,
                expired_count=0,
            )
        slot = self._slots[idx]
        return ExplainResult(
            claim=slot["claim"],
            has_evidence=slot["evidence_count"] > 0,
            supporting_count=slot["evidence_count"],
            attacking_count=0,
            truth_state="true",
            confidence=slot["confidence"],
            expired_count=0,
        )

    def revise(self, belief_id: str, evidence: list[dict]) -> ReviseResult:
        for slot in self._slots:
            if slot["id"] == belief_id:
                slot["evidence_count"] = slot.get("evidence_count", 0) + len(evidence)
                slot["confidence"] = min(0.5 + 0.1 * slot["evidence_count"], 0.95)
                return ReviseResult(
                    belief_id=belief_id,
                    truth_state="true",
                    confidence=slot["confidence"],
                    superseded_count=0,
                )
        return ReviseResult(
            belief_id=belief_id,
            truth_state=None,
            confidence=None,
            superseded_count=0,
        )

    def reset(self) -> None:
        self._slots.clear()
