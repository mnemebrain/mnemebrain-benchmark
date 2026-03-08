"""OpenAI embeddings RAG adapter for BMB benchmark.

Requires OPENAI_API_KEY environment variable and openai package.
"""

from __future__ import annotations

import os
from uuid import uuid4

import numpy as np
from openai import OpenAI

from mnemebrain_benchmark.interface import (
    Capability,
    MemorySystem,
    QueryResult,
    StoreResult,
)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class OpenAIRAGAdapter(MemorySystem):
    """RAG baseline using real OpenAI embeddings."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        threshold: float = 0.80,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY required")
        self._client = OpenAI(api_key=key)
        self._model = model
        self._threshold = threshold
        self._entries: list[dict] = []

    def name(self) -> str:
        return "openai_rag"

    def capabilities(self) -> set[Capability]:
        return {Capability.STORE, Capability.QUERY}

    def _embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=[text],
            model=self._model,
        )
        return list(response.data[0].embedding)

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        embedding = self._embed(claim)
        for i, entry in enumerate(self._entries):
            sim = _cosine_sim(embedding, entry["embedding"])
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
        embedding = self._embed(claim)
        results = []
        for entry in self._entries:
            sim = _cosine_sim(embedding, entry["embedding"])
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
