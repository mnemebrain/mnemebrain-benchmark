"""Real Mem0 API adapter for BMB benchmark.

Requires MEM0_API_KEY environment variable and mem0ai package.
"""

from __future__ import annotations

import logging
import os
import time
from uuid import uuid4

from mem0 import MemoryClient

from mnemebrain_benchmark.interface import (
    Capability,
    ExplainResult,
    MemorySystem,
    QueryResult,
    RetractResult,
    ReviseResult,
    StoreResult,
)

logger = logging.getLogger(__name__)


class Mem0Adapter(MemorySystem):
    """Real Mem0 API adapter for the BMB benchmark."""

    def __init__(self, api_key: str | None = None, store_delay: float = 1.5) -> None:
        key = api_key or os.environ.get("MEM0_API_KEY")
        if not key:
            raise ValueError("MEM0_API_KEY required")
        self._client = MemoryClient(api_key=key)
        self._store_delay = store_delay
        try:
            self._client.project.update(enable_graph=True)
        except Exception:
            logger.warning("Failed to enable graph mode", exc_info=True)
        self._user_id = f"bmb_{uuid4().hex[:12]}"
        self._store_to_memory: dict[str, str] = {}

    def name(self) -> str:
        return "mem0"

    def capabilities(self) -> set[Capability]:
        return {
            Capability.STORE,
            Capability.QUERY,
            Capability.RETRACT,
            Capability.EXPLAIN,
            Capability.REVISE,
        }

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        content = claim
        if evidence:
            details = "; ".join(e.get("content", "") for e in evidence)
            content = f"{claim}. Evidence: {details}"

        result = self._client.add(
            messages=[{"role": "user", "content": content}],
            user_id=self._user_id,
        )

        time.sleep(self._store_delay)

        memory_id = None
        if isinstance(result, dict) and "results" in result:
            results = result["results"]
            if results:
                memory_id = results[0].get("id") or results[0].get("memory_id")

        if not memory_id:
            search_result = self._client.search(
                query=claim,
                filters={"user_id": self._user_id},
                limit=1,
            )
            if isinstance(search_result, dict) and search_result.get("results"):
                memory_id = search_result["results"][0].get("id")

        store_id = str(uuid4())
        if memory_id:
            self._store_to_memory[store_id] = memory_id

        return StoreResult(
            belief_id=store_id,
            merged=False,
            contradiction_detected=False,
            truth_state="true",
            confidence=0.7,
        )

    def query(self, claim: str) -> list[QueryResult]:
        result = self._client.search(
            query=claim,
            filters={"user_id": self._user_id},
            limit=5,
        )

        results_list = []
        if isinstance(result, dict) and "results" in result:
            for item in result["results"]:
                memory_text = item.get("memory", "")
                memory_id = item.get("id", "")
                score = item.get("score", 0)
                if score >= 0.3:
                    results_list.append(
                        QueryResult(
                            belief_id=str(memory_id),
                            claim=memory_text,
                            confidence=0.7,
                            truth_state="true",
                        )
                    )

        return results_list

    def retract(self, belief_id: str) -> RetractResult:
        memory_id = self._store_to_memory.get(belief_id)
        if memory_id:
            try:
                self._client.delete(memory_id=memory_id)
                return RetractResult(affected_beliefs=1, truth_states_changed=1)
            except Exception:
                logger.warning("Failed to delete memory %s", memory_id, exc_info=True)
        return RetractResult(affected_beliefs=0, truth_states_changed=0)

    def explain(self, claim: str) -> ExplainResult:
        result = self._client.search(
            query=claim,
            filters={"user_id": self._user_id},
            limit=5,
        )

        memories = []
        if isinstance(result, dict) and "results" in result:
            memories = result["results"]

        if not memories:
            return ExplainResult(
                claim=claim,
                has_evidence=False,
                supporting_count=0,
                attacking_count=0,
                truth_state=None,
                confidence=None,
                expired_count=0,
            )

        return ExplainResult(
            claim=claim,
            has_evidence=True,
            supporting_count=len(memories),
            attacking_count=0,
            truth_state="true",
            confidence=0.7,
            expired_count=0,
        )

    def revise(self, belief_id: str, evidence: list[dict]) -> ReviseResult:
        memory_id = self._store_to_memory.get(belief_id)
        if memory_id:
            content = "; ".join(e.get("content", "") for e in evidence)
            try:
                self._client.update(memory_id=memory_id, data=content)
            except Exception:
                logger.warning("Failed to update memory, falling back to add", exc_info=True)
                self._client.add(
                    messages=[{"role": "user", "content": content}],
                    user_id=self._user_id,
                )

        return ReviseResult(
            belief_id=belief_id,
            truth_state="true",
            confidence=0.7,
            superseded_count=0,
        )

    def reset(self) -> None:
        try:
            self._client.delete_all(user_id=self._user_id)
        except Exception:
            logger.warning("Failed to delete all memories", exc_info=True)
        self._store_to_memory.clear()
        self._user_id = f"bmb_{uuid4().hex[:12]}"
