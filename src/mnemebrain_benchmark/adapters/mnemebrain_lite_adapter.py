"""MnemeBrain Lite adapter -- uses mnemebrain_core.BeliefMemory directly (no HTTP)."""

from __future__ import annotations

import shutil
import tempfile
from datetime import timedelta
from uuid import UUID

from mnemebrain_core.engine import compute_confidence, compute_truth_state
from mnemebrain_core.memory import BeliefMemory
from mnemebrain_core.models import BeliefType
from mnemebrain_core.providers.base import EmbeddingProvider as LiteEmbeddingProvider
from mnemebrain_core.providers.base import EvidenceInput

from mnemebrain_benchmark.interface import (
    Capability,
    ExplainResult,
    MemorySystem,
    QueryResult,
    RetractResult,
    ReviseResult,
    StoreResult,
)
from mnemebrain_benchmark.protocols import EmbeddingProvider as BenchmarkEmbeddingProvider


class _EmbedderBridge(LiteEmbeddingProvider):
    """Adapts benchmark EmbeddingProvider to mnemebrain-lite EmbeddingProvider."""

    def __init__(self, benchmark_embedder: BenchmarkEmbeddingProvider) -> None:
        self._inner = benchmark_embedder

    def embed(self, text: str) -> list[float]:
        return self._inner.embed(text)

    def similarity(self, a: list[float], b: list[float]) -> float:
        return self._inner.similarity(a, b)


class MnemeBrainLiteAdapter(MemorySystem):
    """Adapter wrapping mnemebrain-lite's BeliefMemory for the system benchmark.

    Uses the library directly (embedded Kuzu DB), no HTTP server required.
    Supports 7 of 12 capabilities — the core belief engine without
    backend-only features (sandbox, attack, consolidation, hipporag,
    pattern separation).
    """

    def __init__(self, embedder: BenchmarkEmbeddingProvider) -> None:
        self._embedder = embedder
        self._db_dir: str | None = None
        self._memory: BeliefMemory = None  # set in _init_memory
        self._time_offset_days: int = 0
        self._init_memory()

    def _init_memory(self) -> None:
        parent = tempfile.mkdtemp(prefix="mnemebrain_lite_bench_")
        self._db_dir = parent
        db_path = f"{parent}/kuzu_db"
        self._memory = BeliefMemory(
            db_path=db_path,
            embedding_provider=_EmbedderBridge(self._embedder),
        )
        self._time_offset_days = 0

    def name(self) -> str:
        return "mnemebrain_lite"

    def capabilities(self) -> set[Capability]:
        return {
            Capability.STORE,
            Capability.QUERY,
            Capability.RETRACT,
            Capability.EXPLAIN,
            Capability.CONTRADICTION,
            Capability.DECAY,
            Capability.REVISE,
        }

    # -- Core operations --

    @staticmethod
    def _normalize_polarity(raw: str) -> str:
        """Map variant polarity strings to 'supports' or 'attacks'."""
        raw_lower = raw.lower()
        if raw_lower in ("supports", "supporting", "positive", "support"):
            return "supports"
        if raw_lower in ("attacks", "attacking", "negative", "attack"):
            return "attacks"
        return raw_lower  # let validation catch unknown values

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        ev_items = [
            EvidenceInput(
                source_ref=e.get("source_ref", "benchmark"),
                content=e.get("content", claim),
                polarity=self._normalize_polarity(e.get("polarity", "supports")),
                weight=float(e.get("weight", 0.7)),
                reliability=float(e.get("reliability", 0.8)),
            )
            for e in evidence
        ]
        belief_type_str = evidence[0].get("belief_type", "inference") if evidence else "inference"
        belief_type = BeliefType(belief_type_str)

        result = self._memory.believe(
            claim=claim,
            evidence_items=ev_items,
            belief_type=belief_type,
        )
        return StoreResult(
            belief_id=str(result.id),
            merged=False,
            contradiction_detected=result.conflict,
            truth_state=result.truth_state.value,
            confidence=result.confidence,
        )

    def query(self, claim: str) -> list[QueryResult]:
        results = self._memory.search(query=claim, limit=10)
        return [
            QueryResult(
                belief_id=str(belief.id),
                claim=belief.claim,
                confidence=belief.confidence,
                truth_state=belief.truth_state.value,
            )
            for belief, _sim, _conf, _rank in results
        ]

    def retract(self, belief_id: str) -> RetractResult:
        # The benchmark runner passes belief_id (from StoreResult).
        # mnemebrain-lite's retract needs an actual evidence UUID.
        # Try as belief_id first: find all evidence and retract them.
        uid = UUID(belief_id)
        belief = self._memory._store.get(uid)
        if belief is not None:
            count = 0
            for ev in belief.evidence:
                if ev.valid:
                    self._memory.retract(ev.id)
                    count += 1
            return RetractResult(
                affected_beliefs=1 if count > 0 else 0,
                truth_states_changed=1 if count > 0 else 0,
            )
        # Fall back: try as actual evidence UUID
        results = self._memory.retract(uid)
        return RetractResult(
            affected_beliefs=len(results),
            truth_states_changed=len(results),
        )

    def explain(self, claim: str) -> ExplainResult:
        result = self._memory.explain(claim)
        if result is None:
            return ExplainResult(
                claim=claim,
                has_evidence=False,
                supporting_count=0,
                attacking_count=0,
                truth_state="neither",
                confidence=0.0,
                expired_count=0,
            )
        return ExplainResult(
            claim=result.claim,
            has_evidence=(len(result.supporting) + len(result.attacking) + len(result.expired) > 0),
            supporting_count=len(result.supporting),
            attacking_count=len(result.attacking),
            truth_state=result.truth_state.value,
            confidence=result.confidence,
            expired_count=len(result.expired),
        )

    def revise(self, belief_id: str, evidence: list[dict]) -> ReviseResult:
        ev_data = evidence[0] if evidence else {}
        ev_input = EvidenceInput(
            source_ref=ev_data.get("source_ref", "benchmark_revise"),
            content=ev_data.get("content", "revised evidence"),
            polarity=self._normalize_polarity(ev_data.get("polarity", "supports")),
            weight=float(ev_data.get("weight", 0.8)),
            reliability=float(ev_data.get("reliability", 0.9)),
        )
        result = self._memory.revise(UUID(belief_id), ev_input)
        return ReviseResult(
            belief_id=str(result.id),
            truth_state=result.truth_state.value,
            confidence=result.confidence,
            superseded_count=0,
        )

    # -- Decay --

    def set_time_offset_days(self, days: int) -> None:
        """Simulate time passage by shifting evidence timestamps backward.

        The engine computes decay based on (now - evidence.timestamp).
        Shifting timestamps backward makes evidence appear older.
        """
        additional = days - self._time_offset_days
        if additional <= 0:
            return
        self._time_offset_days = days
        delta = timedelta(days=additional)

        beliefs, _ = self._memory.list_beliefs(limit=1000)
        for belief in beliefs:
            # Retrieve the stored embedding so we don't lose it on upsert
            emb = self._get_stored_embedding(str(belief.id))

            for ev in belief.evidence:
                ev.timestamp = ev.timestamp - delta
            belief.truth_state = compute_truth_state(belief.evidence, belief.belief_type)
            belief.confidence = compute_confidence(belief.evidence, belief.belief_type)
            self._memory._store.upsert(belief, embedding=emb)

    def _get_stored_embedding(self, belief_id: str) -> list[float] | None:
        """Retrieve the embedding for a belief from the store."""
        result = self._memory._store._conn.execute(
            "MATCH (b:Belief {id: $id}) RETURN b.embedding",
            parameters={"id": belief_id},
        )
        if result.has_next():
            row = result.get_next()
            emb = row[0]
            if emb and len(emb) > 0:
                return list(emb)
        return None

    # -- Reset --

    def reset(self) -> None:
        if self._memory is not None:
            self._memory.close()
        if self._db_dir is not None:
            shutil.rmtree(self._db_dir, ignore_errors=True)
        self._init_memory()
