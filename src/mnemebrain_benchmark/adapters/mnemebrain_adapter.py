"""MnemeBrain adapter -- uses the mnemebrain Python SDK to talk to the backend."""
from __future__ import annotations

import os

from mnemebrain import MnemeBrainClient, EvidenceInput

from mnemebrain_benchmark.interface import (
    AttackResult,
    Capability,
    ConsolidateResult,
    ExplainResult,
    MemorySystem,
    MemoryTierResult,
    QueryResult,
    RetractResult,
    ReviseResult,
    SandboxResult,
    StoreResult,
)


class MnemeBrainAdapter(MemorySystem):
    """Adapter wrapping the MnemeBrain Python SDK for the system benchmark."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        url = base_url or os.environ.get("MNEMEBRAIN_URL", "http://localhost:8000")
        self._client = MnemeBrainClient(base_url=url, timeout=timeout)

    def name(self) -> str:
        return "mnemebrain"

    def capabilities(self) -> set[Capability]:
        return {
            Capability.STORE,
            Capability.QUERY,
            Capability.RETRACT,
            Capability.EXPLAIN,
            Capability.CONTRADICTION,
            Capability.DECAY,
            Capability.REVISE,
            Capability.SANDBOX,
            Capability.ATTACK,
            Capability.CONSOLIDATION,
            Capability.HIPPORAG,
            Capability.PATTERN_SEPARATION,
        }

    # -- Core operations --

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        ev_items = [
            EvidenceInput(
                source_ref=e.get("source_ref", "benchmark"),
                content=e.get("content", claim),
                polarity=e.get("polarity", "supports"),
                weight=float(e.get("weight", 0.7)),
                reliability=float(e.get("reliability", 0.8)),
            )
            for e in evidence
        ]
        belief_type = evidence[0].get("belief_type", "inference") if evidence else "inference"
        result = self._client.believe(
            claim=claim,
            evidence=ev_items,
            belief_type=belief_type,
        )
        return StoreResult(
            belief_id=result.id,
            merged=False,
            contradiction_detected=result.conflict,
            truth_state=result.truth_state,
            confidence=result.confidence,
        )

    def query(self, claim: str) -> list[QueryResult]:
        response = self._client.search(query=claim)
        return [
            QueryResult(
                belief_id=r.belief_id,
                claim=r.claim,
                confidence=r.confidence,
                truth_state=r.truth_state,
            )
            for r in response.results
        ]

    def retract(self, evidence_id: str) -> RetractResult:
        results = self._client.retract(evidence_id)
        return RetractResult(
            affected_beliefs=len(results),
            truth_states_changed=len(results),
        )

    def explain(self, claim: str) -> ExplainResult:
        result = self._client.explain(claim)
        if result is None:
            return ExplainResult(
                claim=claim,
                has_evidence=False,
                supporting_count=0,
                attacking_count=0,
                truth_state="neither",
                confidence=0.0,
            )
        return ExplainResult(
            claim=result.claim,
            has_evidence=len(result.supporting) + len(result.attacking) + len(result.expired) > 0,
            supporting_count=len(result.supporting),
            attacking_count=len(result.attacking),
            truth_state=result.truth_state,
            confidence=result.confidence,
            expired_count=len(result.expired),
        )

    def revise(self, belief_id: str, evidence: list[dict]) -> ReviseResult:
        ev_data = evidence[0] if evidence else {}
        ev_input = EvidenceInput(
            source_ref=ev_data.get("source_ref", "benchmark_revise"),
            content=ev_data.get("content", "revised evidence"),
            polarity=ev_data.get("polarity", "supports"),
            weight=float(ev_data.get("weight", 0.8)),
            reliability=float(ev_data.get("reliability", 0.9)),
        )
        result = self._client.revise(belief_id, ev_input)
        return ReviseResult(
            belief_id=result.id,
            truth_state=result.truth_state,
            confidence=result.confidence,
            superseded_count=0,
        )

    # -- Decay --

    def set_time_offset_days(self, days: int) -> None:
        self._client.set_time_offset(days)

    # -- Phase 5: Consolidation --

    def consolidate(self) -> ConsolidateResult:
        result = self._client.consolidate()
        return ConsolidateResult(
            semantic_beliefs_created=result.semantic_beliefs_created,
            episodics_pruned=result.episodics_pruned,
            clusters_found=result.clusters_found,
        )

    def get_memory_tier(self, belief_id: str) -> MemoryTierResult:
        result = self._client.get_memory_tier(belief_id)
        return MemoryTierResult(
            belief_id=result.belief_id,
            memory_tier=result.memory_tier,
            consolidated_from_count=result.consolidated_from_count,
        )

    # -- Phase 5: HippoRAG --

    def query_multihop(self, query: str) -> list[QueryResult]:
        response = self._client.query_multihop(query)
        return [
            QueryResult(
                belief_id=r.belief_id,
                claim=r.claim,
                confidence=r.confidence,
                truth_state=r.truth_state,
            )
            for r in response.results
        ]

    # -- Sandbox (benchmark) --

    def sandbox_fork(self, scenario_label: str = "") -> SandboxResult:
        result = self._client.benchmark_sandbox_fork(scenario_label)
        return SandboxResult(
            sandbox_id=result.sandbox_id,
            resolved_truth_state=result.resolved_truth_state,
            canonical_unchanged=result.canonical_unchanged,
        )

    def sandbox_assume(self, sandbox_id: str, belief_id: str, truth_state: str) -> SandboxResult:
        result = self._client.benchmark_sandbox_assume(sandbox_id, belief_id, truth_state)
        return SandboxResult(
            sandbox_id=result.sandbox_id,
            resolved_truth_state=result.resolved_truth_state,
            canonical_unchanged=result.canonical_unchanged,
        )

    def sandbox_resolve(self, sandbox_id: str, belief_id: str) -> SandboxResult:
        result = self._client.benchmark_sandbox_resolve(sandbox_id, belief_id)
        return SandboxResult(
            sandbox_id=result.sandbox_id,
            resolved_truth_state=result.resolved_truth_state,
            canonical_unchanged=result.canonical_unchanged,
        )

    def sandbox_discard(self, sandbox_id: str) -> None:
        self._client.benchmark_sandbox_discard(sandbox_id)

    # -- Attack (benchmark) --

    def add_attack(self, attacker_id: str, target_id: str, attack_type: str, weight: float) -> AttackResult:
        result = self._client.benchmark_attack(attacker_id, target_id, attack_type, weight)
        return AttackResult(
            edge_id=result.edge_id,
            attacker_id=result.attacker_id,
            target_id=result.target_id,
        )

    # -- Reset --

    def reset(self) -> None:
        self._client.reset()
